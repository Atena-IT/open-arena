import json
import logging
import os
from typing import Any

import litellm
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_litellm import ChatLiteLLM
from langchain_mcp_adapters.client import MultiServerMCPClient
from langfuse import Langfuse
from langgraph.graph import MessagesState, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src.llms.types import MCPServerConfig

load_dotenv()
_logger = logging.getLogger(__name__)


class LLMClient:
    """
    Client for interacting with LLMs.

    Supports both the legacy LiteLLM sync API used by `src.main` and the
    newer async LangGraph-based flow used by the structured CLI.
    """

    def __init__(
        self,
        llm_config: dict[str, Any] | None = None,
        mcp_servers: list[MCPServerConfig] | None = None,
    ):
        self.llm_config = llm_config or {}
        self.mcp_servers = mcp_servers if mcp_servers is not None else []
        self.graph = None
        self.mcp_client = None
        self._initialized = False
        self.langfuse = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
            host=os.getenv("LANGFUSE_HOST", ""),
        )

        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "").strip()
        os.environ["HUGGINGFACE_API_KEY"] = os.getenv("HUGGINGFACE_API_KEY", "").strip()

    async def setup(self):
        """
        Initialize the async graph-based client.
        """
        if not self.llm_config:
            raise ValueError("llm_config is required to initialize the graph-based client")

        self.graph, self.mcp_client = await self._create_graph_with_tools(
            self.llm_config,
            self.mcp_servers,
        )
        self._initialized = True
        _logger.debug("LLM client initialized")

    @staticmethod
    def format_messages(system: str, user: str) -> list[dict[str, str]]:
        """
        Format system and user content into a standard message list.
        """
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def chat(self, messages: list[dict[str, Any]], model_config: dict[str, Any]) -> str:
        """
        Legacy synchronous LiteLLM chat API used by `src.main`.
        """
        completion_kwargs: dict[str, Any] = {
            "messages": messages,
            "model": model_config.get("name") or model_config.get("model"),
        }

        for key in (
            "max_tokens",
            "response_format",
            "stream",
            "temperature",
            "tools",
            "tool_choice",
            "top_p",
        ):
            if key in model_config and model_config[key] is not None:
                completion_kwargs[key] = model_config[key]

        response = litellm.completion(**completion_kwargs)
        content = response.choices[0].message.content or ""

        try:
            self.langfuse.create_event(
                name="llm_completion",
                input=messages,
                output=content,
                metadata={"model": completion_kwargs["model"]},
            )
            self.langfuse.flush()
        except Exception:
            _logger.debug("Failed to trace sync completion to Langfuse", exc_info=True)

        return content

    async def chat_with_mcp_tools(
        self,
        messages: list[dict[str, Any]],
        model_config: dict[str, Any],
        mcp_session,
        mcp_tools_openai: list,
        max_steps: int = 8,
    ) -> str:
        """
        Legacy async LiteLLM + MCP tool-calling loop used by `src.main`.
        """
        tool_calls_log: list[dict[str, str]] = []
        for step in range(max_steps):
            completion_kwargs: dict[str, Any] = {
                "messages": messages,
                "model": model_config.get("name") or model_config.get("model"),
                "tools": mcp_tools_openai,
            }

            for key in ("max_tokens", "response_format", "stream", "temperature", "tool_choice", "top_p"):
                if key in model_config and model_config[key] is not None:
                    completion_kwargs[key] = model_config[key]

            response = await litellm.acompletion(**completion_kwargs)
            msg = response.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None) or []

            if not tool_calls:
                final_text = msg.content or ""
                try:
                    self.langfuse.create_event(
                        name="llm_response",
                        input=messages,
                        output={"content": final_text, "tool_calls": tool_calls_log},
                        metadata={"model": completion_kwargs["model"], "steps": step},
                    )
                    self.langfuse.flush()
                except Exception:
                    _logger.debug("Failed to trace MCP completion to Langfuse", exc_info=True)
                return final_text

            messages.append({"role": "assistant", "content": msg.content, "tool_calls": tool_calls})
            for call in tool_calls:
                function_name = call.function.name
                args_raw = call.function.arguments or "{}"
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
                except json.JSONDecodeError:
                    args = {}

                result = await mcp_session.call_tool(function_name, args)
                tool_calls_log.append({"tool": function_name, "content": str(result.content)})
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": function_name,
                        "content": str(result.content),
                    }
                )

        self.langfuse.flush()
        raise RuntimeError(f"Tool loop exceeded max_steps={max_steps}")

    async def _create_graph_with_tools(
        self,
        llm_config: dict[str, Any],
        mcp_servers: list[MCPServerConfig],
    ):
        """
        Create the LangGraph workflow used by the structured CLI.
        """
        model = ChatLiteLLM(**llm_config)
        tools = []
        mcp_client = None

        if mcp_servers:
            servers_config = {}
            for server in mcp_servers:
                server_name = server.get("server_name", f"server_{len(servers_config)}")
                servers_config[server_name] = {
                    "transport": "sse",
                    "url": server["url"],
                    **({"headers": server["headers"]} if "headers" in server else {}),
                }

            mcp_client = MultiServerMCPClient(servers_config)
            tools = await mcp_client.get_tools()
            _logger.debug("Loaded %s tools from MCP servers", len(tools))

        async def call_model(state: MessagesState):
            if tools:
                response = await model.bind_tools(tools).ainvoke(state["messages"])
            else:
                response = await model.ainvoke(state["messages"])
            return {"messages": response}

        builder = StateGraph(MessagesState)
        builder.add_node("call_model", call_model)
        builder.add_edge(START, "call_model")

        if tools:
            builder.add_node("tools", ToolNode(tools))
            builder.add_conditional_edges("call_model", tools_condition)
            builder.add_edge("tools", "call_model")

        return builder.compile(), mcp_client

    def _convert_messages_to_langchain(
        self,
        messages: list[dict[str, str]],
    ) -> list[Any]:
        """
        Convert standard message dictionaries to LangChain message objects.
        """
        langchain_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                langchain_messages.append(SystemMessage(content=content))
            elif role == "user":
                langchain_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                langchain_messages.append(AIMessage(content=content))
            else:
                _logger.warning("Unknown role '%s', treating as user message", role)
                langchain_messages.append(HumanMessage(content=content))

        return langchain_messages

    async def achat(
        self,
        messages: list[dict[str, str]],
    ) -> str:
        """
        Async chat completion for the structured CLI path.
        """
        if not self._initialized or not self.graph:
            raise RuntimeError("LLMClient not initialized. Call setup() before making requests.")

        langchain_messages = self._convert_messages_to_langchain(messages)
        result = await self.graph.ainvoke({"messages": langchain_messages})

        final_message = result["messages"][-1]
        if hasattr(final_message, "content") and final_message.content:
            return str(final_message.content)
        return ""
