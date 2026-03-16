import logging
from typing import Any

from langchain_litellm import ChatLiteLLM
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_mcp_adapters.client import MultiServerMCPClient

from src.llms.types import MCPServerConfig

_logger = logging.getLogger(__name__)

class LLMClient:
    """
    Client for interacting with LLMs using LiteLLM and Langgraph.
    Supports MCP tools from remote servers via SSE.
    """

    def __init__(
        self,
        llm_config: dict[str, Any],
        mcp_servers: list[MCPServerConfig] | None = None
    ):
        """
        Initialize the LLM client.
        
        :param llm_config: LiteLLM model configuration
        :param mcp_servers: Optional list of MCP server configurations
        """
        self.llm_config = llm_config
        self.mcp_servers = mcp_servers if mcp_servers is not None else []
        self.graph = None
        self.mcp_client = None
        self._initialized = False

    async def setup(self):
        """
        Initialize MCP connection and build the graph.
        Call this once after instantiation.
        """
        if self.llm_config:
            self.graph, self.mcp_client = await self._create_graph_with_tools(
                self.llm_config, 
                self.mcp_servers
            )
            self._initialized = True
            _logger.debug("LLM client initialized with MCP tools")

    @staticmethod
    def format_messages(system: str, user: str) -> list[dict[str, str]]:
        """
        Formats messages for chat completion.
        
        :param system: System prompt
        :param user: User message
        :return: Formatted messages list
        """
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    async def _create_graph_with_tools(
        self, 
        llm_config: dict[str, Any],
        mcp_servers: list[MCPServerConfig]
    ):
        """
        Create a LangGraph workflow with MCP tools.
        
        :param llm_config: Model configuration
        :param mcp_servers: List of MCP server configurations
        :return: Compiled graph and MCP client
        """
        # Convert MCP server configs to MultiServerMCPClient format
        servers_config = {}
        for server in mcp_servers:
            server_name = server.get("server_name", f"server_{len(servers_config)}")
            servers_config[server_name] = {
                "transport": "sse",
                "url": server["url"],
                **({"headers": server["headers"]} if "headers" in server else {})
            }
        
        mcp_client = MultiServerMCPClient(servers_config)
        tools = await mcp_client.get_tools()
        
        _logger.debug(f"Loaded {len(tools)} tools from MCP servers")
        
        model = ChatLiteLLM(**llm_config)
        
        # Build the graph
        async def call_model(state: MessagesState):
            response = await model.bind_tools(tools).ainvoke(state["messages"])
            return {"messages": response}
        
        builder = StateGraph(MessagesState)
        builder.add_node("call_model", call_model)
        builder.add_node("tools", ToolNode(tools))
        builder.add_edge(START, "call_model")
        builder.add_conditional_edges("call_model", tools_condition)
        builder.add_edge("tools", "call_model")
        
        return builder.compile(), mcp_client

    def _convert_messages_to_langchain(
        self, 
        messages: list[dict[str, str]]
    ) -> list[Any]:
        """
        Convert standard message format to LangChain messages.
        
        :param messages: List of message dicts with 'role' and 'content'
        :return: List of LangChain message objects
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
                _logger.warning(f"Unknown role '{role}', treating as user message")
                langchain_messages.append(HumanMessage(content=content))
        
        return langchain_messages

    async def achat(
        self,
        messages: list[dict[str, str]],
    ) -> str:
        """
        Async chat completion with optional MCP tools.
        
        :param messages: List of message dicts
        :return: Model response content
        """
        
        if self._initialized and self.graph:
            langchain_messages = self._convert_messages_to_langchain(messages)
            
            result = await self.graph.ainvoke({"messages": langchain_messages})
            
            final_message = result["messages"][-1]
            if hasattr(final_message, "content") and final_message.content:
                return str(final_message.content)
            else:
                return ""
        else:
            raise RuntimeError("LLMClient not initialized. Call setup() before making requests.")
    
    # TODO: astream
