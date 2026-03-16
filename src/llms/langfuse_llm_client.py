from typing import Any

from langfuse.langchain import CallbackHandler

from src.llms.llm_client import LLMClient
from src.llms.types import MCPServerConfig

class LangfuseLLMClient(LLMClient):
    """
    LLM Client with Langfuse observability integration.

    Inherits all functionality from LLMClient and adds automatic
    tracing of LLM calls to Langfuse via LangChain callbacks.
    """

    def __init__(
        self,
        llm_config: dict[str, Any],
        mcp_servers: list[MCPServerConfig] | None = None
    ):
        """
        Initialize LangfuseLLMClient with Langfuse observability.

        :param llm_config: LiteLLM model configuration
        :param mcp_servers: List of MCP server configurations
        """
        super().__init__(llm_config, mcp_servers)
        self.langfuse_handler = CallbackHandler()

    async def achat(
        self,
        messages: list[dict[str, str]]
    ) -> str:
        """
        Async chat completion with Langfuse tracing.

        :param messages: List of message dicts
        :return: Model response content
        """
        if self._initialized and self.graph:
            langchain_messages = self._convert_messages_to_langchain(messages)

            # Invoke graph with Langfuse callback
            result = await self.graph.ainvoke(
                {"messages": langchain_messages},
                config={"callbacks": [self.langfuse_handler]}
            )

            final_message = result["messages"][-1]
            if hasattr(final_message, "content") and final_message.content:
                return str(final_message.content)
            else:
                return ""
        else:
            raise RuntimeError("LLMClient not initialized. Call setup() before making requests.")
