from dotenv import load_dotenv
from typing import List, Dict, Optional, Any

from langfuse import get_client, observe

from src.llms.llm_client import LLMClient
from src.llms.types import MCPServerConfig

load_dotenv()

class LangfuseLLMClient(LLMClient):
    """
    LLM Client with Langfuse observability integration.
    
    Inherits all functionality from LLMClient and adds automatic
    tracing of LLM calls to Langfuse via LiteLLM callbacks.
    """
    
    def __init__(
        self
    ):
        """
        Initialize LangfuseLLMClient with Langfuse observability.
        """
        super().__init__()

        self._langfuse = get_client()

    @observe(as_type="generation", name="litellm")
    async def achat(
        self, 
        messages: List[Dict[str, str]], 
        llm_config: Dict[str, Any],
        mcp_servers: Optional[List[MCPServerConfig]] = None
    ) -> str:
        """
        Async chat completion with optional MCP tools.
        
        :param messages: List of message dicts
        :param llm_config: Model configuration
        :param mcp_servers: Optional list of remote MCP server configurations
        :return: Model response content
        """
        result = await super().achat(
            messages=messages,
            llm_config=llm_config,
            mcp_servers=mcp_servers
        )

        self._langfuse.update_current_generation(
            model=llm_config["model"],
        )

        return result
