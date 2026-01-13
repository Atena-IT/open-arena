import litellm
import logging
import asyncio
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.sse import sse_client
from litellm import experimental_mcp_client

from src.llms.types import MCPServerConfig

load_dotenv()
_logger = logging.getLogger(__name__)

class LLMClient:
    """
    Client for interacting with LLMs using LiteLLM.
    Supports MCP tools from remote servers via SSE.
    """

    @staticmethod
    def format_messages(system: str, user: str) -> List[Dict[str, str]]:
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


    async def _load_mcp_tools(self, mcp_servers: List[MCPServerConfig]) -> List[dict]:
        """
        Load tools from remote MCP servers via SSE.
        
        :param mcp_servers: List of MCP server configurations with 'url' and optional 'headers'
        :return: List of tools in OpenAI format
        """
        all_tools = []
        
        for server_config in mcp_servers:
            server_name = server_config.get("server_name", "unknown")
            url = server_config["url"]
            headers = server_config.get("headers", {})
            
            _logger.debug(f"Loading tools from remote MCP server: {server_name} at {url}")
            
            try:
                async with sse_client(url, headers=headers) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        
                        tools = await experimental_mcp_client.load_mcp_tools(
                            session=session,
                            format="openai"
                        )
                        
                        _logger.info(f"Loaded {len(tools)} tools from {server_name}")

                        all_tools.extend(tools)
                        
            except Exception as e:
                _logger.error(f"Failed to load tools from {server_name}: {e}")
                continue
        
        return all_tools


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
        completion_args = llm_config.copy()
        completion_args["messages"] = messages
        
        if mcp_servers:
            tools = await self._load_mcp_tools(mcp_servers)
            
            if tools:
                completion_args["tools"] = tools
                completion_args["tool_choice"] = llm_config.get("tool_choice", "auto")
        
        response = await litellm.acompletion(**completion_args)

        return response.choices[0].message.content # type: ignore # TODO: type checking
    
    # TODO: astream

    def chat(
        self, 
        messages: List[Dict[str, str]], 
        llm_config: Dict[str, Any],
        mcp_servers: Optional[List[MCPServerConfig]] = None
    ) -> str:
        """
        Synchronous chat completion with optional MCP tools.
        Wrapper around async implementation.
        
        :param messages: List of message dicts
        :param llm_config: Model configuration
        :param mcp_servers: Optional list of remote MCP server configurations
        :return: Model response content
        """
        return asyncio.run(self.achat(messages, llm_config, mcp_servers))
