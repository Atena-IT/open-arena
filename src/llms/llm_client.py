import litellm
import os
import logging
import asyncio
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.sse import sse_client

from litellm import experimental_mcp_client, Tool

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


    async def _load_mcp_tools(self, mcp_servers: List[Dict[str, Any]]) -> List[dict]: # TODO: fix types
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
                        # Initialize the connection
                        await session.initialize()
                        
                        # Load tools from this server
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
        model_config: Dict[str, Any],
        mcp_servers: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Async chat completion with optional MCP tools.
        
        :param messages: List of message dicts
        :param model_config: Model configuration
        :param mcp_servers: Optional list of remote MCP server configurations
        :return: Model response content
        """
        # Build completion arguments
        completion_args = {
            "model": model_config["name"],
            "messages": messages,
            "max_tokens": model_config.get("max_tokens", 500),
            "temperature": model_config.get("temperature", 0.7),
            "stream": model_config.get("stream", False),
        }
        
        # Add response format if specified
        if "response_format" in model_config:
            completion_args["response_format"] = model_config["response_format"]
        
        # Load and add MCP tools if servers are provided
        if mcp_servers:
            _logger.info(f"Loading tools from {len(mcp_servers)} remote MCP server(s)")
            tools = await self._load_mcp_tools(mcp_servers)
            
            if tools:
                completion_args["tools"] = tools
                completion_args["tool_choice"] = model_config.get("tool_choice", "auto")
                _logger.info(f"Using {len(tools)} tools from MCP servers")
        
        response = await litellm.acompletion(**completion_args)

        return response.choices[0].message.content # TODO: type checking


    def chat(
        self, 
        messages: List[Dict[str, str]], 
        model_config: dict,
        mcp_servers: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Synchronous chat completion with optional MCP tools.
        Wrapper around async implementation.
        
        :param messages: List of message dicts
        :param model_config: Model configuration
        :param mcp_servers: Optional list of remote MCP server configurations
        :return: Model response content
        """
        return asyncio.run(self.achat(messages, model_config, mcp_servers))
