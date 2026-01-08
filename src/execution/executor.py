import os
import logging
from typing import List, Callable, Optional, TypeVar, Generic, Dict, Any
from pydantic import BaseModel

from src.llms import LLMClient
from src.execution.types import ExecutionResult

_logger = logging.getLogger(__name__)
T = TypeVar('T', bound=BaseModel)


class Executor(Generic[T]):
    """
    Generic executor for running LLM completions on dataset items.
    Supports optional MCP tools via LiteLLM's built-in bridge.
    """
    
    def __init__(
        self,
        items: List[T],
        user_prompt_fn: Callable[[T], str],
        llm_client: LLMClient,
        system_prompt: str,
        model_config: Dict[str, Any],
        mcp_servers: Optional[List[Dict[str, Any]]] = None
    ):
        """
        :param items: List of dataset items to execute
        :param user_prompt_fn: Function to extract user prompt from each item
        :param llm_client: LLM client for completions
        :param system_prompt: System prompt for all completions
        :param model_config: Model configuration to use
        :param mcp_servers: Optional list of MCP server configurations (LiteLLM format)
        """
        self.items = items
        self.prompt_fn = user_prompt_fn
        self.client = llm_client
        self.system_prompt = system_prompt
        self.model_config = model_config
        self.mcp_servers = mcp_servers
        
        if self.mcp_servers:
            _logger.info(f"Executor configured with {len(self.mcp_servers)} MCP server(s)")
    
    def execute(self) -> List[ExecutionResult[T]]:
        """
        Execute all items in the dataset.
        
        :return: List of execution results
        """
        results = []
        
        for item in self.items:
            try:
                user_prompt = self.prompt_fn(item)
                messages = self.client.format_messages(
                    system=self.system_prompt,
                    user=user_prompt
                )
                
                output = self.client.chat(
                    messages=messages,
                    model_config=self.model_config,
                    mcp_servers=self.mcp_servers
                )
                
                results.append(ExecutionResult(
                    item=item,
                    output=output,
                    model_name=self.model_config["name"]
                ))
            except Exception as e:
                _logger.error(f"Execution failed for item: {e}")
                results.append(ExecutionResult(
                    item=item,
                    output="",
                    model_name=self.model_config["name"],
                    error=str(e)
                ))
        
        return results
