import logging
from typing import List, Optional, TypeVar, Dict, Any

from src.llms import LLMClient
from src.datasets.item_models import DatasetItem
from src.execution import Executor, ExecutionResult
from src.llms.types import MCPServerConfig

_logger = logging.getLogger(__name__)
T = TypeVar('T', bound=DatasetItem)


class GenericExecutor(Executor[T]):
    """
    Generic executor for running LLM completions on dataset items.
    Supports optional MCP tools via LiteLLM's built-in bridge.
    """
    
    def __init__(
        self,
        dataset: List[T],
        llm_client: LLMClient,
        system_prompt: str,
        model_config: Dict[str, Any],
        mcp_servers: Optional[List[MCPServerConfig]]
    ):
        """
        :param dataset: List of items to execute
        :param llm_client: LLM client for completions
        :param system_prompt: System prompt for all completions
        :param model_config: Model configuration to use
        :param mcp_servers: Optional list of MCP server configurations (LiteLLM format)
        """
        self.dataset = dataset
        self.client = llm_client
        self.system_prompt = system_prompt
        self.model_config = model_config
        self.mcp_servers = mcp_servers
        
        if self.mcp_servers:
            _logger.info(f"Executor configured with {len(self.mcp_servers)} MCP server(s)")

    def _execute_item(self, item: T) -> ExecutionResult[T]:
        """
        Execute a single dataset item.
        
        :param item: Dataset item to execute
        :return: ExecutionResult containing item, output, and model name
        """
        user_prompt = item.user_prompt()
        messages = self.client.format_messages(
            system=self.system_prompt,
            user=user_prompt
        )
        
        output = self.client.chat(
            messages=messages,
            model_config=self.model_config,
            mcp_servers=self.mcp_servers
        )
        
        return ExecutionResult(
            item=item,
            output=output,
            model_name=self.model_config["name"]
        )
    
    def execute(self) -> List[ExecutionResult[T]]:
        """
        Execute all items in the dataset.
        
        :return: List of execution results
        """
        results = []
        
        for item in self.dataset:
            try:
                result = self._execute_item(item)
                results.append(result)
            except Exception as e:
                _logger.error(f"Execution failed for item: {e}")
                results.append(ExecutionResult(
                    item=item,
                    output="",
                    model_name=self.model_config["name"],
                    error=str(e)
                ))
        
        return results
