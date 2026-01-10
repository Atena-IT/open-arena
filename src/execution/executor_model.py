from abc import ABC, abstractmethod
import logging
from typing import List, Generic, Optional, TypeVar, Dict, Any

from src.llms import LLMClient
from src.datasets.item_models import DatasetItem
from src.execution.types import ExecutionResult
from src.llms.types import MCPServerConfig

_logger = logging.getLogger(__name__)
T = TypeVar('T', bound=DatasetItem)

class Executor(ABC, Generic[T]):
    """
    Abstract base class for all executors.
    Defines the common interface that all executor implementations must follow.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        system_prompt: str,
        model_config: Dict[str, Any],
        mcp_servers: Optional[List[MCPServerConfig]] = None
    ):
        """
        :param llm_client: LLM client for completions
        :param system_prompt: System prompt for all completions
        :param model_config: Model configuration to use
        :param mcp_servers: Optional list of MCP server configurations (LiteLLM format)
        """
        self.client = llm_client
        self.system_prompt = system_prompt
        self.model_config = model_config
        self.mcp_servers = mcp_servers

    async def _execute_item(self, item: T) -> ExecutionResult[T]:
        """
        Execute a single dataset item.
        
        :param item: Dataset item to execute
        :return: ExecutionResult containing item, output, and model name
        """
        try:
            user_prompt = item.user_prompt()
            messages = self.client.format_messages(
                system=self.system_prompt,
                user=user_prompt
            )
            
            output = await self.client.achat(
                messages=messages,
                model_config=self.model_config,
                mcp_servers=self.mcp_servers
            )
            
            return ExecutionResult(
                item=item,
                output=output,
                model_name=self.model_config["name"]
            )
        except Exception as e:
            _logger.error(f"Execution failed for item: {e}")
            return ExecutionResult(
                item=item,
                output="",
                model_name=self.model_config["name"],
                error=str(e)
            )
    
    @abstractmethod
    async def execute(self) -> List[ExecutionResult[T]]:
        """
        Execute the task on the dataset.
        """
        pass