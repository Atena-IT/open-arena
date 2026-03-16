from abc import ABC, abstractmethod
import logging
from typing import Generic, TypeVar

from src.llms import LLMClient
from src.datasets.item_models import DatasetItem
from src.execution.types import ExecutionResult

_logger = logging.getLogger(__name__)
T = TypeVar('T', bound=DatasetItem)

class Executor(ABC, Generic[T]):
    """
    Abstract base class for all executors.
    Defines the common interface that all executor implementations must follow.
    """

    def __init__(
        self,
        dataset: list[T],
        llm_client: LLMClient,
        system_prompt: str,
    ):
        """
        :param dataset: List of dataset items to execute
        :param llm_client: LLM client for completions
        :param system_prompt: System prompt for all completions
        """
        self.dataset = dataset
        self.client = llm_client
        self.system_prompt = system_prompt

    async def _execute_item(self, item: T) -> ExecutionResult[T]:
        """
        Execute a single dataset item.
        
        :param item: Dataset item to execute
        :return: ExecutionResult containing item, output, and model name
        """
        try:
            user_input = item.input()
            messages = self.client.format_messages(
                system=self.system_prompt,
                user=user_input
            )
            
            output = await self.client.achat(
                messages=messages,
            )
            
            return ExecutionResult(
                item=item,
                output=output,
                model_name=self.client.llm_config["model"]
            )
        
        except Exception as e:
            _logger.error(f"Execution failed for item: {e}")
            return ExecutionResult(
                item=item,
                output="",
                model_name=self.client.llm_config["model"],
                error=str(e)
            )
    
    @abstractmethod
    async def execute(self) -> list[ExecutionResult[T]]:
        """
        Execute the task on the dataset.
        """
        pass