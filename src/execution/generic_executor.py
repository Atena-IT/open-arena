import asyncio
from typing import TypeVar
from tqdm.asyncio import tqdm as async_tqdm

from src.llms import LLMClient
from src.datasets.item_models import DatasetItem
from src.execution.base_executor import Executor
from src.execution.types import ExecutionResult

T = TypeVar('T', bound=DatasetItem)


class GenericExecutor(Executor[T]):
    """
    Generic executor for running LLM completions on dataset items.
    """
    
    def __init__(
        self,
        dataset: list[T],
        llm_client: LLMClient,
        system_prompt: str,
        max_concurrency: int = 50
    ):
        """
        :param dataset: List of items to execute
        :param llm_client: LLM client for completions
        :param system_prompt: System prompt for all completions
        :param max_concurrency: Maximum number of concurrent executions
        """
        super().__init__(
            dataset=dataset,
            llm_client=llm_client,
            system_prompt=system_prompt
        )

        self.dataset = dataset
        self.max_concurrency = max_concurrency
    
    async def execute(self) -> list[ExecutionResult[T]]:
        """
        Execute all items in the dataset in parallel.
        
        :return: List of execution results
        """  
        semaphore = asyncio.Semaphore(self.max_concurrency)
        
        async def execute_with_semaphore(item: T) -> ExecutionResult[T]:
            async with semaphore:
                return await self._execute_item(item)
        
        tasks = [execute_with_semaphore(item) for item in self.dataset]
        
        results = []
        for coro in async_tqdm.as_completed(tasks, total=len(tasks), desc="Executing items"):
            result = await coro
            results.append(result)
        
        return results
