import asyncio
from typing import List, Optional, TypeVar, Dict, Any
from tqdm.asyncio import tqdm as async_tqdm

from src.llms import LLMClient
from src.datasets.item_models import DatasetItem
from src.execution.base_executor import Executor
from src.execution.types import ExecutionResult
from src.llms.types import MCPServerConfig

T = TypeVar('T', bound=DatasetItem)


class GenericExecutor(Executor[T]):
    """
    Generic executor for running LLM completions on dataset items.
    """
    
    def __init__(
        self,
        dataset: List[T],
        llm_client: LLMClient,
        system_prompt: str,
        llm_config: Dict[str, Any],
        mcp_servers: Optional[List[MCPServerConfig]] = None,
        max_concurrency: int = 50
    ):
        """
        :param dataset: List of items to execute
        :param llm_client: LLM client for completions
        :param system_prompt: System prompt for all completions
        :param llm_config: Model configuration to use
        :param mcp_servers: Optional list of MCP server configurations (LiteLLM format)
        :param max_workers: Maximum number of concurrent executions
        """
        super().__init__(
            llm_client=llm_client,
            system_prompt=system_prompt,
            llm_config=llm_config,
            mcp_servers=mcp_servers
        )

        self.dataset = dataset
        self.max_concurrency = max_concurrency
    
    async def execute(self) -> List[ExecutionResult[T]]:
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
