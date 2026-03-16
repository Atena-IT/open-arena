import asyncio
import logging
from typing import TypeVar
from datetime import datetime, timezone
from tqdm.asyncio import tqdm as async_tqdm

from langfuse import get_client

from src.execution.base_executor import Executor
from src.execution.types import ExecutionResult
from src.llms import LangfuseLLMClient
from src.datasets.item_models import DatasetItem

_logger = logging.getLogger(__name__)
T = TypeVar('T', bound=DatasetItem)


class LangfuseExecutor(Executor[T]):
    """
    Executor that runs experiments with Langfuse tracking.
    
    Iterates over local dataset items and creates Langfuse spans with proper
    experiment metadata.
    """
    
    def __init__(
        self,
        dataset: list[T],
        llm_client: LangfuseLLMClient,
        system_prompt: str,
        experiment_name: str | None = None,
        experiment_description: str | None = None,
        max_concurrency: int = 50
    ):
        """
        :param dataset: List of dataset items (must have lf_item_id, lf_dataset_id in metadata)
        :param llm_client: LLM client for completions
        :param system_prompt: System prompt for all completions
        :param experiment_name: Experiment name (auto-generated if None)
        :param experiment_description: Experiment description (auto-generated if None)
        :param max_concurrency: Maximum number of concurrent executions
        """
        super().__init__(
            dataset=dataset,
            llm_client=llm_client,
            system_prompt=system_prompt,
        )
        
        self.experiment_name = experiment_name or f"Experiment-{self.client.llm_config['model']}"
        self.experiment_description = experiment_description or f"Experiment with {self.client.llm_config['model']}"
        self._langfuse = get_client()
        self.max_concurrency = max_concurrency
    
    async def _execute_item_with_langfuse(
        self,
        item: T,
        experiment_run_name: str,
        dataset_id: str
    ) -> ExecutionResult[T]:
        """
        Execute a single item with Langfuse span tracking.
        
        :param item: Dataset item to execute
        :param experiment_run_name: Name of the experiment run
        :param dataset_id: Langfuse dataset ID
        :return: Execution result with Langfuse metadata
        """
        dataset_item_id = item.metadata.get("lf_item_id")
        
        if not dataset_item_id:
            raise ValueError(f"Item must have 'lf_item_id' in metadata")
        
        with self._langfuse.start_as_current_observation(
            as_type="span",
            name="experiment-item-run",
            input=item.input(),
            metadata={
                "experiment_name": self.experiment_name,
                "experiment_run_name": experiment_run_name,
                "dataset_id": dataset_id,
                "dataset_item_id": dataset_item_id,
            }
        ) as root_span:
            result = await self._execute_item(item)
            
            root_span.update(
                output=str(result.output) if result.output is not None else None,
                level="ERROR" if result.error else "DEFAULT"
            )
            
            try:
                self._langfuse.api.dataset_run_items.create(
                    run_name=experiment_run_name,
                    dataset_item_id=dataset_item_id,
                    trace_id=root_span.trace_id,
                    run_description=self.experiment_description,
                )
            except Exception as e:
                _logger.error(f"Failed to create dataset run item for {dataset_item_id}: {e}")
            
            result.metadata["lf_trace_id"] = root_span.trace_id
            result.metadata["lf_observation_id"] = root_span.id
            result.metadata["lf_experiment_name"] = self.experiment_name
            result.metadata["lf_experiment_run_name"] = experiment_run_name
            
            return result
    
    async def execute(self) -> list[ExecutionResult[T]]:
        """
        Execute all items in the dataset in parallel and track with Langfuse.
        
        :return: List of execution results with Langfuse metadata
        """
        if not self.dataset:
            _logger.warning("Dataset is empty, no items to execute")
            return []

        dataset_id = self.dataset[0].metadata.get("lf_dataset_id")
        
        if not dataset_id:
            raise ValueError("Dataset items must have 'lf_dataset_id' and 'lf_dataset_name' in metadata")
        
        experiment_run_name = f"{self.experiment_name} - {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S+00:00')}"
        
        semaphore = asyncio.Semaphore(self.max_concurrency)
        
        async def execute_with_semaphore(item: T) -> ExecutionResult[T]:
            async with semaphore:
                return await self._execute_item_with_langfuse(
                    item,
                    experiment_run_name,
                    dataset_id
                )
        
        tasks = [execute_with_semaphore(item) for item in self.dataset]
        
        results = []
        for coro in async_tqdm.as_completed(tasks, total=len(tasks), desc="Executing items"):
            result = await coro
            results.append(result)
        
        return results        
    
