from threading import Lock
from typing import List, Dict, Any, Optional, Callable, TypeVar

from langfuse import get_client
from langfuse.experiment import ExperimentItem as ExperimentItem

from src.execution.base_executor import Executor
from src.execution.types import ExecutionResult
from src.llms import LangfuseLLMClient
from src.llms.types import MCPServerConfig
from src.datasets.item_models import DatasetItem

T = TypeVar('T', bound=DatasetItem)


class LangfuseExecutor(Executor[T]):
    """
    Executor that runs Langfuse experiments and returns enhanced results.
    
    Implements the Executor interface but fetches dataset from Langfuse
    and integrates with Langfuse's experiment framework for automatic tracing.
    """
    
    def __init__(
        self,
        dataset_name: str,
        llm_client: LangfuseLLMClient,
        system_prompt: str,
        from_langfuse_fn: Callable[[ExperimentItem], T],
        experiment_name: Optional[str] = None,
        experiment_description: Optional[str] = None,
        max_concurrency: int = 50
    ):
        """
        :param dataset_name: Name of Langfuse dataset
        :param llm_client: LLM client for completions
        :param system_prompt: System prompt for all completions
        :param from_langfuse_fn: Function to convert Langfuse DatasetItem to T
        :param experiment_name: Experiment name (auto-generated if None)
        :param experiment_description: Experiment description (auto-generated if None)
        :param langfuse_client: Optional Langfuse client
        :param max_concurrency: Max parallel items to process
        """
        super().__init__(
            llm_client=llm_client,
            system_prompt=system_prompt,
        )
        
        self.dataset_name = dataset_name
        self.experiment_name = experiment_name or f"Experiment-{self.client.llm_config['model']}"
        self.experiment_description = experiment_description or f"Experiment with {self.client.llm_config['model']}"
        self.from_langfuse_fn = from_langfuse_fn
        self.langfuse = get_client()
        self.max_concurrency = max_concurrency

        
        # Storage for results during experiment execution
        self._execution_results_map: Dict[str, ExecutionResult[T]] = {}
        self._results_lock = Lock()  # Thread safety for concurrent task execution
    
    def _create_task_fn(self):
        """
        Create async task function for Langfuse experiment.
        This wraps _execute_item to work with Langfuse's experiment API.
        
        :return: Async task function that takes ExperimentItem and returns output string
        """
        async def task(*, item: ExperimentItem, **kwargs: Any) -> str:
            """
            Async task function for Langfuse experiment.
            Converts Langfuse ExperimentItem -> T -> executes -> returns output.
            """
            converted_item = self.from_langfuse_fn(item)

            result = await self._execute_item(converted_item)
            
            with self._results_lock:
                self._execution_results_map[item.id] = result  # type: ignore # TODO fix
            
            return str(result.output) # TODO: what to return on error? output = None
        
        return task
    
    async def execute(self) -> List[ExecutionResult[T]]:
        """
        Execute Langfuse experiment and return enhanced results.
        
        :return: List of enhanced execution results with Langfuse metadata
        """
        self._execution_results_map.clear()
        
        dataset = self.langfuse.get_dataset(self.dataset_name)

        experiment_results = dataset.run_experiment(
            name=self.experiment_name,
            description=self.experiment_description,
            task=self._create_task_fn(),
            max_concurrency=self.max_concurrency
        )
        
        execution_results: List[ExecutionResult[T]] = []
        
        for item_result in experiment_results.item_results:
            experiment_item_id = item_result.item.id # type: ignore # TODO: fix
            trace_id = item_result.trace_id
            
            if experiment_item_id in self._execution_results_map:
                exec_result = self._execution_results_map[experiment_item_id]
                
                if exec_result.metadata is None:
                    exec_result.metadata = {}
                
                exec_result.metadata["lf_trace_id"] = trace_id
                exec_result.metadata["lf_experiment_id"] = experiment_item_id
                
                execution_results.append(exec_result)
        
        return execution_results        
    