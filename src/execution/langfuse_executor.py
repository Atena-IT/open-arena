from threading import Lock
from typing import List, Dict, Any, Optional, Callable, TypeVar

from langfuse import get_client
from langfuse.experiment import ExperimentItem as ExperimentItem

from src.execution.executor_model import Executor
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
        llm_config: Dict[str, Any],
        from_langfuse_fn: Callable[[ExperimentItem], T],
        mcp_servers: Optional[List[MCPServerConfig]] = None,
        experiment_name: Optional[str] = None,
        experiment_description: Optional[str] = None,
        max_concurrency: int = 50
    ):
        """
        :param dataset_name: Name of Langfuse dataset
        :param llm_client: LLM client for completions
        :param system_prompt: System prompt for all completions
        :param llm_config: Model configuration
        :param from_langfuse_fn: Function to convert Langfuse DatasetItem to T
        :param mcp_servers: Optional MCP server configs
        :param experiment_name: Experiment name (auto-generated if None)
        :param experiment_description: Experiment description (auto-generated if None)
        :param langfuse_client: Optional Langfuse client
        :param max_concurrency: Max parallel items to process
        """
        super().__init__(
            llm_client=llm_client,
            system_prompt=system_prompt,
            llm_config=llm_config,
            mcp_servers=mcp_servers
        )
        
        self.dataset_name = dataset_name
        self.experiment_name = experiment_name or f"Experiment-{llm_config['model']}"
        self.experiment_description = experiment_description or f"Experiment with {llm_config['model']}"
        self.from_langfuse_fn = from_langfuse_fn
        self.langfuse = get_client() #TODO: public key? Other places where this is used?
        self.max_concurrency = max_concurrency

        
        # Storage for results during experiment execution
        self._execution_results: List[ExecutionResult[T]] = []
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
            
            if result.metadata is None:
                result.metadata = {}
            result.metadata["lf_experiment_id"] = item.id # type: ignore # TODO: fix
            
            with self._results_lock:
                self._execution_results.append(result)
            
            return str(result.output) # TODO: what to return on error? output = None
        
        return task
    
    async def execute(self) -> List[ExecutionResult[T]]:
        """
        Execute Langfuse experiment and return enhanced results.
        
        :return: List of enhanced execution results with Langfuse metadata
        """
        self._execution_results.clear()
        
        dataset = self.langfuse.get_dataset(self.dataset_name)

        dataset.run_experiment(
            name=self.experiment_name,
            description=self.experiment_description,
            task=self._create_task_fn(),
            max_concurrency=self.max_concurrency
        )
                
        return self._execution_results        
    