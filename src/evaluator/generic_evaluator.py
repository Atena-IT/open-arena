import asyncio
from typing import List, TypeVar, Dict, Any

from tqdm.asyncio import tqdm as async_tqdm

from src.datasets.item_models import DatasetItem
from src.execution.types import ExecutionResult
from src.evaluator.evaluator_model import Evaluator
from src.evaluator.types import EvaluationResult
from src.llms import LLMClient

T = TypeVar('T', bound=DatasetItem)


class GenericEvaluator(Evaluator[T]):
    """
    Generic evaluator for LLM-as-a-judge evaluation of execution results.
    """
    
    def __init__(
        self,
        results: List[ExecutionResult[T]],
        llm_client: LLMClient,
        system_prompt: str,
        model_config: Dict[str, Any],
        max_concurrency: int = 10
    ):
        """
        :param results: Results from executor to evaluate
        :param llm_client: LLM client for judge completions
        :param system_prompt: System prompt that defines how the judge should evaluate
        :param model_config: Model configuration for the judge to use
        :param max_concurrency: Maximum number of concurrent evaluations
        """
        super().__init__(
            results=results,
            llm_client=llm_client,
            system_prompt=system_prompt,
            model_config=model_config
        )
        self.max_concurrency = max_concurrency
    
    async def evaluate(self) -> List[EvaluationResult[T]]:
        """
        Evaluate all execution results in parallel.
        
        :return: List of evaluation results with scores
        """
        semaphore = asyncio.Semaphore(self.max_concurrency)
        
        async def evaluate_with_semaphore(result: ExecutionResult[T]) -> EvaluationResult[T]:
            async with semaphore:
                return await self._evaluate_result(result)
        
        tasks = [evaluate_with_semaphore(result) for result in self.results]
        
        evaluation_results = []
        for coro in async_tqdm.as_completed(tasks, total=len(tasks), desc="Evaluating results"):
            eval_result = await coro
            evaluation_results.append(eval_result)

        return evaluation_results
