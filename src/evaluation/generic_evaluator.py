import asyncio
from typing import TypeVar

from tqdm.asyncio import tqdm as async_tqdm

from src.datasets.item_models import DatasetItem
from src.execution.types import ExecutionResult
from src.evaluation.base_evaluator import Evaluator
from src.evaluation.types import EvaluationResult
from src.evaluation.methods import EvaluationMethod

T = TypeVar('T', bound=DatasetItem)


class GenericEvaluator(Evaluator[T]):
    """
    Generic evaluator for LLM-as-a-judge evaluation of execution results.
    """

    def __init__(
        self,
        results: list[ExecutionResult[T]],
        method: EvaluationMethod[T],
        max_concurrency: int = 10
    ):
        """
        :param results: Results from executor to evaluate
        :param method: Evaluation method to use
        :param max_concurrency: Maximum number of concurrent evaluations
        """
        super().__init__(results=results, method=method)
        self.max_concurrency = max_concurrency

    async def evaluate(self) -> list[EvaluationResult[T]]:
        """
        Evaluate all execution results in parallel.

        :return: List of evaluation results with scores
        """
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def evaluate_with_semaphore(result: ExecutionResult[T]) -> EvaluationResult[T]:
            async with semaphore:
                return await self.method.evaluate(result)

        tasks = [evaluate_with_semaphore(result) for result in self.results]

        evaluation_results = []
        for coro in async_tqdm.as_completed(tasks, total=len(tasks), desc="Evaluating results"):
            eval_result = await coro
            evaluation_results.append(eval_result)

        return evaluation_results
