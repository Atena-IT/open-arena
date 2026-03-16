from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from src.datasets.item_models import DatasetItem
from src.execution.types import ExecutionResult
from src.evaluation.types import EvaluationResult
from src.evaluation.methods import EvaluationMethod

T = TypeVar('T', bound=DatasetItem)


class Evaluator(ABC, Generic[T]):
    """
    Abstract base class for all evaluators.
    Defines the common interface that all evaluator implementations must follow.
    """
    
    def __init__(
        self,
        results: list[ExecutionResult[T]],
        method: EvaluationMethod[T]
    ):
        """
        :param results: Results from executor to evaluate
        :param method: Evaluation method to use
        """
        self.results = results
        self.method = method
    
    @abstractmethod
    async def evaluate(self) -> list[EvaluationResult[T]]:
        """
        Evaluate the execution results.
        
        :return: List of evaluation results with scores
        """
        pass
