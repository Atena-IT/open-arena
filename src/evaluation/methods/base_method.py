from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from src.datasets.item_models import DatasetItem
from src.execution.types import ExecutionResult
from src.evaluation.types import EvaluationResult

T = TypeVar('T', bound=DatasetItem)


class EvaluationMethod(ABC, Generic[T]):
    """
    Abstract interface for evaluation methods.

    Each method implements a different strategy for evaluating
    execution results.
    """

    def __init__(
        self,
        name: str = "evaluation"
    ):
        self.name = name


    @abstractmethod
    async def evaluate(self, result: ExecutionResult[T]) -> EvaluationResult[T]:
        """
        Evaluate a single execution result.

        :param result: Execution result to evaluate
        :return: Evaluation result with score and explanation
        """
        pass
