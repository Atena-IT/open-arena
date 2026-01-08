from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List

from src.datasets.item_models import DatasetItem
from src.execution.types import ExecutionResult

T = TypeVar('T', bound=DatasetItem)

class Executor(ABC, Generic[T]):
    """
    Abstract base class for all executors.
    Defines the common interface that all executor implementations must follow.
    """
    
    @abstractmethod
    def execute(self) -> List[ExecutionResult[T]]:
        """
        Execute the task on the dataset.
        """
        pass