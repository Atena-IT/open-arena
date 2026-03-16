from typing import Any, TypeVar, Generic
from dataclasses import dataclass, field

from src.datasets.item_models import DatasetItem

T = TypeVar('T', bound=DatasetItem)


# TODO: why not a TypedDict?
@dataclass
class ExecutionResult(Generic[T]):
    """Result of executing a single dataset item."""
    item: T
    output: str | None
    model_name: str
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
