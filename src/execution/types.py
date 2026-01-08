from typing import Dict, Any, TypeVar, Generic, Optional
from dataclasses import dataclass, field

from src.datasets.item_models import DatasetItem

T = TypeVar('T', bound=DatasetItem)


# TODO: why not a TypedDict?
@dataclass
class ExecutionResult(Generic[T]):
    """Result of executing a single dataset item."""
    item: T
    output: str
    model_name: str
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
