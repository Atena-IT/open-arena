from typing import Any, TypeVar, Generic
from dataclasses import dataclass, field
from pydantic import BaseModel, Field

from src.datasets.item_models import DatasetItem

T = TypeVar('T', bound=DatasetItem)


@dataclass
class EvaluationResult(Generic[T]):
    """
    Result of evaluating a single execution result.

    Contains the original execution data plus optional score and explanation.
    """
    item: T
    output: str
    model_name: str
    score: float | None = None
    explanation: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

class JudgeResponse(BaseModel):
    """
    Structured response expected from judge LLM.
    Judge must return JSON with this schema.
    """
    score: int = Field(..., description="Integer score from 1 to 5", ge=1, le=5)
    explanation: str = Field(..., description="Explanation for the score")
