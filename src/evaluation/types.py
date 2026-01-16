from typing import Dict, Any, TypeVar, Generic, Optional
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
    score: Optional[float] = None
    explanation: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class JudgeResponse(BaseModel):
    """
    Structured response expected from judge LLM.
    Judge must return JSON with this schema.
    """
    score: float = Field(..., description="Numerical score (e.g., 1-5)")
    explanation: str = Field(..., description="Explanation for the score")
