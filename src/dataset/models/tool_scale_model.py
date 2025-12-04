from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


""" CLASSES """
class ToolScale(BaseModel):
    id: str = Field(..., description="Unique identifier for the item.")
    description: str = Field(..., description="Text description of the item.")
    user_scenario: Dict[str, Any] = Field(..., description="User scenario details as a dict.")
    initial_state: Optional[Dict[str, Any]] = Field(None, description="Initial state, can be null.")
    evaluation_criteria: Dict[str, Any] = Field(..., description="Evaluation criteria as a dict.")
