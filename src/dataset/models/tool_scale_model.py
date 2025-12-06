from pydantic import BaseModel, Field
from typing import Optional


""" CLASSES """
class ToolScaleItem(BaseModel):
    """
    Represents a single tool scale item.
    Parameters:
        :param id (str): Unique identifier for the item.
        :param description (str): Text description of the item.
        :param user_scenario (dict): User scenario details as a dict.
        :param initial_state (Optional[dict]): Initial state, can be null.
        :param evaluation_criteria (dict): Evaluation criteria as a dict.
    """
    id: str = Field(..., description="Unique identifier for the item.", json_schema_extra={"langfuse_dataset": "metadata"})
    description: str = Field(..., description="Text description of the item.", json_schema_extra={"langfuse_dataset": "input"})
    user_scenario: str = Field(..., description="User scenario details as a dict.", json_schema_extra={"langfuse_dataset": "input"})
    initial_state: Optional[str] = Field(None, description="Initial state, can be null.", json_schema_extra={"langfuse_dataset": "metadata"})
    evaluation_criteria: str = Field(..., description="Evaluation criteria as a dict.", json_schema_extra={"langfuse_dataset": "expected_output"})
