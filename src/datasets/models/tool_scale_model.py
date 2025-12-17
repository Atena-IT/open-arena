from pydantic import Field
from src.datasets.models import DatasetItem
from typing import Any, Optional


""" CLASSES """
class ToolScaleItem(DatasetItem):
    """
    Represents a single tool scale item.
    Parameters:
        :param id (str): Unique identifier for the item.
        :param description (str): Text description of the item.
        :param user_scenario (dict): User scenario details as a dict.
        :param initial_state (Optional[dict]): Initial state can be null.
        :param evaluation_criteria (dict): Evaluation criteria as a dict.
    Methods:
        from_langfuse_item() -> Any: Creates a Pydantic BaseModel object from a Langfuse dataset item.
        user_prompt() -> str: Returns Input string.
    """
    id: str = Field(..., description="Unique identifier for the item.", json_schema_extra={"langfuse_dataset": "metadata"})
    description: Optional[str] = Field(None, description="Text description of the item.", json_schema_extra={"langfuse_dataset": "input"})
    user_scenario: str = Field(..., description="User scenario details as a dict.", json_schema_extra={"langfuse_dataset": "input"})
    initial_state: Optional[str] = Field(None, description="Initial state, can be null.", json_schema_extra={"langfuse_dataset": "metadata"})
    evaluation_criteria: str = Field(..., description="Evaluation criteria as a dict.", json_schema_extra={"langfuse_dataset": "expected_output"})

    @classmethod
    def from_langfuse_item(cls, item: Any) -> "ToolScaleItem":
        """
        Creates a ToolScaleItem from a Langfuse dataset item.
        Expects that:
        - item.input contains fields marked as “input”
        - item.expected_output contains fields marked as “expected_output”
        - item.metadata contains fields marked as “metadata”
        Parameters:
            :param item: The Langfuse dataset item.
        Return:
            :return: ToolScaleItem: The constructed ToolScaleItem instance.
        """
        input_data = getattr(item, "input", {}) or {}
        expected_output = getattr(item, "expected_output", {}) or {}
        metadata = getattr(item, "metadata", {}) or {}
        data = {
            # input
            "user_scenario": input_data.get("user_scenario", ""),

            # expected_output
            "evaluation_criteria": expected_output.get("evaluation_criteria", ""),

            # metadata
            "id": metadata.get("id", ""),
        }
        return cls(**data)


    def user_prompt(self) -> str:
        """
        Return:
            :return: Input string
        """
        return self.user_scenario
