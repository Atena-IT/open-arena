from pydantic import Field
from src.datasets.item_models import DatasetItem
from typing import Any


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
        input() -> str: Returns Input string.
        expected_output() -> str: Returns expected output string (ground truth).
        meta() -> Dict[str, Any]: Returns metadata dictionary.
    """
    id: str = Field(..., description="Unique identifier for the item.")
    description: str | None = Field(None, description="Text description of the item.")
    user_scenario: str = Field(..., description="User scenario details as a dict.")
    initial_state: str | None = Field(None, description="Initial state, can be null.")
    evaluation_criteria: str = Field(..., description="Evaluation criteria as a dict.")

    def input(self) -> str:
        return self.user_scenario
    
    def expected_output(self) -> str:
        return self.evaluation_criteria
    
    def meta(self) -> dict[str, Any]:
        return {
            **self.metadata,
            "description": self.description,
            "initial_state": self.initial_state
        }

