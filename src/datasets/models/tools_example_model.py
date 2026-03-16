from pydantic import Field
from src.datasets.models import DatasetItem
from typing import Any


""" CLASSES """
class ToolsExample(DatasetItem):
    """
    Represents a single tools-evaluation dataset item.
    Parameters:
        :param id (str): Unique identifier for the item.
        :param question (str): The question text.
        :param answer (str): The correct answer.
    Methods:
        from_langfuse_item() -> Any: Creates a Pydantic BaseModel object from a Langfuse dataset item.
        user_prompt() -> str: Returns Input string.
    """
    id: str = Field(..., description="Unique identifier for the item.", json_schema_extra={"langfuse_dataset": "metadata"})
    question: str = Field(..., description="The question text.", json_schema_extra={"langfuse_dataset": "input"})
    answer: str = Field(..., description="The correct answer.", json_schema_extra={"langfuse_dataset": "expected_output"})

    @classmethod
    def from_langfuse_item(cls, item: Any) -> "ToolsExample":
        """
        Creates a ToolsExample from a Langfuse dataset item.
        Expects that:
        - item.input contains fields marked as “input”
        - item.expected_output contains fields marked as “expected_output”
        - item.metadata contains fields marked as “metadata”
        Parameters:
            :param item: The Langfuse dataset item.
        Return:
            :return: ToolsExample: The constructed ToolsExample instance.
        """
        input_data = getattr(item, "input", {}) or {}
        expected_output = getattr(item, "expected_output", {}) or {}
        metadata = getattr(item, "metadata", {}) or {}
        data = {
            # input
            "question": input_data.get("question", ""),

            # expected_output
            "answer": expected_output.get("answer", ""),

            # metadata
            "id": metadata.get("id", ""),
        }
        return cls(**data)


    def user_prompt(self) -> str:
        """
        Return:
            :return: Input string
        """
        return self.question
