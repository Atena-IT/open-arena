from pydantic import Field
from src.datasets.item_models import DatasetItem


""" CLASSES """
class ToolsExample(DatasetItem):
    """
    Represents a single tool scale item.
    Parameters:
        :param id (str): Unique identifier for the item.
        :param question (str): The question text.
        :param answer (str): The correct answer.
    Methods:
        input() -> str: Returns Input string.
        expected_output() -> str: Returns expected output string (ground truth).
        meta() -> Dict[str, Any]: Returns metadata dictionary.
    """
    id: str = Field(..., description="Unique identifier for the item.", json_schema_extra={"langfuse_dataset": "metadata"})
    question: str = Field(..., description="The question text.", json_schema_extra={"langfuse_dataset": "input"})
    answer: str = Field(..., description="The correct answer.", json_schema_extra={"langfuse_dataset": "expected_output"})

    def input(self) -> str:
        return self.question

    def expected_output(self) -> str:
        return self.answer
