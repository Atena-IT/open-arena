from pydantic import Field
from src.datasets.models import DatasetItem
from typing import Any, Dict, Optional


""" CLASSES """
class QAItem(DatasetItem):
    """
    Represents a single question-answer item for the Financial Advisor datasets.
    Parameters:
        :param id (str): Unique identifier for the question.
        :param level (str): Difficulty or level.
        :param topic (str): Topic of the question.
        :param practical (str): Practical/theoretical indicator.
        :param question (str): The question text.
        :param option_a (str): Option A.
        :param option_b (str): Option B.
        :param option_c (str): Option C.
        :param option_d (str): Option D.
        :param answer (str): The correct answer.
        :param theme (str): The theme of the question (filename).
        :param multiple_choice_responses (dict): Model responses for multiple choice.
        :param open_ended_responses (dict): Model responses for open-ended.
        :param multiple_choice_evaluation (dict): Evaluation results for multiple choice.
        :param open_ended_evaluation (dict): Evaluation results for open-ended.
    Methods:
        from_langfuse_item() -> Any: Creates a Pydantic BaseModel object from a Langfuse dataset item.
        user_prompt() -> str: Returns Input string.
    """
    id: str = Field(..., description="Unique identifier for the question.", json_schema_extra={"langfuse_dataset": "metadata"})
    level: str = Field(..., description="Difficulty or level.", json_schema_extra={"langfuse_dataset": "metadata"})
    topic: str = Field(..., description="Topic of the question.", json_schema_extra={"langfuse_dataset": "input"})
    practical: str = Field(..., description="Practical/theoretical indicator.", json_schema_extra={"langfuse_dataset": "metadata"})
    question: str = Field(..., description="The question text.", json_schema_extra={"langfuse_dataset": "input"})
    option_a: str = Field(..., description="Option A.", json_schema_extra={"langfuse_dataset": "input"})
    option_b: str = Field(..., description="Option B.", json_schema_extra={"langfuse_dataset": "input"})
    option_c: str = Field(..., description="Option C.", json_schema_extra={"langfuse_dataset": "input"})
    option_d: str = Field(..., description="Option D.", json_schema_extra={"langfuse_dataset": "input"})
    answer: str = Field(..., description="The correct answer.", json_schema_extra={"langfuse_dataset": "expected_output"})
    theme: Optional[str] = Field(None, description="The theme of the question (filename).", json_schema_extra={"langfuse_dataset": "metadata"})
    multiple_choice_responses: Optional[Dict[str, str]] = Field(default_factory=dict, description="Model responses for multiple choice.", json_schema_extra={"langfuse_dataset": "metadata"})
    open_ended_responses: Optional[Dict[str, str]] = Field(default_factory=dict, description="Model responses for open ended.", json_schema_extra={"langfuse_dataset": "metadata"})
    multiple_choice_evaluation: Optional[Dict[str, str]] = Field(default_factory=dict, description="Evaluation results for multiple choice.", json_schema_extra={"langfuse_dataset": "metadata"})
    open_ended_evaluation: Optional[Dict[str, str]] = Field(default_factory=dict, description="Evaluation results for open ended.", json_schema_extra={"langfuse_dataset": "metadata"})


    @classmethod
    def from_langfuse_item(cls, item: Any) -> "QAItem":
        """
        Creates a QAItem from a Langfuse dataset item.
        Expects that:
        - item.input contains fields marked as “input”
        - item.expected_output contains fields marked as “expected_output”
        - item.metadata contains fields marked as “metadata”
        Parameters:
            :param item: The Langfuse dataset item.
        Return:
            :return: QAItem: The constructed QAItem instance.
        """
        input_data = getattr(item, "input", {}) or {}
        expected_output = getattr(item, "expected_output", {}) or {}
        metadata = getattr(item, "metadata", {}) or {}
        data = {
            # input
            "topic": input_data.get("topic", ""),
            "question": input_data.get("question", ""),
            "option_a": input_data.get("option_a", ""),
            "option_b": input_data.get("option_b", ""),
            "option_c": input_data.get("option_c", ""),
            "option_d": input_data.get("option_d", ""),

            # expected_output
            "answer": expected_output.get("answer", ""),

            # metadata
            "id": metadata.get("id", ""),
            "level": metadata.get("level", ""),
            "practical": metadata.get("practical", ""),
            "theme": metadata.get("theme"),
            "multiple_choice_responses": metadata.get("multiple_choice_responses", {}),
            "open_ended_responses": metadata.get("open_ended_responses", {}),
            "multiple_choice_evaluation": metadata.get("multiple_choice_evaluation", {}),
            "open_ended_evaluation": metadata.get("open_ended_evaluation", {})
        }
        return cls(**data)


    def user_prompt(self) -> str:
        """
        Return:
            :return: Input string
        """
        return f"{self.question}\n"f"A) {self.option_a}\n"f"B) {self.option_b}\n"f"C) {self.option_c}\n"f"D) {self.option_d}"
