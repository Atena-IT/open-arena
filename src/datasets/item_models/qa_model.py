from pydantic import Field
from src.datasets.item_models import DatasetItem
from typing import Any


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
        input() -> str: Returns Input string.
        expected_output() -> str: Returns expected output string (ground truth).
        meta() -> Dict[str, Any]: Returns metadata dictionary.
    """
    id: str = Field(..., description="Unique identifier for the question.")
    level: str = Field(..., description="Difficulty or level.")
    topic: str = Field(..., description="Topic of the question.")
    practical: str = Field(..., description="Practical/theoretical indicator.")
    question: str = Field(..., description="The question text.")
    option_a: str = Field(..., description="Option A.")
    option_b: str = Field(..., description="Option B.")
    option_c: str = Field(..., description="Option C.")
    option_d: str = Field(..., description="Option D.")
    answer: str = Field(..., description="The correct answer.")
    theme: str | None = Field(None, description="The theme of the question (filename).")
    multiple_choice_responses: dict[str, str] | None = Field(default_factory=dict, description="Model responses for multiple choice.")
    open_ended_responses: dict[str, str] | None = Field(default_factory=dict, description="Model responses for open ended.")
    multiple_choice_evaluation: dict[str, str] | None = Field(default_factory=dict, description="Evaluation results for multiple choice.")
    open_ended_evaluation: dict[str, str] | None = Field(default_factory=dict, description="Evaluation results for open ended.")

    def input(self) -> str:
        return (
            f"{self.question}\n"
            f"A) {self.option_a}\n"
            f"B) {self.option_b}\n"
            f"C) {self.option_c}\n"
            f"D) {self.option_d}"
        )

    def expected_output(self) -> str:
        return self.answer

    def meta(self) -> dict[str, Any]:
        return {
            **self.metadata,
            "id": self.id,
            "level": self.level,
            "practical": self.practical,
            "theme": self.theme,
            "multiple_choice_responses": self.multiple_choice_responses,
            "open_ended_responses": self.open_ended_responses,
            "multiple_choice_evaluation": self.multiple_choice_evaluation,
            "open_ended_evaluation": self.open_ended_evaluation
        }
