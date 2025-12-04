from pydantic import BaseModel, Field
from typing import Optional, Dict


class QAItem(BaseModel):
    """
    Represents a single question-answer item for the Financial Advisor dataset.

    Attributes:
        id (str): Unique identifier for the question.
        level (str): Difficulty or level.
        topic (str): Topic of the question.
        practical (str): Practical/theoretical indicator.
        question (str): The question text.
        option_a (str): Option A.
        option_b (str): Option B.
        option_c (str): Option C.
        option_d (str): Option D.
        answer (str): The correct answer.
        theme (str): The theme of the question (filename).
        multiple_choice_responses (dict): Model responses for multiple choice.
        open_ended_responses (dict): Model responses for open ended.
        multiple_choice_evaluation (dict): Evaluation results for multiple choice.
        open_ended_evaluation (dict): Evaluation results for open ended.
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
    theme: str = Field(..., description="The theme of the question (filename).")
    multiple_choice_responses: Optional[Dict[str, str]] = Field(default_factory=dict, description="Model responses for multiple choice.")
    open_ended_responses: Optional[Dict[str, str]] = Field(default_factory=dict, description="Model responses for open ended.")
    multiple_choice_evaluation: Optional[Dict[str, str]] = Field(default_factory=dict, description="Evaluation results for multiple choice.")
    open_ended_evaluation: Optional[Dict[str, str]] = Field(default_factory=dict, description="Evaluation results for open ended.")
