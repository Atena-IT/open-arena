from pydantic import BaseModel, Field
from typing import Optional, Dict


class QAItem(BaseModel):
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
        :param open_ended_responses (dict): Model responses for open ended.
        :param multiple_choice_evaluation (dict): Evaluation results for multiple choice.
        :param open_ended_evaluation (dict): Evaluation results for open ended.
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
