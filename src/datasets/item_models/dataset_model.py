from abc import abstractmethod, ABC
from pydantic import BaseModel
from typing import Any


""" CLASSES """
class DatasetItem(BaseModel, ABC):
    """
    Represents a single dataset item
    Methods:
        from_langfuse_item() -> Any: Creates a Pydantic BaseModel object from a Langfuse dataset item.
        user_prompt() -> str: Returns Input string.
    """


    @classmethod
    @abstractmethod
    def from_langfuse_item(cls, item: Any) -> Any:
        """
        Creates a Pydantic BaseModel object from a Langfuse dataset item.
        Expects that:
        - item.input contains fields marked as “input”
        - item.expected_output contains fields marked as “expected_output”
        - item.metadata contains fields marked as “metadata”
        Parameters:
            :param item: The Langfuse dataset item.
        Return:
            :return: BaseModel: The constructed BaseModel instance.
        """
        raise NotImplementedError


    @abstractmethod
    def user_prompt(self) -> str:
        """
        Return:
            :return: Input string
        """
        raise NotImplementedError
