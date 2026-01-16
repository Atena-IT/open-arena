from abc import abstractmethod, ABC
from pydantic import BaseModel, Field
from typing import Any, Dict


class DatasetItem(BaseModel, ABC):
    """
    Represents a single dataset item
    Methods:
        input() -> str: Returns Input string.
        expected_output() -> str: Returns expected output string (ground truth).
        meta() -> Dict[str, Any]: Returns metadata dictionary.
    """
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Optional metadata for the dataset item")

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
    def input(self) -> str:
        """
        Return:
            :return: Input string
        """
        raise NotImplementedError
    
    @abstractmethod
    def expected_output(self) -> str:
        """
        Return:
            :return: Expected output string (ground truth)
        """
        raise NotImplementedError
    
    def meta(self) -> Dict[str, Any]:
        """
        Return:
            :return: Metadata dictionary
        """
        return self.metadata