from abc import abstractmethod, ABC
from pydantic import BaseModel, Field
from typing import Any


class DatasetItem(BaseModel, ABC):
    """
    Represents a single dataset item
    Methods:
        input() -> str: Returns Input string.
        expected_output() -> str: Returns expected output string (ground truth).
        meta() -> Dict[str, Any]: Returns metadata dictionary.
    """
    metadata: dict[str, Any] = Field(default_factory=dict, description="Optional metadata for the dataset item")

    @classmethod
    def from_langfuse_item(cls, item: Any) -> Any:
        """
        Creates a Pydantic BaseModel object from a Langfuse dataset item.

        Subclasses may override this when they support reconstructing a typed dataset item
        from Langfuse data. The default implementation keeps local file-based item models
        instantiable, but does not provide reconstruction support.

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

    def meta(self) -> dict[str, Any]:
        """
        Return:
            :return: Metadata dictionary
        """
        return self.metadata
