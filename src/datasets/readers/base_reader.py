from abc import ABC, abstractmethod
from typing import Any


class DatasetReader(ABC):
    """Abstract base class for dataset readers."""

    @abstractmethod
    def read(self, file_path: str) -> list[dict[str, Any]]:
        """
        Read data from source and return list of raw dictionaries.

        :param file_path: Path to the data source
        :return: List of dictionaries representing rows/items
        """
        raise NotImplementedError
