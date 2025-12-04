from abc import ABC, abstractmethod
import pandas as pd
from pathlib import Path
from typing import List


""" CLASS """
class DatasetLoader(ABC):
    """
    General class to load the dataset from a specified input path.
    Can be extended to match new input specifications and formats, to be then transformed into QAItem instances.
    Attributes:
        input_path (Path): The path to the dataset files.
    Methods:
        load() -> List[QAItem]: Loads and parses the dataset files into a list of QAItem instances.
    """

    def __init__(self, input_path: str, create_langfuse_dataset_bool: bool = False, dataset_name: str = ""):
        self.input_path = Path(input_path)
        self.create_langfuse_dataset_bool = create_langfuse_dataset_bool
        self.dataset_name = dataset_name

    @abstractmethod
    def load(self):
        """
        Loads and parses the dataset files into a list of QAItem instances.

        Returns:
            List[QAItem]: A list of QAItem instances parsed from the dataset files.
        """
        pass

    @abstractmethod
    def prepare_data(self) -> List:
        pass

    @abstractmethod
    def create_langfuse_dataset(self, dataset_df: pd.DataFrame):
        pass
