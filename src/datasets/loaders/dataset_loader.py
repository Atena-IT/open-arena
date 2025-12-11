from abc import ABC, abstractmethod
import pandas as pd
from pathlib import Path
from typing import List


""" CLASS """
class DatasetLoader(ABC):
    """
    General class to load the datasets from a specified input path.
    Can be extended to match new input specifications and formats, to be then transformed into QAItem instances.
    Parameters:
        :param input_path: The path to the datasets files.
        :param create_langfuse_dataset_bool: Boolean indicating whether to create a Langfuse datasets.
        :param dataset_name: The name of the Langfuse datasets.
    Methods:
        load(): Loads and parses the datasets files into a list of instances.
        prepare_data() -> List: Prepares and returns the datasets as a list of model instances
        create_langfuse_dataset(dataset_df: pd.DataFrame): Creates a Langfuse datasets from the provided DataFrame.
    """
    def __init__(self, input_path: str, create_langfuse_dataset_bool: bool = False, dataset_name: str = ""):
        self.create_langfuse_dataset_bool = create_langfuse_dataset_bool
        self.dataset_name = dataset_name
        self.input_path = Path(input_path)


    @abstractmethod
    def load(self):
        """
        Loads and parses the datasets files into a list of instances.
        """
        raise NotImplementedError


    @abstractmethod
    def prepare_data(self) -> List:
        """
        Prepares and returns the datasets as a list of model instances.
        """
        raise NotImplementedError


    @abstractmethod
    def create_langfuse_dataset(self, dataset_df: pd.DataFrame):
        """
        Creates a Langfuse datasets from the provided DataFrame.
        """
        raise NotImplementedError
