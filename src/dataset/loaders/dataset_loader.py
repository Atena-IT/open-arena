from pathlib import Path
from src.dataset_core.models.qa_model import QAItem
from typing import List


""" CLASS """
class DatasetLoader:
    """
    General class to load the dataset from a specified input path.
    Can be extended to match new input specifications and formats, to be then transformed into QAItem instances.
    Attributes:
        input_path (Path): The path to the dataset files.
    Methods:
        load() -> List[QAItem]: Loads and parses the dataset files into a list of QAItem instances.
    """

    def __init__(self, input_path: str):
        self.input_path = Path(input_path)

    def load(self) -> List[QAItem]:
        """
        Loads and parses the dataset files into a list of QAItem instances.

        Returns:
            List[QAItem]: A list of QAItem instances parsed from the dataset files.
        """
        pass
