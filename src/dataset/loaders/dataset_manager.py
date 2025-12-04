import logging, os
from dataset_core.dataset_loader import DatasetLoader
from dataset_core.models.qa_model import QAItem
from langfuse import Langfuse, get_client
from langfuse.openai import OpenAI
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from typing import List
from urllib.parse import quote


""" CONFIG """
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


""" CLASS """
class DatasetManager(DatasetLoader):
    """
    Specialized class to prepare financial advisor QA dataset.
    Inherits from DatasetLoader to utilize its loading capabilities.
    Attributes:
        input_path (str): The path to the financial advisor dataset files.
    Methods:
        prepare_data() -> List[QAItem]: Prepares and returns the financial advisor QA data as a list of QAItem instances.
    """

    def __init__(self, input_path: str, create_langfuse_dataset_bool: bool = False):


        super().__init__(input_path)
        self.langfuse_client = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_OTEL_HOST"),
        )
        self.create_langfuse_dataset_bool = create_langfuse_dataset_bool

    def prepare_data(self) -> List[QAItem]:
        """
        Prepares and returns the financial advisor QA data as a list of QAItem instances.
        Adds a 'theme' field to each QAItem, which is the filename of the Excel file.
        Logs the head of each dataframe.
        Also creates a Langfuse dataset item for each row. No expected_output is provided.
        Returns:
            List[QAItem]: A list of QAItem instances for financial advisor data.
        """
        items = []
        excel_files = list(Path(self.input_path).glob("*.xlsx"))
        for file in excel_files:
            df = pd.read_excel(file)[:100]
            df = df.rename(
                columns={
                    "ID": "id",
                    "livello": "level",
                    "Argomento": "topic",
                    "pratico": "practical",
                    "Domanda": "question",
                    "opzione A": "option_a",
                    "opzione B": "option_b",
                    "opzione C": "option_c",
                    "opzione D": "option_d",
                    "Risposta": "answer",
                }
            )
            df["theme"] = file.stem
            logger.info(f"Head of {file.name}:")
            logger.info(df.head())
            for _, row in df.iterrows():
                # build QAItem
                item = QAItem(
                    id=str(row["id"]) + "_" + str(row["theme"]),
                    level=str(row["level"]),
                    topic=str(row["topic"]),
                    practical=str(row["practical"]),
                    question=str(row["question"]),
                    option_a=str(row["option_a"]),
                    option_b=str(row["option_b"]),
                    option_c=str(row["option_c"]),
                    option_d=str(row["option_d"]),
                    answer=str(row["answer"]),
                    theme=str(row["theme"]),
                )
                items.append(item)
            self.create_langfuse_dataset(dataset_df=df)
        return items

    def create_langfuse_dataset(self, dataset_df: pd.DataFrame):
        """
        Creates Langfuse dataset items for each row in the provided DataFrame.
        Args:
            dataset_df (pd.DataFrame): The DataFrame containing the dataset rows.
        """
        if self.create_langfuse_dataset_bool:
            logger.info("Creating Langfuse dataset items...")
            dataset_name = "unicredit/fin-adv-dataset"
            for _, row in tqdm(dataset_df.iterrows(), total=len(dataset_df)):
                self.langfuse_client.create_dataset_item(
                    dataset_name=dataset_name,
                    input={
                        "id": str(row["id"]),
                        "question": (
                            None
                            if pd.isna(row.get("question"))
                            else str(row.get("question"))
                        ),
                        "option_a": (
                            None
                            if pd.isna(row.get("option_a"))
                            else str(row.get("option_a"))
                        ),
                        "option_b": (
                            None
                            if pd.isna(row.get("option_b"))
                            else str(row.get("option_b"))
                        ),
                        "option_c": (
                            None
                            if pd.isna(row.get("option_c"))
                            else str(row.get("option_c"))
                        ),
                        "option_d": (
                            None
                            if pd.isna(row.get("option_d"))
                            else str(row.get("option_d"))
                        ),
                    },
                    metadata={
                        "theme": (
                            None if pd.isna(row.get("theme")) else str(row.get("theme"))
                        ),
                        "topic": (
                            None if pd.isna(row.get("topic")) else str(row.get("topic"))
                        ),
                        "level": (
                            None if pd.isna(row.get("level")) else str(row.get("level"))
                        ),
                        "practical": (
                            None
                            if pd.isna(row.get("practical"))
                            else str(row.get("practical"))
                        ),
                    },
                )
