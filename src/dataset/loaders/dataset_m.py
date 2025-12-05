import logging
from pathlib import Path
from typing import Type, List, Dict, Any
import pandas as pd
from pydantic import BaseModel, ValidationError

import logging, os
from src.dataset.loaders import DatasetLoader
from src.dataset.models import QAItem
from langfuse import Langfuse, get_client
from langfuse.openai import OpenAI
from pathlib import Path
from tqdm import tqdm
from typing import List
import re


logger = logging.getLogger(__name__)



def to_snake_case(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[^\w]+", "_", name)  # simboli/spazi → underscore
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name)
    return name.lower()


class GenericDatasetLoader:
    """
    Generic loader that dynamically reads tabular datasets and maps them
    into instances of any Pydantic model.

    Parameters:
        input_path (str): Folder containing Excel/CSV files.
        model_class (Type[BaseModel]): A Pydantic model to instantiate.
        column_mapping (Dict[str, str]): Optional mapping {file col -> model field}.
    """

    def __init__(self, input_path: str, create_langfuse_dataset_bool: bool = False, dataset_name: str = ""):
        self.input_path = input_path
        self.model_class = QAItem
        self.datasets_df: List[pd.DataFrame] = []
        self.create_langfuse_dataset_bool = create_langfuse_dataset_bool
        self.dataset_name = dataset_name

    # -------------------------------------------------

    def load(self):
        """Load all .xlsx files, convert column names to snake_case,
        and keep only model fields."""

        excel_files = list(Path(self.input_path).glob("*.xlsx"))

        for file in excel_files:
            df = pd.read_excel(file)

            # Convert all column names to snake_case automatically
            df.columns = [to_snake_case(c) for c in df.columns]

            # Keep only fields that exist in the Pydantic model
            model_fields = set(self.model_class.__fields__.keys())
            df = df[[col for col in df.columns if col in model_fields]]

            # Add file theme
            df["theme"] = file.stem

            logger.info(f"Loaded file {file.name}, columns: {list(df.columns)}")
            self.datasets_df.append(df)

    # -------------------------------------------------

    def prepare_data(self) -> List[BaseModel]:
        """Convert each row of each DataFrame into a Pydantic model instance."""

        items: List[BaseModel] = []

        model_fields = set(self.model_class.__fields__.keys())

        for df in self.datasets_df:
            for _, row in df.iterrows():

                # Select only fields that exist in model
                row_dict = {field: None if pd.isna(row[field]) else str(row[field])
                    for field in model_fields
                    if field in row
                }

                try:
                    item = self.model_class(**row_dict)
                    items.append(item)

                except ValidationError as e:
                    logger.error(f"Row validation error: {e}")

        if self.create_langfuse_dataset_bool:
            self.create_langfuse_dataset(items)

        return items

    # -------------------------------------------------

    def create_langfuse_dataset(self, items: List[BaseModel]):
        """
        Create a Langfuse dataset by sending each Pydantic model instance.
        Uses only fields from the Pydantic model_class.
        Automatically excludes fields that are None, empty strings,
        empty dicts or empty lists.
        """

        if not self.create_langfuse_dataset_bool:
            return

        logger.info(f"Creating Langfuse dataset '{self.dataset_name}'...")

        # initialize client
        langfuse = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
            host=os.getenv("LANGFUSE_HOST", ""),
        )

        # ensure dataset exists
        try:
            langfuse.get_dataset(self.dataset_name)
            logger.info(f"Dataset '{self.dataset_name}' already exists.")
        except Exception:
            logger.info(f"Dataset '{self.dataset_name}' not found. Creating new one...")
            langfuse.create_dataset(name=self.dataset_name)

        # convert each item
        for item in tqdm(items, desc="Uploading Langfuse dataset items"):
            raw = item.dict()

            # remove empty / None fields
            cleaned = {
                k: v for k, v in raw.items()
                if v not in (None, "", [], {})  # no empty values
            }

            # send all fields to input (Approach A)
            langfuse.create_dataset_item(
                dataset_name=self.dataset_name,
                input=cleaned,
                expected_output={},
                metadata={},
            )



