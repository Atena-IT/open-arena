import logging
from pathlib import Path
from typing import Type, List, Dict, Any
import pandas as pd
from pydantic import BaseModel, ValidationError
from src.dataset.models import QAItem


logger = logging.getLogger(__name__)


class GenericDatasetLoader:
    """
    Generic loader that dynamically reads tabular datasets and maps them
    into instances of any Pydantic model.

    Parameters:
        input_path (str): Folder containing Excel/CSV files.
        model_class (Type[BaseModel]): A Pydantic model to instantiate.
        column_mapping (Dict[str, str]): Optional mapping {file col -> model field}.
    """

    def __init__(self, input_path: str, column_mapping: Dict[str, str] = None):
        self.input_path = input_path
        self.model_class = QAItem
        self.column_mapping = column_mapping or {}
        self.datasets_df: List[pd.DataFrame] = []

    # -------------------------------------------------

    def load(self):
        """Load all .xlsx files and apply column mappings."""
        excel_files = list(Path(self.input_path).glob("*.xlsx"))

        for file in excel_files:
            df = pd.read_excel(file)

            # Apply dynamic renaming
            if self.column_mapping:
                df = df.rename(columns=self.column_mapping)

            # Add "file theme"
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

        return items




