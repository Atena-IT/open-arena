import pandas as pd
import logging
from typing import Any
from .base_reader import DatasetReader
from .utils import to_snake_case

_logger = logging.getLogger(__name__)


class ExcelReader(DatasetReader):
    """Reads Excel files and returns raw dictionaries."""

    def read(self, file_path: str) -> list[dict[str, Any]]:
        """
        Read Excel file and return list of dictionaries.

        :param file_path: Path to Excel file
        :return: List of row dictionaries with snake_case keys
        """
        df = pd.read_excel(file_path)
        _logger.debug(f"Read Excel file: {file_path}, shape: {df.shape}")

        # Convert columns to snake_case
        original_columns = list(df.columns)
        df.columns = [to_snake_case(col) for col in df.columns]
        _logger.debug(f"Converted columns from {original_columns} to {list(df.columns)}")

        # Convert to list of dicts
        records = df.to_dict('records')

        # Clean up NaN values
        cleaned_records = []
        for record in records:
            cleaned = {k: (None if pd.isna(v) else v) for k, v in record.items()}
            cleaned_records.append(cleaned)

        _logger.debug(f"Converted {len(cleaned_records)} records from Excel")
        return cleaned_records
