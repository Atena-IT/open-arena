from pathlib import Path
from typing import Any, Generic, TypeVar
from pydantic import ValidationError
import logging

from src.datasets.item_models import DatasetItem
from src.datasets.readers.base_reader import DatasetReader
from src.datasets.types import DatasetConfig

_logger = logging.getLogger(__name__)
T = TypeVar('T', bound=DatasetItem)


class DatasetLoader(Generic[T]):
    """
    Generic dataset loader that uses a Reader strategy to load data
    and validates it against a Pydantic model.
    """
    
    def __init__(
        self,
        item_model: type[T],
        reader: DatasetReader,
        config: DatasetConfig,
        input_path: str = "."
    ):
        """
        :param item_model: Pydantic model class for dataset items
        :param reader: Reader instance to use for loading data
        :param config: Dataset configuration (name and source file)
        :param input_path: Base path for data files
        """
        self.item_model = item_model
        self.reader = reader
        self.config = config
        self.dataset_name = config["dataset_name"]
        self.source_file = config["source_file"]
        self.input_path = Path(input_path)
        self._raw_data: list[dict[str, Any]] = []
        self._items: list[T] = []
    
    def load_raw(self) -> list[dict[str, Any]]:
        """
        Load raw data from source using the configured reader.
        
        :return: List of raw dictionaries
        """
        file_path = self.input_path / self.source_file
        _logger.debug(f"Loading raw data from {file_path}")
        self._raw_data = self.reader.read(str(file_path))
        _logger.debug(f"Loaded {len(self._raw_data)} raw items from {self.source_file}")
        return self._raw_data
    
    def validate_and_prepare(self, raw_data: list[dict[str, Any]] | None = None) -> list[T]:
        """
        Validate raw dictionaries against the Pydantic model.
        
        :param raw_data: Optional raw data to validate (uses loaded data if None)
        :return: List of validated Pydantic model instances
        """
        data_to_validate = raw_data if raw_data is not None else self._raw_data
        _logger.debug(f"Validating {len(data_to_validate)} items against {self.item_model.__name__}")
        
        validated_items: list[T] = []
        for idx, record in enumerate(data_to_validate):
            try:
                # Convert all values to strings (matching current behavior)
                processed = {k: str(v) if v is not None else None for k, v in record.items()}
                item = self.item_model(**processed)
                validated_items.append(item)
            except ValidationError as e:
                _logger.error(f"Validation error at row {idx}: {e}")
        
        self._items = validated_items
        _logger.debug(f"Validated {len(validated_items)}/{len(data_to_validate)} items successfully")
        return validated_items
    
    def load(self) -> list[T]:
        """
        Convenience method: load and validate in one step.
        
        :return: List of validated Pydantic model instances
        """
        self.load_raw()
        return self.validate_and_prepare()
    
    @property
    def items(self) -> list[T]:
        """Get the currently loaded and validated items."""
        return self._items
    
    @property
    def raw_data(self) -> list[dict[str, Any]]:
        """Get the raw loaded data."""
        return self._raw_data
