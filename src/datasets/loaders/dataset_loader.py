import logging
from abc import ABC
from pathlib import Path
from typing import Any, Generic, TypeVar

from pydantic import ValidationError

from src.datasets.item_models import DatasetItem
from src.datasets.readers.base_reader import DatasetReader
from src.datasets.types import DatasetConfig

_logger = logging.getLogger(__name__)
T = TypeVar("T", bound=DatasetItem)


class DatasetLoader(ABC, Generic[T]):
    """
    Shared dataset loader base.

    Supports both the legacy loader API used by `src.main` and the newer
    reader + Pydantic validation flow used by the structured CLI.
    """

    def __init__(
        self,
        item_model: type[T] | None = None,
        reader: DatasetReader | None = None,
        config: DatasetConfig | None = None,
        input_path: str = ".",
        dataset_config: dict | None = None,
        dataset_name: str = "",
    ):
        self.input_path = Path(input_path)

        self.item_model = item_model
        self.reader = reader
        self.config = config
        self.dataset_config = dataset_config
        self.dataset_name = dataset_name
        self.source_file = ""

        self._raw_data: list[dict[str, Any]] = []
        self._items: list[T] = []

        if config is not None:
            self.dataset_name = config["dataset_name"]
            self.source_file = config["source_file"]

    def load_raw(self) -> list[dict[str, Any]]:
        """
        Load raw records using the configured reader strategy.
        """
        if self.reader is None or self.item_model is None or not self.source_file:
            raise ValueError("Reader-based loading requires item_model, reader, and config")

        file_path = self.input_path / self.source_file
        _logger.debug("Loading raw data from %s", file_path)
        self._raw_data = self.reader.read(str(file_path))
        _logger.debug("Loaded %s raw items from %s", len(self._raw_data), self.source_file)
        return self._raw_data

    def validate_and_prepare(self, raw_data: list[dict[str, Any]] | None = None) -> list[T]:
        """
        Validate raw dictionaries against the configured Pydantic item model.
        """
        if self.item_model is None:
            raise ValueError("Validation requires an item_model")

        data_to_validate = raw_data if raw_data is not None else self._raw_data
        _logger.debug("Validating %s items against %s", len(data_to_validate), self.item_model.__name__)

        validated_items: list[T] = []
        for idx, record in enumerate(data_to_validate):
            try:
                processed = {k: str(v) if v is not None else None for k, v in record.items()}
                item = self.item_model(**processed)
                validated_items.append(item)
            except ValidationError as e:
                _logger.error("Validation error at row %s: %s", idx, e)

        self._items = validated_items
        _logger.debug("Validated %s/%s items successfully", len(validated_items), len(data_to_validate))
        return validated_items

    def load(self) -> list[T]:
        """
        Reader-based convenience method for the structured CLI path.
        Legacy subclasses override this method with their own loading logic.
        """
        if self.reader is None or self.item_model is None or self.config is None:
            raise NotImplementedError("Legacy loaders must implement load()")

        self.load_raw()
        return self.validate_and_prepare()

    def prepare_data(self):
        """
        Legacy hook implemented by `GenericDatasetLoader`.
        """
        raise NotImplementedError

    def create_langfuse_dataset(self, *args, **kwargs):
        """
        Legacy hook implemented by `GenericDatasetLoader`.
        """
        raise NotImplementedError

    @property
    def items(self) -> list[T]:
        return self._items

    @property
    def raw_data(self) -> list[dict[str, Any]]:
        return self._raw_data
