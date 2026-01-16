import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Type, Generic, Any, Optional, TypeVar, TypedDict
from langfuse import get_client
from tqdm import tqdm

from src.datasets.item_models import DatasetItem
from src.datasets.loaders.dataset_loader import DatasetLoader
from src.datasets.readers.base_reader import DatasetReader
from src.datasets.types import DatasetConfig

_logger = logging.getLogger(__name__)
T = TypeVar('T', bound=DatasetItem)


class _LangfuseDatasetItem(TypedDict):
    """Structure for Langfuse dataset items."""
    input: Any
    expected_output: Any
    metadata: Any

class LangfuseLoader(DatasetLoader[T]):
    """
    DatasetLoader with Langfuse integration.
    Extends base loader to upload validated items to Langfuse.
    """
    
    def __init__(
        self,
        item_model: Type[T],
        reader: DatasetReader,
        config: DatasetConfig,
        input_path: str = ".",
        max_items: Optional[int] = None,
        max_workers: int = 12,
    ):
        """
        :param item_model: Pydantic model class for dataset items
        :param reader: Reader instance to use for loading data
        :param config: Dataset configuration
        :param input_path: Base path for data files
        :param max_items: Maximum number of items to upload (None = all)
        :param max_workers: Number of parallel workers for upload
        """
        super().__init__(item_model, reader, config, input_path)
        
        self.max_items = max_items
        self.max_workers = max_workers
        self.dataset_description = config.get("dataset_description", "")
        
        self.langfuse = get_client()
    
    def convert_type(self, item: T) -> _LangfuseDatasetItem:
        """
        Convert item fields into Langfuse dataset item format based on
        json_schema_extra metadata in the Pydantic model.
        
        :param item: Pydantic model instance
        :return: LangfuseDatasetItem with 'input', 'expected_output', 'metadata' keys
        """
        raw = item.model_dump()
        
        # Remove None, empty strings, empty lists, empty dicts
        cleaned = {k: v for k, v in raw.items() if v not in (None, "", [], {})}
        
        input_data = {}
        expected_output_data = {}
        metadata_data = {}
        
        for field_name, field_def in type(item).model_fields.items():
            value = cleaned.get(field_name)
            if value is None:
                continue
            
            # Default to "input" if no metadata specified
            category = "input"
            if (
                field_def.json_schema_extra 
                and isinstance(field_def.json_schema_extra, dict) 
                and "langfuse_dataset" in field_def.json_schema_extra
            ):
                category = field_def.json_schema_extra["langfuse_dataset"]
            
            if category == "input":
                input_data[field_name] = value
            elif category == "expected_output":
                expected_output_data[field_name] = value
            elif category == "metadata":
                metadata_data[field_name] = value
        
        return {
            "input": input_data,
            "expected_output": expected_output_data,
            "metadata": metadata_data
        }
    
    def _ensure_dataset_exists(self):
        """Ensure the Langfuse dataset exists, create if not."""
        try:
            self.langfuse.get_dataset(self.dataset_name)
            _logger.debug(f"Dataset '{self.dataset_name}' already exists on Langfuse")
        except Exception:
            _logger.debug(f"Creating new Langfuse dataset '{self.dataset_name}'")
            self.langfuse.create_dataset(
                name=self.dataset_name,
                description=self.dataset_description
            )
    
    def _upload(self) -> List[T]:
        """
        Upload items to Langfuse dataset.
        
        :param items: Optional list of items to upload (uses self._items if None)
        :return: List of uploaded items with lf_item_id added to metadata
        """
        if not self._items:
            _logger.warning("No items to upload to Langfuse")
            return []
        
        if self.max_items:
            self._items = self._items[:self.max_items]
        
        _logger.debug(f"Uploading {len(self._items)} items to Langfuse dataset '{self.dataset_name}'")
        
        self._ensure_dataset_exists()
        
        # Upload items in parallel
        def upload_item(item: T) -> T:
            langfuse_item = self.convert_type(item)
            created = self.langfuse.create_dataset_item(
                dataset_name=self.dataset_name,
                **langfuse_item
            )
            
            # Add Langfuse item ID to metadata
            if item.metadata is None:
                item.metadata = {}
            item.metadata["lf_item_id"] = created.id
            item.metadata["lf_dataset_name"] = created.dataset_name
            item.metadata["lf_dataset_id"] = created.dataset_id
            
            return item
        
        created_items: List[T] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(upload_item, item) for item in self._items]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Uploading to Langfuse"
            ):
                try:
                    created_items.append(future.result())
                except Exception as e:
                    _logger.error(f"Ulpoad failed for item: {e}")
        
        _logger.debug(f"Successfully uploaded {len(created_items)} items to Langfuse")
        return created_items

    def load(self) -> List[T]:
        """
        Load, validate, and upload items to Langfuse in one step.
        
        :return: List of validated and uploaded Pydantic model instances with lf_item_id in metadata
        """
        super().load()
        return self._upload()