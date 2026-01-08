import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Type, Generic, Any, Optional, TypeVar, TypedDict
from langfuse import Langfuse
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

@dataclass
class LangfuseDatasetItemMeta(TypedDict, Generic[T]):
    """Langfuse metadata wrapper for dataset items."""
    item: T
    lf_item_id: str

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
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: Optional[str] = None
    ):
        """
        :param item_model: Pydantic model class for dataset items
        :param reader: Reader instance to use for loading data
        :param config: Dataset configuration
        :param input_path: Base path for data files
        :param max_items: Maximum number of items to upload (None = all)
        :param max_workers: Number of parallel workers for upload
        :param public_key: Langfuse public key (defaults to env var)
        :param secret_key: Langfuse secret key (defaults to env var)
        :param host: Langfuse host (defaults to env var)
        """
        super().__init__(item_model, reader, config, input_path)
        
        self.max_items = max_items
        self.max_workers = max_workers
        self.dataset_description = config.get("dataset_description", "")
        
        # Initialize Langfuse client
        self.langfuse = Langfuse(
            public_key=public_key or os.getenv("LANGFUSE_PUBLIC_KEY", ""),
            secret_key=secret_key or os.getenv("LANGFUSE_SECRET_KEY", ""),
            host=host or os.getenv("LANGFUSE_HOST", ""),
        )
    
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
        
        # Categorize fields
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
            _logger.info(f"Creating new Langfuse dataset '{self.dataset_name}'")
            self.langfuse.create_dataset(
                name=self.dataset_name,
                description=self.dataset_description
            )
    
    def upload(self, items: Optional[List[T]] = None) -> List[LangfuseDatasetItemMeta[T]]:
        """
        Upload items to Langfuse dataset.
        
        :param items: Optional list of items to upload (uses self._items if None)
        :return: List of created Langfuse dataset items
        """
        items_to_upload = items if items is not None else self._items
        
        if not items_to_upload:
            _logger.warning("No items to upload to Langfuse")
            return []
        
        # Apply max_items limit if specified
        if self.max_items:
            items_to_upload = items_to_upload[:self.max_items]
        
        _logger.info(f"Uploading {len(items_to_upload)} items to Langfuse dataset '{self.dataset_name}'")
        
        # Ensure dataset exists
        self._ensure_dataset_exists()
        
        # Upload items in parallel
        def upload_item(item: T) -> LangfuseDatasetItemMeta[T]:
            langfuse_item = self.convert_type(item)
            created = self.langfuse.create_dataset_item(
                dataset_name=self.dataset_name,
                **langfuse_item
            )
            return {
                "item": item, 
                "lf_item_id": created.id
            }
        
        created_items: List[LangfuseDatasetItemMeta[T]] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(upload_item, item) for item in items_to_upload]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Uploading to Langfuse"
            ):
                created_items.append(future.result())
        
        _logger.info(f"Successfully uploaded {len(created_items)} items to Langfuse")
        return created_items
