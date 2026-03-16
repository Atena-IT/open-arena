import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TypeVar
from langfuse import get_client
from tqdm import tqdm

from src.datasets.item_models import DatasetItem
from src.datasets.loaders.dataset_loader import DatasetLoader
from src.datasets.readers.base_reader import DatasetReader
from src.datasets.types import DatasetConfig

_logger = logging.getLogger(__name__)
T = TypeVar('T', bound=DatasetItem)

class LangfuseLoader(DatasetLoader[T]):
    """
    DatasetLoader with Langfuse integration.
    Extends base loader to upload validated items to Langfuse.
    """
    
    def __init__(
        self,
        item_model: type[T],
        reader: DatasetReader,
        config: DatasetConfig,
        input_path: str = ".",
        max_items: int | None = None,
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
    
    def _upload(self) -> list[T]:
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
            created = self.langfuse.create_dataset_item(
                dataset_name=self.dataset_name,
                input=item.input(),
                expected_output=item.expected_output(),
                metadata=item.meta()
            )
            
            item.metadata["lf_item_id"] = created.id
            item.metadata["lf_dataset_name"] = created.dataset_name
            item.metadata["lf_dataset_id"] = created.dataset_id
            
            return item
        
        created_items: list[T] = []
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
                    _logger.error(f"Upload failed for item: {e}")
        
        _logger.debug(f"Successfully uploaded {len(created_items)} items to Langfuse")
        return created_items

    def load(self) -> list[T]:
        """
        Load, validate, and upload items to Langfuse in one step.
        
        :return: List of validated and uploaded Pydantic model instances with lf_item_id in metadata
        """
        super().load()
        return self._upload()