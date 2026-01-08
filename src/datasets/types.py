from typing import TypedDict, NotRequired


class DatasetConfig(TypedDict):
    """Configuration for a dataset (metadata only)."""
    dataset_name: str
    source_file: str
    dataset_description: NotRequired[str]
