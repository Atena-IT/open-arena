# License Apache 2.0: (c) 2026 Atena-Reply

"""The dataset providers, and a registry entry built into one of them.

A dataset entry names its provider by `type`; every other key of its flat
mapping is a setting of that provider or a template, except what the sweep
reads at the top level (the kind block, `reward`, `metrics`), which is
stripped before the provider sees it.
"""

from __future__ import annotations

from typing import Any

from openarena.config.entries import KIND_BLOCKS
from openarena.datasets.braintrust_dataset import BraintrustDataset
from openarena.datasets.dataset import Dataset
from openarena.datasets.folder_dataset import FolderDataset
from openarena.datasets.harbor_tasks_dataset import HarborTasksDataset
from openarena.datasets.huggingface_dataset import HuggingFaceDataset
from openarena.datasets.langfuse_dataset import LangfuseDataset
from openarena.datasets.langsmith_dataset import LangSmithDataset
from openarena.datasets.local_dataset import LocalDataset
from openarena.datasets.mlflow_dataset import MLflowDataset
from openarena.datasets.opik_dataset import OpikDataset
from openarena.datasets.phoenix_dataset import PhoenixDataset
from openarena.datasets.unity_catalog_dataset import UnityCatalogDataset

PROVIDERS: dict[str, type[Dataset]] = {
    "braintrust": BraintrustDataset,
    "folder": FolderDataset,
    "harbor_tasks": HarborTasksDataset,  # task folders, one row each, scored by their own tests
    "huggingface": HuggingFaceDataset,
    "langfuse": LangfuseDataset,
    "langsmith": LangSmithDataset,
    "local": LocalDataset,
    "mlflow": MLflowDataset,
    "opik": OpikDataset,
    "phoenix": PhoenixDataset,
    "unity_catalog": UnityCatalogDataset,
}
# The name the platform and the checks read the registry under.
_DATASET_TYPES = PROVIDERS
# What an entry carries for the sweep, not for the provider.
SWEEP_KEYS = (*KIND_BLOCKS, "reward", "metrics")


def get(name: str) -> type[Dataset]:
    """The provider registered as `name`; `KeyError` naming the known ones otherwise."""
    if name not in PROVIDERS:
        raise KeyError(f"Unknown dataset provider {name!r}. Known: {', '.join(sorted(PROVIDERS))}")
    return PROVIDERS[name]


def provider_settings(entry: dict[str, Any]) -> dict[str, Any]:
    """An entry's keys as the provider's constructor takes them: the sweep's keys out, an embedding dataset's shapes in."""
    settings = {key: value for key, value in entry.items() if key not in SWEEP_KEYS}
    if entry.get("embedding") is not None:
        # An embedding dataset's rows are a query and the chunks to search, its target the gold chunks: the program's
        # own shapes unless the entry says otherwise, since `{query, chunks}` is no chat message.
        from openarena.programs.embedding_program import INPUT_SCHEMA, OUTPUT_SCHEMA

        if settings.get("input_schema") is None and settings.get("input_data_model") is None:
            settings["input_schema"] = INPUT_SCHEMA
        if settings.get("output_template") is not None and settings.get("output_schema") is None and settings.get("output_data_model") is None:
            settings["output_schema"] = OUTPUT_SCHEMA
    return settings


def load_dataset_from_config(config: dict, name: str | None = None, yaml_path: str = "<config>") -> Dataset:
    """The dataset `name` of a parsed config (`{"datasets": {name: entry}}`, `default` naming one), built by its provider.

    `yaml_path` only labels the errors. A caller building many datasets
    parses the file once and calls this per dataset.
    """
    entries = config.get("datasets")
    if not entries:
        raise ValueError(f"{yaml_path}: missing or empty `datasets:` section.")
    name = config.get("default") if name is None else name
    if name is None:
        raise ValueError(f"{yaml_path}: no `name` provided and no `default` key in config.")
    if name not in entries:
        raise KeyError(f"{yaml_path}: dataset {name!r} not found. Available: {', '.join(sorted(entries))}")
    entry = dict(entries[name])
    provider = entry.pop("type", None)
    if provider is None:
        raise ValueError(f"{yaml_path}: dataset {name!r} is missing a `type` field.")
    try:
        cls = get(provider)
    except KeyError as e:
        raise ValueError(f"{yaml_path}: dataset {name!r}: {e}") from e
    return cls(**provider_settings(entry))
