# License Apache 2.0: (c) 2026 Athena-Reply

import yaml

from src.datasets.braintrust_dataset import BraintrustDataset
from src.datasets.dataset import Dataset
from src.datasets.folder_dataset import FolderDataset
from src.datasets.huggingface_dataset import HuggingFaceDataset
from src.datasets.langfuse_dataset import LangfuseDataset
from src.datasets.langsmith_dataset import LangSmithDataset
from src.datasets.local_dataset import LocalDataset
from src.datasets.opik_dataset import OpikDataset
from src.datasets.phoenix_dataset import PhoenixDataset

_DATASET_TYPES = {
    "braintrust": BraintrustDataset,
    "folder": FolderDataset,
    "huggingface": HuggingFaceDataset,
    "langfuse": LangfuseDataset,
    "langsmith": LangSmithDataset,
    "local": LocalDataset,
    "opik": OpikDataset,
    "phoenix": PhoenixDataset,
}


def get(name: str) -> type[Dataset]:
    """Look up a `Dataset` subclass by its snake_case provider name.

    Args:
        name: Provider identifier, e.g. `"huggingface"`.

    Returns:
        The matching `Dataset` subclass (uninstantiated).

    Raises:
        KeyError: If no provider is registered under `name`.
    """
    cls = _DATASET_TYPES.get(name)
    if cls is None:
        known = ", ".join(sorted(_DATASET_TYPES))
        raise KeyError(f"Unknown dataset provider {name!r}. Known: {known}")
    return cls


def load_dataset_from_yaml(yaml_path: str, name: str | None = None) -> Dataset:
    """Instantiate a dataset declared in a YAML config.

    Expected layout:

    ```yaml
    datasets:
      gsm8k_train:
        type: huggingface
        path: gsm8k
        name: main
        split: train
        input_template: |
          {"question": {{ question | tojson }}}
        output_template: |
          {"answer": {{ answer.split("####")[-1].strip() | tojson }}}
        batch_size: 8
        limit: 1000

    default: gsm8k_train
    ```

    The `type` key dispatches to a `Dataset` subclass via `_DATASET_TYPES`
    (`huggingface`, `local`, `folder`, `langfuse`, `langsmith`, `opik`,
    `phoenix`, `braintrust`). All other keys are forwarded verbatim as
    keyword arguments to the subclass constructor — so any provider-
    specific parameter (HuggingFace `revision` / `streaming` / `data_files`
    / `token`, Langfuse `dataset_name` / `host`, …) can be set from YAML.
    Note that the HF `name` parameter coexists with the top-level dataset
    selector key (the YAML mapping key under `datasets:`); they don't
    collide because they live at different nesting levels.

    `input_data_model` / `output_data_model` are intentionally not exposed
    here — they default to `synalinks.ChatMessages` inside the dataset.

    For task-specific structured output, set `input_schema:` and
    `output_schema:` instead — each takes a literal JSON Schema (as a YAML
    map or as a JSON-encoded string). Rows are then wrapped as
    `synalinks.JsonDataModel(schema=..., json=...)`, the schema flows into
    the LM as a structured-output constraint, and `y_true` / `y_pred` no
    longer carry ChatMessage auxiliary noise (`thinking`, `tool_calls`, …).

    Args:
        yaml_path: Filesystem path to the YAML config.
        name: Selector under `datasets:`. When `None`, the value of the
            top-level `default` key is used.

    Returns:
        A constructed `Dataset` instance, ready to be called: `ds()` returns
        the generator passed to `program.fit(x=...)`.
    """
    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    entries = config.get("datasets")
    if not entries:
        raise ValueError(f"{yaml_path}: missing or empty `datasets:` section.")

    if name is None:
        name = config.get("default")
        if name is None:
            raise ValueError(
                f"{yaml_path}: no `name` provided and no `default` key in config."
            )

    if name not in entries:
        available = ", ".join(sorted(entries))
        raise KeyError(f"{yaml_path}: dataset {name!r} not found. Available: {available}")

    entry = dict(entries[name])
    type_name = entry.pop("type", None)
    if type_name is None:
        raise ValueError(f"{yaml_path}: dataset {name!r} is missing a `type` field.")

    # `generator:` and `reward:` are consumed by main.py (per-dataset
    # Generator kwargs and Reward spec), not by the Dataset constructor —
    # strip them before forwarding kwargs.
    entry.pop("generator", None)
    entry.pop("reward", None)

    try:
        cls = get(type_name)
    except KeyError as e:
        raise ValueError(f"{yaml_path}: dataset {name!r}: {e}") from e

    return cls(**entry)
