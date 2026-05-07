# License Apache 2.0: (c) 2026 Athena-Reply

from datetime import datetime
from datetime import timezone

from src.datasets.dataset import Dataset


def _coerce_as_of(as_of):
    """Pass tag strings through; coerce naive datetimes to UTC."""
    if as_of is None or isinstance(as_of, str):
        return as_of
    if isinstance(as_of, datetime):
        if as_of.tzinfo is None:
            return as_of.replace(tzinfo=timezone.utc)
        return as_of
    raise TypeError(
        f"`as_of` must be str (version tag), datetime, or None — "
        f"got {type(as_of).__name__}"
    )


class LangSmithDataset(Dataset):
    """Streaming dataset backed by a LangSmith dataset.

    See https://docs.smith.langchain.com/evaluation/concepts#datasets. Each
    LangSmith `Example` exposes `inputs` (dict), `outputs` (dict | None),
    `metadata` (dict), and `id`; those names become the variables available
    to the Jinja2 `input_template` / `output_template`. Auth comes from env
    vars (`LANGSMITH_API_KEY`, `LANGSMITH_ENDPOINT`); set them in `.env`.

    Example:

    ```python
    ds = LangSmithDataset(
        dataset_name="mmlu_subset",
        input_template='{"messages":[{"role":"user","content":{{ inputs.question | tojson }}}]}',
        output_template='{"role":"assistant","content":{{ outputs.answer | tojson }}}',
        batch_size=1,
    )
    program.evaluate(x=ds)
    ```

    Args:
        dataset_name (str): The LangSmith dataset name.
        as_of (str | datetime): Optional. Point-in-time snapshot — either a
            version tag string (e.g. `"prod"`) or a UTC tz-aware datetime
            for time-travel. Defaults to `None` (latest state).
        input_data_model (DataModel): See `Dataset`. Defaults to
            `synalinks.ChatMessages`.
        input_template (str): See `Dataset`.
        output_data_model (DataModel): See `Dataset`. Defaults to
            `synalinks.ChatMessage`.
        output_template (str): See `Dataset`.
        batch_size (int): Examples per yielded batch. Defaults to `1`.
        limit (int): Optional. See `Dataset`. Caps how many items are
            consumed; required for `__len__` since LangSmith doesn't
            expose a count under `as_of` without a full scan.
    """

    def __init__(
        self,
        dataset_name,
        *,
        as_of=None,
        input_data_model=None,
        input_schema=None,
        input_template=None,
        output_data_model=None,
        output_schema=None,
        output_template=None,
        batch_size=1,
        limit: int = None,
        repeat: int = 1,
    ):
        super().__init__(
            input_data_model=input_data_model,
            input_schema=input_schema,
            input_template=input_template,
            output_data_model=output_data_model,
            output_schema=output_schema,
            output_template=output_template,
            batch_size=batch_size,
            limit=limit,
            repeat=repeat,
        )
        self.dataset_name = dataset_name
        self.as_of = _coerce_as_of(as_of)

        # Imported lazily so the project doesn't require the langsmith package
        # unless this dataset type is actually used.
        from langsmith import Client

        self._client = Client()

    def _iter_rows(self):
        for ex in self._client.list_examples(
            dataset_name=self.dataset_name,
            as_of=self.as_of,
        ):
            yield {
                "inputs": ex.inputs or {},
                "outputs": ex.outputs or {},
                "metadata": ex.metadata or {},
                "id": str(ex.id),
            }

    def __len__(self):
        if self.limit is None:
            raise NotImplementedError(
                "LangSmith dataset length is unknown without a full scan; "
                "set `limit` for a bounded epoch."
            )
        return self._total_batches(self.limit)
