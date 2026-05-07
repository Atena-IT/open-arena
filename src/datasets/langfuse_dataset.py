# License Apache 2.0: (c) 2026 Athena-Reply

from datetime import datetime
from datetime import timezone

from src.datasets.dataset import Dataset


def _coerce_version(version):
    """Accept str (ISO 8601) or datetime; return tz-aware UTC datetime or None."""
    if version is None or isinstance(version, datetime):
        if isinstance(version, datetime) and version.tzinfo is None:
            return version.replace(tzinfo=timezone.utc)
        return version
    parsed = datetime.fromisoformat(str(version).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


class LangfuseDataset(Dataset):
    """Streaming dataset backed by a Langfuse dataset.

    See https://langfuse.com/docs/evaluation/experiments/datasets. Each
    Langfuse `DatasetItem` exposes `input`, `expected_output`, `metadata`,
    and `id`; those names become the variables available to the Jinja2
    `input_template` / `output_template`. Auth comes from env vars
    (`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`); set
    them in `.env`.

    Example:

    ```python
    ds = LangfuseDataset(
        dataset_name="mmlu_subset",
        input_template='{"messages":[{"role":"user","content":{{ input | tojson }}}]}',
        output_template='{"role":"assistant","content":{{ expected_output | tojson }}}',
        batch_size=1,
    )
    program.evaluate(x=ds)
    ```

    Args:
        dataset_name (str): The Langfuse dataset name
            (`langfuse.get_dataset(name)`).
        version (str | datetime): Optional. The UTC point-in-time snapshot
            to pin items to (Langfuse datasets are versioned by timestamp,
            not by string tag — see `Langfuse.get_dataset(version=...)`).
            Accepts an ISO 8601 string (`"2026-04-15T00:00:00Z"`), a YAML
            timestamp (already parsed to `datetime`), or `None` for the
            latest state.
        streaming (bool): If `True` (default), fetch items one page at a
            time as the dataset is iterated — items don't need to fit in
            memory and `__len__` is unknown unless `limit` is set. If
            `False`, prefetch the entire dataset upfront via
            `Langfuse.get_dataset(...)`.
        page_size (int): Items per page request when streaming. Defaults
            to `50`, matching the SDK's `fetch_items_page_size` default.
        input_data_model (DataModel): See `Dataset`. Defaults to
            `synalinks.ChatMessages`.
        input_template (str): See `Dataset`.
        output_data_model (DataModel): See `Dataset`. Defaults to
            `synalinks.ChatMessage`.
        output_template (str): See `Dataset`.
        batch_size (int): Examples per yielded batch. Defaults to `1`.
        limit (int): Optional. See `Dataset`. Caps how many items are
            consumed.
    """

    def __init__(
        self,
        dataset_name,
        *,
        version=None,
        streaming=True,
        page_size=50,
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
        self.version = _coerce_version(version)
        self.streaming = streaming
        self.page_size = page_size

        # Imported lazily so the project doesn't require the langfuse package
        # unless this dataset type is actually used.
        from langfuse import get_client

        self._client = get_client()
        self._dataset = (
            None
            if streaming
            else self._client.get_dataset(dataset_name, version=self.version)
        )

    def _iter_rows(self):
        if not self.streaming:
            for item in self._dataset.items:
                yield self._item_to_row(item)
            return

        from urllib.parse import quote

        encoded_name = quote(self.dataset_name, safe="")
        page = 1
        while True:
            page_data = self._client.api.dataset_items.list(
                dataset_name=encoded_name,
                page=page,
                limit=self.page_size,
                version=self.version,
            )
            for item in page_data.data:
                yield self._item_to_row(item)
            if page_data.meta.total_pages <= page:
                break
            page += 1

    @staticmethod
    def _item_to_row(item):
        return {
            "input": item.input,
            "expected_output": item.expected_output,
            "metadata": item.metadata,
            "id": item.id,
        }

    def __len__(self):
        if self.streaming and self.limit is None:
            raise NotImplementedError(
                "Streaming Langfuse datasets have unknown length; set `limit` "
                "or use `streaming=False`."
            )
        if self.limit is not None:
            n = self.limit
        else:
            n = len(self._dataset.items)
        return self._total_batches(n)
