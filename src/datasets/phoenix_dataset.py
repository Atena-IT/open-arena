# License Apache 2.0: (c) 2026 Athena-Reply

from src.datasets.dataset import Dataset


class PhoenixDataset(Dataset):
    """Streaming dataset backed by an Arize Phoenix dataset.

    See https://docs.arize.com/phoenix/datasets-and-experiments. Each
    Phoenix `DatasetExample` exposes `id`, `input` (dict), `output` (dict),
    and `metadata` (dict); those names become the variables available to
    the Jinja2 `input_template` / `output_template`. Auth comes from env
    vars (`PHOENIX_API_KEY`, `PHOENIX_COLLECTOR_ENDPOINT` or
    `PHOENIX_HOST` + `PHOENIX_PORT`); set them in `.env`.

    Example:

    ```python
    ds = PhoenixDataset(
        dataset_name="mmlu_subset",
        input_template='{"messages":[{"role":"user","content":{{ input.question | tojson }}}]}',
        output_template='{"role":"assistant","content":{{ output.answer | tojson }}}',
        batch_size=1,
    )
    program.evaluate(x=ds)
    ```

    Args:
        dataset_name (str): The Phoenix dataset name (or ID).
        version_id (str): Optional. The version ID to pin items to a
            snapshot. Defaults to `None` (latest version).
        splits (list): Optional. List of split names to restrict to.
        input_data_model (DataModel): See `Dataset`. Defaults to
            `synalinks.ChatMessages`.
        input_template (str): See `Dataset`.
        output_data_model (DataModel): See `Dataset`. Defaults to
            `synalinks.ChatMessage`.
        output_template (str): See `Dataset`.
        batch_size (int): Examples per yielded batch. Defaults to `1`.
        limit (int): Optional. See `Dataset`. Caps how many items are
            consumed. Phoenix returns the full version in one call, so
            this is enforced client-side.
    """

    def __init__(
        self,
        dataset_name,
        *,
        version_id=None,
        splits=None,
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
        self.version_id = version_id
        self.splits = splits

        # Imported lazily so the project doesn't require the phoenix client
        # unless this dataset type is actually used.
        from phoenix.client import Client

        self._client = Client()
        self._dataset = self._client.datasets.get_dataset(
            dataset=dataset_name,
            version_id=version_id,
            splits=splits,
        )

    def _iter_rows(self):
        for ex in self._dataset.examples:
            yield {
                "id": ex["id"],
                "input": ex.get("input") or {},
                "output": ex.get("output") or {},
                "metadata": ex.get("metadata") or {},
            }

    def __len__(self):
        n = self.limit if self.limit is not None else self._dataset.example_count
        return self._total_batches(n)
