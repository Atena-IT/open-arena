# License Apache 2.0: (c) 2026 Athena-Reply

from src.datasets.dataset import Dataset


class OpikDataset(Dataset):
    """Streaming dataset backed by an Opik (Comet) dataset.

    See https://www.comet.com/docs/opik/evaluation/manage_datasets. Each
    Opik dataset item is exposed as a dict — the standard fields are
    `input`, `expected_output`, `metadata`, plus `id` — and any custom
    fields the user added at insertion time are forwarded verbatim. Those
    keys become the Jinja2 variables available to `input_template` /
    `output_template`. Auth comes from env vars (`OPIK_API_KEY`,
    `OPIK_WORKSPACE`, `OPIK_URL_OVERRIDE`); set them in `.env`.

    Example:

    ```python
    ds = OpikDataset(
        dataset_name="mmlu_subset",
        input_template='{"messages":[{"role":"user","content":{{ input.question | tojson }}}]}',
        output_template='{"role":"assistant","content":{{ expected_output.answer | tojson }}}',
        batch_size=1,
    )
    program.evaluate(x=ds)
    ```

    Args:
        dataset_name (str): The Opik dataset name.
        version (str): Optional. The version name (e.g. `"v1"`) to pin
            items to a named snapshot via `Dataset.get_version_view`.
            Defaults to `None` (latest live state).
        filter_string (str): Optional. Opik OQL filter applied server-side
            (e.g. `'tags contains "regression"'`).
        input_data_model (DataModel): See `Dataset`. Defaults to
            `synalinks.ChatMessages`.
        input_template (str): See `Dataset`.
        output_data_model (DataModel): See `Dataset`. Defaults to
            `synalinks.ChatMessage`.
        output_template (str): See `Dataset`.
        batch_size (int): Examples per yielded batch. Defaults to `1`.
        limit (int): Optional. See `Dataset`. Caps how many items are
            consumed. When set, it is forwarded to Opik as `nb_samples` so
            only that many rows are fetched.
    """

    def __init__(
        self,
        dataset_name,
        *,
        version=None,
        filter_string=None,
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
        self.version = version
        self.filter_string = filter_string

        # Imported lazily so the project doesn't require the opik package
        # unless this dataset type is actually used.
        from opik import Opik

        self._client = Opik()
        ds = self._client.get_dataset(name=dataset_name)
        self._dataset = ds.get_version_view(version) if version else ds

    def _iter_rows(self):
        # `get_items` returns a fully-materialized list, so passing
        # `nb_samples=limit` keeps the wire payload bounded.
        yield from self._dataset.get_items(
            nb_samples=self.limit,
            filter_string=self.filter_string,
        )

    def __len__(self):
        if self.limit is not None:
            n = self.limit
        else:
            n = self._dataset.dataset_items_count
        return self._total_batches(n)
