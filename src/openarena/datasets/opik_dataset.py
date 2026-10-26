# License Apache 2.0: (c) 2026 Atena-Reply

from openarena.datasets.platform import PlatformDataset


class OpikDataset(PlatformDataset):
    """An Opik (Comet) dataset.

    See https://www.comet.com/docs/opik/evaluation/manage_datasets. Each
    item is handed to the templates as the mapping Opik keeps: `input`,
    `expected_output`, `metadata`, `id`, and any custom field added at
    insertion. Auth comes from the environment (`OPIK_API_KEY`,
    `OPIK_WORKSPACE`, `OPIK_URL_OVERRIDE`).

    Args:
        dataset_name (str): The Opik dataset name.
        version (str): Optional. A version name (`"v1"`) to read the
            dataset's snapshot of that name. None reads the live state.
        filter_string (str): Optional. An Opik OQL filter, applied on the
            server (`'tags contains "regression"'`).
        input_data_model (DataModel): See `Dataset`.
        input_schema (dict | str): See `Dataset`.
        input_template (str): See `Dataset`.
        output_data_model (DataModel): See `Dataset`.
        output_schema (dict | str): See `Dataset`.
        output_template (str): See `Dataset`.
        tools_template (str): See `Dataset`.
        batch_size (int): Examples per yielded batch. Defaults to 1.
        limit (int): Optional. See `Dataset`; asked of Opik too, so only
            that many items come over the wire.
        repeat (int): See `Dataset`.
    """

    provider = sdk = package = extra = "opik"

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
        tools_template=None,
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
            tools_template=tools_template,
            batch_size=batch_size,
            limit=limit,
            repeat=repeat,
            dataset_name=dataset_name,
            version=version,
            filter_string=filter_string,
        )

    def connect(self):
        dataset = self.load_sdk().Opik().get_dataset(name=self.dataset_name)
        return dataset.get_version_view(self.version) if self.version else dataset

    def records(self):
        # Opik materializes the whole list: `nb_samples` keeps it to the rows the run reads.
        return self.client.get_items(nb_samples=self.limit, filter_string=self.filter_string)

    def count(self):
        return self.client.dataset_items_count
