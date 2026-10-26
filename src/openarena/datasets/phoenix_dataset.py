# License Apache 2.0: (c) 2026 Atena-Reply

from openarena.datasets.platform import PlatformDataset


class PhoenixDataset(PlatformDataset):
    """An Arize Phoenix dataset.

    See https://docs.arize.com/phoenix/datasets-and-experiments. Each
    example is handed to the templates as `id`, `input`, `output` and
    `metadata` (each a mapping, empty when the example has none). Auth
    comes from the environment (`PHOENIX_API_KEY`, and
    `PHOENIX_COLLECTOR_ENDPOINT` or `PHOENIX_HOST` + `PHOENIX_PORT`).

    Args:
        dataset_name (str): The Phoenix dataset name, or its id.
        version_id (str): Optional. The version to read. None reads the
            latest.
        splits (list): Optional. The split names to keep.
        input_data_model (DataModel): See `Dataset`.
        input_schema (dict | str): See `Dataset`.
        input_template (str): See `Dataset`.
        output_data_model (DataModel): See `Dataset`.
        output_schema (dict | str): See `Dataset`.
        output_template (str): See `Dataset`.
        tools_template (str): See `Dataset`.
        batch_size (int): Examples per yielded batch. Defaults to 1.
        limit (int): Optional. See `Dataset`. Phoenix sends the whole
            version in one call; the cap applies on this side.
        repeat (int): See `Dataset`.
    """

    provider = sdk = extra = "phoenix"
    package = "arize-phoenix-client"

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
            version_id=version_id,
            splits=splits,
        )

    def connect(self):
        self.load_sdk()
        from phoenix.client import Client

        return Client().datasets.get_dataset(dataset=self.dataset_name, version_id=self.version_id, splits=self.splits)

    def records(self):
        return self.client.examples

    def row(self, example):
        return {"id": example["id"], "input": example.get("input") or {}, "output": example.get("output") or {}, "metadata": example.get("metadata") or {}}

    def count(self):
        return self.client.example_count
