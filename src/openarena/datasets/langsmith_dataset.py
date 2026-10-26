# License Apache 2.0: (c) 2026 Atena-Reply

from datetime import datetime

from openarena.datasets.platform import PlatformDataset, utc


class LangSmithDataset(PlatformDataset):
    """A LangSmith dataset.

    See https://docs.smith.langchain.com/evaluation/concepts#datasets. Each
    example is handed to the templates as `inputs`, `outputs`, `metadata`
    (each a mapping, empty when the example has none) and `id`. Auth comes
    from the environment (`LANGSMITH_API_KEY`, `LANGSMITH_ENDPOINT`).

    Args:
        dataset_name (str): The LangSmith dataset name.
        as_of (str | datetime): Optional. A version tag (`"prod"`), or a
            UTC datetime to read the dataset as it was then. None reads
            the latest state.
        input_data_model (DataModel): See `Dataset`.
        input_schema (dict | str): See `Dataset`.
        input_template (str): See `Dataset`.
        output_data_model (DataModel): See `Dataset`.
        output_schema (dict | str): See `Dataset`.
        output_template (str): See `Dataset`.
        tools_template (str): See `Dataset`.
        batch_size (int): Examples per yielded batch. Defaults to 1.
        limit (int): Optional. See `Dataset`. Needed for a length: LangSmith
            gives no count under `as_of` without a full scan.
        repeat (int): See `Dataset`.
    """

    provider = sdk = package = extra = "langsmith"

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
        tools_template=None,
        batch_size=1,
        limit: int = None,
        repeat: int = 1,
    ):
        if not isinstance(as_of, (str, datetime, type(None))):
            raise TypeError(f"`as_of` must be str (version tag), datetime, or None — got {type(as_of).__name__}")
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
            as_of=as_of if isinstance(as_of, str) else utc(as_of),
        )

    def connect(self):
        return self.load_sdk().Client()

    def records(self):
        return self.client.list_examples(dataset_name=self.dataset_name, as_of=self.as_of)

    def row(self, example):
        return {"inputs": example.inputs or {}, "outputs": example.outputs or {}, "metadata": example.metadata or {}, "id": str(example.id)}
