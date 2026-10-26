# License Apache 2.0: (c) 2026 Atena-Reply

from urllib.parse import quote

from openarena.datasets.platform import PlatformDataset, utc


class LangfuseDataset(PlatformDataset):
    """A Langfuse dataset.

    See https://langfuse.com/docs/evaluation/experiments/datasets. Each
    item is handed to the templates as `input`, `expected_output`,
    `metadata` and `id`. Auth comes from the environment
    (`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`).

    Args:
        dataset_name (str): The Langfuse dataset name.
        version (str | datetime): Optional. The UTC point in time to pin the
            items to: Langfuse versions a dataset by timestamp, so an ISO 8601
            string (`"2026-04-15T00:00:00Z"`) or a YAML timestamp. None reads
            the latest state.
        streaming (bool): Read the items page by page as the rows are
            iterated (the default), or the whole dataset at once
            (`False`), which is also when the length is known without a
            `limit`.
        page_size (int): Items per page when streaming. Defaults to 50,
            the SDK's own default.
        input_data_model (DataModel): See `Dataset`.
        input_schema (dict | str): See `Dataset`.
        input_template (str): See `Dataset`.
        output_data_model (DataModel): See `Dataset`.
        output_schema (dict | str): See `Dataset`.
        output_template (str): See `Dataset`.
        tools_template (str): See `Dataset`.
        batch_size (int): Examples per yielded batch. Defaults to 1.
        limit (int): Optional. See `Dataset`.
        repeat (int): See `Dataset`.
    """

    provider = sdk = package = extra = "langfuse"

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
            version=utc(version),
            streaming=streaming,
            page_size=page_size,
        )

    def connect(self):
        client = self.load_sdk().get_client()
        # Whole dataset: fetched once here, its items read back as many times as the rows are.
        self.snapshot = None if self.streaming else client.get_dataset(self.dataset_name, version=self.version)
        return client

    def records(self):
        if self.snapshot is not None:
            return self.snapshot.items
        return self._pages()

    def _pages(self):
        """The items page after page, under the dataset's URL-encoded name, at the pinned version."""
        name = quote(self.dataset_name, safe="")
        page, pages = 1, 1
        while page <= pages:
            listing = self.client.api.dataset_items.list(dataset_name=name, page=page, limit=self.page_size, version=self.version)
            yield from listing.data
            pages = listing.meta.total_pages
            page += 1

    def row(self, item):
        return {"input": item.input, "expected_output": item.expected_output, "metadata": item.metadata, "id": item.id}

    def count(self):
        return None if self.snapshot is None else len(self.snapshot.items)
