# License Apache 2.0: (c) 2026 Atena-Reply

from openarena.datasets.platform import PlatformDataset


class BraintrustDataset(PlatformDataset):
    """A Braintrust dataset.

    See https://www.braintrust.dev/docs/guides/datasets. Each record is
    handed to the templates as `id`, `input`, `expected` and `metadata`
    (a mapping, empty when the record has none). Auth comes from the
    environment (`BRAINTRUST_API_KEY`, and `BRAINTRUST_APP_URL`,
    `BRAINTRUST_ORG_NAME` when needed).

    Args:
        project (str): The Braintrust project name; or give `project_id`.
        dataset_name (str): The dataset's name within the project.
        version (str | int): Optional. A transaction id to pin the records
            to. None reads the latest state.
        project_id (str): Optional. The project's UUID, instead of its
            name; wins over `project` when both are set.
        org_name (str): Optional. The organization, for a user in several.
            Falls back to `BRAINTRUST_ORG_NAME`.
        fetch_batch_size (int): Optional. Records per request to the API;
            the SDK's own default when unset.
        input_data_model (DataModel): See `Dataset`.
        input_schema (dict | str): See `Dataset`.
        input_template (str): See `Dataset`.
        output_data_model (DataModel): See `Dataset`.
        output_schema (dict | str): See `Dataset`.
        output_template (str): See `Dataset`.
        tools_template (str): See `Dataset`.
        batch_size (int): Examples per yielded batch. Defaults to 1.
        limit (int): Optional. See `Dataset`. Needed for a length: Braintrust
            gives no count without a full scan.
        repeat (int): See `Dataset`.
    """

    provider = sdk = package = extra = "braintrust"

    def __init__(
        self,
        project=None,
        dataset_name=None,
        *,
        version=None,
        project_id=None,
        org_name=None,
        fetch_batch_size=None,
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
        if project is None and project_id is None:
            raise ValueError("BraintrustDataset requires `project` or `project_id`.")
        if dataset_name is None:
            raise ValueError("BraintrustDataset requires `dataset_name`.")
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
            project=project,
            dataset_name=dataset_name,
            version=version,
            project_id=project_id,
            org_name=org_name,
            fetch_batch_size=fetch_batch_size,
        )

    def connect(self):
        return self.load_sdk().init_dataset(
            project=self.project, project_id=self.project_id, name=self.dataset_name, version=self.version, org_name=self.org_name
        )

    def records(self):
        request = {} if self.fetch_batch_size is None else {"batch_size": self.fetch_batch_size}
        return self.client.fetch(**request)

    def row(self, record):
        return {"id": record.get("id"), "input": record.get("input"), "expected": record.get("expected"), "metadata": record.get("metadata") or {}}
