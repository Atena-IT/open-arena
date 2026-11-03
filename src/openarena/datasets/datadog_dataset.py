# License Apache 2.0: (c) 2026 Atena-Reply

import os

from openarena.datasets.platform import PlatformDataset, listing

DEFAULT_PROJECT = "default-project"


def _named(found: list[dict], name: str) -> list[dict]:
    """The hits of a `filter[name]` lookup that carry exactly `name`; one without a name is taken at its word."""
    return [f for f in found if (f.get("attributes") or {}).get("name", name) == name]


def _get(client, path, params):
    import httpx

    response = httpx.get(f"{client['api']}/{path}", params=params, headers=client["headers"], timeout=30)
    response.raise_for_status()
    return response.json()


def _paged(client, path, params):
    """Every entry under `path`, page after page, following `meta.after` as the cursor."""
    while True:
        page = _get(client, path, params)
        yield from page["data"]
        cursor = (page.get("meta") or {}).get("after")
        if not cursor:
            return
        params = {**params, "page[cursor]": cursor}


def _reach() -> dict:
    """The API's address on `DD_SITE` and the two keys' headers; `OSError` naming a key that is not set."""
    for key in ("DD_API_KEY", "DD_APP_KEY"):
        if not os.environ.get(key):
            raise OSError(f"{key} is not set. A Datadog dataset is read with an API key and an application key.")
    site = os.environ.get("DD_SITE") or "datadoghq.com"
    return {
        "api": f"https://api.{site}/api/v2/llm-obs/v1",
        "headers": {"DD-API-KEY": os.environ["DD_API_KEY"], "DD-APPLICATION-KEY": os.environ["DD_APP_KEY"]},
    }


def _project(client, name) -> str:
    """The id of the project called `name`. Looked up, never created: ddtrace's pull_dataset would create a missing one in the customer's org."""
    projects = _named(_get(client, "projects", {"filter[name]": name})["data"], name)
    if not projects:
        raise ValueError(f"no Datadog project named {name!r}; set `project_name` or DD_LLMOBS_PROJECT_NAME.")
    return projects[0]["id"]


def _project_name(given=None) -> str:
    return given or os.environ.get("DD_LLMOBS_PROJECT_NAME") or DEFAULT_PROJECT


class DatadogDataset(PlatformDataset):
    """A Datadog Agent Observability (LLM Observability) dataset.

    See https://docs.datadoghq.com/llm_observability/experiments/datasets/.
    Read over Datadog's REST API, no SDK: the project and the dataset are
    found by name, never created, and the records are paged. Each record is
    handed to the templates as `input_data`, `expected_output`, `metadata`
    (a mapping, empty when the record has none) and `id`, ddtrace's names.
    Auth comes from the environment: `DD_API_KEY`, `DD_APP_KEY` and
    `DD_SITE` (`datadoghq.com` by default).

    Args:
        dataset_name (str): The Datadog dataset name.
        project_name (str): Optional. The project that holds the dataset.
            Defaults to `DD_LLMOBS_PROJECT_NAME`, else `default-project`.
        version (int): Optional. The dataset version to read. None reads
            the current one.
        input_data_model (DataModel): See `Dataset`.
        input_schema (dict | str): See `Dataset`.
        input_template (str): See `Dataset`.
        output_data_model (DataModel): See `Dataset`.
        output_schema (dict | str): See `Dataset`.
        output_template (str): See `Dataset`.
        tools_template (str): See `Dataset`.
        batch_size (int): Records per yielded batch. Defaults to 1.
        limit (int): Optional. See `Dataset`. Needed for a length: the API
            gives no record count.
        repeat (int): See `Dataset`.
    """

    provider = "datadog"

    def __init__(
        self,
        dataset_name,
        *,
        project_name=None,
        version: int = None,
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
        if isinstance(version, str) and version.strip().isdigit():
            version = int(version)  # a quoted YAML value, or the assistant's string argument
        if version is not None and (
            not isinstance(version, int) or isinstance(version, bool)
        ):
            raise TypeError(f"`version` must be an int; got {type(version).__name__}")
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
            project_name=_project_name(project_name),
            version=version,
        )

    def connect(self):
        self.client = _reach()
        project = _project(self.client, self.project_name)
        datasets = _named(_get(self.client, f"{project}/datasets", {"filter[name]": self.dataset_name})["data"], self.dataset_name)
        if not datasets:
            raise ValueError(
                f"no dataset named {self.dataset_name!r} in the Datadog project {self.project_name!r}."
            )
        self.client["records"] = f"{project}/datasets/{datasets[0]['id']}/records"
        return self.client

    def records(self):
        params = {} if self.version is None else {"filter[version]": self.version}
        return _paged(self.client, self.client["records"], params)

    @classmethod
    def list_datasets(cls, query="", limit=100):
        """The datasets of the configured project (`DD_LLMOBS_PROJECT_NAME`, else `default-project`); see `PlatformDataset`."""
        client = _reach()
        found = (d.get("attributes") or {} for d in _paged(client, f"{_project(client, _project_name())}/datasets", {}))
        return listing(((a.get("name", ""), a.get("description"), None) for a in found), query, limit)

    def row(self, record):
        # The API reference has a record's fields at the top level, ddtrace reads them under `attributes`.
        fields = record.get("attributes", record)
        return {
            "input_data": fields.get("input"),
            "expected_output": fields.get("expected_output"),
            "metadata": fields.get("metadata") or {},
            "id": str(record["id"]),
        }
