# License Apache 2.0: (c) 2026 Atena-Reply

"""A Unity Catalog table as a dataset: the table found through the catalog's REST API, its files read from object storage."""

from __future__ import annotations

import os

from openarena.datasets.platform import PlatformDataset

# Where the catalog is, and how the files behind a table are reached: all from the environment.
API_URL = "UNITY_CATALOG_API_URL"  # the REST base URL, `/api/2.1/unity-catalog` included
TOKEN = "UC_TOKEN"  # the bearer token of the REST API
# What `metadata` keeps of a resolved table.
METADATA_KEYS = ("storage.uri", "content_hash", "table_type", "data_source_format")
# The formats the catalog may say a table's files are in; anything else is read as Delta, the usual case.
PLAIN_FORMATS = {"PARQUET", "CSV", "JSON", "ORC", "AVRO"}


def table_parts(full_name: str) -> tuple[str, str, str]:
    """`(catalog, schema, table)` of a fully-qualified table name."""
    parts = tuple(full_name.split("."))
    if len(parts) != 3:
        raise ValueError(f"`table` must be fully-qualified as 'catalog.schema.table'; got {full_name!r}.")
    return parts


def storage_options() -> dict:
    """How object storage is reached, for Delta Lake and pyarrow: the AWS variables, and `S3_ENDPOINT` for an S3-compatible store."""
    options = {
        "region": os.environ.get("AWS_DEFAULT_REGION", "fr-par"),
        "aws_access_key_id": os.environ.get("AWS_ACCESS_KEY_ID", ""),
        "aws_secret_access_key": os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
    }
    if endpoint := os.environ.get("S3_ENDPOINT", ""):
        # A MinIO or Scaleway endpoint, over plain HTTP when it says so.
        options.update(endpoint_url=endpoint, allow_http=endpoint.startswith("http://"))
    return options


def lookup_table(full_name: str) -> dict:
    """What the catalog says of a table, over REST: `{storage.uri, content_hash, table_type, data_source_format}`."""
    import httpx

    api_url = os.environ.get(API_URL, "").rstrip("/")
    token = os.environ.get(TOKEN, "")
    if not api_url:
        raise OSError(
            f"{API_URL} is not set. It must point to the Unity Catalog REST API base URL, e.g. 'https://my-uc-host/api/2.1/unity-catalog'."
        )
    if not token:
        raise OSError(f"{TOKEN} is not set. Provide a bearer token for the Unity Catalog REST API.")
    response = httpx.get(f"{api_url}/tables/{full_name}", headers={"Authorization": f"Bearer {token}"}, timeout=30)
    response.raise_for_status()
    info = response.json()
    # The table's version token: the last Delta commit, else when the catalog last saw it change.
    stamp = (info.get("properties") or {}).get("delta.lastCommitTimestamp", info.get("updated_at", ""))
    return {
        "storage.uri": info.get("storage_location", ""),
        "content_hash": str(stamp),
        "table_type": info.get("table_type", ""),
        "data_source_format": info.get("data_source_format", ""),
    }


def filesystem_for(path: str, options: dict):
    """The pyarrow filesystem `path` lives on: S3 from `options` for an `s3://` path, else whatever the path says."""
    from pyarrow import fs

    if not path.startswith("s3://"):
        return fs.FileSystem.from_uri(path)[0]
    s3 = {"region": options.get("region", "fr-par")}
    for option, key in (("aws_access_key_id", "access_key"), ("aws_secret_access_key", "secret_key"), ("endpoint_url", "endpoint_override")):
        if options.get(option):
            s3[key] = options[option]
    return fs.S3FileSystem(**s3)


class UnityCatalogDataset(PlatformDataset):
    """A Delta or Parquet table governed by Unity Catalog (Databricks or OSS).

    Each row of the table is handed to the templates as a mapping of its
    columns. The table is found through the catalog's REST API (its
    storage location, its version token), then read from object storage
    with `deltalake` and `pyarrow`, record batch by record batch, so the
    table never has to fit in memory. The catalog is reached with
    `UNITY_CATALOG_API_URL` and `UC_TOKEN`; the storage with
    `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`
    (`fr-par` unless set) and, for an S3-compatible store, `S3_ENDPOINT`.
    What the catalog said of the table is kept in `metadata`, for the run
    to record where its rows came from.

    Args:
        table (str): The table, fully qualified: `catalog.schema.table`.
        version (int): Optional. The Delta version to read. Not with
            `as_of`.
        as_of (str): Optional. The UTC time (ISO 8601) to read the Delta
            table as of. Not with `version`.
        storage_path (str): Optional. Where the table's files are
            (`s3://...`), given directly: the catalog is not asked.
        read_batch_size (int): Rows per record batch read from storage.
            Defaults to 10,000. Not the run's `batch_size`.
        input_data_model (DataModel): See `Dataset`.
        input_schema (dict | str): See `Dataset`.
        input_template (str): See `Dataset`.
        output_data_model (DataModel): See `Dataset`.
        output_schema (dict | str): See `Dataset`.
        output_template (str): See `Dataset`.
        tools_template (str): See `Dataset`.
        batch_size (int): Examples per yielded batch. Defaults to 1.
        limit (int): Optional. See `Dataset`. Needed for a length: a table's
            row count takes a full scan.
        repeat (int): See `Dataset`.
    """

    provider = "unity_catalog"
    sdk = package = "deltalake"
    extra = "unity-catalog"

    def __init__(
        self,
        table,
        *,
        version=None,
        as_of=None,
        storage_path=None,
        read_batch_size=10_000,
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
        if version is not None and as_of is not None:
            raise ValueError("Pass either `version` (int) or `as_of` (ISO 8601 str), not both.")
        if version is not None and not isinstance(version, int):
            raise TypeError(f"`version` must be an int; got {type(version).__name__}.")
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
            table=table,
            version=version,
            as_of=as_of,
            storage_path=storage_path,
            read_batch_size=read_batch_size,
        )

    def connect(self) -> str:
        """Where the table's files are: the path given, else what the catalog says; `metadata` keeps the rest."""
        table_parts(self.table)  # the name is checked even when the catalog is not asked
        self.metadata = {"storage.uri": self.storage_path} if self.storage_path else lookup_table(self.table)
        location = self.metadata["storage.uri"]
        if not location:
            raise ValueError(f"Could not determine a storage path for table {self.table!r}. Check {API_URL}, {TOKEN}, and the table name.")
        return location

    @property
    def is_delta(self) -> bool:
        """Whether the files are a Delta table: what the catalog says, and Delta when it says nothing."""
        return self.metadata.get("data_source_format", "").upper() not in PLAIN_FORMATS

    def records(self):
        options = storage_options()
        reader = self._delta_batches if self.is_delta else self._parquet_batches
        for batch in reader(self.client, options):
            yield from batch.to_pylist()

    def _delta_batches(self, path, options):
        deltalake = self.load_sdk()
        pinned = {"version": self.version} if self.version is not None else {}
        table = deltalake.DeltaTable(path, storage_options=options, **pinned)
        if self.as_of is not None:
            table.load_as_version(self.as_of)
        self.metadata["delta_version"] = table.version()
        return table.to_pyarrow_dataset().to_batches(batch_size=self.read_batch_size)

    def _parquet_batches(self, path, options):
        import pyarrow.dataset as pyarrow_dataset

        files = pyarrow_dataset.dataset(path.removeprefix("s3://"), filesystem=filesystem_for(path, options), format="parquet")
        return files.to_batches(batch_size=self.read_batch_size)
