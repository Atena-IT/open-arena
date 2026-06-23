# License Apache 2.0: (c) 2026 Athena-Reply

import os
import re

from src.datasets.dataset import Dataset


def _sanitize_name(name: str) -> str:
    """Sanitize a Unity Catalog identifier for use as a Python/file name.

    Converts a full three-part table name (``catalog.schema.table``) or
    any identifier fragment into a filesystem- and Python-safe string by
    replacing dots, slashes, and hyphens with underscores:

    - ``/`` → ``__``
    - ``-`` → ``_``
    - ``.`` → ``_``

    Example::

        >>> _sanitize_name("my-catalog.my_schema.my-table")
        'my_catalog_my_schema_my_table'
    """
    name = name.replace("/", "__")
    name = name.replace("-", "_")
    name = name.replace(".", "_")
    return name


def _parse_table(full_name: str):
    """Split ``catalog.schema.table`` into a ``(catalog, schema, table)`` tuple.

    Raises:
        ValueError: If *full_name* does not contain exactly two dots.
    """
    parts = full_name.split(".")
    if len(parts) != 3:
        raise ValueError(
            f"`table` must be fully-qualified as 'catalog.schema.table'; "
            f"got {full_name!r}."
        )
    return tuple(parts)  # (catalog, schema, table)


class UnityCatalogDataset(Dataset):
    """Streaming dataset backed by a Unity Catalog (Databricks or OSS) table.

    Each row in the target table becomes a raw ``dict`` whose keys are the
    column names.  The ``_iter_rows()`` implementation resolves the table to
    its backing storage path (``s3://…``) via the Unity Catalog REST API,
    reads the Delta or Parquet data with ``deltalake`` + ``pyarrow``, and
    streams rows without pulling the full table into memory first.

    Auth & connectivity
    -------------------
    * **UC REST API** — read from ``UNITY_CATALOG_API_URL`` (must already
      include the ``/api/2.1/unity-catalog`` prefix) and bearer token
      ``UC_TOKEN``.
    * **S3 / object storage** — reads from ``AWS_ACCESS_KEY_ID``,
      ``AWS_SECRET_ACCESS_KEY``, and optionally ``S3_ENDPOINT`` (for
      MinIO/Scaleway-compatible stores; region defaults to ``fr-par``).

    Resolution flow
    ---------------
    1. ``GET /tables/{catalog}.{schema}.{table}`` → ``storage_location``
       (e.g. ``s3://bucket/path/to/table``) and ``content_hash`` (the UC
       table version token).
    2. If *version* or *as_of* is given, it is forwarded to the Delta reader
       as a version integer or timestamp string.
    3. The Delta table is opened via ``deltalake.DeltaTable``; if the path
       does not contain a ``_delta_log/`` directory the reader falls back to
       plain Parquet via ``pyarrow.dataset``.
    4. Rows are streamed in record-batch chunks (controlled by *batch_size*
       at the storage layer — distinct from the synalinks ``batch_size``).

    The resolved ``content_hash`` / ``storage_location`` are stored as
    ``self.metadata`` for downstream pinning / reproducibility logging.

    Example YAML config::

        datasets:
          uc_eval_set:
            type: unity_catalog
            table: my_catalog.eval_schema.questions_v2
            version: 3
            input_template: |
              {"messages": [{"role": "user", "content": {{ question | tojson }}}]}
            output_template: |
              {"role": "assistant", "content": {{ reference_answer | tojson }}}
            batch_size: 8
            limit: 500

    Python usage::

        ds = UnityCatalogDataset(
            table="my_catalog.eval_schema.questions_v2",
            version=3,
            input_template='{"messages":[{"role":"user","content":{{ question | tojson }}}]}',
            batch_size=8,
        )
        program.evaluate(x=ds())

    Args:
        table (str): Fully-qualified table name in ``catalog.schema.table``
            form.  Required.
        version (int): Optional. Pin the Delta table to this numeric version.
            Mutually exclusive with *as_of*.
        as_of (str): Optional. Pin the Delta table to a UTC timestamp string
            (ISO 8601, e.g. ``"2026-04-15T00:00:00Z"``).  Mutually exclusive
            with *version*.
        storage_path (str): Optional explicit ``s3://…`` path override.
            When given, the UC REST lookup is skipped entirely.
        read_batch_size (int): Number of rows per Arrow record batch when
            streaming the Delta/Parquet source.  Defaults to ``10_000``.
            Distinct from the synalinks ``batch_size``.
        input_data_model: See :class:`~src.datasets.dataset.Dataset`.
        input_schema: See :class:`~src.datasets.dataset.Dataset`.
        input_template (str): See :class:`~src.datasets.dataset.Dataset`.
        output_data_model: See :class:`~src.datasets.dataset.Dataset`.
        output_schema: See :class:`~src.datasets.dataset.Dataset`.
        output_template (str): See :class:`~src.datasets.dataset.Dataset`.
        batch_size (int): Examples per yielded synalinks batch. Defaults to ``1``.
        limit (int): Optional cap on raw rows consumed.
        repeat (int): Repeat expansion per raw row.  Defaults to ``1``.
    """

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
        if version is not None and as_of is not None:
            raise ValueError(
                "Pass either `version` (int) or `as_of` (ISO 8601 str), not both."
            )
        if version is not None and not isinstance(version, int):
            raise TypeError(f"`version` must be an int; got {type(version).__name__}.")

        self.table = table
        self.version = version
        self.as_of = as_of
        self.storage_path = storage_path
        self.read_batch_size = read_batch_size

        # Parsed components — set eagerly so errors surface at construction time.
        self._catalog, self._schema, self._table = _parse_table(table)

        # Will be populated during the first _iter_rows() call (or
        # eagerly here when storage_path is given so no network call is needed).
        self.metadata: dict = {}

        if storage_path is not None:
            # Caller-supplied path: skip the REST lookup.
            self.metadata["storage.uri"] = storage_path
            self._resolved_path = storage_path
        else:
            # Imported lazily so the project doesn't require httpx unless
            # this dataset type is actually used.
            import httpx

            api_url = os.environ.get("UNITY_CATALOG_API_URL", "").rstrip("/")
            uc_token = os.environ.get("UC_TOKEN", "")
            if not api_url:
                raise EnvironmentError(
                    "UNITY_CATALOG_API_URL is not set. "
                    "It must point to the Unity Catalog REST API base URL, "
                    "e.g. 'https://my-uc-host/api/2.1/unity-catalog'."
                )
            if not uc_token:
                raise EnvironmentError(
                    "UC_TOKEN is not set. "
                    "Provide a bearer token for the Unity Catalog REST API."
                )

            full_name = f"{self._catalog}.{self._schema}.{self._table}"
            url = f"{api_url}/tables/{full_name}"
            resp = httpx.get(
                url,
                headers={"Authorization": f"Bearer {uc_token}"},
                timeout=30,
            )
            resp.raise_for_status()
            table_info = resp.json()

            storage_location = table_info.get("storage_location", "")
            content_hash = table_info.get("properties", {}).get(
                "delta.lastCommitTimestamp",
                table_info.get("updated_at", ""),
            )

            self.metadata = {
                "storage.uri": storage_location,
                "content_hash": str(content_hash),
                "table_type": table_info.get("table_type", ""),
                "data_source_format": table_info.get("data_source_format", ""),
            }
            self._resolved_path = storage_location

        if not self._resolved_path:
            raise ValueError(
                f"Could not determine a storage path for table {table!r}. "
                "Check UNITY_CATALOG_API_URL, UC_TOKEN, and the table name."
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _s3_storage_options(self) -> dict:
        """Build the storage-options dict for Delta Lake / PyArrow S3 access."""
        opts = {
            "region": os.environ.get("AWS_DEFAULT_REGION", "fr-par"),
            "aws_access_key_id": os.environ.get("AWS_ACCESS_KEY_ID", ""),
            "aws_secret_access_key": os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
        }
        endpoint = os.environ.get("S3_ENDPOINT", "")
        if endpoint:
            opts["endpoint_url"] = endpoint
            # Some MinIO / Scaleway setups require path-style addressing.
            opts["allow_http"] = endpoint.startswith("http://")
        return opts

    def _is_delta(self, path: str) -> bool:
        """Return True when *path* looks like a Delta table (has _delta_log)."""
        # We attempt a lightweight HEAD on the _delta_log/_last_checkpoint
        # or simply trust the data_source_format hint from UC metadata.
        fmt = self.metadata.get("data_source_format", "").upper()
        if fmt in ("DELTA",):
            return True
        if fmt in ("PARQUET", "CSV", "JSON", "ORC", "AVRO"):
            return False
        # If format is unknown, default to Delta (most common in UC).
        return True

    # ------------------------------------------------------------------
    # Core generator
    # ------------------------------------------------------------------

    def _iter_rows(self):
        """Yield raw row dicts from the resolved UC table storage location."""
        path = self._resolved_path
        storage_options = self._s3_storage_options()

        if self._is_delta(path):
            yield from self._iter_delta(path, storage_options)
        else:
            yield from self._iter_parquet(path, storage_options)

    def _iter_delta(self, path: str, storage_options: dict):
        """Stream rows from a Delta table via ``deltalake``."""
        # Imported lazily so deltalake is an optional dependency.
        try:
            from deltalake import DeltaTable
        except ImportError as exc:
            raise ImportError(
                "The `deltalake` package is required to read Delta tables. "
                "Install it with: pip install 'open-arena[unity_catalog]'"
            ) from exc

        kwargs: dict = {"storage_options": storage_options}
        if self.version is not None:
            kwargs["version"] = self.version

        dt = DeltaTable(path, **kwargs)

        # Optionally pin to a timestamp (as_of).
        if self.as_of is not None:
            # deltalake accepts ISO 8601 strings via load_as_version.
            dt.load_as_version(self.as_of)

        # Update metadata with the resolved Delta version.
        self.metadata["delta_version"] = dt.version()

        # Convert to an Arrow dataset and stream in batches.
        arrow_ds = dt.to_pyarrow_dataset()
        for batch in arrow_ds.to_batches(batch_size=self.read_batch_size):
            for row in batch.to_pylist():
                yield row

    def _iter_parquet(self, path: str, storage_options: dict):
        """Stream rows from a plain Parquet dataset via ``pyarrow.dataset``."""
        try:
            import pyarrow.dataset as pad
        except ImportError as exc:
            raise ImportError(
                "The `pyarrow` package is required to read Parquet tables. "
                "Install it with: pip install 'open-arena[unity_catalog]'"
            ) from exc

        # Build the filesystem for S3-compatible stores.
        fs = self._build_pyarrow_fs(path, storage_options)
        # Strip the s3:// scheme — PyArrow wants a bare path when fs is given.
        bare_path = re.sub(r"^s3://", "", path)

        ds = pad.dataset(bare_path, filesystem=fs, format="parquet")
        for batch in ds.to_batches(batch_size=self.read_batch_size):
            for row in batch.to_pylist():
                yield row

    @staticmethod
    def _build_pyarrow_fs(path: str, storage_options: dict):
        """Return a PyArrow filesystem for the given *path* and *storage_options*."""
        import pyarrow.fs as paf

        if not path.startswith("s3://"):
            # Local or other filesystem — let PyArrow auto-detect.
            fs, _ = paf.FileSystem.from_uri(path)
            return fs

        s3_kwargs: dict = {
            "region": storage_options.get("region", "fr-par"),
        }
        access_key = storage_options.get("aws_access_key_id", "")
        secret_key = storage_options.get("aws_secret_access_key", "")
        if access_key:
            s3_kwargs["access_key"] = access_key
        if secret_key:
            s3_kwargs["secret_key"] = secret_key
        endpoint = storage_options.get("endpoint_url", "")
        if endpoint:
            s3_kwargs["endpoint_override"] = endpoint

        return paf.S3FileSystem(**s3_kwargs)

    # ------------------------------------------------------------------
    # Length
    # ------------------------------------------------------------------

    def __len__(self):
        if self.limit is not None:
            return self._total_batches(self.limit)
        raise NotImplementedError(
            "UnityCatalogDataset length is unknown without a full scan; "
            "set `limit` to get a bounded epoch."
        )
