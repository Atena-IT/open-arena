# License Apache 2.0: (c) 2026 Athena-Reply
"""Unit tests for UnityCatalogDataset.

All external I/O (Unity Catalog REST API, Delta Lake, S3/PyArrow) is
mocked so no live services are required.
"""
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


# ---------------------------------------------------------------------------
# Helpers to stub heavy optional deps before any real import
# ---------------------------------------------------------------------------

def _make_arrow_batch(rows: list[dict]):
    """Return a mock Arrow RecordBatch whose to_pylist() yields *rows*."""
    batch = MagicMock()
    batch.to_pylist.return_value = rows
    return batch


def _stub_httpx(monkeypatch, table_info: dict):
    """Patch httpx.get to return *table_info* as JSON."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = table_info
    mock_httpx = MagicMock()
    mock_httpx.get.return_value = mock_resp
    monkeypatch.setattr("src.datasets.unity_catalog_dataset.httpx", mock_httpx, raising=False)
    return mock_httpx


# ---------------------------------------------------------------------------
# Tests for pure helper functions
# ---------------------------------------------------------------------------

class TestSanitizeName:
    def test_dots_become_underscores(self):
        from src.datasets.unity_catalog_dataset import _sanitize_name
        assert _sanitize_name("a.b.c") == "a_b_c"

    def test_slashes_become_double_underscores(self):
        from src.datasets.unity_catalog_dataset import _sanitize_name
        assert _sanitize_name("a/b") == "a__b"

    def test_hyphens_become_underscores(self):
        from src.datasets.unity_catalog_dataset import _sanitize_name
        assert _sanitize_name("my-catalog.my-schema.my-table") == "my_catalog_my_schema_my_table"

    def test_combined(self):
        from src.datasets.unity_catalog_dataset import _sanitize_name
        assert _sanitize_name("cat-1.sch/em.tab-le") == "cat_1_sch__em_tab_le"


class TestParseTable:
    def test_valid_full_name(self):
        from src.datasets.unity_catalog_dataset import _parse_table
        assert _parse_table("cat.sch.tbl") == ("cat", "sch", "tbl")

    def test_missing_parts_raises(self):
        from src.datasets.unity_catalog_dataset import _parse_table
        with pytest.raises(ValueError, match="fully-qualified"):
            _parse_table("just_two.parts")

    def test_too_many_parts_raises(self):
        from src.datasets.unity_catalog_dataset import _parse_table
        with pytest.raises(ValueError, match="fully-qualified"):
            _parse_table("a.b.c.d")


# ---------------------------------------------------------------------------
# Tests for constructor validation (no network needed when storage_path given)
# ---------------------------------------------------------------------------

class TestUnityCatalogDatasetInit:
    """Constructor-level checks that require NO network access."""

    def _make(self, **extra):
        from src.datasets.unity_catalog_dataset import UnityCatalogDataset
        defaults = dict(
            table="my_catalog.eval_schema.questions_v2",
            storage_path="s3://bucket/path/to/table",
            input_template='{"messages":[{"role":"user","content":{{ question | tojson }}}]}',
        )
        defaults.update(extra)
        return UnityCatalogDataset(**defaults)

    def test_table_parsed_correctly(self):
        ds = self._make()
        assert ds._catalog == "my_catalog"
        assert ds._schema == "eval_schema"
        assert ds._table == "questions_v2"

    def test_storage_path_stored(self):
        ds = self._make(storage_path="s3://b/p")
        assert ds._resolved_path == "s3://b/p"
        assert ds.metadata["storage.uri"] == "s3://b/p"

    def test_version_int_accepted(self):
        ds = self._make(version=3)
        assert ds.version == 3

    def test_version_and_as_of_mutually_exclusive(self):
        from src.datasets.unity_catalog_dataset import UnityCatalogDataset
        with pytest.raises(ValueError, match="not both"):
            UnityCatalogDataset(
                table="c.s.t",
                storage_path="s3://b/p",
                input_template='{"x": {{ x | tojson }}}',
                version=1,
                as_of="2026-01-01T00:00:00Z",
            )

    def test_version_must_be_int(self):
        from src.datasets.unity_catalog_dataset import UnityCatalogDataset
        with pytest.raises(TypeError, match="int"):
            UnityCatalogDataset(
                table="c.s.t",
                storage_path="s3://b/p",
                input_template='{"x": {{ x | tojson }}}',
                version="latest",
            )

    def test_invalid_table_name_raises(self):
        from src.datasets.unity_catalog_dataset import UnityCatalogDataset
        with pytest.raises(ValueError, match="fully-qualified"):
            UnityCatalogDataset(
                table="only.two",
                storage_path="s3://b/p",
                input_template='{"x": {{ x | tojson }}}',
            )

    def test_missing_input_template_raises(self):
        from src.datasets.unity_catalog_dataset import UnityCatalogDataset
        with pytest.raises(ValueError, match="input_template"):
            UnityCatalogDataset(
                table="c.s.t",
                storage_path="s3://b/p",
            )


# ---------------------------------------------------------------------------
# Tests for env-var validation (REST path, no storage_path override)
# ---------------------------------------------------------------------------

class TestEnvVarValidation:
    def test_missing_api_url_raises(self, monkeypatch):
        monkeypatch.delenv("UNITY_CATALOG_API_URL", raising=False)
        monkeypatch.delenv("UC_TOKEN", raising=False)

        from src.datasets.unity_catalog_dataset import UnityCatalogDataset
        with pytest.raises(EnvironmentError, match="UNITY_CATALOG_API_URL"):
            UnityCatalogDataset(
                table="c.s.t",
                input_template='{"x": {{ x | tojson }}}',
            )

    def test_missing_uc_token_raises(self, monkeypatch):
        monkeypatch.setenv("UNITY_CATALOG_API_URL", "https://uc.example.com/api/2.1/unity-catalog")
        monkeypatch.delenv("UC_TOKEN", raising=False)

        from src.datasets.unity_catalog_dataset import UnityCatalogDataset
        with pytest.raises(EnvironmentError, match="UC_TOKEN"):
            UnityCatalogDataset(
                table="c.s.t",
                input_template='{"x": {{ x | tojson }}}',
            )


# ---------------------------------------------------------------------------
# Tests for UC REST resolution
# ---------------------------------------------------------------------------

class TestUCRestResolution:
    """Verify table name, version pinning, and metadata extraction."""

    def test_rest_lookup_sets_metadata(self, monkeypatch):
        monkeypatch.setenv("UNITY_CATALOG_API_URL", "https://uc.example.com/api/2.1/unity-catalog")
        monkeypatch.setenv("UC_TOKEN", "tok-123")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "key")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")

        table_info = {
            "storage_location": "s3://my-bucket/path/to/table",
            "table_type": "MANAGED",
            "data_source_format": "DELTA",
            "updated_at": "2026-04-15T00:00:00Z",
            "properties": {"delta.lastCommitTimestamp": "42"},
        }

        # Stub httpx before importing UnityCatalogDataset to capture lazy import
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = table_info
        mock_httpx = MagicMock()
        mock_httpx.get.return_value = mock_resp

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            from importlib import reload
            import src.datasets.unity_catalog_dataset as uc_mod
            reload(uc_mod)

            ds = uc_mod.UnityCatalogDataset(
                table="my_catalog.eval_schema.questions_v2",
                input_template='{"q": {{ question | tojson }}}',
            )

        assert ds._resolved_path == "s3://my-bucket/path/to/table"
        assert ds.metadata["storage.uri"] == "s3://my-bucket/path/to/table"
        assert ds.metadata["content_hash"] == "42"
        assert ds.metadata["data_source_format"] == "DELTA"

    def test_correct_api_url_constructed(self, monkeypatch):
        monkeypatch.setenv("UNITY_CATALOG_API_URL", "https://uc.example.com/api/2.1/unity-catalog")
        monkeypatch.setenv("UC_TOKEN", "tok-abc")

        table_info = {
            "storage_location": "s3://b/t",
            "table_type": "MANAGED",
            "data_source_format": "DELTA",
            "properties": {},
        }
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = table_info
        mock_httpx = MagicMock()
        mock_httpx.get.return_value = mock_resp

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            from importlib import reload
            import src.datasets.unity_catalog_dataset as uc_mod
            reload(uc_mod)
            uc_mod.UnityCatalogDataset(
                table="cat.sch.tbl",
                input_template='{"x": {{ x | tojson }}}',
            )

        called_url = mock_httpx.get.call_args[0][0]
        assert called_url == "https://uc.example.com/api/2.1/unity-catalog/tables/cat.sch.tbl"
        assert mock_httpx.get.call_args[1]["headers"]["Authorization"] == "Bearer tok-abc"


# ---------------------------------------------------------------------------
# Tests for row streaming (Delta path)
# ---------------------------------------------------------------------------

class TestDeltaRowStreaming:
    """Mock deltalake and pyarrow to verify _iter_rows() yields dicts."""

    def _make_ds(self, rows, version=None, as_of=None):
        """Build a UnityCatalogDataset with storage_path and mock deltalake."""
        from src.datasets.unity_catalog_dataset import UnityCatalogDataset

        ds = UnityCatalogDataset(
            table="c.s.t",
            storage_path="s3://bucket/delta_table",
            version=version,
            as_of=as_of,
            input_template='{"question": {{ question | tojson }}}',
            batch_size=2,
        )
        ds.metadata["data_source_format"] = "DELTA"

        # Build Arrow batch stubs
        batches = [_make_arrow_batch(rows)]

        # Stub Arrow dataset
        arrow_ds = MagicMock()
        arrow_ds.to_batches.return_value = batches

        # Stub DeltaTable
        mock_dt = MagicMock()
        mock_dt.version.return_value = 7
        mock_dt.to_pyarrow_dataset.return_value = arrow_ds

        mock_deltalake = MagicMock()
        mock_deltalake.DeltaTable.return_value = mock_dt

        ds._mock_dt = mock_dt
        ds._mock_deltalake = mock_deltalake
        return ds, mock_deltalake

    def test_rows_yielded_from_delta(self):
        rows = [{"question": "What is 2+2?", "answer": "4"},
                {"question": "Sky color?", "answer": "blue"}]
        ds, mock_deltalake = self._make_ds(rows)

        with patch.dict("sys.modules", {"deltalake": mock_deltalake}):
            yielded = list(ds._iter_delta("s3://bucket/delta_table", {}))

        assert yielded == rows

    def test_version_passed_to_deltatabel(self):
        rows = [{"question": "Q1"}]
        ds, mock_deltalake = self._make_ds(rows, version=5)

        with patch.dict("sys.modules", {"deltalake": mock_deltalake}):
            list(ds._iter_delta("s3://bucket/t", {}))

        call_kwargs = mock_deltalake.DeltaTable.call_args[1]
        assert call_kwargs["version"] == 5

    def test_as_of_loads_version(self):
        rows = [{"question": "Q1"}]
        ds, mock_deltalake = self._make_ds(rows, as_of="2026-01-15T00:00:00Z")
        mock_dt = mock_deltalake.DeltaTable.return_value

        with patch.dict("sys.modules", {"deltalake": mock_deltalake}):
            list(ds._iter_delta("s3://bucket/t", {}))

        mock_dt.load_as_version.assert_called_once_with("2026-01-15T00:00:00Z")

    def test_delta_version_stored_in_metadata(self):
        rows = [{"question": "Q1"}]
        ds, mock_deltalake = self._make_ds(rows)

        with patch.dict("sys.modules", {"deltalake": mock_deltalake}):
            list(ds._iter_delta("s3://bucket/t", {}))

        assert ds.metadata["delta_version"] == 7

    def test_missing_deltalake_raises_helpful_error(self):
        from src.datasets.unity_catalog_dataset import UnityCatalogDataset
        ds = UnityCatalogDataset(
            table="c.s.t",
            storage_path="s3://b/t",
            input_template='{"x": {{ x | tojson }}}',
        )
        ds.metadata["data_source_format"] = "DELTA"

        with patch.dict("sys.modules", {"deltalake": None}):
            with pytest.raises(ImportError, match="deltalake"):
                list(ds._iter_delta("s3://b/t", {}))


# ---------------------------------------------------------------------------
# Tests for row-to-batch pipeline (base Dataset templating)
# ---------------------------------------------------------------------------

class TestTemplatePipeline:
    """End-to-end: rows flow from _iter_rows() through Jinja2 → batches."""

    def test_rows_rendered_and_batched(self):
        """Verify the base Dataset batching logic consumes UC rows correctly."""
        from src.datasets.unity_catalog_dataset import UnityCatalogDataset

        ds = UnityCatalogDataset(
            table="c.s.t",
            storage_path="s3://b/t",
            input_template='{"messages":[{"role":"user","content":{{ question | tojson }}}]}',
            batch_size=2,
            # Set limit so __len__ is defined and list() can pre-allocate.
            limit=3,
        )
        ds.metadata["data_source_format"] = "PARQUET"

        raw_rows = [
            {"question": "What is AI?"},
            {"question": "What is ML?"},
            {"question": "What is DL?"},
        ]

        # Stub _iter_rows to return our rows directly (bypass S3/Delta)
        ds._iter_rows = lambda: iter(raw_rows)

        batches = list(ds)
        # 3 rows, batch_size=2 → 2 batches: (2,) and (1,)
        assert len(batches) == 2
        x0, = batches[0]
        x1, = batches[1]
        assert x0.shape == (2,)
        assert x1.shape == (1,)

    def test_limit_caps_rows(self):
        from src.datasets.unity_catalog_dataset import UnityCatalogDataset

        ds = UnityCatalogDataset(
            table="c.s.t",
            storage_path="s3://b/t",
            input_template='{"messages":[{"role":"user","content":{{ question | tojson }}}]}',
            batch_size=10,
            limit=2,
        )
        raw_rows = [{"question": f"Q{i}"} for i in range(10)]
        ds._iter_rows = lambda: iter(raw_rows)

        batches = list(ds)
        assert len(batches) == 1
        (x,) = batches[0]
        assert x.shape == (2,)

    def test_repeat_expands_rows(self):
        from src.datasets.unity_catalog_dataset import UnityCatalogDataset

        ds = UnityCatalogDataset(
            table="c.s.t",
            storage_path="s3://b/t",
            input_template='{"messages":[{"role":"user","content":{{ question | tojson }}}]}',
            batch_size=6,
            repeat=3,
            # Set limit so __len__ is defined and list() can pre-allocate.
            limit=2,
        )
        raw_rows = [{"question": "Q1"}, {"question": "Q2"}]
        ds._iter_rows = lambda: iter(raw_rows)

        batches = list(ds)
        assert len(batches) == 1
        (x,) = batches[0]
        # 2 rows × 3 repeats = 6
        assert x.shape == (6,)


# ---------------------------------------------------------------------------
# Tests for __len__
# ---------------------------------------------------------------------------

class TestLength:
    def test_len_with_limit(self):
        from src.datasets.unity_catalog_dataset import UnityCatalogDataset
        ds = UnityCatalogDataset(
            table="c.s.t",
            storage_path="s3://b/t",
            input_template='{"x": {{ x | tojson }}}',
            batch_size=4,
            limit=8,
        )
        assert len(ds) == 2

    def test_len_without_limit_raises(self):
        from src.datasets.unity_catalog_dataset import UnityCatalogDataset
        ds = UnityCatalogDataset(
            table="c.s.t",
            storage_path="s3://b/t",
            input_template='{"x": {{ x | tojson }}}',
            batch_size=4,
        )
        with pytest.raises(NotImplementedError, match="limit"):
            len(ds)


# ---------------------------------------------------------------------------
# Registry test
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_unity_catalog_in_dataset_types(self):
        # Force a clean import to avoid stale state from module reloads in
        # TestUCRestResolution (which patches sys.modules["httpx"]).
        import importlib
        import src.datasets as ds_pkg
        importlib.reload(ds_pkg)
        assert "unity_catalog" in ds_pkg._DATASET_TYPES

    def test_registry_points_to_correct_class(self):
        import importlib
        import src.datasets as ds_pkg
        import src.datasets.unity_catalog_dataset as uc_mod
        importlib.reload(uc_mod)
        importlib.reload(ds_pkg)
        assert ds_pkg._DATASET_TYPES["unity_catalog"] is uc_mod.UnityCatalogDataset

    def test_get_returns_class(self):
        import importlib
        import src.datasets as ds_pkg
        import src.datasets.unity_catalog_dataset as uc_mod
        importlib.reload(uc_mod)
        importlib.reload(ds_pkg)
        assert ds_pkg.get("unity_catalog") is uc_mod.UnityCatalogDataset
