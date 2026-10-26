# License Apache 2.0: (c) 2026 Atena-Reply

"""The providers that read an evaluation platform, each against a fake of its SDK.

What each one is held to: the call it makes on the SDK and with what, the
row it hands the templates for each record (the names a user's template may
use), how it pins a version, and when it knows its length. No SDK is
installed here: each is a module stood up for the test, so what is checked
is the provider's side of the contract alone.
"""

import os
import sys
import types
from datetime import UTC, datetime, timedelta, timezone
from types import SimpleNamespace

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import pytest

from openarena.datasets import get

TEMPLATES = {"input_template": '{"messages": [{"role": "user", "content": {{ input | tojson }}}]}'}


def _module(monkeypatch, name, **attrs):
    module = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(module, k, v)
    monkeypatch.setitem(sys.modules, name, module)
    return module


# -- Langfuse --------------------------------------------------------------------


def _langfuse(monkeypatch, pages, seen):
    def list_items(**kwargs):
        seen.append(("list", kwargs))
        return SimpleNamespace(data=pages[kwargs["page"] - 1], meta=SimpleNamespace(total_pages=len(pages)))

    class Client:
        api = SimpleNamespace(dataset_items=SimpleNamespace(list=list_items))

        def get_dataset(self, name, version=None):
            seen.append(("get_dataset", name, version))
            return SimpleNamespace(items=[item for page in pages for item in page])

    _module(monkeypatch, "langfuse", get_client=Client)


def _lf_item(n):
    return SimpleNamespace(input={"question": f"q{n}"}, expected_output={"answer": f"a{n}"}, metadata={"n": n}, id=f"item-{n}")


def test_langfuse_streams_page_by_page_under_the_encoded_name_and_the_pinned_version(monkeypatch):
    seen = []
    _langfuse(monkeypatch, [[_lf_item(1), _lf_item(2)], [_lf_item(3)]], seen)
    ds = get("langfuse")(dataset_name="my bench", version="2026-04-15T00:00:00Z", page_size=2, **TEMPLATES)
    rows = list(ds._iter_rows())
    assert rows[0] == {"input": {"question": "q1"}, "expected_output": {"answer": "a1"}, "metadata": {"n": 1}, "id": "item-1"}
    assert [r["id"] for r in rows] == ["item-1", "item-2", "item-3"]
    pinned = datetime(2026, 4, 15, tzinfo=UTC)
    assert seen == [
        ("list", {"dataset_name": "my%20bench", "page": 1, "limit": 2, "version": pinned}),
        ("list", {"dataset_name": "my%20bench", "page": 2, "limit": 2, "version": pinned}),
    ]
    with pytest.raises(NotImplementedError, match="unknown length; set `limit`"):
        len(ds)
    assert len(get("langfuse")(dataset_name="b", limit=3, batch_size=2, **TEMPLATES)) == 2


def test_langfuse_without_streaming_fetches_the_whole_dataset_once_and_knows_its_length(monkeypatch):
    seen = []
    _langfuse(monkeypatch, [[_lf_item(1), _lf_item(2)]], seen)
    ds = get("langfuse")(dataset_name="bench", streaming=False, batch_size=1, **TEMPLATES)
    assert seen == [("get_dataset", "bench", None)]
    assert [r["id"] for r in ds._iter_rows()] == ["item-1", "item-2"]
    assert len(ds) == 2 and not any(call[0] == "list" for call in seen)


def test_langfuse_versions_are_utc_datetimes_however_written():
    from openarena.datasets.platform import utc

    assert utc(None) is None
    assert utc("2026-04-15T00:00:00Z") == datetime(2026, 4, 15, tzinfo=UTC)
    assert utc("2026-04-15T02:00:00+02:00") == datetime(2026, 4, 15, tzinfo=UTC)
    assert utc(datetime(2026, 4, 15)) == datetime(2026, 4, 15, tzinfo=UTC)  # a YAML timestamp, naive: read as UTC
    aware = datetime(2026, 4, 15, tzinfo=timezone(timedelta(hours=2)))
    assert utc(aware) is aware


# -- LangSmith -------------------------------------------------------------------


def _langsmith(monkeypatch, examples, seen):
    class Client:
        def list_examples(self, **kwargs):
            seen.append(kwargs)
            return iter(examples)

    _module(monkeypatch, "langsmith", Client=Client)


def test_langsmith_lists_the_examples_as_of_a_tag_with_empty_sides_as_empty_mappings(monkeypatch):
    seen = []
    _langsmith(monkeypatch, [SimpleNamespace(inputs={"question": "q"}, outputs=None, metadata=None, id=1234)], seen)
    ds = get("langsmith")(dataset_name="bench", as_of="prod", **TEMPLATES)
    assert list(ds._iter_rows()) == [{"inputs": {"question": "q"}, "outputs": {}, "metadata": {}, "id": "1234"}]
    assert seen == [{"dataset_name": "bench", "as_of": "prod"}]
    with pytest.raises(NotImplementedError, match="set `limit`"):
        len(ds)
    assert len(get("langsmith")(dataset_name="bench", limit=5, batch_size=2, **TEMPLATES)) == 3


def test_langsmith_as_of_is_a_tag_or_a_utc_datetime(monkeypatch):
    _langsmith(monkeypatch, [], [])
    assert get("langsmith")(dataset_name="b", **TEMPLATES).as_of is None
    assert get("langsmith")(dataset_name="b", as_of="prod", **TEMPLATES).as_of == "prod"
    assert get("langsmith")(dataset_name="b", as_of=datetime(2026, 4, 15), **TEMPLATES).as_of == datetime(2026, 4, 15, tzinfo=UTC)
    with pytest.raises(TypeError, match="`as_of` must be str"):
        get("langsmith")(dataset_name="b", as_of=3, **TEMPLATES)


# -- Braintrust ------------------------------------------------------------------


def _braintrust(monkeypatch, records, seen):
    def init_dataset(**kwargs):
        seen.append(("init", kwargs))

        def fetch(**kw):
            seen.append(("fetch", kw))
            return iter(records)

        return SimpleNamespace(fetch=fetch)

    _module(monkeypatch, "braintrust", init_dataset=init_dataset)


def test_braintrust_opens_the_dataset_by_project_and_name_and_fetches_its_records(monkeypatch):
    seen = []
    _braintrust(monkeypatch, [{"id": "r1", "input": "q", "expected": "a", "metadata": None, "extra": "dropped"}], seen)
    ds = get("braintrust")(project="p", dataset_name="bench", version="42", org_name="org", fetch_batch_size=100, **TEMPLATES)
    assert list(ds._iter_rows()) == [{"id": "r1", "input": "q", "expected": "a", "metadata": {}}]
    assert seen == [
        ("init", {"project": "p", "project_id": None, "name": "bench", "version": "42", "org_name": "org"}),
        ("fetch", {"batch_size": 100}),
    ]
    seen.clear()
    list(get("braintrust")(project_id="uuid", dataset_name="bench", **TEMPLATES)._iter_rows())
    assert seen[1] == ("fetch", {})  # no fetch size given: the SDK's own
    with pytest.raises(NotImplementedError, match="set `limit`"):
        len(ds)


def test_braintrust_needs_a_project_and_a_dataset_name(monkeypatch):
    _braintrust(monkeypatch, [], [])
    with pytest.raises(ValueError, match="requires `project` or `project_id`"):
        get("braintrust")(dataset_name="bench", **TEMPLATES)
    with pytest.raises(ValueError, match="requires `dataset_name`"):
        get("braintrust")(project="p", **TEMPLATES)


# -- Opik ------------------------------------------------------------------------


def _opik(monkeypatch, items, seen, count=7):
    class Versioned:
        def __init__(self, version=None):
            self.version = version
            self.dataset_items_count = count

        def get_version_view(self, version):
            seen.append(("version", version))
            return Versioned(version)

        def get_items(self, **kwargs):
            seen.append(("items", self.version, kwargs))
            return list(items)

    class Opik:
        def get_dataset(self, name):
            seen.append(("get_dataset", name))
            return Versioned()

    _module(monkeypatch, "opik", Opik=Opik)


def test_opik_reads_a_version_view_when_pinned_and_forwards_the_limit_and_filter(monkeypatch):
    seen = []
    items = [{"id": "i1", "input": {"q": "x"}, "expected_output": {"a": "y"}, "metadata": {}, "custom": 1}]
    _opik(monkeypatch, items, seen)
    ds = get("opik")(dataset_name="bench", version="v1", filter_string='tags contains "r"', limit=10, batch_size=5, **TEMPLATES)
    assert list(ds._iter_rows()) == items  # every field forwarded verbatim, custom ones included
    assert seen == [("get_dataset", "bench"), ("version", "v1"), ("items", "v1", {"nb_samples": 10, "filter_string": 'tags contains "r"'})]
    assert len(ds) == 2
    seen.clear()
    live = get("opik")(dataset_name="bench", **TEMPLATES)
    list(live._iter_rows())
    assert seen == [("get_dataset", "bench"), ("items", None, {"nb_samples": None, "filter_string": None})]
    assert len(live) == 7  # the platform's count, one row per batch


# -- Phoenix ---------------------------------------------------------------------


def _phoenix(monkeypatch, examples, seen):
    def get_dataset(**kwargs):
        seen.append(kwargs)
        return SimpleNamespace(examples=examples, example_count=len(examples))

    class Client:
        datasets = SimpleNamespace(get_dataset=get_dataset)

    _module(monkeypatch, "phoenix")
    _module(monkeypatch, "phoenix.client", Client=Client)


def test_phoenix_reads_one_version_of_the_dataset_with_empty_sides_as_empty_mappings(monkeypatch):
    seen = []
    _phoenix(monkeypatch, [{"id": "e1", "input": {"q": "x"}, "output": None, "metadata": None}, {"id": "e2", "input": {}, "output": {"a": 1}}], seen)
    ds = get("phoenix")(dataset_name="bench", version_id="v9", splits=["test"], batch_size=1, **TEMPLATES)
    assert seen == [{"dataset": "bench", "version_id": "v9", "splits": ["test"]}]
    assert list(ds._iter_rows()) == [
        {"id": "e1", "input": {"q": "x"}, "output": {}, "metadata": {}},
        {"id": "e2", "input": {}, "output": {"a": 1}, "metadata": {}},
    ]
    assert len(ds) == 2 and len(get("phoenix")(dataset_name="bench", limit=1, **TEMPLATES)) == 1


# -- MLflow ----------------------------------------------------------------------


def test_mlflow_reads_the_source_server_unless_told_where(monkeypatch):
    import synalinks

    seen = []
    monkeypatch.setattr(synalinks.datasets.MLflowDataset, "__init__", lambda self, **kwargs: seen.append(kwargs))
    monkeypatch.setenv("MLFLOW_SOURCE_TRACKING_URI", "http://source:5000")
    get("mlflow")(name="golden", **TEMPLATES)
    assert seen[-1]["tracking_uri"] == "http://source:5000" and seen[-1]["name"] == "golden"
    get("mlflow")(dataset_id="d1", tracking_uri="http://own:5000", **TEMPLATES)
    assert seen[-1]["tracking_uri"] == "http://own:5000" and seen[-1]["dataset_id"] == "d1"
    monkeypatch.delenv("MLFLOW_SOURCE_TRACKING_URI")
    get("mlflow")(name="golden", **TEMPLATES)
    assert seen[-1]["tracking_uri"] is None  # the framework's default server
    assert set(seen[-1]) >= {"input_template", "output_template", "tools_template", "batch_size", "limit", "repeat"}


# -- Unity Catalog ---------------------------------------------------------------


def _uc(**kwargs):
    return get("unity_catalog")(**{**TEMPLATES, "input_template": '{"q": {{ q | tojson }}}', **kwargs})


def _no_rest(monkeypatch):
    import httpx

    def refuse(*args, **kwargs):
        raise AssertionError("the REST API must not be called")

    monkeypatch.setattr(httpx, "get", refuse)


def test_unity_catalog_names_are_three_parts():
    from openarena.datasets.unity_catalog_dataset import table_parts

    assert table_parts("cat.sch.tbl") == ("cat", "sch", "tbl")
    with pytest.raises(ValueError, match="fully-qualified as 'catalog.schema.table'; got 'sch.tbl'"):
        table_parts("sch.tbl")


def test_unity_catalog_with_a_storage_path_skips_the_rest_lookup(monkeypatch):
    _no_rest(monkeypatch)
    ds = _uc(table="cat.sch.tbl", storage_path="s3://bucket/tbl", version=3, read_batch_size=50)
    assert ds.metadata == {"storage.uri": "s3://bucket/tbl"} and ds.client == "s3://bucket/tbl"
    assert ds.table == "cat.sch.tbl" and ds.version == 3 and ds.read_batch_size == 50
    with pytest.raises(NotImplementedError, match="set `limit`"):
        len(ds)
    assert len(_uc(table="cat.sch.tbl", storage_path="s3://b/t", limit=10, batch_size=4)) == 3


def test_unity_catalog_refuses_a_pin_it_cannot_read_and_a_name_it_cannot_parse(monkeypatch):
    _no_rest(monkeypatch)
    with pytest.raises(ValueError, match="either `version` \\(int\\) or `as_of`"):
        _uc(table="c.s.t", storage_path="s3://b/t", version=1, as_of="2026-01-01T00:00:00Z")
    with pytest.raises(TypeError, match="`version` must be an int; got str"):
        _uc(table="c.s.t", storage_path="s3://b/t", version="3")
    with pytest.raises(ValueError, match="fully-qualified"):
        _uc(table="just_a_table", storage_path="s3://b/t")


def test_unity_catalog_resolves_the_table_over_rest_with_the_token(monkeypatch):
    import httpx

    monkeypatch.setenv("UNITY_CATALOG_API_URL", "https://uc.example/api/2.1/unity-catalog/")
    monkeypatch.setenv("UC_TOKEN", "secret")
    seen = []

    def fake_get(url, headers=None, timeout=None):
        seen.append((url, headers, timeout))
        payload = {
            "storage_location": "s3://bucket/path",
            "properties": {"delta.lastCommitTimestamp": "1700000000"},
            "updated_at": "ignored when the commit stamp is there",
            "table_type": "EXTERNAL",
            "data_source_format": "DELTA",
        }
        return SimpleNamespace(json=lambda: payload, raise_for_status=lambda: None)

    monkeypatch.setattr(httpx, "get", fake_get)
    ds = _uc(table="cat.sch.tbl")
    assert seen == [("https://uc.example/api/2.1/unity-catalog/tables/cat.sch.tbl", {"Authorization": "Bearer secret"}, 30)]
    assert ds.metadata == {"storage.uri": "s3://bucket/path", "content_hash": "1700000000", "table_type": "EXTERNAL", "data_source_format": "DELTA"}
    assert ds.client == "s3://bucket/path"

    monkeypatch.setattr(httpx, "get", lambda url, headers=None, timeout=None: SimpleNamespace(json=lambda: {"updated_at": "2026"}, raise_for_status=lambda: None))
    with pytest.raises(ValueError, match="Could not determine a storage path for table 'cat.sch.tbl'"):
        _uc(table="cat.sch.tbl")


def test_unity_catalog_says_which_setting_is_missing(monkeypatch):
    _no_rest(monkeypatch)
    monkeypatch.delenv("UNITY_CATALOG_API_URL", raising=False)
    monkeypatch.setenv("UC_TOKEN", "secret")
    with pytest.raises(EnvironmentError, match="UNITY_CATALOG_API_URL is not set"):
        _uc(table="c.s.t")
    monkeypatch.setenv("UNITY_CATALOG_API_URL", "https://uc.example/api/2.1/unity-catalog")
    monkeypatch.delenv("UC_TOKEN", raising=False)
    with pytest.raises(EnvironmentError, match="UC_TOKEN is not set"):
        _uc(table="c.s.t")


def test_unity_catalog_storage_options_come_from_the_environment(monkeypatch):
    from openarena.datasets.unity_catalog_dataset import storage_options

    _no_rest(monkeypatch)
    ds = _uc(table="c.s.t", storage_path="s3://b/t")
    for name in ("AWS_DEFAULT_REGION", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "S3_ENDPOINT"):
        monkeypatch.delenv(name, raising=False)
    assert storage_options() == {"region": "fr-par", "aws_access_key_id": "", "aws_secret_access_key": ""}
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-west-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AK")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "SK")
    monkeypatch.setenv("S3_ENDPOINT", "http://minio:9000")
    assert storage_options() == {"region": "eu-west-1", "aws_access_key_id": "AK", "aws_secret_access_key": "SK", "endpoint_url": "http://minio:9000", "allow_http": True}
    monkeypatch.setenv("S3_ENDPOINT", "https://s3.example")
    assert storage_options()["allow_http"] is False
    # The format hint decides the reader; unknown means Delta.
    for hint, delta in (("DELTA", True), ("PARQUET", False), ("csv", False), ("", True)):
        ds.metadata["data_source_format"] = hint
        assert ds.is_delta is delta, hint


def test_unity_catalog_streams_a_delta_table_at_the_pinned_version(monkeypatch):
    import pyarrow as pa
    import pyarrow.dataset as pad

    _no_rest(monkeypatch)
    seen = []

    class DeltaTable:
        def __init__(self, path, **kwargs):
            seen.append(("open", path, kwargs))

        def load_as_version(self, version):
            seen.append(("as_of", version))

        def version(self):
            return 7

        def to_pyarrow_dataset(self):
            return pad.dataset(pa.Table.from_pylist([{"q": "a"}, {"q": "b"}, {"q": "c"}]))

    _module(monkeypatch, "deltalake", DeltaTable=DeltaTable)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AK")
    ds = _uc(table="c.s.t", storage_path="s3://b/t", version=3, read_batch_size=2)
    assert list(ds._iter_rows()) == [{"q": "a"}, {"q": "b"}, {"q": "c"}]
    assert seen[0][1] == "s3://b/t" and seen[0][2]["version"] == 3 and seen[0][2]["storage_options"]["aws_access_key_id"] == "AK"
    assert ds.metadata["delta_version"] == 7 and not any(call[0] == "as_of" for call in seen)
    by_time = _uc(table="c.s.t", storage_path="s3://b/t", as_of="2026-01-01T00:00:00Z")
    list(by_time._iter_rows())
    assert ("as_of", "2026-01-01T00:00:00Z") in seen and "version" not in seen[-2][2]


def test_unity_catalog_streams_plain_parquet_through_a_filesystem_for_the_path(monkeypatch, tmp_path):
    import pyarrow as pa
    import pyarrow.fs as paf
    import pyarrow.parquet as pq

    _no_rest(monkeypatch)
    pq.write_table(pa.Table.from_pylist([{"q": "a"}, {"q": "b"}]), tmp_path / "part-0.parquet")
    ds = _uc(table="c.s.t", storage_path=str(tmp_path), read_batch_size=1)
    ds.metadata["data_source_format"] = "PARQUET"
    assert list(ds._iter_rows()) == [{"q": "a"}, {"q": "b"}]
    # An s3 path gets an S3 filesystem built from the storage options, the scheme stripped.
    from openarena.datasets.unity_catalog_dataset import filesystem_for

    seen = []
    monkeypatch.setattr(paf, "S3FileSystem", lambda **kwargs: seen.append(kwargs) or "s3fs")
    options = {"region": "eu-west-1", "aws_access_key_id": "AK", "aws_secret_access_key": "SK", "endpoint_url": "http://minio:9000"}
    assert filesystem_for("s3://b/t", options) == "s3fs"
    assert seen == [{"region": "eu-west-1", "access_key": "AK", "secret_key": "SK", "endpoint_override": "http://minio:9000"}]
    assert filesystem_for("s3://b/t", {}) == "s3fs" and seen[-1] == {"region": "fr-par"}


def test_unity_catalog_without_deltalake_names_its_extra(monkeypatch):
    _no_rest(monkeypatch)
    monkeypatch.setitem(sys.modules, "deltalake", None)
    with pytest.raises(ImportError, match=r"openarena\[unity-catalog\]"):
        list(_uc(table="c.s.t", storage_path="s3://b/t")._iter_rows())
