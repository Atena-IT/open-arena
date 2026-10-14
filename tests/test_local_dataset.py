# License Apache 2.0: (c) 2026 Atena-Reply

"""The local file provider reads a JSON list as well as JSONL, and says what a JSON file must hold."""

import json
import os

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import pytest

from openarena.datasets.local_dataset import LocalDataset

TEMPLATES = dict(input_template='{"q": {{ q | tojson }}}', output_template='{"a": {{ a | tojson }}}')
ROWS = [{"q": "2+2", "a": "4"}, {"q": "3+3", "a": "6"}]


def _rows(ds):
    return [json.loads(x.get_json() and json.dumps(x.get_json())) for batch in ds for x in batch[0]]


def test_a_json_list_reads_like_jsonl(tmp_path):
    (tmp_path / "list.json").write_text(json.dumps(ROWS))
    (tmp_path / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in ROWS))
    as_list = LocalDataset(tmp_path / "list.json", batch_size=2, **TEMPLATES)
    as_lines = LocalDataset(tmp_path / "rows.jsonl", batch_size=2, **TEMPLATES)
    assert list(as_list._iter_rows()) == list(as_lines._iter_rows()) == ROWS
    assert len(as_list) == 1  # two rows, one batch


def test_a_json_object_with_one_list_inside_is_accepted_and_the_rest_refused(tmp_path):
    (tmp_path / "wrapped.json").write_text(json.dumps({"version": 1, "data": ROWS}))
    assert list(LocalDataset(tmp_path / "wrapped.json", **TEMPLATES)._iter_rows()) == ROWS
    (tmp_path / "two.json").write_text(json.dumps({"a": ROWS, "b": ROWS}))
    with pytest.raises(ValueError, match="exactly one list"):
        list(LocalDataset(tmp_path / "two.json", **TEMPLATES)._iter_rows())
    (tmp_path / "scalars.json").write_text(json.dumps([1, 2, 3]))
    with pytest.raises(ValueError, match="list of objects"):
        list(LocalDataset(tmp_path / "scalars.json", **TEMPLATES)._iter_rows())


def test_a_jsonl_or_csv_file_has_a_length_without_a_limit(tmp_path):
    """The trainer asks every dataset its length; a jsonl or csv file without `limit` must answer, not raise."""
    from openarena.datasets.local_dataset import LocalDataset

    rows = tmp_path / "rows.jsonl"
    rows.write_text('{"q": "a", "a": "1"}\n\n{"q": "b", "a": "2"}\n{"q": "c", "a": "3"}\n')
    ds = LocalDataset(path=str(rows), input_template='{"q": {{ q | tojson }}}', batch_size=2)
    assert len(ds) == 2  # 3 rows, batches of 2
    table = tmp_path / "rows.csv"
    table.write_text("q,a\nx,1\ny,2\n")
    assert len(LocalDataset(path=str(table), input_template='{"q": {{ q | tojson }}}')) == 2


def test_a_bad_file_is_named_with_its_line_before_a_run(tmp_path):
    import pytest

    from openarena.datasets.local_dataset import LocalDataset

    def check(name, text):
        path = tmp_path / name
        path.write_text(text)
        return LocalDataset(path=str(path), input_template="{}").check()

    assert check("ok.jsonl", '{"q": 1}\n{"q": 2}\n') == 2
    with pytest.raises(ValueError, match=r"bad.jsonl, line 2: not valid JSON"):
        check("bad.jsonl", '{"q": 1}\n{"q": 2,}\n')
    with pytest.raises(ValueError, match=r"list.jsonl, line 1: each line must be one JSON object"):
        check("list.jsonl", "[1, 2]\n")
    with pytest.raises(ValueError, match=r"wide.csv, line 3: more values than the header"):
        check("wide.csv", "q,a\nx,1\ny,2,3\n")
    with pytest.raises(ValueError, match="has no rows"):
        check("empty.jsonl", "\n")
