# License Apache 2.0: (c) 2026 Atena-Reply

"""The engine's `Dataset` and its file providers, as a run reads them.

What every provider inherits: a hand-written schema closed at the door, the
templates rendered over each row, `limit`, `repeat` and `batch_size` shaping
the batches, the length the trainer asks for. Then the registry that turns a
YAML entry into a provider, and the three providers that read files: `local`,
`folder`, `huggingface` (its loader faked, no download).
"""

import json
import os

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import jinja2
import pytest
import synalinks

from openarena.datasets import get, load_dataset_from_config
from openarena.datasets.dataset import Dataset, _close_objects, _coerce_schema
from openarena.datasets.folder_dataset import FolderDataset
from openarena.datasets.huggingface_dataset import HuggingFaceDataset
from openarena.datasets.local_dataset import LocalDataset

ROWS = [{"q": "2+2", "a": "4"}, {"q": "3+3", "a": "6"}, {"q": "4+4", "a": "8"}]
TEMPLATES = {"input_template": '{"q": {{ q | tojson }}}', "output_template": '{"a": {{ a | tojson }}}'}
Q_SCHEMA = {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}
A_SCHEMA = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}


class Rows(Dataset):
    """A dataset over rows held in memory: the base class, and nothing else. Schemas unless told otherwise."""

    def __init__(self, rows=ROWS, **kwargs):
        super().__init__(**{**TEMPLATES, "input_schema": Q_SCHEMA, "output_schema": A_SCHEMA, **kwargs})
        self.rows = list(rows)

    def _iter_rows(self):
        yield from self.rows

    def __len__(self):
        return self._total_batches(len(self.rows) if self.limit is None else min(self.limit, len(self.rows)))


def _json(batch):
    return [[item.get_json() for item in side] for side in batch]


# -- schemas closed at the door --------------------------------------------------


def test_every_object_of_a_hand_written_schema_is_closed_at_every_level():
    schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "object", "properties": {"city": {"type": "string"}}},
            "steps": {"type": "array", "items": {"type": "object", "properties": {"text": {"type": "string"}}}},
            "either": {"anyOf": [{"type": "object", "properties": {}}, {"type": "string"}]},
        },
        "$defs": {"Call": {"properties": {"name": {"type": "string"}}}},
    }
    closed = _close_objects(schema)
    assert closed["additionalProperties"] is False
    assert closed["properties"]["answer"]["additionalProperties"] is False
    assert closed["properties"]["steps"]["items"]["additionalProperties"] is False
    assert closed["properties"]["either"]["anyOf"][0]["additionalProperties"] is False
    assert "additionalProperties" not in closed["properties"]["either"]["anyOf"][1]
    assert closed["$defs"]["Call"]["additionalProperties"] is False  # `properties` alone marks an object
    assert "additionalProperties" not in schema  # the schema given is left as it was
    assert "additionalProperties" not in closed["properties"]["steps"]  # an array is not an object


def test_an_explicit_additional_properties_is_kept_whatever_it_says():
    schema = {"type": "object", "properties": {"x": {"type": "object", "additionalProperties": True}}, "additionalProperties": {"type": "string"}}
    closed = _close_objects(schema)
    assert closed["additionalProperties"] == {"type": "string"}
    assert closed["properties"]["x"]["additionalProperties"] is True


def test_a_schema_may_be_a_json_string_and_nothing_else():
    assert _coerce_schema(None) is None
    assert _coerce_schema(json.dumps(Q_SCHEMA)) == {**Q_SCHEMA, "additionalProperties": False}
    for not_a_schema in (42, [Q_SCHEMA]):
        with pytest.raises(TypeError, match="dict or JSON string"):
            _coerce_schema(not_a_schema)


def test_the_dataset_reads_its_schemas_closed_and_the_rows_against_them():
    ds = Rows(input_schema=json.dumps(Q_SCHEMA), output_schema=A_SCHEMA, batch_size=3)
    assert ds.input_schema["additionalProperties"] is False and ds.output_schema["additionalProperties"] is False
    assert ds.input_data_model is None and ds.output_data_model is None
    [(x, y)] = list(ds)
    assert x[0].get_schema() == ds.input_schema and y[0].get_schema() == ds.output_schema
    assert _json((x, y)) == [[{"q": r["q"]} for r in ROWS], [{"a": r["a"]} for r in ROWS]]


def test_without_a_schema_the_rows_are_chat_messages():
    ds = Rows(
        input_template='{"messages": [{"role": "user", "content": {{ q | tojson }}}]}',
        output_template='{"role": "assistant", "content": {{ a | tojson }}}',
        input_schema=None,
        output_schema=None,
        batch_size=1,
    )
    assert ds.input_data_model is synalinks.ChatMessages and ds.output_data_model is synalinks.ChatMessage
    (x, y) = next(iter(ds))
    assert isinstance(x[0], synalinks.ChatMessages) and isinstance(y[0], synalinks.ChatMessage)
    assert y[0].get_json()["content"] == "4"


# -- rows into batches -------------------------------------------------------------


def test_batches_are_filled_to_batch_size_and_the_last_one_flushed():
    assert [len(x) for x, _ in Rows(batch_size=2)] == [2, 1]
    assert len(Rows(batch_size=2)) == 2
    [(x, _)] = list(Rows(batch_size=None))  # no size: everything in one batch
    assert len(x) == 3 and len(Rows(batch_size=None)) == 1
    assert Rows(batch_size=None, rows=[])._total_batches(0) == 0


def test_limit_caps_the_rows_read_and_repeat_copies_each_one_in_a_row():
    assert _json(next(iter(Rows(limit=2, batch_size=5))))[0] == [{"q": "2+2"}, {"q": "3+3"}]
    assert len(Rows(limit=2, batch_size=5)) == 1
    [(x, y)] = list(Rows(repeat=2, batch_size=None))
    assert [r["q"] for r in _json((x,))[0]] == ["2+2", "2+2", "3+3", "3+3", "4+4", "4+4"]
    assert [r["a"] for r in _json((y,))[0]] == ["4", "4", "6", "6", "8", "8"]  # the target repeated with its row
    assert len(Rows(repeat=2, batch_size=4)) == 2  # 3 rows x 2 = 6, in batches of 4
    with pytest.raises(ValueError, match="`repeat` must be a positive int"):
        Rows(repeat=0)


def test_an_inputs_only_dataset_yields_batches_without_targets():
    ds = Rows(output_template=None, output_schema=None, batch_size=None)
    [batch] = list(ds)
    assert len(batch) == 1 and ds.output_data_model is None
    (x,) = ds.materialize()
    assert len(x) == 3


def test_a_template_needs_every_variable_it_names_and_the_input_template_is_required():
    with pytest.raises(jinja2.UndefinedError):
        list(Rows(input_template='{"q": "{{ question }}"}'))
    with pytest.raises(TypeError, match="StrictUndefined is not JSON serializable"):  # the same miss, through `tojson`
        list(Rows(input_template='{"q": {{ question | tojson }}}'))
    with pytest.raises(ValueError, match="`input_template` is required"):
        Rows(input_template=None)
    with pytest.raises(ValueError, match="not both"):
        Rows(input_data_model=synalinks.ChatMessages)  # beside the schema
    with pytest.raises(ValueError, match="require `output_template`"):
        Rows(output_template=None, output_schema=A_SCHEMA)


# -- what a template sees ----------------------------------------------------------


def test_the_whole_row_is_row_for_a_column_no_name_can_reach():
    ds = Rows(rows=[{"Question Text": "2+2", "a": "4"}], input_template='{"q": {{ row["Question Text"] | tojson }}}')
    assert _json(next(iter(ds)))[0] == [{"q": "2+2"}]


def test_a_column_named_row_keeps_its_own_value():
    ds = Rows(rows=[{"row": "2+2", "a": "4"}], input_template='{"q": {{ row | tojson }}}')
    assert _json(next(iter(ds)))[0] == [{"q": "2+2"}]


def test_fromjson_reads_a_column_that_holds_json_text():
    calls = json.dumps([{"name": "get_weather", "arguments": {"city": "Rome"}}])
    ds = Rows(rows=[{"q": "Weather?", "a": calls}], output_template='{"a": {{ (a | fromjson)[0].name | tojson }}}')
    assert _json(next(iter(ds)))[1] == [{"a": "get_weather"}]


def test_fromliteral_reads_a_list_written_as_python_writes_it():
    ds = Rows(rows=[{"q": "Cost?", "a": "['$6', '$7']"}], output_template='{"a": {{ (a | fromliteral)[1] | tojson }}}')
    assert _json(next(iter(ds)))[1] == [{"a": "$7"}]


# -- images and audio, as parts of the message ------------------------------------

PNG = b"\x89PNG\r\n\x1a\n" + bytes(8)
WITH_MEDIA = '{"messages": [{"role": "user", "content": [{"type": "text", "text": {{ q | tojson }}}, {{ m | %s | tojson }}]}]}'


def _part(media, filter="as_image"):
    """The media part the model is sent for one row whose column `m` holds `media`."""
    ds = Rows(rows=[{"q": "What is shown?", "m": media, "a": "4"}], input_template=WITH_MEDIA % filter, input_schema=None)
    [[x], _] = _json(next(iter(ds)))
    text, part = x["messages"][0]["content"]
    assert text == {"type": "text", "text": "What is shown?"}
    return part


def test_an_image_held_as_bytes_is_inlined_and_one_elsewhere_is_left_to_be_read_with_its_batch():
    import base64

    inline = f"data:image/png;base64,{base64.b64encode(PNG).decode()}"
    assert _part({"bytes": PNG, "path": None}) == {"type": "image_url", "image_url": {"url": inline}}  # the type read off its first bytes
    assert _part({"bytes": PNG, "path": "chart.jpg"})["image_url"]["url"].startswith("data:image/jpeg;base64,")  # else its name's
    assert _part("https://example.com/cat.png") == {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}}


def test_an_audio_part_keeps_its_files_format():
    import base64

    assert _part({"bytes": b"fLaC", "path": "clip.flac"}, "as_audio") == {
        "type": "input_audio",
        "input_audio": {"data": base64.b64encode(b"fLaC").decode(), "format": "flac"},
    }
    assert _part("https://example.com/a.mp3?sig=1", "as_audio") == {
        "type": "input_audio",
        "input_audio": {"url": "https://example.com/a.mp3?sig=1", "format": "mp3"},
    }
    with pytest.raises(ValueError, match="no image or audio"):
        _part(None, "as_audio")


def test_a_folder_datasets_relative_image_is_read_from_the_folder(tmp_path):
    (tmp_path / "1.json").write_text(json.dumps({"q": "What is shown?", "image": "images/1.png"}))
    ds = FolderDataset(tmp_path, input_template=WITH_MEDIA.replace("m |", "image |") % "as_image")
    [x] = _json(next(iter(ds)))[0]  # inputs only: no expected answer
    assert x["messages"][0]["content"][1]["image_url"]["url"] == f"file://{tmp_path / 'images' / '1.png'}"


def test_hugging_face_images_and_audio_are_read_undecoded():
    import datasets

    from openarena.datasets.huggingface_dataset import undecoded

    # Stored as the Hub stores them, `{bytes, path}`, then typed: decoding either would need Pillow or torchcodec.
    rows = datasets.Dataset.from_dict(
        {"q": ["What?"], "image": [{"bytes": PNG, "path": "a.png"}], "audio": [{"bytes": b"RIFF", "path": "a.wav"}]}
    ).cast_column("image", datasets.Image()).cast_column("audio", datasets.Audio())
    row = undecoded(datasets.DatasetDict({"test": rows}))["test"][0]
    assert row["image"] == {"bytes": PNG, "path": "a.png"} and row["audio"] == {"bytes": b"RIFF", "path": "a.wav"}


# -- the registry ----------------------------------------------------------------


def test_the_registry_names_every_provider_and_lists_them_on_a_miss():
    assert get("local") is LocalDataset and get("folder") is FolderDataset and get("huggingface") is HuggingFaceDataset
    with pytest.raises(KeyError, match="Unknown dataset provider 'hugingface'. Known: braintrust, folder, harbor_tasks, huggingface, langfuse, langsmith, local, mlflow, opik, phoenix, unity_catalog"):
        get("hugingface")


def _write_rows(tmp_path, name="rows.jsonl", rows=ROWS):
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return str(path)


def test_an_entry_is_built_by_its_type_with_the_engines_keys_stripped(tmp_path):
    entry = {
        "type": "local",
        "path": _write_rows(tmp_path),
        **TEMPLATES,
        "input_schema": Q_SCHEMA,
        "output_schema": json.dumps(A_SCHEMA),
        "batch_size": 2,
        # What the sweep reads, not the provider: stripped before the constructor sees them.
        "generator": {"instructions": "Answer."},
        "reward": "exact",
        "metrics": ["answer_accuracy"],
    }
    ds = load_dataset_from_config({"datasets": {"qa": entry}, "default": "qa"})  # `default` picks the entry
    assert isinstance(ds, LocalDataset) and ds.batch_size == 2
    assert ds.output_schema == _close_objects(A_SCHEMA)  # a JSON string in YAML, read as the schema
    assert [len(x) for x, _ in ds] == [2, 1]


def test_what_an_entry_must_say_and_how_a_miss_is_named(tmp_path):
    entry = {"type": "local", "path": _write_rows(tmp_path), **TEMPLATES}
    with pytest.raises(ValueError, match="<config>: missing or empty `datasets:` section"):
        load_dataset_from_config({"datasets": {}})
    with pytest.raises(ValueError, match="no `name` provided and no `default` key"):
        load_dataset_from_config({"datasets": {"qa": entry}})
    with pytest.raises(KeyError, match="datasets.yaml: dataset 'qb' not found. Available: qa"):
        load_dataset_from_config({"datasets": {"qa": entry}}, "qb", yaml_path="datasets.yaml")
    with pytest.raises(ValueError, match="dataset 'qa' is missing a `type` field"):
        load_dataset_from_config({"datasets": {"qa": {k: v for k, v in entry.items() if k != "type"}}}, "qa")
    with pytest.raises(ValueError, match="dataset 'qa': \"Unknown dataset provider 'locl'"):
        load_dataset_from_config({"datasets": {"qa": {**entry, "type": "locl"}}}, "qa")


def test_an_embedding_entry_gets_the_programs_shapes_unless_it_brings_its_own(tmp_path):
    from openarena.programs.embedding_program import INPUT_SCHEMA, OUTPUT_SCHEMA

    rows = _write_rows(tmp_path, rows=[{"query": "q", "passages": ["a", "b"], "gold": ["a"]}])
    entry = {
        "type": "local",
        "path": rows,
        "input_template": '{"query": {{ query | tojson }}, "chunks": {{ passages | tojson }}}',
        "output_template": '{"chunks": {{ gold | tojson }}}',
        "embedding": {"k": 1},
    }
    ds = load_dataset_from_config({"datasets": {"p": entry}}, "p")
    assert ds.input_schema == _close_objects(INPUT_SCHEMA) and ds.output_schema == _close_objects(OUTPUT_SCHEMA)
    assert [len(x) for x, _ in ds] == [1]
    # Inputs only: no output schema is forced on a dataset without targets.
    alone = load_dataset_from_config({"datasets": {"p": {k: v for k, v in entry.items() if k != "output_template"}}}, "p")
    assert alone.output_schema is None
    own = {"type": "object", "properties": {"query": {"type": "string"}, "chunks": {"type": "array"}}}
    assert load_dataset_from_config({"datasets": {"p": {**entry, "input_schema": own}}}, "p").input_schema == _close_objects(own)


# -- local files -----------------------------------------------------------------


def test_a_csv_or_tsv_file_reads_by_its_delimiter_with_every_value_a_string(tmp_path):
    (tmp_path / "rows.csv").write_text("q,a\n2+2,4\n3+3,6\n")
    (tmp_path / "rows.tsv").write_text("q\ta\n2+2\t4\n3+3\t6\n")
    expected = [{"q": "2+2", "a": "4"}, {"q": "3+3", "a": "6"}]
    assert list(LocalDataset(tmp_path / "rows.csv", **TEMPLATES)._iter_rows()) == expected
    assert list(LocalDataset(tmp_path / "rows.tsv", **TEMPLATES)._iter_rows()) == expected
    assert LocalDataset(tmp_path / "rows.tsv", **TEMPLATES).format == "csv"


def test_a_parquet_file_streams_its_rows_and_knows_its_length_from_its_metadata(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.Table.from_pylist(ROWS), tmp_path / "rows.parquet", row_group_size=2)
    ds = LocalDataset(tmp_path / "rows.parquet", batch_size=2, **TEMPLATES)
    assert list(ds._iter_rows()) == ROWS
    assert len(ds) == 2 and ds.check() == 3
    assert len(LocalDataset(tmp_path / "rows.parquet", batch_size=2, limit=1, **TEMPLATES)) == 1


def test_the_format_is_read_off_the_extension_unless_given(tmp_path):
    rows = tmp_path / "rows.txt"
    rows.write_text("\n".join(json.dumps(r) for r in ROWS))
    with pytest.raises(ValueError, match="Unsupported format for 'rows.txt'"):
        LocalDataset(rows, **TEMPLATES)
    assert list(LocalDataset(rows, format="jsonl", **TEMPLATES)._iter_rows()) == ROWS
    with pytest.raises(FileNotFoundError):
        LocalDataset(tmp_path / "missing.jsonl", **TEMPLATES)


def test_a_jsonl_row_count_follows_the_file(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(ROWS[0]) + "\n")
    ds = LocalDataset(path, **TEMPLATES)
    assert len(ds) == 1
    path.write_text("\n".join(json.dumps(r) for r in ROWS) + "\n")
    assert len(ds) == 3  # recounted: the file changed
    assert len(LocalDataset(path, limit=2, **TEMPLATES)) == 2


# -- folders ---------------------------------------------------------------------


def _folder(tmp_path):
    root = tmp_path / "cases"
    (root / "nested").mkdir(parents=True)
    (root / "b.json").write_text(json.dumps({"question": "B?", "answer": "b"}))
    (root / "a.yaml").write_text("question: A?\nanswer: a\n")
    (root / "c.txt").write_text("plain text")
    (root / "d.json").write_text(json.dumps([1, 2]))
    (root / "nested" / "e.md").write_text("# nested")
    return root


def test_each_file_is_a_row_in_name_order_with_its_fields_or_its_text(tmp_path):
    root = _folder(tmp_path)
    ds = FolderDataset(root, input_template='{"f": {{ _filename | tojson }}}')
    rows = list(ds._iter_rows())
    assert [r["_filename"] for r in rows] == ["a.yaml", "b.json", "c.txt", "d.json"]  # sorted, top level only
    assert rows[0] == {"_filename": "a.yaml", "_stem": "a", "_path": "a.yaml", "question": "A?", "answer": "a"}
    assert rows[1]["question"] == "B?" and rows[2]["text"] == "plain text"
    assert rows[3]["record"] == [1, 2]  # parsed, not a mapping: exposed whole
    assert len(ds) == 4 and len(FolderDataset(root, limit=2, input_template="{}")) == 2


def test_a_pattern_and_recursion_pick_the_files_and_the_path_is_relative_posix(tmp_path):
    root = _folder(tmp_path)
    assert [r["_filename"] for r in FolderDataset(root, pattern="*.json", input_template="{}")._iter_rows()] == ["b.json", "d.json"]
    deep = list(FolderDataset(root, pattern="*.md", recursive=True, input_template="{}")._iter_rows())
    assert deep == [{"_filename": "e.md", "_stem": "e", "_path": "nested/e.md", "text": "# nested"}]


def test_a_format_given_applies_to_every_file_and_an_unknown_one_is_refused(tmp_path):
    root = tmp_path / "notes"
    root.mkdir()
    (root / "x.cfg").write_text("key=value")
    with pytest.raises(ValueError, match="Cannot infer format for 'x.cfg'"):
        list(FolderDataset(root, input_template="{}")._iter_rows())
    [row] = FolderDataset(root, format="text", input_template="{}")._iter_rows()
    assert row["text"] == "key=value"
    with pytest.raises(ValueError, match="Unsupported `format`='xml'"):
        FolderDataset(root, format="xml", input_template="{}")
    with pytest.raises(NotADirectoryError):
        FolderDataset(root / "x.cfg", input_template="{}")
    with pytest.raises(FileNotFoundError):
        FolderDataset(root / "nope", input_template="{}")


# -- HuggingFace -----------------------------------------------------------------


class _Splits(dict):
    """What `load_dataset` returns without a split: a mapping of split name to rows."""


def _hf(monkeypatch, splits, seen):
    from synalinks.src.datasets import huggingface_dataset as module

    def load_dataset(path, **kwargs):
        seen.append((path, kwargs))
        if kwargs.get("split"):
            return list(splits[kwargs["split"]])
        return _Splits(splits)

    monkeypatch.setattr(module, "load_dataset", load_dataset)


def test_without_a_split_every_split_is_read_in_order_and_the_loader_gets_its_kwargs(monkeypatch):
    seen = []
    _hf(monkeypatch, {"train": ROWS[:2], "test": ROWS[2:]}, seen)
    ds = HuggingFaceDataset("org/bench", name="main", revision="v1", streaming=False, data_files="x.jsonl", token="t", **TEMPLATES)
    assert seen == [("org/bench", {"name": "main", "split": None, "revision": "v1", "streaming": False, "data_files": "x.jsonl", "token": "t"})]
    assert ds.load_kwargs == {"data_files": "x.jsonl", "token": "t"}
    assert list(ds._iter_rows()) == ROWS
    assert len(ds) == 3  # the splits' lengths, summed
    only = HuggingFaceDataset("org/bench", split="test", streaming=False, **TEMPLATES)
    assert list(only._iter_rows()) == ROWS[2:] and len(only) == 1


def test_a_template_that_does_not_parse_fails_before_the_source_is_opened(monkeypatch):
    seen = []
    _hf(monkeypatch, {"train": ROWS}, seen)
    with pytest.raises(jinja2.TemplateSyntaxError):
        HuggingFaceDataset("org/bench", split="train", **{**TEMPLATES, "output_template": '{"a": {{ a | tojson }'})
    assert seen == []  # nothing downloaded for a dataset that cannot render


def test_a_streaming_source_has_a_length_only_under_a_limit(monkeypatch):
    _hf(monkeypatch, {"train": ROWS}, [])
    with pytest.raises(NotImplementedError, match="unknown length"):
        len(HuggingFaceDataset("org/bench", split="train", **TEMPLATES))
    assert len(HuggingFaceDataset("org/bench", split="train", limit=2, batch_size=1, **TEMPLATES)) == 2


# -- Harbor task folders ---------------------------------------------------------


def _tasks(tmp_path):
    root = tmp_path / "tasks"
    for name, instruction in (("b-task", "Do B."), ("a-task", None), ("c-task", "Do C.")):
        (root / name).mkdir(parents=True)
        (root / name / "task.toml").write_text("")
        if instruction:
            (root / name / "instruction.md").write_text(f"  {instruction}\n")
    (root / "a-task" / "steps" / "02-second").mkdir(parents=True)
    (root / "a-task" / "steps" / "01-first").mkdir()
    (root / "a-task" / "steps" / "01-first" / "instruction.md").write_text("First step.")
    (root / "a-task" / "steps" / "02-second" / "instruction.md").write_text("Second step.")
    (root / "not-a-task").mkdir()
    (root / "stray.txt").write_text("")
    return root


def test_each_task_folder_is_a_row_named_by_its_folder_with_its_instruction(tmp_path):
    from openarena.datasets.harbor_tasks_dataset import (
        HarborTasksDataset,
        instruction_of,
        task_folders,
    )

    root = _tasks(tmp_path)
    assert [p.name for p in task_folders(root)] == ["a-task", "b-task", "c-task"]  # a folder with a task.toml, by name
    assert [p.name for p in task_folders(root, ["c-task", "a-task"])] == ["c-task", "a-task"]  # the named ones, in that order
    assert task_folders(tmp_path / "nowhere") == []
    with pytest.raises(FileNotFoundError, match="no Harbor task 'z-task' under .* \\(a task is a folder with a task.toml\\)"):
        task_folders(root, ["a-task", "z-task"])
    assert instruction_of(root / "b-task") == "Do B." and instruction_of(root / "a-task") == "First step."  # a multi-step task: its first step's
    assert instruction_of(root / "not-a-task") == ""
    ds = HarborTasksDataset(root, tasks=["b-task", "a-task"], batch_size=1, input_template="ignored", reward="ignored too")
    rows = list(ds._iter_rows())
    assert [(r["task"], r["question"]) for r in rows] == [("b-task", "Do B."), ("a-task", "First step.")]
    assert rows[0]["path"] == str((root / "b-task").resolve())
    assert len(ds) == 2 and len(HarborTasksDataset(root, limit=1)) == 1 and len(HarborTasksDataset(root, repeat=2)) == 6
    (x, y) = next(iter(ds))
    assert x[0].get_json() == rows[0] and y[0].get_json() == {"task": "b-task"}
    with pytest.raises(FileNotFoundError, match="no Harbor tasks folder"):
        HarborTasksDataset(tmp_path / "nowhere")
