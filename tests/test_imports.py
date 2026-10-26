# License Apache 2.0: (c) 2026 Atena-Reply

"""Imports: benchmarks scored elsewhere read as runs of the board, and the files other tools write read into that shape.

`parse` and what it refuses, an import as a run (`run_of`), the folder of
imports read once per version of each file (`load`, `runs_in`), a saved
leaderboard exported as an import that gives the same board back (`export`),
and `read` over the three shapes it knows: the arena's own JSON, a tidy CSV,
lm-evaluation-harness results.
"""

import json
import logging

import pytest
from pydantic import ValidationError

from openarena import imports
from openarena import leaderboard as lb
from openarena.import_readers import NotRecognised, _slug, guess_direction, read
from openarena.imports import (
    MAX_BYTES,
    ImportFile,
    NothingToExport,
    export,
    load,
    parse,
    run_of,
    runs_in,
)


def _row(model, dataset, metric="reward", value=0.5, direction="max", **more):
    return {"model": model, "dataset": dataset, "metric": metric, "value": value, "direction": direction, **more}


def _doc(**overrides):
    doc = {
        "meta": {"name": "bench", "title": "The bench", "source": "a harness", "tags": ["external"]},
        "rows": [_row("a", "d1", value=0.9), _row("a", "d1", "cost", 1.0, "min"), _row("a", "d2", value=0.4), _row("b", "d1", value=0.5), _row("b", "d2", value=0.8, ci_low=0.7, ci_high=0.9, n=100)],
        "datasets": {"d1": {"group": "reasoning", "tags": ["english"]}, "d2": {"group": "knowledge"}},
        "models": {"a": {"model_id": "openai/gpt-4o"}},
    }
    return {**doc, **overrides}


def _bytes(doc):
    return json.dumps(doc).encode()


# -- parse ------------------------------------------------------------------------------


def test_an_import_is_a_runs_result_with_a_reward_per_cell_and_nothing_unknown():
    imported = parse(_bytes(_doc()))
    assert isinstance(imported, ImportFile) and imported.meta.name == "bench" and len(imported.rows) == 5
    assert imported.rows[4].ci_low == 0.7 and imported.rows[4].n == 100 and imported.datasets["d1"].group == "reasoning"
    with pytest.raises(ValueError, match=f"over the {MAX_BYTES} limit"):
        parse(b" " * (MAX_BYTES + 1))
    with pytest.raises(ValueError, match="NaN is not a JSON number"):
        parse(_bytes(_doc()).replace(b"0.9", b"NaN"))
    with pytest.raises(ValueError, match="Infinity is not a JSON number"):
        parse(_bytes(_doc(rows=[_row("a", "d", value=1e400)])))
    with pytest.raises(ValidationError, match="finite"):
        ImportFile.model_validate(_doc(rows=[_row("a", "d1", value=float("inf"))], datasets={}, models={}))
    with pytest.raises(ValidationError, match="extra_forbidden"):
        parse(_bytes(_doc(extra="no")))


def _refused(match, **overrides):
    with pytest.raises(ValidationError, match=match):
        parse(_bytes(_doc(**overrides)))


def test_what_an_import_may_not_say():
    _refused("1 cell\\(s\\) have no `reward` row, e.g. model 'c' on dataset 'd1'", rows=[*_doc()["rows"], _row("c", "d1", "cost", 1.0, "min")])
    _refused("rows\\[1\\] repeats model 'a', dataset 'd1', metric 'reward'", rows=[_row("a", "d1"), _row("a", "d1", value=0.6)], datasets={}, models={})
    _refused("`datasets` names \\['d9'\\], which no row scores", datasets={"d9": {}})
    _refused("`models` names \\['z'\\], which no row scores", models={"z": {}})
    _refused("`leaderboard.datasets` names \\['d9'\\]", leaderboard={"datasets": ["d9"]})
    _refused("`leaderboard.dataset_shares` names \\['d9'\\], which is none of \\['d1', 'd2'\\]", leaderboard={"dataset_shares": {"d9": 2}})
    _refused("`leaderboard.group_shares` names \\['other'\\], which is none of \\['knowledge', 'reasoning'\\]", leaderboard={"group_shares": {"other": 2}})
    _refused("charts\\[0\\] plots 'latency', which no row records: one of \\['cost', 'reward'\\]", charts=[{"type": "bar", "title": "t", "metric": "latency"}])
    _refused("charts\\[0\\] stacks by group, but \\['d2'\\] have no `datasets.<name>.group`", datasets={"d1": {"group": "g"}}, charts=[{"type": "stacked-bar", "title": "t", "metric": "reward", "stack": "group"}])
    _refused("`stack` is set on a `stacked-bar` or a `heatmap`, and only there", charts=[{"type": "bar", "title": "t", "metric": "reward", "stack": "group"}])
    _refused("`x` is set on a `scatter`, and only there", charts=[{"type": "scatter", "title": "t", "metric": "reward"}])
    _refused("url", meta={"name": "b", "title": "t", "url": "ftp://x"})


def test_an_import_is_a_run_of_the_board_under_its_own_prefix():
    run = run_of(parse(_bytes(_doc())))
    assert run["id"] == "import:bench" and run["model_ids"] == {"a": "openai/gpt-4o"}
    assert run["groups"] == {"bench/d1": "bench/reasoning", "bench/d2": "bench/knowledge"}
    assert {r["dataset"] for r in run["result"]["rows"]} == {"bench/d1", "bench/d2"}
    assert run["result"]["rows"][4] == {**_row("b", "bench/d2", value=0.8), "ci_low": 0.7, "ci_high": 0.9, "n": 100}
    data = lb.board([run], eol=False)
    assert data["datasets"] == ["bench/d1", "bench/d2"] and data["groups"] == run["groups"]
    assert [r["model"] for r in lb.rank(data, shares={"reward": 100})["rows"]] == ["a", "b"]


# -- the folder of imports --------------------------------------------------------------


def test_a_file_is_parsed_once_per_version_and_a_broken_one_is_skipped_with_a_warning(tmp_path, monkeypatch, caplog):
    parsed = []
    real = imports.parse
    monkeypatch.setattr(imports, "parse", lambda raw: parsed.append(1) or real(raw))
    monkeypatch.setattr(imports, "_LOADED", type(imports._LOADED)())
    good, bad = tmp_path / "bench.json", tmp_path / "broken.json"
    good.write_bytes(_bytes(_doc()))
    bad.write_text("{not json")
    assert load(good).meta.name == "bench" and load(good).meta.name == "bench" and parsed == [1]
    with caplog.at_level(logging.WARNING, logger="openarena.imports"):
        runs = runs_in(tmp_path)
    assert [r["id"] for r in runs] == ["import:bench"] and len(parsed) == 2  # the broken file was tried, the good one was cached
    assert "import broken.json skipped" in caplog.text
    assert runs_in(tmp_path)[0] is runs[0] and len(parsed) == 3  # the run is cached too; the broken file is tried again
    good.write_bytes(_bytes(_doc(meta={"name": "bench2", "title": "t"})))
    assert runs_in(tmp_path)[0]["id"] == "import:bench2" and len(parsed) == 5  # the file changed: parsed again
    bad.unlink()
    assert len(runs_in(tmp_path)) == 1 and bad not in imports._LOADED
    assert runs_in(tmp_path / "nowhere") == []


def test_the_cache_keeps_the_most_recent_imports(tmp_path, monkeypatch):
    monkeypatch.setattr(imports, "_LOADED", type(imports._LOADED)())
    monkeypatch.setattr(imports, "MAX_CACHED", 2)
    paths = []
    for i in range(3):
        path = tmp_path / f"b{i}.json"
        path.write_bytes(_bytes(_doc(meta={"name": f"b{i}", "title": "t"})))
        load(path)
        paths.append(path)
    assert list(imports._LOADED) == paths[1:]


# -- export ----------------------------------------------------------------------------


def _board_data():
    rows = [_row("a", "d1", value=0.9), _row("a", "d1", "cost", 1.0, "min"), _row("a", "d2", value=0.4), _row("b", "d1", value=0.5)]
    return lb.board([{"id": "run", "result": {"rows": rows}, "model_ids": {"a": "openai/gpt-4o"}}], eol=False)


def test_a_saved_leaderboard_exports_as_an_import_that_gives_the_same_board_back():
    board = {"name": "weekly", "title": "Weekly", "description": "the week", "tags": ["imported", "ops"], "datasets": ["d1"], "models": [], "shares": {"reward": 100}}
    document = export(_board_data(), board)
    assert document["meta"] == {"name": "weekly", "title": "Weekly", "source": "arena", "tags": ["ops"], "description": "the week", "provisional": False}
    assert sorted((r["model"], r["metric"], r["value"]) for r in document["rows"]) == [("a", "cost", 1.0), ("a", "reward", 0.9), ("b", "reward", 0.5)]
    assert document["models"]["a"]["model_id"] == "openai/gpt-4o" and list(document["models"]) == ["a"]  # the id a registry name ran as
    assert document["leaderboard"]["datasets"] == ["d1"] and document["leaderboard"]["shares"] == {"reward": 100} and document["leaderboard"]["tags"] == ["ops"]
    again = run_of(parse(_bytes(document)))
    assert {r["dataset"] for r in again["result"]["rows"]} == {"weekly/d1"}
    with pytest.raises(NothingToExport):
        export(_board_data(), {"name": "x", "title": "x", "models": ["nobody"]})
    everything = export(_board_data(), {"name": "all", "title": "All"})  # no scope: every dataset, every model
    assert {r["dataset"] for r in everything["rows"]} == {"d1", "d2"} and everything["leaderboard"]["datasets"] == []


def test_a_board_over_one_import_exports_under_the_imports_own_names():
    source = parse(_bytes(_doc(charts=[{"type": "bar", "title": "Reward", "metric": "reward"}, {"type": "scatter", "title": "Cost", "metric": "reward", "x": "cost"}])))
    data = lb.board([run_of(source)], eol=False)
    board = {"name": "bench-board", "title": "Bench", "tags": [], "datasets": ["bench/d2"], "models": ["a", "b"], "shares": {}, "group_shares": {"bench/knowledge": 2, "bench/reasoning": 1}, "dataset_shares": {"bench/d2": 3, "bench/d1": 1}}
    document = export(data, board, source=source)
    assert document["meta"]["name"] == "bench" and document["meta"]["source"] == "a harness"
    assert {r["dataset"] for r in document["rows"]} == {"d2"}
    assert list(document["datasets"]) == ["d2"] and document["datasets"]["d2"]["group"] == "knowledge"
    assert list(document["models"]) == ["a"] and document["models"]["a"]["model_id"] == "openai/gpt-4o"
    assert document["leaderboard"]["datasets"] == ["d2"] and document["leaderboard"]["group_shares"] == {"knowledge": 2} and document["leaderboard"]["dataset_shares"] == {"d2": 3}
    assert [c["title"] for c in document["charts"]] == ["Reward"]  # the scatter plots cost, which d2 has no row of


# -- the readers -------------------------------------------------------------------------


def test_the_arenas_own_json_is_read_as_it_is_with_its_corrections_applied():
    found = read(_bytes(_doc(leaderboard={"name": "bench-board", "title": "B"})), "bench.json", directions={"cost": "max", "junk": "sideways"}, meta={"name": "mine", "title": ""})
    assert found["format"] == "arena" and found["guessed"] == [] and found["warnings"] == []
    assert found["document"]["meta"]["name"] == "mine" and found["document"]["meta"]["title"] == "The bench"  # a blank correction changes nothing
    assert found["document"]["leaderboard"]["name"] == "mine"  # renamed, its board goes by the new name
    assert {r["metric"]: r["direction"] for r in found["document"]["rows"]}["cost"] == "max"
    # What the user may correct, each metric with its direction as corrected.
    assert found["metrics"] == [{"name": "cost", "direction": "max", "guessed": False}, {"name": "reward", "direction": "max", "guessed": False}]
    assert isinstance(found["imported"], ImportFile)
    with pytest.raises(ValueError, match="corrections are"):
        read(_bytes(_doc()), directions=["cost"])
    with pytest.raises(ValueError, match="over the"):
        read(b" " * (MAX_BYTES + 1))


def test_a_tidy_csv_is_read_with_the_directions_guessed_from_the_metric_names():
    csv = "﻿Model,Task,Metric,Value,direction,primary,ci_low,ci_high,n,notes\n" "a,d1,acc,0.9,,true,0.8,1.0,100,x\n" "a,d1,latency_ms,120,,,,,,\n" "b,d1,acc,0.5,MAX,yes,,,,\n" "b,d1,latency_ms,,,,,,,\n" "b,d2,acc,0.7,,,,,,\n"
    found = read(csv.encode(), "My Results v2.csv")
    assert found["format"] == "csv" and found["document"]["meta"] == {"name": "My-Results-v2", "title": "My Results v2"}
    rows = {(r["model"], r["dataset"], r["metric"]): r for r in found["document"]["rows"]}
    assert rows[("a", "d1", "acc")] == {"model": "a", "dataset": "d1", "metric": "acc", "value": 0.9, "direction": "max", "ci_low": 0.8, "ci_high": 1.0, "n": 100}
    assert rows[("a", "d1", "latency_ms")]["direction"] == "min" and rows[("a", "d1", "reward")]["value"] == 0.9  # the primary metric is the reward
    assert rows[("b", "d2", "reward")]["value"] == 0.7  # a lone metric is the reward
    assert ("b", "d1", "latency_ms") not in rows  # a blank value is no row
    assert found["guessed"] == ["acc", "latency_ms"] and found["warnings"] == ["columns not read: notes"]
    assert found["metrics"][0] == {"name": "acc", "direction": "max", "guessed": True}
    corrected = read(csv.encode(), "r.csv", directions={"acc": "min"})
    assert corrected["guessed"] == ["latency_ms"] and corrected["metrics"][0]["guessed"] is False
    with pytest.raises(ValueError, match="cells with several metrics and none marked primary, e.g. dataset 'd1'"):
        read(b"model,dataset,metric,value\na,d1,acc,0.9\na,d1,f1,0.8\n")
    with pytest.raises(ValueError, match="line 2: could not convert string to float"):
        read(b"model,dataset,metric,value\na,d1,acc,high\n")
    with pytest.raises(NotRecognised, match="not a tidy CSV"):
        read(b"model,score\na,1\n")


def test_lm_evaluation_harness_results_are_read_task_by_task():
    results = {
        "model_name": "hf/qwen",
        "results": {
            "gsm8k": {"alias": "gsm8k", "exact_match,strict-match": 0.6, "exact_match_stderr,strict-match": 0.01, "exact_match,flexible-extract": 0.7, "ok": True},
            "wikitext": {"word_perplexity,none": 12.5, "alias": " - wikitext"},
        },
        "higher_is_better": {"gsm8k": {"exact_match": True}},
        "n-samples": {"gsm8k": {"original": 1319, "effective": 100}},
    }
    found = read(_bytes(results), "results_2026.json")
    assert found["format"] == "lm-evaluation-harness" and found["document"]["meta"] == {"name": "results_2026", "title": "hf/qwen (lm-evaluation-harness)", "source": "lm-evaluation-harness"}
    rows = {(r["dataset"], r["metric"]): r for r in found["document"]["rows"]}
    assert set(rows) == {("gsm8k", "exact_match@strict-match"), ("gsm8k", "exact_match@flexible-extract"), ("gsm8k", "reward"), ("wikitext", "word_perplexity"), ("wikitext", "reward")}
    assert rows[("gsm8k", "reward")]["value"] == 0.6 and rows[("gsm8k", "exact_match@strict-match")]["n"] == 100  # the first metric is the reward
    assert rows[("wikitext", "word_perplexity")]["direction"] == "min" and found["guessed"] == ["word_perplexity"]
    assert found["warnings"] == [
        "each task's first metric is its reward: check it is the one to rank on",
        "dataset 'gsm8k' mixes exact_match@flexible-extract, exact_match@strict-match: check every model was measured the same way",
    ]
    with pytest.raises(ValueError, match="malformed lm-evaluation-harness results"):
        read(_bytes({"results": {"gsm8k": [1, 2]}}))
    assert read(_bytes({"config": {"model_args": "pretrained=x"}, "results": {"t": {"acc,none": 1}}}))["document"]["rows"][0]["model"] == "pretrained=x"


def test_the_shapes_the_readers_do_not_know_and_the_helpers_they_share():
    for raw in (b"<html>", b'{"other": 1}', b"[1, 2]", b"\x00\xff"):
        with pytest.raises(NotRecognised):
            read(raw)
    assert guess_direction("TTFT@p50") == "min" and guess_direction("accuracy") == "max" and guess_direction("cost_usd") == "min"
    assert _slug("My Results (v2).json") == "My-Results-v2" and _slug("") == "import" and _slug("...") == "import"
    mixed = read(b"model,dataset,metric,value,primary\na,d,acc@a,0.9,1\nb,d,acc@b,0.5,1\n")
    assert mixed["warnings"] == ["dataset 'd' mixes acc@a, acc@b: check every model was measured the same way"]
