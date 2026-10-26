# License Apache 2.0: (c) 2026 Atena-Reply

"""`run_sweep`, end to end on a faked model, and the pieces that turn its trials into the result.

The sweep is run on a local dataset with the language model's call replaced:
no provider, no key. What is held: the result's rows (one per model, dataset
and metric, with its status and direction), the events a run page is told,
the samples it is handed, that a second run on the same state re-runs
nothing, and that `no_cache` does. Then the helpers under it, each on its
own: the long-form rows, the Pareto frontier, the cells and errors read off
a tuner's trials, the metric merge, the grid hand-out.
"""

import asyncio
import os
from types import SimpleNamespace

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import pytest
import synalinks
import yaml
from synalinks.src.backend import JsonDataModel

from openarena import evaluate
from openarena.evaluate import (
    SweepCancelled,
    _as_json,
    _first_by_alias,
    _hand_out,
    _num_batches,
    _ProgressCallback,
    _usage,
    run_sweep,
)
from openarena.results import SweepResults

Q_SCHEMA = {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]}
A_SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}
ROWS = [{"question": "2 + 2?", "answer": "4"}, {"question": "3 + 3?", "answer": "6"}, {"question": "5 + 5?", "answer": "10"}]


def _fake_model(monkeypatch, calls):
    """The model's call replaced: `openai/gpt-4o-mini` answers right, every other model answers "0"."""

    async def fake_call(self, messages, schema=None, **kwargs):
        calls.append(self.model)
        question = messages.get_json()["messages"][-1]["content"]
        right = next(r["answer"] for r in ROWS if r["question"] in question)
        answer = right if self.model == "openai/gpt-4o-mini" else "0"
        return JsonDataModel(json={"answer": answer}, schema=schema)

    monkeypatch.setattr(synalinks.LanguageModel, "__call__", fake_call)


def _config(tmp_path):
    rows = tmp_path / "rows.jsonl"
    rows.write_text("\n".join(__import__("json").dumps(r) for r in ROWS) + "\n")
    config = {
        "models": {"right": {"model": "openai/gpt-4o-mini"}, "wrong": {"model": "openai/gpt-4o", "params": {"temperature": 0.0}}},
        "rewards": {"exact": {"type": "exact_match"}},
        "datasets": {
            "qa": {
                "type": "local",
                "config": {
                    "path": str(rows),
                    "input_template": '{"question": {{ question | tojson }}}',
                    "output_template": '{"answer": {{ answer | tojson }}}',
                    "input_schema": Q_SCHEMA,
                    "output_schema": A_SCHEMA,
                },
                "reward": "exact",
                "metrics": [{"class": "answer_accuracy", "alias": "acc"}],
            }
        },
        "evaluation": {"name": "e", "language_models": ["right", "wrong"], "datasets": ["qa"], "batch_size": 2},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _cells(result):
    return {(r["model"], r["metric"]): (r["value"], r["status"], r["direction"]) for r in result["rows"]}


def test_a_sweep_scores_every_model_on_every_dataset_and_tells_the_run_as_it_goes(tmp_path, monkeypatch):
    calls, events, samples = [], [], []
    _fake_model(monkeypatch, calls)
    result = asyncio.run(
        run_sweep(str(_config(tmp_path)), state_dir=tmp_path / "state", on_event=events.append, on_sample=samples.append, samples_per_cell=2)
    )
    assert result["meta"]["models"] == ["right", "wrong"] and result["meta"]["datasets"] == ["qa"]
    assert result["meta"]["metrics"] == ["reward", "acc"]
    assert result["meta"]["directions"] == {"qa": "max"} and result["meta"]["metric_directions"] == {"acc": "max"}
    assert result["meta"]["objectives_by_ds"] == {"qa": [["reward", "max"]]} and result["meta"]["frontier_by_ds"] == {"qa": []}
    cells = _cells(result)
    assert set(cells) == {("right", "reward"), ("right", "acc"), ("wrong", "reward"), ("wrong", "acc")}
    assert all(status == "COMPLETED" and direction == "max" for _, status, direction in cells.values())
    assert {key: value for key, (value, _, _) in cells.items()} == pytest.approx(
        {("right", "reward"): 1.0, ("right", "acc"): 1.0, ("wrong", "reward"): 0.0, ("wrong", "acc"): 0.0}, abs=1e-6
    )
    assert all(r["dataset"] == "qa" and r["on_frontier"] is False for r in result["rows"])
    # Each model was asked each row, by its own id; the model with params was built with them.
    assert sorted(calls) == ["openai/gpt-4o"] * 3 + ["openai/gpt-4o-mini"] * 3

    names = [e["event"] for e in events]
    assert names[:3] == ["sweep_start", "dataset_start", "dataset_ready"] and names[-2:] == ["dataset_end", "sweep_end"]
    # The run's batch size, 2, over 3 rows: two batches per cell, model-major.
    cells = [(e["model"], e["batches_total"]) for e in events if e["event"] == "cell_start"]
    assert cells == [("right", 2), ("wrong", 2)]
    assert [(e["model"], e["batch"]) for e in events if e["event"] == "batch"] == [("right", 1), ("right", 2), ("wrong", 1), ("wrong", 2)]
    ends = {e["model"]: e["metrics"] for e in events if e["event"] == "cell_end"}
    assert ends["right"]["reward"] == 1.0 and ends["wrong"]["reward"] == 0.0 and "acc" in ends["right"]
    [end] = [e for e in events if e["event"] == "dataset_end"]
    assert end["statuses"] == {"right": "COMPLETED", "wrong": "COMPLETED"} and end["rewards"] == {"right": 1.0, "wrong": 0.0} and end["errors"] == {}
    # The first rows of each cell: the same index is the same row for every model.
    assert [(s["model"], s["index"], s["output"]["answer"], s["reward"]) for s in samples] == [
        ("right", 0, "4", 1.0), ("right", 1, "6", 1.0), ("wrong", 0, "0", 0.0), ("wrong", 1, "0", 0.0),
    ]
    assert samples[0]["input"] == {"question": "2 + 2?"} and samples[0]["target"] == {"answer": "4"} and samples[0]["dataset"] == "qa"


def test_a_second_run_on_the_same_state_reads_the_scores_back_and_no_cache_runs_again(tmp_path, monkeypatch):
    calls, events = [], []
    _fake_model(monkeypatch, calls)
    config, state = str(_config(tmp_path)), tmp_path / "state"
    first = asyncio.run(run_sweep(config, state_dir=state))
    assert len(calls) == 6
    again = asyncio.run(run_sweep(config, state_dir=state, on_event=events.append))
    assert len(calls) == 6  # nothing re-run
    assert again["rows"] == first["rows"]
    [resumed] = [e for e in events if e["event"] == "dataset_resumed"]
    assert resumed == {"event": "dataset_resumed", "dataset": "qa", "scored": ["right", "wrong"], "retrying": []}
    assert not any(e["event"] == "cell_start" for e in events)
    fresh = asyncio.run(run_sweep(config, state_dir=state, no_cache=True))
    assert len(calls) == 12 and fresh["rows"] == first["rows"]


def test_an_event_callback_can_cancel_the_sweep(tmp_path, monkeypatch):
    import keras_tuner  # after the engine, which sets the Keras backend

    _fake_model(monkeypatch, [])

    def cancel(event):
        if event["event"] == "cell_start":
            raise SweepCancelled("stop")

    with pytest.raises(SweepCancelled):
        asyncio.run(run_sweep(str(_config(tmp_path)), state_dir=tmp_path / "state", on_event=cancel))
    assert issubclass(SweepCancelled, keras_tuner.errors.FatalError)  # what makes the tuner re-raise it


# -- the pieces ------------------------------------------------------------------


class _Metrics:
    def __init__(self, values):
        self.values = values

    def get_best_value(self, key):
        if key not in self.values:
            raise KeyError(key)
        return self.values[key]


def _oracle(**trials):
    """A dataset's tuner: `model=(status, values, message)` per trial, plus one of another search space."""
    stale = SimpleNamespace(hyperparameters={}, status="COMPLETED", metrics=_Metrics({"reward": 1.0}), message=None)
    return SimpleNamespace(trials={
        **{model: SimpleNamespace(hyperparameters={"language_model": model}, status=status, metrics=_Metrics(values), message=message) for model, (status, values, message) in trials.items()},
        "stale": stale,
    })


def test_every_trial_is_read_into_one_row_per_model_dataset_and_metric():
    results = SweepResults(["a", "b", "c"], ["d", "e"])
    results.read(
        "d",
        _oracle(
            a=("COMPLETED", {"reward": 0.5, "acc": 0.25}, None),
            b=("FAILED", {}, "Traceback (most recent call last):\n  File x\nRuntimeError: no row got an answer\n"),
            c=("INVALID", {}, "  "),
        ),
        [("acc", "max"), ("cost", "min")],
        "min",
        [("reward", "min")],
    )
    assert results.metrics == ["reward", "acc", "cost"] and results.errors == {"d": {"b": "RuntimeError: no row got an answer", "c": "  "}}
    assert results.status("a", "d") == "COMPLETED" and results.status("b", "d") == "FAILED" and results.status("a", "e") is None
    assert results.value("a", "d") == 0.5 and results.value("a", "d", "acc") == 0.25 and results.value("a", "d", "cost") is None
    rows = results.rows()
    assert [(r["model"], r["dataset"], r["metric"]) for r in rows[:4]] == [("a", "d", "reward"), ("a", "d", "acc"), ("a", "d", "cost"), ("b", "d", "reward")]
    assert len(rows) == 3 * 2 * 3 and [r["dataset"] for r in rows[9:12]] == ["e"] * 3
    assert rows[0] == {"model": "a", "dataset": "d", "metric": "reward", "value": 0.5, "status": "COMPLETED", "direction": "min", "on_frontier": False}
    assert rows[2]["direction"] == "min" and rows[3]["value"] is None and rows[3]["status"] == "FAILED"
    assert rows[9]["status"] is None and rows[9]["direction"] == "max"  # a dataset never read: no reward direction either
    assert results.meta("c.yaml") == {
        "config": "c.yaml", "models": ["a", "b", "c"], "datasets": ["d", "e"], "metrics": ["reward", "acc", "cost"],
        "directions": {"d": "min"}, "metric_directions": {"acc": "max", "cost": "min"},
        "objectives_by_ds": {"d": [["reward", "min"]]}, "frontier_by_ds": {"d": []},
    }


def test_the_frontier_keeps_the_models_no_other_dominates_and_leaves_out_the_unscored():
    results = SweepResults(["cheap", "good", "bad", "tie", "half"], ["d"])
    results.read(
        "d",
        _oracle(
            cheap=("COMPLETED", {"reward": 0.5, "cost": 0.1}, None),
            good=("COMPLETED", {"reward": 0.9, "cost": 1.0}, None),
            bad=("COMPLETED", {"reward": 0.4, "cost": 2.0}, None),
            tie=("COMPLETED", {"reward": 0.9, "cost": 1.0}, None),
            half=("COMPLETED", {"reward": 1.0}, None),  # no cost: cannot be compared
        ),
        [("cost", "min")],
        "max",
        [("reward", "max"), ("cost", "min")],
    )
    assert results.frontier("d") == ["cheap", "good", "tie"] and results.frontier("e") == []
    assert {r["model"] for r in results.rows() if r["on_frontier"]} == {"cheap", "good", "tie"}
    assert results.meta("c.yaml")["frontier_by_ds"] == {"d": ["cheap", "good", "tie"]}
    alone = SweepResults(["bad"], ["d"])
    alone.read("d", _oracle(bad=("COMPLETED", {"reward": 0.4, "cost": 2.0}, None)), [("cost", "min")], "max", [("reward", "max")])
    assert alone.frontier("d") == []  # one objective: no frontier to speak of


def test_metric_lists_merge_by_alias_first_wins():
    per_dataset = [("acc", "ds-acc", "max", True), ("cost", "ds-cost", "min", False)]
    everywhere = [("acc", "global-acc", "max", False), ("latency", "global-latency", "min", False)]
    assert _first_by_alias(per_dataset, everywhere) == [per_dataset[0], per_dataset[1], everywhere[1]]
    assert _first_by_alias([], []) == []


def test_the_grid_hands_out_one_trial_per_model_then_stops():
    oracle = SimpleNamespace()
    _hand_out(oracle, ["a", "b"])
    assert oracle.populate_space("t0") == {"status": "RUNNING", "values": {"language_model": "a"}}
    assert oracle.populate_space("t1") == {"status": "RUNNING", "values": {"language_model": "b"}}
    assert oracle.populate_space("t2") == {"status": "STOPPED", "values": None}


def test_a_dataset_without_a_length_has_no_batch_count():
    class Unknown:
        def __len__(self):
            raise NotImplementedError

    assert _num_batches(Unknown()) is None and _num_batches(object()) is None and _num_batches([1, 2]) == 2


def test_the_progress_callback_reports_each_batch_with_its_usage_and_the_cell_with_its_metrics():
    events = []
    callback = _ProgressCallback(events.append, "d", "m", [1, 2, 3])
    callback.on_test_begin()
    callback.on_test_batch_end(0, {"reward": 0.5, "cost": 0.01, "input_tokens": 12, "note": "text"})
    callback.on_test_batch_end(1, {})
    callback.on_test_end({"reward": 0.75, "acc": 1, "name": "x"})
    assert events == [
        {"event": "batch", "dataset": "d", "model": "m", "batch": 1, "batches_total": 3, "reward": 0.5, "usage": {"cost": 0.01, "input_tokens": 12.0}},
        {"event": "batch", "dataset": "d", "model": "m", "batch": 2, "batches_total": 3, "reward": None, "usage": {}},
        {"event": "cell_end", "dataset": "d", "model": "m", "metrics": {"reward": 0.75, "acc": 1.0}},
    ]
    assert _usage({"cost": "1", "reward_cost": 2, "other": 3}) == {"reward_cost": 2.0}


def test_a_symbolic_batch_has_no_json():
    assert _as_json(None) is None
    assert _as_json(synalinks.SymbolicDataModel(schema=Q_SCHEMA)) is None
    assert _as_json(JsonDataModel(json={"question": "q"}, schema=Q_SCHEMA)) == {"question": "q"}
    assert evaluate.DEFAULT_STATE_DIR.name == ".open-arena"


def test_a_cell_that_failed_is_run_again_on_resume_and_scores_once_the_model_answers(tmp_path, monkeypatch):
    """A model down on the first run: its cell FAILED, with why; the resume retries that cell alone."""
    calls, events = [], []
    _fake_model(monkeypatch, calls)
    down = {"openai/gpt-4o"}
    real_call = synalinks.LanguageModel.__call__

    async def flaky(self, messages, schema=None, **kwargs):
        if self.model in down:
            raise RuntimeError("connection refused")
        return await real_call(self, messages, schema=schema, **kwargs)

    monkeypatch.setattr(synalinks.LanguageModel, "__call__", flaky)
    config, state = str(_config(tmp_path)), tmp_path / "state"
    first = asyncio.run(run_sweep(config, state_dir=state, on_event=events.append))
    cells = _cells(first)
    assert cells[("right", "reward")] == (1.0, "COMPLETED", "max") and cells[("wrong", "reward")][:2] == (None, "FAILED")
    [end] = [e for e in events if e["event"] == "dataset_end"]
    assert end["statuses"] == {"right": "COMPLETED", "wrong": "FAILED"} and end["errors"] == {"wrong": "RuntimeError: connection refused"}
    down.clear()
    events.clear()
    again = asyncio.run(run_sweep(config, state_dir=state, on_event=events.append))
    [resumed] = [e for e in events if e["event"] == "dataset_resumed"]
    assert resumed["scored"] == ["right"] and resumed["retrying"] == ["wrong"]
    assert [e["model"] for e in events if e["event"] == "cell_start"] == ["wrong"]
    assert _cells(again)[("wrong", "reward")] == (0.0, "COMPLETED", "max") and calls.count("openai/gpt-4o-mini") == 3


def test_metrics_are_built_from_the_rewards_or_the_frameworks_and_a_bad_one_fails_early():
    from openarena.config import MetricEntry
    from openarena.evaluate import _instantiate_metrics

    entries = [MetricEntry.model_validate(m) for m in ("cost", {"class": "exact_match", "alias": "em"}, {"class": "avg_latency", "direction": "max"})]
    built = _instantiate_metrics(entries, context="metrics", target=["answer"])
    assert [(alias, direction, objective) for alias, _, direction, objective in built] == [("cost", "min", False), ("em", "max", False), ("avg_latency", "max", False)]
    assert isinstance(built[0][1], synalinks.metrics.Cost) and isinstance(built[1][1], synalinks.metrics.MeanMetricWrapper)
    assert built[1][1]._fn.fields == ["answer"]  # a reward metric is scoped to the answer like the reward
    judged = _instantiate_metrics([MetricEntry.model_validate({"class": "lm_as_judge", "language_model": {"model": "openai/gpt-4o", "temperature": 0.0}})], context="metrics")
    assert judged[0][1]._fn.name == "lm_as_judge"
    with pytest.raises(ValueError, match="Could not interpret metric identifier: nope"):
        _instantiate_metrics([MetricEntry.model_validate("nope")], context="metrics")
    with pytest.raises(ValueError, match="metrics: metric alias 'reward' collides with the primary metric"):
        _instantiate_metrics([MetricEntry.model_validate({"class": "exact_match", "alias": "reward"})], context="metrics")


def test_an_unknown_reward_is_left_for_the_builder_to_refuse():
    from openarena.evaluate import _scope_to_answer

    assert _scope_to_answer({"name": "nope"}, ["answer"]) == {"name": "nope"}
    assert _scope_to_answer({"name": "harbor_verifier"}, ["answer"]) == {"name": "harbor_verifier"}  # self-scored: reads no target
    assert _scope_to_answer({"name": "exact_match"}, []) == {"name": "exact_match"}  # no target fields known
