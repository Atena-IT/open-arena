# License Apache 2.0: (c) 2026 Atena-Reply

"""What a sweep holds while it runs: one cell's worth, not the grid's.

A sweep over many models is run on a faked model, with a judge that holds a
sandbox (`deep_agent_as_judge`): each cell's program, its reward and the
judge's sandbox are built as the cell starts and closed as it ends. Read at
every cell: at most that one cell's sandbox is open, and the process's
resident memory stays within a small bound however many cells ran. Once the
sweep returns, cancelled or not, it has left no sandbox, thread or file
descriptor behind.
"""

import asyncio
import gc
import json
import os
import sys
import threading

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import pytest
import synalinks
import yaml
from synalinks.src.backend import JsonDataModel
from synalinks.src.sandboxes.sandbox import Sandbox
from synalinks.src.utils.confinement_utils import confinement_available

from openarena.evaluate import SweepCancelled, run_sweep

_CONFINES, _WHY_NOT = confinement_available()
pytestmark = [
    pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc/self"),
    # The judge's sandbox confines its code, and refuses to start where it cannot (a container, a runner
    # with user namespaces restricted): the CI workflow enables them, so these run there.
    pytest.mark.skipif(not _CONFINES, reason=f"sandbox confinement unavailable: {_WHY_NOT}"),
]

MODELS = 12
# Resident memory the sweep may add from its first cell's end to its last.
RSS_BOUND_MB = 64
Q_SCHEMA = {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]}
A_SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}
ROWS = [{"question": f"{n} + {n}?", "answer": str(2 * n)} for n in range(3)]


def _rss_mb():
    with open("/proc/self/status") as status:
        return next(int(line.split()[1]) / 1024 for line in status if line.startswith("VmRSS:"))


def _fds():
    return len(os.listdir("/proc/self/fd"))


def _filled(schema):
    """A value of `schema`'s shape: what the faked model answers, for the program and the judge alike."""
    kind = schema.get("type")
    if "enum" in schema:
        return schema["enum"][0]
    if kind == "object" or "properties" in schema:
        props = schema.get("properties") or {}
        if "role" in props and "content" in props:  # a chat message: the agent's last turn
            return {"role": "assistant", "content": "done"}
        return {name: _filled(sub) for name, sub in props.items()}
    if kind == "array":
        return []
    if kind == "number":
        return max(schema.get("minimum", 0.0), min(schema.get("maximum", 1.0), 1.0))
    if kind == "integer":
        return int(schema.get("minimum", 1))
    if kind == "boolean":
        return True
    if "anyOf" in schema:
        return _filled(schema["anyOf"][0])
    return "done"


def _fake_model(monkeypatch):
    async def fake_call(self, messages, schema=None, **kwargs):
        if schema is None:
            return JsonDataModel(json={"role": "assistant", "content": "done"}, schema=synalinks.ChatMessage.get_schema())
        return JsonDataModel(json=_filled(schema), schema=schema)

    monkeypatch.setattr(synalinks.LanguageModel, "__call__", fake_call)


def _config(tmp_path, models=MODELS):
    tmp_path.mkdir(parents=True, exist_ok=True)
    rows = tmp_path / "rows.jsonl"
    rows.write_text("\n".join(json.dumps(r) for r in ROWS) + "\n")
    models = {f"m{i}": {"model": "openai/gpt-4o-mini", "params": {"temperature": i / models}} for i in range(models)}
    config = {
        "models": {**models, "judge": {"model": "openai/gpt-4o"}},
        "rewards": {"judged": {"type": "deep_agent_as_judge", "config": {"language_model": "judge"}}},
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
                "reward": "judged",
            }
        },
        "evaluation": {"name": "e", "language_models": list(models), "datasets": ["qa"], "batch_size": 3},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return str(path)


def _baseline(tmp_path):
    """What the process holds after a one-cell sweep: what a first sweep starts once (a shared worker pool) is not a leak."""
    asyncio.run(run_sweep(_config(tmp_path / "warm-up", models=1), state_dir=tmp_path / "warm-up" / "state"))
    gc.collect()
    return set(Sandbox.live_sandboxes), threading.active_count(), _fds()


def _left_nothing_behind(baseline):
    gc.collect()
    live, threads, fds = baseline
    assert set(Sandbox.live_sandboxes) - live == set()
    assert threading.active_count() <= threads
    assert _fds() <= fds


def test_a_sweep_holds_one_cells_sandbox_at_a_time_and_its_memory_stays_flat(tmp_path, monkeypatch):
    _fake_model(monkeypatch)
    baseline = _baseline(tmp_path)
    open_at_start, rss_at_end = [], []

    def watch(event):
        if event["event"] == "cell_start":
            open_at_start.append(len(set(Sandbox.live_sandboxes) - baseline[0]))
        elif event["event"] == "cell_end":
            rss_at_end.append(_rss_mb())

    result = asyncio.run(run_sweep(_config(tmp_path), state_dir=tmp_path / "state", on_event=watch))

    statuses = {r["model"]: r["status"] for r in result["rows"]}
    assert len(statuses) == MODELS and set(statuses.values()) == {"COMPLETED"}, statuses
    # The cell's own judge sandbox, never the grid's: built as the cell starts, closed as it ends.
    assert len(open_at_start) == MODELS and max(open_at_start) <= 1, open_at_start
    grown = max(rss_at_end) - rss_at_end[0]
    assert grown < RSS_BOUND_MB, f"resident memory grew {grown:.0f} MB over {MODELS} cells"
    _left_nothing_behind(baseline)


def test_a_cancelled_sweep_leaves_nothing_open(tmp_path, monkeypatch):
    _fake_model(monkeypatch)
    baseline = _baseline(tmp_path)
    started = []

    def cancel(event):
        if event["event"] == "cell_start":
            started.append(event["model"])
            if len(started) == 3:
                raise SweepCancelled("stop")

    with pytest.raises(SweepCancelled):
        asyncio.run(run_sweep(_config(tmp_path), state_dir=tmp_path / "state", on_event=cancel))
    # The cancelled cell's program, built and never run, is closed with the rest.
    _left_nothing_behind(baseline)
