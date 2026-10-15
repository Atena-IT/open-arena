# License Apache 2.0: (c) 2026 Atena-Reply

"""What a run does when a call, a row or a cell cannot be answered: a row scores 0, a cell fails alone, the run says so."""

import asyncio
import json

import yaml
from click.testing import CliRunner

from openarena.cli.main import cli, init
from openarena.programs.a2a_program import A2aAgent

_SCHEMAS = {
    "input_template": '{"question": {{ question | tojson }}}',
    "output_template": '{"answer": {{ answer | tojson }}}',
    "input_schema": {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]},
    "output_schema": {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]},
}


def test_a_row_the_a2a_agent_does_not_answer_predicts_nothing():
    class Input:
        def get_json(self):
            return {"question": "2 + 2?"}

    agent = A2aAgent(url="http://agent.invalid", schema=_SCHEMAS["output_schema"], timeout_sec=5)

    async def times_out(text):
        raise TimeoutError

    agent.ask = times_out
    assert asyncio.run(agent.call(Input())) is None
    assert agent.last_failure == "no answer within 5s"


def _workspace(tmp_path):
    (tmp_path / "rows.jsonl").write_text(json.dumps({"question": "2 + 2?", "answer": "4"}) + "\n")
    registries = {
        # openai without a key fails every call locally: no network, no bill.
        "models": {"nokey": {"model": "openai/gpt-4o-mini"}},
        "rewards": {"exact": {"type": "exact_match"}},
        "mcp_servers": {"ghost": {"transport": "stdio", "command": "not-a-command-on-any-path", "args": []}},
        "datasets": {
            "qa": {"type": "local", "config": {"path": str(tmp_path / "rows.jsonl"), **_SCHEMAS}, "reward": "exact"},
            "tooled": {
                "type": "local",
                "config": {"path": str(tmp_path / "rows.jsonl"), **_SCHEMAS},
                "agent": {"mcp_servers": ["ghost"]},
                "reward": "exact",
            },
        },
        "evaluations": {"e": {"language_models": ["nokey"], "datasets": ["qa", "tooled"]}},
    }
    configs = tmp_path / "configs"
    configs.mkdir()
    for section, entries in registries.items():
        (configs / f"{section}.yaml").write_text(yaml.safe_dump({section: entries}))
    return configs


def test_cells_that_cannot_answer_fail_alone_and_the_run_exits_1(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    configs, state = _workspace(tmp_path), tmp_path / "state"
    run = CliRunner().invoke(cli, ["evaluations", "run", "e", "--configs", str(configs), "--state-dir", str(state)])
    said = " ".join(run.output.split())
    assert run.exit_code == 1, run.output
    # Every call failed: the cell fails with the provider's error, not a 0 a resume would keep.
    assert "no row got an answer" in said and "OPENAI_API_KEY" in said
    # A server that cannot start fails its own cell; the run went on and wrote the other.
    assert "could not load the MCP tools of ['ghost']" in said
    assert "2 cell(s) failed" in said
    rows = json.loads((state / "e" / "result.json").read_text())["rows"]
    assert {(r["dataset"], r["status"], r["value"]) for r in rows} == {("qa", "FAILED", None), ("tooled", "FAILED", None)}


def test_a_run_whose_cells_all_scored_exits_0(tmp_path, monkeypatch):
    import openarena.evaluate as evaluate

    async def fake_sweep(config_path, **kwargs):
        return {
            "meta": {"models": ["qwen38"], "datasets": ["gsm8k-20"], "metrics": ["reward"]},
            "rows": [{"model": "qwen38", "dataset": "gsm8k-20", "metric": "reward", "value": 0.0, "status": "COMPLETED"}],
        }

    monkeypatch.setattr(evaluate, "run_sweep", fake_sweep)
    init(tmp_path)
    run = CliRunner().invoke(cli, ["evaluations", "run", "smoke", "--configs", str(tmp_path / "configs"), "--state-dir", str(tmp_path / "state")])
    assert run.exit_code == 0, run.output


def test_a_cell_is_judged_on_its_rows_answers_not_on_its_model_calls():
    import keras_tuner
    import pytest
    from synalinks.src.backend import JsonDataModel

    from openarena.evaluate import _NoAnswerGuard

    class Program:
        """An agent's program, built: its model calls (tool requests) went through, its final answers did not."""

        _compile_metrics = None

        def _flatten_modules(self):
            return []

        async def compute_metrics(self, x, y, y_pred):
            return {}

    row = JsonDataModel(json={"question": "2 + 2?"}, schema=_SCHEMAS["input_schema"])
    program = Program()
    guard = _NoAnswerGuard(program)
    asyncio.run(program.compute_metrics([row, row], [None, None], [None, None]))
    with pytest.raises(keras_tuner.errors.FailedTrialError, match=r"all 2 row\(s\) failed"):
        guard.on_test_end()

    answered = Program()
    guard = _NoAnswerGuard(answered)
    asyncio.run(answered.compute_metrics([row, row], [None, None], [None, JsonDataModel(json={"answer": "4"}, schema=_SCHEMAS["output_schema"])]))
    guard.on_test_end()  # one row answered: the cell keeps its score
