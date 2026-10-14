# License Apache 2.0: (c) 2026 Atena-Reply

"""`openarena init`, `evaluations run` and `leaderboards run`: the templates it writes are a config the engine accepts, and the commands read them."""

import json
import os
import re

import pytest
import yaml
from click.testing import CliRunner

from openarena.cli.main import cli, evaluation_config, init, templates
from openarena.config import Config


def _read(dest, section):
    return yaml.safe_load((dest / "configs" / f"{section}.yaml").read_text())[section]


@pytest.mark.parametrize("template", templates())
def test_every_saved_evaluation_validates(tmp_path, template):
    init(tmp_path, template=template)
    registries = {s: _read(tmp_path, s) for s in ("models", "rewards", "datasets", "mcp_servers", "a2a_agents", "environments")}
    for name, evaluation in _read(tmp_path, "evaluations").items():
        Config.model_validate({"evaluation": {"name": name, **evaluation}, **registries})


def test_refuses_to_overwrite_without_force(tmp_path):
    init(tmp_path)
    (tmp_path / "configs" / "models.yaml").write_text("models: {}\n")
    with pytest.raises(FileExistsError, match="configs/models.yaml"):
        init(tmp_path)
    assert (tmp_path / "configs" / "models.yaml").read_text() == "models: {}\n"
    init(tmp_path, force=True)
    assert "qwen38" in _read(tmp_path, "models")


def test_cli(tmp_path):
    runner = CliRunner()
    dest = str(tmp_path / "arena")
    ok = runner.invoke(cli, ["init", dest])
    assert ok.exit_code == 0, ok.output
    assert "gsm8k-20" in ok.output
    again = runner.invoke(cli, ["init", dest])
    assert again.exit_code == 1 and "--force" in again.output
    assert runner.invoke(cli, ["init", dest, "--force"]).exit_code == 0
    assert runner.invoke(cli, ["init", str(tmp_path / "other"), "--template", "nope"]).exit_code == 2


def test_init_writes_a_workspace(tmp_path):
    """configs/, a Harbor task the `harbor_tasks` dataset runs, and the build context of the environment it names."""
    init(tmp_path)
    task = tmp_path / "tasks" / "hello-world"
    for path in ("task.toml", "instruction.md", "environment/Dockerfile", "tests/test.sh", "solution/solve.sh"):
        assert (task / path).is_file(), path
    assert os.access(task / "tests" / "test.sh", os.X_OK) and os.access(task / "solution" / "solve.sh", os.X_OK)
    assert (tmp_path / "environments" / "python" / "environment" / "requirements.txt").is_file()
    assert _read(tmp_path, "datasets")["hello-tasks"]["config"]["path"] == "tasks"
    assert "python" in _read(tmp_path, "environments")
    # What is optional is there to read, commented: an empty registry still says how to fill it.
    assert _read(tmp_path, "mcp_servers") == {} and "# fetch:" in (tmp_path / "configs" / "mcp_servers.yaml").read_text()
    assert _read(tmp_path, "a2a_agents") == {}


def test_a_harbor_tasks_run_takes_the_task_folders(tmp_path, monkeypatch):
    import openarena.evaluate as evaluate

    async def fake_sweep(config_path, **kwargs):
        return {"meta": {"models": ["gpt-4o-mini"], "datasets": [], "metrics": []}, "rows": []}

    monkeypatch.setattr(evaluate, "run_sweep", fake_sweep)
    init(tmp_path)
    state = tmp_path / "state"
    ok = CliRunner().invoke(cli, ["evaluations", "run", "harbor", "--configs", str(tmp_path / "configs"), "--state-dir", str(state)])
    assert ok.exit_code == 0, ok.output
    assert (state / "harbor" / "tasks" / "hello-world" / "task.toml").is_file()
    assert (state / "harbor" / "environments" / "python" / "environment" / "requirements.txt").is_file()


def test_evaluation_config_takes_what_the_evaluation_names(tmp_path):
    init(tmp_path)
    raw = evaluation_config(tmp_path / "configs", "smoke")
    assert raw["evaluation"]["name"] == "smoke"
    assert list(raw["datasets"]) == ["gsm8k-20"]
    assert list(raw["rewards"]) == ["exact-match"]
    Config.model_validate(raw)
    with pytest.raises(KeyError, match="smoke"):
        evaluation_config(tmp_path / "configs", "nope")


def test_run_by_name(tmp_path, monkeypatch):
    import openarena.evaluate as evaluate

    seen = {}

    async def fake_sweep(config_path, **kwargs):
        seen["config"] = Config.load(config_path)
        seen.update(kwargs)
        return {
            "meta": {"models": ["qwen38"], "datasets": ["gsm8k-20"], "metrics": ["reward"]},
            "rows": [{"model": "qwen38", "dataset": "gsm8k-20", "metric": "reward", "value": 0.75}],
        }

    monkeypatch.setattr(evaluate, "run_sweep", fake_sweep)
    configs, state = tmp_path / "configs", tmp_path / "state"
    init(tmp_path)
    runner = CliRunner()
    ok = runner.invoke(cli, ["evaluations", "run", "smoke", "--configs", str(configs), "--state-dir", str(state)])
    assert ok.exit_code == 0, ok.output
    assert "0.7500" in ok.output
    assert seen["config"].evaluation.name == "smoke"
    assert seen["state_dir"] == state / "smoke"
    assert (state / "smoke" / "config.yaml").is_file()
    assert json.loads((state / "smoke" / "result.json").read_text())["rows"][0]["value"] == 0.75

    unknown = runner.invoke(cli, ["evaluations", "run", "nope", "--configs", str(configs)])
    assert unknown.exit_code == 1 and "smoke" in unknown.output


def test_a_harbor_run_takes_its_environments_folder(tmp_path, monkeypatch):
    import openarena.evaluate as evaluate

    async def fake_sweep(config_path, **kwargs):
        return {"meta": {"models": ["m"], "datasets": ["tasks"], "metrics": ["reward"]}, "rows": []}

    monkeypatch.setattr(evaluate, "run_sweep", fake_sweep)
    configs, state = tmp_path / "configs", tmp_path / "state"
    configs.mkdir()
    registries = {
        "models": {"m": {"model": "ollama/qwen3"}},
        "rewards": {"judge": {"type": "lm_as_judge", "config": {"language_model": "m"}}},
        "environments": {"py": {"docker_image": "python:3.12-slim"}},
        "datasets": {"tasks": {"type": "local", "config": {"path": "x.jsonl", "input_template": '{"question": {{ q | tojson }}}'}, "harbor": {"environment": "py"}, "reward": "judge"}},
        "evaluations": {"h": {"language_models": ["m"], "datasets": ["tasks"]}},
    }
    for section, entries in registries.items():
        (configs / f"{section}.yaml").write_text(yaml.safe_dump({section: entries}))
    context = tmp_path / "environments" / "py" / "environment"
    context.mkdir(parents=True)
    (context / "requirements.txt").write_text("numpy\n")

    ok = CliRunner().invoke(cli, ["evaluations", "run", "h", "--configs", str(configs), "--state-dir", str(state)])
    assert ok.exit_code == 0, ok.output
    assert (state / "h" / "environments" / "py" / "environment" / "requirements.txt").read_text() == "numpy\n"


def _result(cells):
    """A `run_sweep` result: `cells` is `{(model, dataset): {metric: value}}`."""
    rows = [
        {"model": m, "dataset": d, "metric": k, "value": v, "direction": "min" if k == "cost" else "max"}
        for (m, d), metrics in cells.items()
        for k, v in metrics.items()
    ]
    return {"meta": {}, "rows": rows}


def _save_run(state, name, cells, models, mtime):
    run = state / name
    run.mkdir(parents=True)
    (run / "config.yaml").write_text(yaml.safe_dump({"models": {m: {"model": f"ollama/{m}"} for m in models}}))
    (run / "result.json").write_text(json.dumps(_result(cells)))
    os.utime(run / "result.json", (mtime, mtime))


def test_leaderboards_run(tmp_path):
    configs, state = tmp_path / "configs", tmp_path / "state"
    init(tmp_path)
    # An older run of `a` on gsm8k-20 is replaced by the newer one.
    _save_run(state, "old", {("a", "gsm8k-20"): {"reward": 0.1}}, ["a"], 1_000)
    _save_run(
        state,
        "smoke",
        {
            ("a", "gsm8k-20"): {"reward": 0.9},
            ("b", "gsm8k-20"): {"reward": 0.6},
            ("c", "other"): {"reward": 1.0},
        },
        ["a", "b", "c"],
        2_000,
    )
    runner = CliRunner()
    args = ["--configs", str(configs), "--state-dir", str(state)]
    out = tmp_path / "math.json"
    ok = runner.invoke(cli, ["leaderboards", "run", "math", *args, "--no-eol", "--json", str(out)])
    assert ok.exit_code == 0, ok.output
    ranking = json.loads(out.read_text())
    assert [r["model"] for r in ranking["rows"]] == ["a", "b"]
    assert ranking["rows"][0]["values"]["reward"] == 0.9
    assert ranking["rows"][0]["score"] == 1.0 and ranking["rows"][1]["score"] == pytest.approx(0.05)
    assert ranking["shares"] == {"reward": 100.0}

    # `overall` spans both datasets: nobody is scored on both, so nobody ranks, and each says why.
    ok = runner.invoke(cli, ["leaderboards", "run", "overall", *args, "--no-eol"])
    assert ok.exit_code == 0, ok.output
    assert "not scored on other" in ok.output and "not scored on gsm8k-20" in ok.output

    listed = runner.invoke(cli, ["leaderboards", "list", "--configs", str(configs)])
    assert "overall" in listed.output and "math" in listed.output
    assert "all" in listed.output  # `overall` scopes nothing: every dataset and model
    unknown = runner.invoke(cli, ["leaderboards", "run", "nope", *args])
    assert unknown.exit_code == 1 and "overall" in unknown.output
    empty = runner.invoke(cli, ["leaderboards", "run", "math", "--configs", str(configs), "--state-dir", str(tmp_path / "none")])
    assert empty.exit_code == 1 and "evaluations run" in empty.output


def test_values_read_like_the_results_table():
    from openarena.cli.main import _fmt

    assert [_fmt(v) for v in (0.0, 1.0, 0.125, 365.0, 12345.0, 2.5)] == ["0.0000", "1.0000", "0.1250", "365", "12,345", "2.5000"]



def test_a_failed_cell_says_why(tmp_path, monkeypatch):
    import openarena.evaluate as evaluate

    async def fake_sweep(config_path, *, on_event, **kwargs):
        on_event({"event": "dataset_end", "dataset": "gsm8k-20", "errors": {"qwen38": "AuthenticationError: no OPENAI_API_KEY"}})
        return {"meta": {"models": ["qwen38"], "datasets": ["gsm8k-20"], "metrics": ["reward"]}, "rows": []}

    monkeypatch.setattr(evaluate, "run_sweep", fake_sweep)
    init(tmp_path)
    ok = CliRunner().invoke(cli, ["evaluations", "run", "smoke", "--configs", str(tmp_path / "configs"), "--state-dir", str(tmp_path / "state")])
    assert ok.exit_code == 0, ok.output
    assert "qwen38 failed: AuthenticationError: no OPENAI_API_KEY" in " ".join(ok.output.split())


def test_a_model_over_budget_is_named(tmp_path):
    configs, state = tmp_path / "configs", tmp_path / "state"
    init(tmp_path)
    (configs / "leaderboards.yaml").write_text(yaml.safe_dump({"leaderboards": {"cheap": {
        "title": "Cheap", "datasets": ["gsm8k-20"], "budgets": [{"metric": "cost", "op": "lte", "value": 0.05}],
    }}}))
    _save_run(state, "smoke", {("a", "gsm8k-20"): {"reward": 0.9, "cost": 0.2}, ("b", "gsm8k-20"): {"reward": 0.5, "cost": 0.01}}, ["a", "b"], 1_000)
    ok = CliRunner().invoke(cli, ["leaderboards", "run", "cheap", "--configs", str(configs), "--state-dir", str(state), "--no-eol"])
    assert ok.exit_code == 0, ok.output
    assert "Over budget a: cost at most 0.05" in ok.output


def test_init_writes_the_agent_guide_and_skills(tmp_path):
    """Ready for a coding agent: AGENTS.md, CLAUDE.md pointing at it, and the skills in `.agents/skills/`."""
    from openarena.cli.main import SKILLS

    init(tmp_path)
    assert (tmp_path / "CLAUDE.md").is_symlink() and os.readlink(tmp_path / "CLAUDE.md") == "AGENTS.md"
    assert (tmp_path / "CLAUDE.md").read_text() == (tmp_path / "AGENTS.md").read_text()
    for path in (".env.template", ".gitignore", "README.md"):
        assert (tmp_path / path).is_file(), path
    copied = sorted(p.name for p in (tmp_path / ".agents" / "skills").iterdir())
    assert copied == sorted(p.name for p in SKILLS.glob("openarena-*")) and len(copied) >= 12
    for name in copied:
        assert (tmp_path / ".agents" / "skills" / name / "SKILL.md").read_text() == (SKILLS / name / "SKILL.md").read_text()
    # Every skill AGENTS.md lists is one the workspace has.
    listed = set(re.findall(r"`(openarena-[a-z0-9-]+)`", (tmp_path / "AGENTS.md").read_text()))
    assert listed == set(copied)


def test_a_run_reads_the_workspace_env(tmp_path, monkeypatch):
    import openarena.evaluate as evaluate

    seen = {}

    async def fake_sweep(config_path, **kwargs):
        seen["key"] = os.environ.get("ARENA_TEST_KEY")
        seen["kept"] = os.environ.get("ARENA_TEST_KEPT")
        return {"meta": {"models": [], "datasets": [], "metrics": []}, "rows": []}

    monkeypatch.setattr(evaluate, "run_sweep", fake_sweep)
    monkeypatch.delenv("ARENA_TEST_KEY", raising=False)
    monkeypatch.setenv("ARENA_TEST_KEPT", "shell")
    init(tmp_path)
    (tmp_path / ".env").write_text("ARENA_TEST_KEY=from-env-file\nARENA_TEST_KEPT=file\n")
    ok = CliRunner().invoke(cli, ["evaluations", "run", "smoke", "--configs", str(tmp_path / "configs"), "--state-dir", str(tmp_path / "state")])
    assert ok.exit_code == 0, ok.output
    assert seen == {"key": "from-env-file", "kept": "shell"}  # never over what the shell set
    monkeypatch.delenv("ARENA_TEST_KEY", raising=False)


def test_init_leaves_a_claude_md_already_there_alone(tmp_path):
    """Whatever is at CLAUDE.md, a file or a link leading nowhere, is in the way: nothing is written without --force."""
    (tmp_path / "CLAUDE.md").write_text("my own notes\n")
    with pytest.raises(FileExistsError, match="CLAUDE.md"):
        init(tmp_path)
    assert (tmp_path / "CLAUDE.md").read_text() == "my own notes\n"
    assert not (tmp_path / "AGENTS.md").exists()

    other = tmp_path / "other"
    other.mkdir()
    (other / "CLAUDE.md").symlink_to("nowhere.md")
    with pytest.raises(FileExistsError, match="CLAUDE.md"):
        init(other)
    assert os.readlink(other / "CLAUDE.md") == "nowhere.md"
