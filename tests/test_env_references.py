# License Apache 2.0: (c) 2026 Atena-Reply

"""`${NAME}` in a config: the environment variable when a run reads it, a warning from `check` when it is set nowhere."""

import pytest
import yaml
from click.testing import CliRunner

from openarena.cli.main import cli, init
from openarena.config import Config
from openarena.config.check import check_folder
from openarena.config.files import expand, references

_AGENT = {"a2a_agents": {"bot": {"url": "${BOT_URL}", "headers": {"Authorization": "Bearer ${BOT_TOKEN}"}}}}


def test_a_reference_is_replaced_and_an_escaped_one_kept():
    value = {"h": ["Bearer ${T}", "cost: $${T}"], "t": "{{ question }}", "n": 3}
    assert expand(value, {"T": "s3cret"}) == {"h": ["Bearer s3cret", "cost: ${T}"], "t": "{{ question }}", "n": 3}
    assert references(value) == [("h[0]", "T")]


def test_a_variable_set_nowhere_is_refused_with_where_it_is_used():
    with pytest.raises(ValueError, match=r"BOT_TOKEN \(at a2a_agents\.bot\.headers\.Authorization\)"):
        expand(_AGENT, {"BOT_URL": "https://bot.example.com"})
    kept = expand(_AGENT, {}, keep_unset=True)
    assert kept["a2a_agents"]["bot"]["headers"]["Authorization"] == "Bearer ${BOT_TOKEN}"


def test_a_run_reads_the_values_and_its_snapshot_keeps_the_references(tmp_path, monkeypatch):
    import openarena.evaluate as evaluate

    seen = {}

    async def fake_sweep(config_path, **kwargs):
        seen["config"] = Config.load(config_path)
        return {"meta": {"models": [], "datasets": [], "metrics": ["reward"]}, "rows": []}

    monkeypatch.setattr(evaluate, "run_sweep", fake_sweep)
    monkeypatch.delenv("ARENA_TEST_TEMPERATURE", raising=False)
    init(tmp_path)
    models = yaml.safe_load((tmp_path / "configs" / "models.yaml").read_text())
    name = next(iter(models["models"]))
    models["models"][name]["description"] = "judged at ${ARENA_TEST_TEMPERATURE}"
    (tmp_path / "configs" / "models.yaml").write_text(yaml.safe_dump(models))
    args = ["evaluations", "run", "smoke", "--configs", str(tmp_path / "configs"), "--state-dir", str(tmp_path / "state")]

    refused = CliRunner().invoke(cli, args)
    assert refused.exit_code == 1 and "ARENA_TEST_TEMPERATURE" in refused.output

    (tmp_path / ".env").write_text("ARENA_TEST_TEMPERATURE=0.0\n")
    ran = CliRunner().invoke(cli, args)
    monkeypatch.delenv("ARENA_TEST_TEMPERATURE", raising=False)  # set by the run's `.env`
    assert ran.exit_code == 0, ran.output
    assert seen["config"].models[name].description == "judged at 0.0"
    assert "${ARENA_TEST_TEMPERATURE}" in (tmp_path / "state" / "smoke" / "config.yaml").read_text()


def test_check_warns_about_a_variable_set_nowhere_and_reads_the_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("BOT_URL", raising=False)
    monkeypatch.delenv("BOT_TOKEN", raising=False)
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "a2a_agents.yaml").write_text(yaml.safe_dump(_AGENT))

    def problems():
        return [(p.level, p.where) for r in check_folder(configs) for p in r.problems]

    # Unset, the reference is a warning, and the url's own rule is not applied to `${BOT_URL}`'s text.
    assert sorted(problems()) == [
        ("warning", "a2a_agents.bot.headers.Authorization"),
        ("warning", "a2a_agents.bot.url"),
    ]
    (tmp_path / ".env").write_text("BOT_URL=https://bot.example.com\nBOT_TOKEN=t\n")
    assert problems() == []
    (tmp_path / ".env").write_text("BOT_URL=bot.example.com\nBOT_TOKEN=t\n")
    assert [level for level, where in problems() if where.startswith("a2a_agents.bot.url")] == ["error"]


def test_an_unset_value_is_not_judged_by_its_entry_rules_either(tmp_path, monkeypatch):
    # A streamable-HTTP server checks its url's scheme on the entry, not the field: an unset `${MCP_URL}` must not trip it.
    monkeypatch.delenv("MCP_URL", raising=False)
    configs = tmp_path / "configs"
    configs.mkdir()
    servers = {"mcp_servers": {"docs": {"transport": "streamable_http", "url": "${MCP_URL}"}}}
    (configs / "mcp_servers.yaml").write_text(yaml.safe_dump(servers))
    assert [(p.level, p.where) for r in check_folder(configs) for p in r.problems] == [("warning", "mcp_servers.docs.url")]
    # An error of its own on the same entry still shows.
    servers["mcp_servers"]["docs"]["timeout"] = 5
    (configs / "mcp_servers.yaml").write_text(yaml.safe_dump(servers))
    assert ("error", "mcp_servers.docs.timeout") in [(p.level, p.where) for r in check_folder(configs) for p in r.problems]


def test_a_task_folder_named_by_a_variable_is_copied_into_the_run(tmp_path, monkeypatch):
    import openarena.evaluate as evaluate

    async def fake_sweep(config_path, **kwargs):
        return {"meta": {"models": [], "datasets": [], "metrics": ["reward"]}, "rows": []}

    monkeypatch.setattr(evaluate, "run_sweep", fake_sweep)
    monkeypatch.delenv("TASK_DIR", raising=False)
    init(tmp_path)
    path = tmp_path / "configs" / "datasets.yaml"
    datasets = yaml.safe_load(path.read_text())
    [name] = [n for n, e in datasets["datasets"].items() if e.get("type") == "harbor_tasks"]
    datasets["datasets"][name]["config"]["path"] = "${TASK_DIR}"
    path.write_text(yaml.safe_dump(datasets))
    (tmp_path / ".env").write_text("TASK_DIR=tasks\n")
    state = tmp_path / "state"
    ran = CliRunner().invoke(cli, ["evaluations", "run", "harbor", "--configs", str(tmp_path / "configs"), "--state-dir", str(state)])
    monkeypatch.delenv("TASK_DIR", raising=False)  # set by the run's `.env`
    assert ran.exit_code == 0, ran.output
    assert (state / "harbor" / "tasks" / "hello-world" / "task.toml").is_file()
    assert "${TASK_DIR}" in (state / "harbor" / "config.yaml").read_text()
