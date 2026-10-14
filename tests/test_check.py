# License Apache 2.0: (c) 2026 Atena-Reply

"""`openarena <registry> check FILE`: every problem of a registry file, where it is and what is wrong."""

import pytest
import yaml
from click.testing import CliRunner

from openarena.config import KIND_BLOCKS
from openarena.config.check import check_file, check_folder
from openarena.cli.main import cli, init


def _write(folder, section, entries):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{section}.yaml"
    path.write_text(yaml.safe_dump({section: entries}, sort_keys=False))
    return path


def _problems(report):
    return {(p.level, p.where) for p in report.problems}


def _message(report, where):
    return next(p.message for p in report.problems if p.where == where)


def _fix(report, where):
    return next(p.fix for p in report.problems if p.where == where)


@pytest.fixture
def configs(tmp_path):
    """The default template, which is correct as written."""
    init(tmp_path)
    return tmp_path / "configs"


def test_the_template_is_clean(configs):
    reports = check_folder(configs)
    assert {r.section for r in reports} == {
        "models", "rewards", "datasets", "mcp_servers", "a2a_agents", "environments", "evaluations", "leaderboards"
    }
    assert [p for r in reports for p in r.problems] == []


def test_the_file_itself(tmp_path):
    missing = check_file(tmp_path / "nope.yaml")
    assert missing.errors and "no file" in missing.errors[0].message

    twice = tmp_path / "models.yaml"
    twice.write_text("models:\n  a: {model: x/y}\n  a: {model: x/z}\n")
    report = check_file(twice)
    assert "'a' is written twice" in report.errors[0].message
    assert report.errors[0].line == 3 and "keep one 'a'" in report.errors[0].fix

    other = tmp_path / "rewards.yaml"
    other.write_text("models: {}\n")
    wrong = check_file(other).errors[0]
    assert "no `rewards:`" in wrong.message and "rewards:\n  exact-match:" in wrong.fix

    broken = tmp_path / "datasets.yaml"
    broken.write_text("datasets: [a, b\n")
    syntax = check_file(broken).errors[0]
    assert "not valid YAML" in syntax.message and syntax.line == 2 and "close every" in syntax.fix


def test_models(tmp_path):
    report = check_file(
        _write(tmp_path, "models", {"ok": {"model": "ollama/qwen3", "tags": ["local"]}, "no-id": {"temprature": 0}, "bad name": {"model": "a/b"}}))
    assert _problems(report) == {
        ("error", "models.no-id.model"),
        ("error", "models.no-id.temprature"),  # strict: a field the model does not know
        ("error", "models.bad name"),
    }


def test_rewards(configs):
    models = yaml.safe_load((configs / "models.yaml").read_text())["models"]
    models["vectors"] = {"model": "ollama/nomic-embed-text", "type": "embedding"}
    _write(configs, "models", models)
    report = check_file(
        _write(
            configs,
            "rewards",
            {
                "judge": {"type": "lm_as_judge", "config": {"language_model": "vectors", "instruction": "grade"}},
                "typo": {"type": "exact_matsh"},
                "by-id": {"type": "lm_as_judge", "config": {"language_model": "openai/gpt-4o"}},
            },
        ))
    assert _problems(report) == {
        ("error", "rewards.judge.config.language_model"),
        ("error", "rewards.judge.config.instruction"),
        ("error", "rewards.typo.type"),
        ("error", "rewards.by-id.config.language_model"),  # a judge is a models key, never a raw id
    }
    assert _fix(report, "rewards.judge.config.instruction") == "rename `instruction:` to `instructions:` under rewards.judge.config"
    assert _fix(report, "rewards.typo.type") == "change 'exact_matsh' to 'exact_match'"
    assert "an embedding model" in _message(report, "rewards.judge.config.language_model")
    assert _fix(report, "rewards.judge.config.language_model").startswith("name a language model instead: ")
    assert _problems(check_file(_write(configs, "rewards", {"old": {"type": "metric_reward"}}))) == {("error", "rewards.old.type")}


def test_datasets(configs):
    ds = yaml.safe_load((configs / "datasets.yaml").read_text())["datasets"]["gsm8k-20"]
    templates = {k: v for k, v in ds["config"].items() if k.endswith(("_template", "_schema"))}
    (configs / "mcp_servers.yaml").unlink()  # no `mcp_servers` registry beside it
    report = check_file(
        _write(
            configs,
            "datasets",
            {
                "ok": ds,
                "misspelled": {**ds, "type": "local", "config": {**templates, "pth": "rows.jsonl"}},
                "no-template": {**ds, "config": {k: v for k, v in ds["config"].items() if k != "input_template"}},
                "unknown-reward": {**ds, "reward": "exact-mach"},
                "agent": {**{k: v for k, v in ds.items() if k != "generator"}, "agent": {"mcp_servers": ["sandbox"]}},
                "two-kinds": {**ds, "tools": {"tool_schemas": [{"name": "f"}]}},
                "provider": {**ds, "type": "hugingface"},
            },
        ))
    problems = _problems(report)
    # A misspelled required setting is one problem, fixed by the rename.
    assert ("error", "datasets.misspelled.config.path") not in problems
    assert _fix(report, "datasets.misspelled.config.pth") == "rename `pth:` to `path:` under datasets.misspelled.config"
    assert "input_template: '{\"question\": {{ question | tojson }}}'" in _fix(report, "datasets.no-template")
    assert _fix(report, "datasets.unknown-reward.reward") == "change 'exact-mach' to 'exact-match'"
    # No `mcp_servers` registry beside it: said once, the reference left unchecked.
    assert ("warning", "mcp_servers") in problems
    assert not any(p.where.startswith("datasets.agent") for p in report.problems)
    assert any(p.where == "datasets.two-kinds" and "mutually exclusive" in p.message for p in report.problems)
    assert _fix(report, "datasets.provider.type") == "change 'hugingface' to 'huggingface'"
    assert not any(p.where.startswith("datasets.ok") for p in report.problems)


def test_evaluations_check_every_reference_and_the_whole(configs):
    models = yaml.safe_load((configs / "models.yaml").read_text())["models"]
    models["vectors"] = {"model": "ollama/nomic-embed-text", "type": "embedding"}
    _write(configs, "models", models)
    report = check_file(
        _write(
            configs,
            "evaluations",
            {
                "broken": {"language_models": ["qwen38", "gpt-5"], "datasets": ["gsm8k-20", "missing"], "batch_size": 0},
                "wrong-type": {"language_models": ["vectors"], "datasets": ["gsm8k-20"]},
            },
        ))
    assert _problems(report) == {
        ("error", "evaluations.broken.batch_size"),
        ("error", "evaluations.broken.language_models[1]"),
        ("error", "evaluations.broken.datasets[1]"),
        # Validated whole, as a run would: an LLM sweep cannot take an embedding model.
        ("error", "evaluations.wrong-type"),
    }
    assert "embedding model" in _message(report, "evaluations.wrong-type")
    assert _fix(report, "evaluations.broken.batch_size") == "set `batch_size:` to 1 or more"
    assert "use one of the models there" in _fix(report, "evaluations.broken.language_models[1]")


def test_leaderboards_warn_on_what_scores_can_outlive(configs):
    report = check_file(
        _write(configs, "leaderboards", {"old": {"title": "Old", "datasets": ["retired"], "shares": {"reward": 0}}, "untitled": {}}))
    assert _problems(report) == {
        ("warning", "leaderboards.old.datasets[0]"),
        ("error", "leaderboards.old.shares"),
        ("error", "leaderboards.untitled.title"),
    }


def test_mcp_servers_and_a2a_agents(tmp_path):
    mcp = check_file(
        _write(
            tmp_path,
            "mcp_servers",
            {
                "ok": {"transport": "stdio", "command": "python", "args": ["server.py"]},
                "no-url": {"transport": "sse"},
                "mixed": {"transport": "stdio", "command": "python", "url": "http://x"},
                "odd": {"transport": "carrier-pigeon"},
            },
        ))
    assert _problems(mcp) == {
        ("error", "mcp_servers.no-url"),
        ("error", "mcp_servers.mixed"),
        ("error", "mcp_servers.odd.transport"),
    }
    assert _fix(mcp, "mcp_servers.no-url") == "add `url: https://<host>/mcp`"
    assert _fix(mcp, "mcp_servers.mixed").startswith("remove `url:`, or set `transport:` to one that reads them")
    a2a = check_file(
        _write(
            tmp_path,
            "a2a_agents",
            {"ok": {"url": "http://localhost:9999"}, "bad": {"headers": {"A": 1}}, "ftp": {"url": "ftp://x"}},
        ))
    assert _problems(a2a) == {("error", "a2a_agents.bad.url"), ("error", "a2a_agents.bad.headers.A"), ("error", "a2a_agents.ftp.url")}


def test_cli(configs):
    runner = CliRunner()
    ok = runner.invoke(cli, ["check", str(configs / "datasets.yaml")])
    assert ok.exit_code == 0, ok.output
    assert "3 datasets, no problems" in ok.output
    assert runner.invoke(cli, ["check", str(configs)]).exit_code == 0

    _write(configs, "rewards", {"exact-match": {"type": "exact_matsh"}})
    bad = runner.invoke(cli, ["check", str(configs / "rewards.yaml")])
    assert bad.exit_code == 1
    assert "rewards.exact-match.type" in bad.output and "line 3" in bad.output
    assert "fix" in bad.output and "change 'exact_matsh' to 'exact_match'" in bad.output
    assert runner.invoke(cli, ["check", str(configs)]).exit_code == 1

    # A warning alone does not fail the check: a leaderboard scoped to a dataset since removed.
    _write(configs, "leaderboards", {"old": {"title": "Old", "datasets": ["retired"]}})
    warned = runner.invoke(cli, ["check", str(configs / "leaderboards.yaml")])
    assert warned.exit_code == 0 and "1 warning" in warned.output

    listed = runner.invoke(cli, ["mcp-servers", "list", "--configs", str(configs)])
    assert listed.exit_code == 0


def test_models_call_options_belong_under_params(tmp_path):
    report = check_file(_write(tmp_path, "models", {"qwen": {"model": "ollama/qwen3", "temprature": 0}}))
    assert _problems(report) == {("error", "models.qwen.temprature")}
    assert _fix(report, "models.qwen.temprature") == "move it under `params:` (spelled `temperature:`):\nparams:\n  temperature: 0"


def test_every_problem_says_how_to_fix_it_and_where(configs):
    """The point of the check: each problem comes with the change to make, at a line of the file."""
    models = yaml.safe_load((configs / "models.yaml").read_text())["models"]
    _write(configs, "models", {**models, "vectors": {"model": "ollama/nomic", "type": "embedding"}, "bad": {"temprature": 0, "colour": "red"}})
    ds = yaml.safe_load((configs / "datasets.yaml").read_text())["datasets"]["gsm8k-20"]
    _write(
        configs,
        "datasets",
        {
            "a": {**ds, "type": "local", "config": {"pth": "x"}, "reward": "nope"},
            "b": {**ds, "config": {k: v for k, v in ds["config"].items() if k != "input_template"}},
            "c": {**ds, "tools": {"tool_schemas": []}},
            "d": {**ds, "a2a": {"agent": "x"}},
        },
    )
    _write(configs, "rewards", {"j": {"type": "lm_as_judge", "config": {"language_model": "vectors", "colour": 1}}, "k": {"type": "nope"}})
    _write(configs, "evaluations", {"e": {"language_models": ["gpt-5"], "datasets": ["zz"], "batch_size": 0, "seed": "x"}})
    _write(configs, "leaderboards", {"bad name": {"shares": {"reward": -1}}})
    _write(configs, "mcp_servers", {"m": {"transport": "sse", "command": "x", "env": {"A": 1}}, "n": {"transport": "pigeon"}})
    _write(configs, "a2a_agents", {"a": {"url": "x", "timeout_sec": 0}})
    reports = check_folder(configs)
    problems = [p for r in reports for p in r.problems]
    assert len(problems) > 20
    assert [p for p in problems if not p.fix] == []
    assert [p for p in problems if not p.line] == []


def test_tags_the_app_refuses_are_errors_with_the_spelling_it_takes(tmp_path):
    report = check_file(
        _write(
            tmp_path,
            "models",
            {
                "tidy": {"model": "a/b", "tags": ["Local", "local", "Open Weights", "tool_use"]},  # the app only tidies these
                "bad": {"model": "a/b", "tags": ["bad!tag", "Français", "!!!", "x" * 41]},
                "scalar": {"model": "a/b", "tags": "Open Weights"},
                "many": {"model": "a/b", "tags": [f"t{i}" for i in range(33)]},
            },
        ))
    fixes = {p.where: p.fix for p in report.problems}
    assert all(p.level == "error" for p in report.problems)
    assert not any(w.startswith("models.tidy") for w in fixes)
    assert fixes["models.bad.tags[0]"] == "write it as `bad-tag`"
    assert fixes["models.bad.tags[1]"] == "write it as `francais`"
    assert fixes["models.bad.tags[2]"].startswith("remove '!!!'")
    assert fixes["models.bad.tags[3]"] == f"write it as `{'x' * 40}`"
    assert fixes["models.scalar.tags"] == "write it as a list: `tags: [open-weights]`"
    assert fixes["models.many.tags"] == "keep at most 32; remove 1"


def test_the_app_and_the_check_share_the_tag_rule():
    """What `check` refuses is what the app's registries refuse: one rule, `openarena.config.tags`."""
    from openarena.config.tags import normalize_tag, suggest_tag

    for raw in ("bad!tag", "Français", "C#", "x" * 41):
        with pytest.raises(ValueError):
            normalize_tag(raw)
        assert normalize_tag(suggest_tag(raw)) == suggest_tag(raw)


def test_check_registry_needs_no_file():
    """The backend's entry point: a registry it holds, the others to check references against."""
    from openarena.config import check_registry

    alone = check_registry("datasets", {"d": {"type": "local", "config": {"path": "x"}, "generator": {}, "reward": "r"}})
    assert alone.path is None
    assert {p.where for p in alone.problems} == {"datasets.d"}  # no question; no registries given: no reference checked

    report = check_registry(
        "rewards",
        {"judge": {"type": "lm_as_judge", "config": {"language_model": "vectors"}}},
        registries={"models": {"vectors": {"model": "ollama/nomic", "type": "embedding"}}},
    )
    assert [(p.where, p.line) for p in report.problems] == [("rewards.judge.config.language_model", None)]

    missing = check_registry("evaluations", {"e": {"language_models": ["m"], "datasets": ["d"]}}, registries={"models": {"m": {"model": "a/b"}}})
    assert [(p.level, p.where) for p in missing.problems] == [("warning", "datasets")]
    with pytest.raises(ValueError, match="no registry"):
        check_registry("widgets", {})


def test_check_is_one_command_for_a_folder_or_a_file(configs, tmp_path):
    runner = CliRunner()
    assert "check" not in cli.commands["datasets"].commands  # no per-registry check
    ok = runner.invoke(cli, ["check", str(configs / "rewards.yaml")])
    assert ok.exit_code == 0 and "2 rewards, no problems" in ok.output
    assert runner.invoke(cli, ["check", str(tmp_path / "nope")]).exit_code == 1


def test_a_registry_is_read_only_from_its_own_file(configs, tmp_path):
    """The name says the registry: `<registry>.yaml`, holding that one key. Nothing else is read."""
    misnamed = tmp_path / "dataset.yaml"
    misnamed.write_text("datasets: {}\n")
    report = check_file(misnamed)
    assert [(p.level, p.fix) for p in report.problems] == [("error", "rename it to `datasets.yaml`")]
    assert check_file(tmp_path / "notes.yml").errors  # not even there: said so
    # A registry key in another registry's file is not read from there.
    (configs / "rewards.yaml").write_text("rewards:\n  exact-match: {type: exact_match}\nmodels:\n  extra: {model: a/b}\n")
    report = check_file(configs / "rewards.yaml")
    assert [(p.where, p.fix) for p in report.problems] == [("models", "move `models:` and its entries to models.yaml")]
    # A folder names the YAML files nothing reads.
    (configs / "my-models.yaml").write_text("models: {}\n")
    stray = [r for r in check_folder(configs) if r.path.name == "my-models.yaml"]
    assert stray and stray[0].problems[0].fix == "rename it to `models.yaml`"
    # A run reads each registry from its own file only.
    from openarena.config.files import registries

    assert "extra" not in registries(configs)["models"]


def test_an_a2a_sweep_names_a2a_agents_on_its_axis(configs):
    ds = yaml.safe_load((configs / "datasets.yaml").read_text())["datasets"]["gsm8k-20"]
    ds = {k: v for k, v in ds.items() if k not in KIND_BLOCKS}
    _write(configs, "datasets", {"tasks": {**ds, "a2a": {}}})
    _write(configs, "a2a_agents", {"helper": {"url": "http://localhost:9999"}})
    _write(configs, "evaluations", {"e": {"language_models": ["helper", "ghost"], "datasets": ["tasks"]}})
    report = check_file(configs / "evaluations.yaml")
    assert _problems(report) == {("error", "evaluations.e.language_models[1]")}
