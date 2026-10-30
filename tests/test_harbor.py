# License Apache 2.0: (c) 2026 Atena-Reply

import json
import os
import posixpath
import shutil
import subprocess
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import openarena
import pytest
import yaml
from openarena.config import ClockSpec, EnvironmentSpec
from openarena.config.check import check_file
from openarena.config.errors import Invalid
from openarena.harbor import CLOCK_FILE, PROBE_NOTES, clock_arm_command, probe_facts, probe_notes, task_clock, task_files, virtual_now
from pydantic import ValidationError



def _write(folder, section, entries):
    path = folder / f"{section}.yaml"
    path.write_text(yaml.safe_dump({section: entries}, sort_keys=False))
    return path


def test_an_environment_needs_one_image():
    with pytest.raises(ValidationError, match="needs an image"):
        EnvironmentSpec()
    with pytest.raises(ValidationError, match="not both"):
        EnvironmentSpec(docker_image="python:3.12", dockerfile="FROM python:3.12")
    assert EnvironmentSpec(docker_image="python:3.12").target == "python:3.12"
    assert EnvironmentSpec(dockerfile="# base\nFROM ubuntu:24.04\nRUN true").target == "Dockerfile (FROM ubuntu:24.04)"


def test_allowed_hosts_go_with_an_allowlist():
    with pytest.raises(ValidationError, match="lets nothing through"):
        EnvironmentSpec(docker_image="x", network_mode="allowlist")
    with pytest.raises(ValidationError, match="only read with"):
        EnvironmentSpec(docker_image="x", allowed_hosts=["pypi.org"])
    assert EnvironmentSpec(docker_image="x", network_mode="allowlist", allowed_hosts=[" pypi.org ", ""]).allowed_hosts == ["pypi.org"]


def test_a_clock_has_an_aware_start_and_a_bounded_rate():
    clock = ClockSpec(start="2026-01-05T10:00:00+01:00", rate=60)
    assert clock.start == datetime(2026, 1, 5, 9, 0, tzinfo=timezone.utc)
    assert ClockSpec(start="2026-01-05T09:00:00Z").rate == 0
    assert ClockSpec(start="2026-01-05T09:00:00Z", rate=86400).rate == 86400
    for rate in (-1, 86400.5):
        with pytest.raises(ValidationError):
            ClockSpec(start="2026-01-05T09:00:00Z", rate=rate)
    with pytest.raises(ValidationError, match="offset"):
        ClockSpec(start="2026-01-05T09:00:00")
    with pytest.raises(ValidationError):
        ClockSpec(start="2026-01-05T09:00:00Z", speed=2)


def test_a_clock_needs_a_posix_sandbox():
    with pytest.raises(ValidationError, match="Windows"):
        EnvironmentSpec(docker_image="x", os="windows", clock={"start": "2026-01-05T09:00:00Z"})
    assert EnvironmentSpec(docker_image="x").clock is None


POSIX_SH = pytest.mark.skipif(shutil.which("sh") is None, reason="the arm is a POSIX shell command")


@POSIX_SH
def test_the_arm_writes_the_clock_once(tmp_path):
    path = tmp_path / "arena" / "clock.json"
    command = clock_arm_command(ClockSpec(start="2026-01-05T09:00:00Z", rate=60), str(path))
    assert subprocess.run(["sh", "-c", command]).returncode == 0
    first = json.loads(path.read_text())
    assert first["t0"] == 1767603600 and first["rate"] == 60 and first["ahead"] == 0
    assert abs(first["anchor"] - time.time()) < 5
    assert oct(path.stat().st_mode & 0o777) == "0o644"
    time.sleep(1.1)
    assert subprocess.run(["sh", "-c", command]).returncode == 0
    assert json.loads(path.read_text()) == first  # a healthcheck retry does not move the anchor


@POSIX_SH
@pytest.mark.skipif(getattr(os, "geteuid", lambda: -1)() == 0, reason="root writes anywhere")
def test_an_arm_that_cannot_write_fails(tmp_path):
    locked = tmp_path / "locked"
    locked.mkdir(mode=0o555)
    command = clock_arm_command(ClockSpec(start="2026-01-05T09:00:00Z"), str(locked / "arena" / "clock.json"))
    assert subprocess.run(["sh", "-c", command]).returncode != 0


def test_virtual_time_follows_the_formula():
    clock = {"t0": 1000.0, "rate": 60.0, "anchor": 500.0, "ahead": 30.0}
    assert virtual_now(clock, 510.0) == 1000 + 10 * 60 + 30
    assert virtual_now({**clock, "rate": 0.0}, 9999.0) == 1030


CLOCK = {"start": "2026-01-05T09:00:00Z", "rate": 60}


def test_a_clock_arms_in_the_healthcheck_and_keeps_theirs():
    spec = EnvironmentSpec(docker_image="x", clock=CLOCK, healthcheck={"command": "false || true", "retries": 7})
    hc = tomllib.loads(task_files(spec)["task.toml"])["environment"]["healthcheck"]
    # Theirs in a subshell: an arm that failed cannot be rescued by an `||` of theirs.
    assert hc["command"] == clock_arm_command(spec.clock) + " && (false || true\n)"
    assert hc["retries"] == 7
    bare = tomllib.loads(task_files(EnvironmentSpec(docker_image="x", clock=CLOCK))["task.toml"])["environment"]["healthcheck"]
    assert bare["command"] == clock_arm_command(ClockSpec(**CLOCK)) and bare["retries"] == 3
    assert spec.healthcheck.command == "false || true"  # the entry keeps what was written


def test_no_clock_writes_the_same_task_as_before():
    spec = EnvironmentSpec(docker_image="x", healthcheck={"command": "test -d /app"})
    assert task_files(spec)["task.toml"] == (
        '[environment]\nbuild_timeout_sec = 600\ndocker_image = "x"\nos = "linux"\nnetwork_mode = "public"\n\n'
        '[environment.healthcheck]\ncommand = "test -d /app"\ninterval_sec = 5\ntimeout_sec = 30\nstart_period_sec = 0\n'
        "start_interval_sec = 5\nretries = 3\n"
    )


def test_the_check_reads_the_sandbox_clock():
    out = 'os: Debian\nclock: 1000 {"t0": 1767603600, "rate": 60, "anchor": 990, "ahead": 0}\n'
    assert probe_facts(out) == {"os": "Debian", "clock": "2026-01-05T09:10:00Z (rate 60)"}
    assert "clock" not in probe_facts("os: Debian\n")
    assert probe_facts("clock: garbage\n") == {"clock": "unreadable"}


def test_task_files_read_back_as_harbor_toml():
    spec = EnvironmentSpec(
        dockerfile="FROM python:3.12-slim\nWORKDIR /app",
        cpus=2,
        memory_mb=4096,
        network_mode="allowlist",
        allowed_hosts=["pypi.org"],
        env={"PIP_NO_CACHE": "1"},
        workdir="/app",
        healthcheck={"command": "test -d /app"},
        mcp_servers=["sandbox", "search"],
    )
    servers = {
        "sandbox": {"transport": "stdio", "command": "python", "args": ["server.py"]},
        "search": {"transport": "streamable_http", "url": "http://search:8000/mcp"},
    }
    files = task_files(spec, servers)
    assert files["environment/Dockerfile"] == "FROM python:3.12-slim\nWORKDIR /app\n"
    env = tomllib.loads(files["task.toml"])["environment"]
    assert env["cpus"] == 2 and env["memory_mb"] == 4096 and env["build_timeout_sec"] == 600
    assert env["network_mode"] == "allowlist" and env["allowed_hosts"] == ["pypi.org"]
    assert env["env"] == {"PIP_NO_CACHE": "1"} and "docker_image" not in env
    assert env["healthcheck"]["command"] == "test -d /app" and env["healthcheck"]["retries"] == 3
    assert env["mcp_servers"] == [
        {"name": "sandbox", "transport": "stdio", "command": "python", "args": ["server.py"]},
        {"name": "search", "transport": "streamable-http", "url": "http://search:8000/mcp"},
    ]
    assert "environment/Dockerfile" not in task_files(EnvironmentSpec(docker_image="python:3.12"))


def test_task_files_refuse_a_server_harbor_cannot_reach():
    spec = EnvironmentSpec(docker_image="x", mcp_servers=["ws"])
    with pytest.raises(Invalid, match="no MCP server"):
        task_files(spec, {})
    with pytest.raises(Invalid, match="over 'websocket'"):
        task_files(spec, {"ws": {"transport": "websocket", "url": "ws://x"}})


def test_check_reports_environments_and_their_servers(tmp_path):
    _write(tmp_path, "mcp_servers", {"sandbox": {"transport": "stdio", "command": "python"}})
    report = check_file(
        _write(
            tmp_path,
            "environments",
            {
                "ok": {"docker_image": "python:3.12", "mcp_servers": ["sandbox"]},
                "no-image": {"cpus": 2},
                "typo": {"docker_image": "x", "mcp_servers": ["sandbx"], "cpu": 2},
            },
        ))
    assert {(p.level, p.where) for p in report.problems} == {
        ("error", "environments.no-image"),
        ("error", "environments.typo.cpu"),
        ("error", "environments.typo.mcp_servers[0]"),
    }


def _trajectory():
    """A trajectory built from Harbor's own models, so the scorer reads the format Harbor writes."""
    from harbor.models.trajectories import Agent, Observation, ObservationResult, Step, ToolCall, Trajectory

    return Trajectory(
        session_id="s",
        agent=Agent(name="terminus-2", version="1"),
        steps=[
            Step(step_id=1, source="user", message="What is in /app/out.txt?"),
            Step(
                step_id=2, source="agent", message="Reading it.",
                tool_calls=[ToolCall(tool_call_id="c1", function_name="bash", arguments={"cmd": "cat /app/out.txt"})],
                observation=Observation(results=[ObservationResult(source_call_id="c1", content="hello")]),
            ),
            Step(step_id=3, source="agent", message="hello"),
        ],
    ).model_dump(mode="json")


def test_a_trajectory_reads_as_chat_messages_and_its_last_word(tmp_path):
    import json

    from openarena.harbor import final_answer, find_trajectory, trajectory_messages

    logs = tmp_path / "agent"
    (logs / "nested").mkdir(parents=True)
    (logs / "nested" / "trajectory.json").write_text(json.dumps(_trajectory()))
    messages = trajectory_messages(find_trajectory(logs))
    assert final_answer(messages) == "hello"
    assert messages == [
        {"role": "user", "content": "What is in /app/out.txt?"},
        {"role": "assistant", "content": "Reading it.", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "bash", "arguments": {"cmd": "cat /app/out.txt"}}},
        ]},
        {"role": "tool", "tool_call_id": "c1", "content": "hello"},
        {"role": "assistant", "content": "hello"},
    ]
    with pytest.raises(FileNotFoundError):
        find_trajectory(tmp_path / "none")


def test_an_agent_refuses_a_model_or_a_network_it_cannot_work_with():
    from openarena.harbor import agent_problem, agents

    by_name = {a["name"]: a for a in agents()}
    assert by_name["terminus-2"]["runs"] == "host" and by_name["claude-code"] == {
        "name": "claude-code", "runs": "sandbox", "provider": "anthropic", "params": ["reasoning_effort"],
    }
    public, offline = EnvironmentSpec(docker_image="x"), EnvironmentSpec(docker_image="x", network_mode="no-network")
    assert agent_problem("claude-code", "anthropic/claude-sonnet-4-5", public) is None
    assert "anthropic models only" in agent_problem("claude-code", "openai/gpt-4o", public)
    assert "no-network" in agent_problem("codex", "openai/gpt-4o", offline)
    assert "cannot reach 'ollama/qwen3'" in agent_problem("opencode", "ollama/qwen3", public)
    assert "cannot reach 'sglang/kev'" in agent_problem("opencode", "sglang/kev", public)
    # terminus-2 calls its model from the arena: any model, any network.
    assert agent_problem("terminus-2", "ollama/qwen3", offline) is None


def test_a_sandbox_provider_refuses_what_it_cannot_give():
    from openarena.harbor import environment_problems, sandbox_capabilities

    assert sandbox_capabilities("e2b") == {"installed": True, "gpus": False, "no_network": True, "allowlist": True}
    assert environment_problems(EnvironmentSpec(docker_image="x", gpus=1), "e2b") == ["e2b sandboxes have no GPUs, and the environment asks for 1"]
    assert environment_problems(EnvironmentSpec(docker_image="x", network_mode="allowlist", allowed_hosts=["pypi.org"]), "e2b") == []
    # Docker decides per host: nothing refused up front.
    assert environment_problems(EnvironmentSpec(docker_image="x", gpus=1), "docker") == []


def test_the_configured_provider_runs_with_its_options_and_the_fallback_without(monkeypatch):
    from openarena.harbor import environment_config

    monkeypatch.setenv("ARENA_HARBOR_SANDBOX", "ack")
    monkeypatch.setenv("ARENA_HARBOR_SANDBOX_OPTIONS", '{"namespace": "arena-sandboxes", "node_selector": {"pool": "sandbox"}}')
    assert environment_config("ack").kwargs == {"namespace": "arena-sandboxes", "node_selector": {"pool": "sandbox"}}
    # The local Docker a trial falls back to is another provider: those settings are not its own.
    assert environment_config("docker").kwargs == {}
    monkeypatch.delenv("ARENA_HARBOR_SANDBOX_OPTIONS")
    assert environment_config("ack").kwargs == {}


@pytest.mark.parametrize("raw", ['{"namespace": ', '["arena-sandboxes"]', '"arena-sandboxes"'])
def test_sandbox_options_are_a_json_object_or_refused(monkeypatch, raw):
    from openarena.harbor import sandbox_options

    monkeypatch.setenv("ARENA_HARBOR_SANDBOX_OPTIONS", raw)
    with pytest.raises(ValueError, match="ARENA_HARBOR_SANDBOX_OPTIONS"):
        sandbox_options()


def test_malformed_options_stop_the_run_before_any_row_whichever_provider_runs(monkeypatch):
    import asyncio

    from openarena.programs import harbor_program

    monkeypatch.setenv("ARENA_HARBOR_SANDBOX", "e2b")
    monkeypatch.setenv("ARENA_HARBOR_SANDBOX_OPTIONS", '["arena-sandboxes"]')
    # E2B has no key here, so trials would fall back to the local Docker, which gets no options:
    # the mistake is still reported, not hidden by the fallback.
    monkeypatch.setattr(harbor_program, "ready", lambda name: name == "docker")
    issue = harbor_program.sandbox_issue()
    assert issue is not None and "ARENA_HARBOR_SANDBOX_OPTIONS" in issue and "list" in issue
    # And the cell fails before its first row, as an unknown provider does.
    with pytest.raises(ValueError, match="ARENA_HARBOR_SANDBOX_OPTIONS"):
        asyncio.run(harbor_program.build_harbor("openai/gpt-4o", None, {}, None))
    monkeypatch.setenv("ARENA_HARBOR_SANDBOX_OPTIONS", '{"namespace": "arena-sandboxes"}')
    assert harbor_program.sandbox_issue() is None


def test_a_trial_hands_its_provider_the_options(monkeypatch, tmp_path):
    import asyncio

    from harbor.trial.trial import Trial
    from openarena.programs import harbor_program

    seen = {}

    class Ran:
        async def run(self):
            return "result"

    async def create(config):
        seen["config"] = config
        return Ran()

    monkeypatch.setattr(Trial, "create", staticmethod(create))
    monkeypatch.setattr(harbor_program, "sandbox", lambda: "ack")
    monkeypatch.setenv("ARENA_HARBOR_SANDBOX", "ack")
    monkeypatch.setenv("ARENA_HARBOR_SANDBOX_OPTIONS", '{"namespace": "arena-sandboxes"}')
    (tmp_path / "task").mkdir()
    result = asyncio.run(harbor_program.run_trial(tmp_path / "task", tmp_path / "trials", "t", "terminus-2", "openai/gpt-4o"))
    assert result == "result"
    assert seen["config"].environment.type == "ack"
    assert seen["config"].environment.kwargs == {"namespace": "arena-sandboxes"}


@POSIX_SH
def test_their_healthcheck_runs_after_the_arm_even_with_a_comment(tmp_path):
    folder = str(tmp_path / "arena")
    for theirs, code in (("true  # wait for the API", 0), ("false  # wait for the API", 1)):
        spec = EnvironmentSpec(docker_image="x", clock=CLOCK, healthcheck={"command": theirs})
        hc = tomllib.loads(task_files(spec)["task.toml"])["environment"]["healthcheck"]
        assert subprocess.run(["sh", "-c", hc["command"].replace(posixpath.dirname(CLOCK_FILE), folder)]).returncode == code


@POSIX_SH
def test_the_arm_replaces_a_clock_left_with_other_settings(tmp_path):
    path = tmp_path / "arena" / "clock.json"
    path.parent.mkdir()
    path.write_text('{"t0": 0, "rate": 1, "anchor": 0, "ahead": 0}\n')  # baked into the image, say
    assert subprocess.run(["sh", "-c", clock_arm_command(ClockSpec(**CLOCK), str(path))]).returncode == 0
    clock = json.loads(path.read_text())
    assert clock["t0"] == 1767603600 and clock["rate"] == 60 and abs(clock["anchor"] - time.time()) < 5


@POSIX_SH
def test_arms_started_together_all_succeed(tmp_path):
    path = tmp_path / "arena" / "clock.json"
    command = clock_arm_command(ClockSpec(**CLOCK), str(path))
    runs = [subprocess.Popen(["sh", "-c", command], stderr=subprocess.DEVNULL) for _ in range(30)]
    assert [run.wait() for run in runs] == [0] * 30
    assert json.loads(path.read_text())["t0"] == 1767603600
    assert sorted(p.name for p in path.parent.iterdir()) == ["clock.json"]


def test_the_init_template_names_the_clock():
    template = Path(openarena.__file__).parent / "cli" / "templates" / "default" / "configs" / "environments.yaml"
    assert "#     clock: {start:" in template.read_text()


def test_the_check_notes_an_unreadable_clock():
    assert any("clock" in note for note in probe_notes({"clock": "unreadable"}))
    assert probe_notes({"clock": "2026-01-05T09:00:00Z (rate 0)", "tmux": "tmux 3.4"}) == []
    assert probe_notes({"tmux": "missing"}) == [PROBE_NOTES["tmux"]]


STEPPED = '''
[metadata.arena.clock]
start = "2026-03-02T08:00:00Z"
rate = 0
[metadata.arena.clock.steps]
day2 = "2026-03-03T08:00:00Z"
day3 = "2026-03-04T08:00:00Z"
[[steps]]
name = "day1"
[[steps]]
name = "day2"
[[steps]]
name = "day3"
'''


def _task(folder, toml_text):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "task.toml").write_text(toml_text)
    return folder


def test_a_task_declares_a_clock_and_its_step_times(tmp_path):
    clock = task_clock(_task(tmp_path / "week", STEPPED))
    assert clock.rate == 0 and clock.steps["day3"] == datetime(2026, 3, 4, 8, tzinfo=timezone.utc)
    assert task_clock(_task(tmp_path / "plain", "[agent]\ntimeout_sec = 60\n")) is None


@pytest.mark.parametrize(("change", "says"), [
    (('day2 = "2026-03-03T08:00:00Z"', 'day9 = "2026-03-03T08:00:00Z"'), "day9"),  # not a step
    (('day3 = "2026-03-04T08:00:00Z"', 'day3 = "2026-03-02T09:00:00Z"'), "day3"),  # before day2
    (('day2 = "2026-03-03T08:00:00Z"', 'day2 = "2026-03-01T08:00:00Z"'), "start"),  # before start
    (("rate = 0", "rate = 60"), "rate"),  # a running clock with steps
    (("rate = 0", "rate = 0\nspeed = 2"), "speed"),  # an unknown key
    (('day2 = "2026-03-03T08:00:00Z"', 'day2 = "2026-03-03T08:00:00"'), "offset"),  # no offset
    (('name = "day2"', 'name = "day2"\n[steps.agent]\nuser = "agent"'), "user"),  # a non-root agent
])
def test_a_wrong_task_clock_is_refused_naming_the_task(tmp_path, change, says):
    folder = _task(tmp_path / "week", STEPPED.replace(*change))
    with pytest.raises(Invalid, match=says) as refused:
        task_clock(folder)
    assert str(refused.value).startswith("task 'week': ")
