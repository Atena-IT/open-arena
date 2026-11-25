# License Apache 2.0: (c) 2026 Atena-Reply

import json
import os
import posixpath
import shutil
import subprocess
import time
import tomllib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import openarena
import pytest
import yaml
from openarena.config import ClockSpec, EnvironmentSpec
from openarena.config.check import check_file
from openarena.config.errors import Invalid
from openarena.harbor import (
    CLOCK_FILE,
    PROBE_NOTES,
    arm_task,
    armed_healthcheck_command,
    clock_arm_command,
    clock_jump_command,
    probe_facts,
    probe_notes,
    task_clock,
    task_files,
    virtual_now,
)
from openarena.programs.harbor_program import agent_kwargs, copy_task
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


def test_an_acp_registry_agent_or_a_class_of_ones_own_is_named_by_shape_and_never_imported(monkeypatch):
    import sys

    from openarena.config import HarborBlock

    assert HarborBlock(agent="acp:claude-acp").agent == "acp:claude-acp"
    assert HarborBlock(agent=" acp:gemini@0.4.1 ").agent == "acp:gemini@0.4.1"
    block = HarborBlock(agent="our_agents.harbor:OurAgent", agent_version="1.2.0", agent_env={"OUR_TOKEN": "t"})
    assert (block.agent_version, block.agent_env) == ("1.2.0", {"OUR_TOKEN": "t"})
    # Validating, describing and checking it imports nothing: the class runs only inside Harbor's trial.
    assert "our_agents" not in sys.modules
    for bad in ("nope", "acp:Claude ACP", "acp:", "our_agents.OurAgent", "our-agents:Agent"):
        with pytest.raises(ValidationError, match="no Harbor agent"):
            HarborBlock(agent=bad)
    with pytest.raises(ValidationError, match="pinned in its name"):
        HarborBlock(agent="acp:gemini", agent_version="0.4.1")
    with pytest.raises(ValidationError, match="not environment variable names"):
        HarborBlock(agent="acp:gemini", agent_env={"MY KEY": "v"})
    # A dataset without the new keys reads as before.
    assert HarborBlock().model_dump()["agent"] == "terminus-2" and HarborBlock().agent_env == {}


def test_an_acp_agent_runs_in_the_sandbox_and_a_class_is_checked_by_harbor_itself():
    import sys

    from openarena.harbor import agent_info, agent_problem

    public = EnvironmentSpec(docker_image="x")
    assert agent_info("acp:claude-acp")["runs"] == "sandbox"
    assert agent_info("our_agents.harbor:OurAgent") == {"name": "our_agents.harbor:OurAgent", "runs": None, "provider": None, "params": []}
    assert agent_info("nope") is None
    # Harbor's ACP runner splits the id into provider and model, and calls it from the sandbox.
    assert "names no provider" in agent_problem("acp:claude-acp", "claude-sonnet-4-5", public)
    assert "cannot reach 'ollama/qwen3'" in agent_problem("acp:opencode", "ollama/qwen3", public)
    assert agent_problem("acp:claude-acp", "anthropic/claude-sonnet-4-5", public) is None
    # A class of one's own: where it runs is unknown until Harbor imports it in the trial.
    assert agent_problem("our_agents.harbor:OurAgent", "ollama/qwen3", public) is None
    assert "our_agents" not in sys.modules
    # It gets none of the model's settings, and nothing is imported to find out what it takes.
    assert agent_kwargs("our_agents.harbor:OurAgent", {"temperature": 0.2}) == {}
    assert "our_agents" not in sys.modules


def test_a_trial_hands_harbor_the_agents_name_version_and_environment(tmp_path, monkeypatch):
    import asyncio

    from harbor.trial.trial import Trial

    from openarena.programs import harbor_program

    seen = {}

    class Ran:
        async def run(self):
            return "done"

    async def create(config):
        seen["agent"] = config.agent
        return Ran()

    monkeypatch.setattr(Trial, "create", staticmethod(create))
    out = asyncio.run(
        harbor_program.run_trial(
            tmp_path / "task", tmp_path / "trials", "t1", "acp:gemini@0.4.1", "google/gemini-2.5-pro",
            kwargs={"version": "x"}, provider="docker", env={"GEMINI_FLAG": "1"},
        )
    )
    assert out == "done"
    assert (seen["agent"].name, seen["agent"].env, seen["agent"].kwargs) == ("acp:gemini@0.4.1", {"GEMINI_FLAG": "1"}, {"version": "x"})


def test_hermes_is_an_agent_in_the_sandbox_on_any_provider():
    from openarena.config import HarborBlock
    from openarena.harbor import agent_problem, agents

    assert HarborBlock(agent="hermes").agent == "hermes"
    assert next(a for a in agents() if a["name"] == "hermes") == {"name": "hermes", "runs": "sandbox", "provider": None, "params": []}
    public, offline = EnvironmentSpec(docker_image="x"), EnvironmentSpec(docker_image="x", network_mode="no-network")
    assert agent_problem("hermes", "openrouter/qwen/qwen3-vl-8b-instruct", public) is None
    assert "no-network" in agent_problem("hermes", "anthropic/claude-sonnet-4-5", offline)
    assert "cannot reach 'ollama/qwen3'" in agent_problem("hermes", "ollama/qwen3", public)
    assert "names no provider" in agent_problem("hermes", "gpt-4o", public)
    assert "names no provider" in agent_problem("opencode", "gpt-4o", public)
    # It takes none of the model's params: they are dropped, not passed.
    assert agent_kwargs("hermes", {"temperature": 0.2, "max_tokens": 512}) == {}


# Recorded by pi: lines 746 to 751 of packages/coding-agent/test/fixtures/large-session.jsonl in pi v1.6.0 (MIT), after its
# header, moved to the session format pi 1.6 writes (version 3: each entry an `id` and its `parentId`), as pi migrates them.
PI_SESSION = [
    {'type': 'session', 'id': 'd703a1a9-1b7b-4fb1-b512-c9738b1fe617', 'timestamp': '2025-11-20T23:33:50.805Z', 'cwd': '/Users/badlogic/workspaces/pi-mono', 'provider': 'anthropic', 'modelId': 'claude-sonnet-4-5', 'thinkingLevel': 'off', 'version': 3},
    {'type': 'message', 'id': 'a1b2c3d4', 'parentId': None, 'timestamp': '2025-11-21T01:33:30.786Z', 'message': {'role': 'user', 'content': [{'type': 'text', 'text': 'delete the file'}], 'timestamp': 1763688810764}},
    {'type': 'message', 'id': 'b2c3d4e5', 'parentId': 'a1b2c3d4', 'timestamp': '2025-11-21T01:33:37.208Z', 'message': {'role': 'assistant', 'content': [{'type': 'toolCall', 'id': 'toolu_01Kh71uh1ch6ko73QcHPuPAJ', 'name': 'bash', 'arguments': {'command': 'rm ~/.pi/agent/themes/test.json'}}], 'api': 'anthropic-messages', 'provider': 'anthropic', 'model': 'claude-sonnet-4-5', 'usage': {'input': 3, 'output': 61, 'cacheRead': 0, 'cacheWrite': 135129, 'cost': {'input': 9e-06, 'output': 0.000915, 'cacheRead': 0, 'cacheWrite': 0.50673375, 'total': 0.50765775}}, 'stopReason': 'toolUse', 'timestamp': 1763688810766}},
    {'type': 'message', 'id': 'c3d4e5f6', 'parentId': 'b2c3d4e5', 'timestamp': '2025-11-21T01:33:37.225Z', 'message': {'role': 'toolResult', 'toolCallId': 'toolu_01Kh71uh1ch6ko73QcHPuPAJ', 'toolName': 'bash', 'content': [{'type': 'text', 'text': '(no output)'}], 'isError': False, 'timestamp': 1763688817218}},
    {'type': 'message', 'id': 'd4e5f6a7', 'parentId': 'c3d4e5f6', 'timestamp': '2025-11-21T01:33:40.875Z', 'message': {'role': 'assistant', 'content': [{'type': 'toolCall', 'id': 'toolu_013q2kM3qFk8PAXFhBqsj3t3', 'name': 'bash', 'arguments': {'command': 'ls ~/.pi/agent/themes/'}}], 'api': 'anthropic-messages', 'provider': 'anthropic', 'model': 'claude-sonnet-4-5', 'usage': {'input': 5, 'output': 58, 'cacheRead': 135129, 'cacheWrite': 74, 'cost': {'input': 1.5e-05, 'output': 0.00087, 'cacheRead': 0.0405387, 'cacheWrite': 0.0002775, 'total': 0.041701199999999994}}, 'stopReason': 'toolUse', 'timestamp': 1763688817218}},
    {'type': 'message', 'id': 'e5f6a7b8', 'parentId': 'd4e5f6a7', 'timestamp': '2025-11-21T01:33:40.901Z', 'message': {'role': 'toolResult', 'toolCallId': 'toolu_013q2kM3qFk8PAXFhBqsj3t3', 'toolName': 'bash', 'content': [{'type': 'text', 'text': '(no output)'}], 'isError': False, 'timestamp': 1763688820896}},
    {'type': 'message', 'id': 'f6a7b8c9', 'parentId': 'e5f6a7b8', 'timestamp': '2025-11-21T01:33:45.787Z', 'message': {'role': 'assistant', 'content': [{'type': 'text', 'text': 'Done! The `test.json` theme file has been deleted. If you had it selected when it was deleted, Pi should have automatically fallen back to the dark theme and logged: `[theme] Theme file deleted, falling back to dark theme`'}], 'api': 'anthropic-messages', 'provider': 'anthropic', 'model': 'claude-sonnet-4-5', 'usage': {'input': 4, 'output': 54, 'cacheRead': 135203, 'cacheWrite': 73, 'cost': {'input': 1.2e-05, 'output': 0.0008100000000000001, 'cacheRead': 0.0405609, 'cacheWrite': 0.00027375, 'total': 0.04165665}}, 'stopReason': 'stop', 'timestamp': 1763688820896}},
]


def _pi_trial(tmp_path, events, output=""):
    """A pi trial's folder after Harbor's own pi agent turned its session into `trajectory.json`, as it does after a run."""
    from harbor.agents.installed.pi import Pi
    from harbor.models.agent.context import AgentContext

    logs = tmp_path / "trial" / "agent"
    (logs / "pi" / "sessions").mkdir(parents=True)
    (logs / "pi" / "sessions" / "session.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    (logs / "pi.txt").write_text(output)
    Pi(logs_dir=logs, model_name="anthropic/claude-sonnet-4-5").populate_context_post_run(AgentContext())
    return tmp_path / "trial"


def test_a_recorded_pi_session_reads_as_its_trajectory_and_last_word(tmp_path):
    from types import SimpleNamespace

    from openarena.harbor import find_trajectory, trajectory_messages
    from openarena.programs.harbor_program import prediction, run_error

    trial = _pi_trial(tmp_path, PI_SESSION)
    messages = trajectory_messages(find_trajectory(trial / "agent"))
    assert [m["role"] for m in messages] == ["user", "assistant", "tool", "assistant", "tool", "assistant"]
    assert messages[1]["tool_calls"][0]["function"] == {"name": "bash", "arguments": {"command": "rm ~/.pi/agent/themes/test.json"}}
    predicted = prediction(trial, SimpleNamespace(exception_info=None))
    assert predicted["agent"]["name"] == "pi" and predicted["answer"].startswith("Done! The `test.json` theme file has been deleted.")
    assert run_error(predicted, trial / "agent") is None


def test_a_pi_run_that_ended_on_an_error_did_not_answer_whatever_its_exit_code(tmp_path):
    from types import SimpleNamespace

    from openarena.programs.harbor_program import COLLECTED_ANSWER, prediction, run_error

    failed = {"role": "assistant", "content": [], "api": "anthropic-messages", "provider": "anthropic", "model": "claude-sonnet-4-5",
              "stopReason": "error", "errorMessage": "401 invalid x-api-key", "timestamp": 1763688820896}
    events = [*PI_SESSION[:2], {"type": "message", "id": "b2c3d4e5", "parentId": "a1b2c3d4", "timestamp": "2025-11-21T01:33:37.208Z", "message": failed}]
    # pi's JSON events, as Harbor tees them into pi.txt; pi exits 0 all the same.
    output = json.dumps({"type": "message_end", "message": failed}) + "\n" + json.dumps({"type": "agent_settled", "aborted": False}) + "\n"
    trial = _pi_trial(tmp_path, events, output)
    with pytest.raises(ValueError, match="the agent's last model call failed: 401 invalid x-api-key"):
        prediction(trial, SimpleNamespace(exception_info=None))
    # A task folder's own tests score what the agent did in the sandbox, answer or not.
    assert prediction(trial, SimpleNamespace(exception_info=None), verified=True)["answer"] == ""
    # A log that cannot be read leaves the trajectory's own word.
    (trial / "agent" / "pi.txt").unlink()
    (trial / "agent" / "pi.txt").mkdir()
    assert run_error(json.loads((trial / "agent" / "trajectory.json").read_text()), trial / "agent") == "the agent's last model call failed"
    (trial / "agent" / "pi.txt").rmdir()
    (trial / "agent" / "pi.txt").write_text(output)
    # What it wrote before the error is still its answer, scored; the error goes to the trial's line.
    (trial / "verifier").mkdir()
    (trial / "verifier" / COLLECTED_ANSWER).write_text("4\n")
    predicted = prediction(trial, SimpleNamespace(exception_info=None), "/app/answer.txt")
    assert predicted["answer"] == "4"
    assert run_error(predicted, trial / "agent") == "the agent's last model call failed: 401 invalid x-api-key"


def test_a_trial_without_a_trajectory_says_what_the_agent_printed(tmp_path):
    from types import SimpleNamespace

    from openarena.programs.harbor_program import prediction

    logs = tmp_path / "trial" / "agent"
    logs.mkdir(parents=True)
    (logs / "pi.txt").write_text('Unknown provider "nebius". Use --list-models to see available providers/models.\n')
    with pytest.raises(ValueError, match='left no trajectory: the agent\'s output ends: Unknown provider "nebius"'):
        prediction(tmp_path / "trial", SimpleNamespace(exception_info=None))


def test_pi_calls_only_the_providers_it_knows_and_takes_the_reasoning_effort(monkeypatch):
    from openarena.config import HarborBlock
    from openarena.harbor import agent_problem, agents

    for env in ("OPENAI_BASE_URL", "OPENAI_API_BASE", "OPENROUTER_BASE_URL", "GOOGLE_BASE_URL"):
        monkeypatch.delenv(env, raising=False)
    assert HarborBlock(agent="pi").agent == "pi"
    assert next(a for a in agents() if a["name"] == "pi") == {"name": "pi", "runs": "sandbox", "provider": None, "params": ["reasoning_effort"]}
    # pi's `thinking` answers to `reasoning_effort` too; the rest it does not take.
    assert agent_kwargs("pi", {"reasoning_effort": "high", "temperature": 0.2}) == {"reasoning_effort": "high"}
    assert agent_kwargs("pi", {"reasoning_effort": "disable"}) == {"reasoning_effort": "off"}
    public = EnvironmentSpec(docker_image="x")
    for model in ("openrouter/qwen/qwen3-vl-8b-instruct", "openai/gpt-4.1-mini", "gemini/gemini-2.5-flash", "together_ai/x/y"):
        assert agent_problem("pi", model, public) is None, model
    assert "knows no provider 'nebius'" in agent_problem("pi", "nebius/meta-llama/x", public)
    assert "knows no provider 'fireworks_ai'" in agent_problem("pi", "fireworks_ai/accounts/x", public)
    assert "names no provider" in agent_problem("pi", "gpt-4o", public)
    assert "cannot reach 'hosted_vllm/x'" in agent_problem("pi", "hosted_vllm/x", public)
    # Harbor reads the first of OPENAI_BASE_URL and OPENAI_API_BASE that is set: a blank one is no endpoint, and hides the other.
    monkeypatch.setenv("OPENAI_BASE_URL", "")
    monkeypatch.setenv("OPENAI_API_BASE", "https://gateway.example/v1")
    assert agent_problem("pi", "openai/gpt-4.1-mini", public) is None
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example/v1")
    assert "custom endpoint OPENAI_BASE_URL sets" in agent_problem("pi", "openai/gpt-4.1-mini", public)


def test_a_sandbox_provider_refuses_what_it_cannot_give():
    from openarena.harbor import environment_problems, sandbox_capabilities

    assert sandbox_capabilities("e2b") == {"installed": True, "gpus": False, "no_network": True, "allowlist": True}
    assert environment_problems(EnvironmentSpec(docker_image="x", gpus=1), "e2b") == ["e2b sandboxes have no GPUs, and the environment asks for 1"]
    assert environment_problems(EnvironmentSpec(docker_image="x", network_mode="allowlist", allowed_hosts=["pypi.org"]), "e2b") == []
    # Docker decides per host: nothing refused up front.
    assert environment_problems(EnvironmentSpec(docker_image="x", gpus=1), "docker") == []


def test_a_provider_is_installed_only_when_its_sdk_imports(monkeypatch):
    import importlib.machinery
    import sys
    import types

    import harbor.environments.e2b
    from openarena.harbor import environment_problems, sandbox_capabilities, sandbox_not_installed

    # E2B's SDK comes with the engine; Docker needs none.
    assert sandbox_capabilities("e2b")["installed"] and sandbox_capabilities("docker")["installed"]
    # Harbor loads its E2B module without the SDK and records the failed import as `_HAS_E2B`.
    monkeypatch.setattr(harbor.environments.e2b, "_HAS_E2B", False)
    assert sandbox_capabilities("e2b")["installed"] is False
    assert environment_problems(EnvironmentSpec(docker_image="x"), "e2b") == ["e2b's SDK is not installed (pip install 'harbor[e2b]')"]
    # Harbor's Kubernetes module imports its SDK as it loads: without it, the module itself fails, and the reason is kept.
    monkeypatch.delitem(sys.modules, "harbor.environments.ack", raising=False)
    monkeypatch.setitem(sys.modules, "kubernetes", None)
    assert sandbox_capabilities("ack")["installed"] is False
    assert sandbox_not_installed("ack").startswith("ack cannot be loaded: ") and "kubernetes" in sandbox_not_installed("ack")
    # cua-cloud imports its SDK only once a sandbox starts: there, missing, or a stub with no spec to tell.
    sdk = types.ModuleType("cua_train")
    sdk.__spec__ = importlib.machinery.ModuleSpec("cua_train", None)
    monkeypatch.setitem(sys.modules, "cua_train", sdk)
    assert sandbox_capabilities("cua-cloud")["installed"] is True
    monkeypatch.setitem(sys.modules, "cua_train", None)
    assert sandbox_not_installed("cua-cloud") == "cua-cloud's SDK is not installed (pip install 'harbor[cua]')"
    monkeypatch.setitem(sys.modules, "cua_train", types.ModuleType("cua_train"))
    assert sandbox_capabilities("cua-cloud")["installed"] is False
    # A provider Harbor does not have.
    assert sandbox_capabilities("nope")["installed"] is False
    assert "'nope' is not a valid EnvironmentType" in sandbox_not_installed("nope")


def test_an_environment_on_a_provider_without_its_sdk_says_so(monkeypatch):
    import harbor.environments.e2b
    from openarena.harbor import environment_problems
    from openarena.programs import harbor_program

    # The environment names E2B, whose key is set but whose SDK is missing.
    spec = EnvironmentSpec(docker_image="x", sandbox="e2b")
    monkeypatch.setattr(harbor.environments.e2b, "_HAS_E2B", False)
    monkeypatch.setattr(harbor_program, "ready", lambda name: True)
    missing = "e2b's SDK is not installed (pip install 'harbor[e2b]')"
    # Its tasks run on the local Docker, and the note gives that reason, not missing credentials.
    assert harbor_program.sandbox(spec.sandbox) == "docker"
    assert harbor_program.sandbox_note(spec.sandbox) == f"{missing}, so tasks run on the local Docker instead"
    assert harbor_program.sandbox_issue(spec.sandbox) is None
    # Asked of E2B itself, the environment check refuses it for the same reason.
    assert environment_problems(spec, spec.sandbox) == [missing]
    # With no Docker to fall back to either, no task can start, and the reason is the same.
    monkeypatch.setattr(harbor_program, "ready", lambda name: False)
    assert harbor_program.sandbox_issue(spec.sandbox) == f"{missing}, and the local Docker it falls back to is not available to the backend either"


def test_a_provider_whose_sdk_breaks_on_import_falls_back_with_the_reason(monkeypatch, tmp_path):
    from harbor.environments import factory
    from harbor.models.environment_type import EnvironmentType
    from openarena.harbor import sandbox_capabilities
    from openarena.programs import harbor_program

    # An SDK installed but broken: the provider's module raises something other than ImportError as it loads.
    (tmp_path / "broken_modal_env.py").write_text('raise TypeError("protobuf clash")\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setitem(factory._ENVIRONMENT_REGISTRY, EnvironmentType.MODAL, factory._EnvEntry("broken_modal_env", "ModalEnvironment", "modal"))
    monkeypatch.setattr(harbor_program, "ready", lambda name: name == "docker")
    assert sandbox_capabilities("modal")["installed"] is False
    assert harbor_program.sandbox("modal") == "docker"
    assert harbor_program.sandbox_note("modal") == "modal cannot be loaded: protobuf clash (pip install 'harbor[modal]'), so tasks run on the local Docker instead"
    assert harbor_program.sandbox_issue("modal") is None


def test_a_provider_needs_the_sdk_of_its_base_class_too(monkeypatch, tmp_path):
    import importlib

    from harbor.environments import factory
    from harbor.models.environment_type import EnvironmentType
    from openarena.harbor import sandbox_not_installed

    # A provider built on another's class, each module with its own SDK flag (as wandb on cwsandbox, before Harbor 0.24).
    (tmp_path / "base_env.py").write_text("_HAS_BASE_SDK = False\nclass BaseEnv:\n    pass\n")
    (tmp_path / "child_env.py").write_text("from base_env import BaseEnv\n_HAS_CHILD_SDK = True\nclass ChildEnv(BaseEnv):\n    pass\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setitem(factory._ENVIRONMENT_REGISTRY, EnvironmentType.MODAL, factory._EnvEntry("child_env", "ChildEnv", "modal"))
    assert sandbox_not_installed("modal") == "modal's SDK is not installed (pip install 'harbor[modal]')"
    monkeypatch.setattr(importlib.import_module("base_env"), "_HAS_BASE_SDK", True)
    assert sandbox_not_installed("modal") is None


def test_every_harbor_provider_with_an_sdk_records_whether_it_imports():
    import sys

    from harbor.environments.factory import _ENVIRONMENT_REGISTRY, _load_environment_class
    from openarena.harbor import _LAZY_SDKS

    # A module found with no `_HAS_*` flag counts as installed: a Harbor that renames a flag must fail here, not
    # call that provider installed again. A provider with no extra needs no SDK; a module that imports its SDK as it
    # loads fails without it, and is skipped, or loads with it and is named here.
    eager = {"islo", "tensorlake", "runta", "prime", "smol"}
    unflagged = []
    for kind, entry in _ENVIRONMENT_REGISTRY.items():
        if entry.pip_extra is None or kind.value in _LAZY_SDKS or kind.value in eager:
            continue
        try:
            cls = _load_environment_class(kind)
        except ImportError:
            continue
        if not any(key.startswith("_HAS_") for key in vars(sys.modules[cls.__module__])):
            unflagged.append(kind.value)
    assert unflagged == []


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
    monkeypatch.setattr(harbor_program, "sandbox", lambda name=None: "ack")
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


TUESDAY = datetime(2026, 3, 3, 8, tzinfo=timezone.utc)


def _armed(tmp_path):
    path = tmp_path / "arena" / "clock.json"
    assert subprocess.run(["sh", "-c", clock_arm_command(ClockSpec(start="2026-03-02T08:00:00Z"), str(path))]).returncode == 0
    return path


@POSIX_SH
def test_the_jump_moves_a_frozen_clock_to_the_step_time_once(tmp_path):
    path = _armed(tmp_path)
    armed_text = path.read_text()
    before = json.loads(armed_text)
    jump = clock_jump_command(TUESDAY, str(path))
    assert subprocess.run(["sh", "-c", jump]).returncode == 0
    after = json.loads(path.read_text())
    assert virtual_now(after, time.time()) == TUESDAY.timestamp()
    assert {k: after[k] for k in ("t0", "rate", "anchor")} == {k: before[k] for k in ("t0", "rate", "anchor")}
    assert path.read_text().startswith(armed_text[: armed_text.index(' "anchor"')])  # the arm's check of t0 and rate still matches
    assert subprocess.run(["sh", "-c", jump]).returncode == 0 and json.loads(path.read_text()) == after
    assert sorted(p.name for p in path.parent.iterdir()) == ["clock.json"]


@POSIX_SH
def test_the_jump_never_goes_back_and_needs_the_clock(tmp_path):
    path = _armed(tmp_path)
    assert subprocess.run(["sh", "-c", clock_jump_command(TUESDAY, str(path))]).returncode == 0
    back = subprocess.run(["sh", "-c", clock_jump_command(TUESDAY - timedelta(hours=1), str(path))], capture_output=True, text=True)
    assert back.returncode != 0 and "past" in back.stderr
    assert subprocess.run(["sh", "-c", clock_jump_command(TUESDAY, str(tmp_path / "none.json"))], capture_output=True).returncode != 0
    (tmp_path / "garbage.json").write_text("not a clock")
    assert subprocess.run(["sh", "-c", clock_jump_command(TUESDAY, str(tmp_path / "garbage.json"))], capture_output=True).returncode != 0
    assert sorted(p.name for p in path.parent.iterdir()) == ["clock.json"]


@pytest.mark.skipif(shutil.which("busybox") is None, reason="busybox is not installed")
def test_the_jump_runs_under_busybox(tmp_path):
    path = _armed(tmp_path)
    assert subprocess.run(["busybox", "sh", "-c", clock_jump_command(TUESDAY, str(path))]).returncode == 0
    assert virtual_now(json.loads(path.read_text()), time.time()) == TUESDAY.timestamp()


def _stepped_task(folder):
    """A three-day task with a clock, the author's healthcheck, other config and a day3 setup.sh with no shebang."""
    _task(folder, STEPPED + '\n[environment.healthcheck]\ncommand = "test -d /app"\nretries = 5\n\n[verifier.env]\nTOKEN = "x"\n\n[metadata]\nowner = "qa"\n')
    (folder / "steps" / "day3" / "workdir").mkdir(parents=True)
    (folder / "steps" / "day3" / "workdir" / "setup.sh").write_text("echo day3\n")
    return folder


def _files(folder):
    return {p.relative_to(folder): p.read_bytes() for p in folder.rglob("*") if p.is_file()}


def test_a_copy_is_armed_and_the_source_left_alone(tmp_path):
    from harbor.models.task.config import TaskConfig

    source = _stepped_task(tmp_path / "src")
    before = _files(source)
    copy = copy_task(source, tmp_path / "copy")
    assert _files(source) == before
    armed = TaskConfig.model_validate_toml((copy / "task.toml").read_text())
    original = TaskConfig.model_validate_toml((source / "task.toml").read_text())
    clock = task_clock(source)
    assert armed.environment.healthcheck.command == armed_healthcheck_command(clock, "test -d /app")
    assert armed.environment.healthcheck.retries == original.environment.healthcheck.retries == 5
    assert armed.model_dump(exclude={"environment": {"healthcheck"}}) == original.model_dump(exclude={"environment": {"healthcheck"}})
    assert (copy / "steps/day2/workdir/setup.sh").read_text() == f"#!/bin/bash\n{clock_jump_command(clock.steps['day2'])}\n"
    assert (copy / "steps/day3/workdir/setup.sh").read_text() == f"{clock_jump_command(clock.steps['day3'])}\necho day3\n"
    assert not (copy / "steps/day1/workdir/setup.sh").exists()
    assert oct((copy / "steps/day2/workdir/setup.sh").stat().st_mode & 0o777) == "0o755"


def test_the_jump_goes_after_the_authors_shebang(tmp_path):
    source = _task(tmp_path / "src", STEPPED)
    (source / "steps" / "day2" / "workdir").mkdir(parents=True)
    (source / "steps" / "day2" / "workdir" / "setup.sh").write_text("#!/bin/sh\necho hi\n")
    copy = copy_task(source, tmp_path / "copy")
    jump = clock_jump_command(task_clock(source).steps["day2"])
    assert (copy / "steps/day2/workdir/setup.sh").read_text() == f"#!/bin/sh\n{jump}\necho hi\n"


def test_arming_twice_changes_nothing_and_a_plain_task_is_copied_as_is(tmp_path):
    copy = copy_task(_stepped_task(tmp_path / "src"), tmp_path / "copy")
    once = _files(copy)
    assert arm_task(copy) is False
    assert _files(copy) == once
    plain = _task(tmp_path / "plain", "[agent]\ntimeout_sec = 60\n")
    assert _files(copy_task(plain, tmp_path / "plain-copy")) == _files(plain)


def test_rearming_with_a_changed_clock_replaces_the_old_arm_and_jumps(tmp_path):
    from harbor.models.task.config import TaskConfig

    copy = copy_task(_stepped_task(tmp_path / "src"), tmp_path / "copy")  # an export, armed for Monday
    toml_path = copy / "task.toml"
    toml_path.write_text(toml_path.read_text().replace("2026-03-02T08:00:00Z", "2026-03-01T08:00:00Z").replace("2026-03-04T08:00:00Z", "2026-03-05T08:00:00Z"))
    assert arm_task(copy) is True  # re-imported and edited: armed for the new clock
    clock = task_clock(copy)
    healthcheck = TaskConfig.model_validate_toml(toml_path.read_text()).environment.healthcheck
    assert healthcheck.command == armed_healthcheck_command(clock, "test -d /app")  # one arm, the author's check kept
    assert (copy / "steps/day3/workdir/setup.sh").read_text() == f"{clock_jump_command(clock.steps['day3'])}\necho day3\n"
    assert (copy / "steps/day2/workdir/setup.sh").read_text() == f"#!/bin/bash\n{clock_jump_command(clock.steps['day2'])}\n"


def test_other_arena_metadata_is_not_a_clock(tmp_path):
    assert task_clock(_task(tmp_path / "label", '[metadata]\narena = "benchmark-a"\n')) is None
    assert task_clock(_task(tmp_path / "table", '[metadata.arena]\nsuite = "mail"\n')) is None


@pytest.mark.parametrize(("change", "says"), [
    (('[[steps]]\nname = "day1"', '[environment]\nos = "windows"\n\n[[steps]]\nname = "day1"'), "Windows"),  # a POSIX clock
    (('day2 = "2026-03-03T08:00:00Z"', 'day2 = "2026-03-03T08:00:00.750Z"'), "second"),  # a fraction of a second
    (('start = "2026-03-02T08:00:00Z"', 'start = "2026-03-02T08:00:00.5Z"'), "second"),
])
def test_a_task_clock_needs_linux_and_whole_seconds(tmp_path, change, says):
    folder = _task(tmp_path / "week", STEPPED.replace(*change))
    with pytest.raises(Invalid, match=says) as refused:
        task_clock(folder)
    assert str(refused.value).startswith("task 'week': ")


# -- a sandbox provider per environment ----------------------------------------


def test_an_environment_names_its_sandbox_or_keeps_the_configured_one():
    assert EnvironmentSpec(docker_image="x").sandbox is None
    assert EnvironmentSpec(docker_image="x", sandbox=" Docker ").sandbox == "docker"
    assert EnvironmentSpec(docker_image="x", sandbox="  ").sandbox is None
    with pytest.raises(ValidationError, match="Harbor has no sandbox 'dokcer'"):
        EnvironmentSpec(docker_image="x", sandbox="dokcer")


def test_check_refuses_an_unknown_sandbox_with_the_known_ones(tmp_path):
    report = check_file(_write(tmp_path, "environments", {"ok": {"docker_image": "x", "sandbox": "kata"}, "typo": {"docker_image": "x", "sandbox": "dokcer"}}))
    [problem] = report.problems
    assert (problem.level, problem.where) == ("error", "environments.typo.sandbox")
    assert problem.message == "Harbor has no sandbox 'dokcer'"
    assert "docker" in problem.fix and "e2b" in problem.fix and "kata" in problem.fix


def test_a_named_provider_wins_over_the_configured_one(monkeypatch):
    from openarena.programs import harbor_program

    monkeypatch.setenv("ARENA_HARBOR_SANDBOX", "modal")
    assert harbor_program.configured_sandbox() == "modal"
    assert harbor_program.configured_sandbox("Daytona") == "daytona"
    assert harbor_program.configured_sandbox("") == "modal"
    monkeypatch.delenv("ARENA_HARBOR_SANDBOX")
    assert harbor_program.configured_sandbox() == "e2b"
    with pytest.raises(ValueError, match="no sandbox 'nope'; use one of: .*docker"):
        harbor_program.configured_sandbox("nope")
    monkeypatch.setenv("ARENA_HARBOR_SANDBOX", "nope")
    with pytest.raises(ValueError, match="set ARENA_HARBOR_SANDBOX to one of"):
        harbor_program.configured_sandbox()


def test_a_named_provider_falls_back_to_docker_on_its_own(monkeypatch):
    from openarena.programs import harbor_program

    monkeypatch.setenv("ARENA_HARBOR_SANDBOX", "e2b")
    monkeypatch.setattr(harbor_program, "ready", lambda name: name in ("e2b", "docker"))
    monkeypatch.setattr(harbor_program, "sandbox_not_installed", lambda name: None)  # every SDK here: this is about keys
    assert harbor_program.sandbox() == "e2b" and harbor_program.sandbox_note() is None
    # Daytona has no key here: its tasks run on the local Docker, whatever the configured provider does.
    assert harbor_program.sandbox("daytona") == "docker"
    assert harbor_program.sandbox_note("daytona") == "daytona has no credentials set, so tasks run on the local Docker instead"
    assert harbor_program.sandbox_issue("daytona") is None
    assert "no sandbox 'nope'" in harbor_program.sandbox_issue("nope")
    monkeypatch.setattr(harbor_program, "ready", lambda name: False)
    assert "daytona has no credentials set" in harbor_program.sandbox_issue("daytona")
    assert "the docker sandbox provider cannot start" in harbor_program.sandbox_issue("docker")


def test_options_go_only_to_the_configured_provider(monkeypatch):
    from openarena.harbor import environment_config

    monkeypatch.setenv("ARENA_HARBOR_SANDBOX", "ack")
    monkeypatch.setenv("ARENA_HARBOR_SANDBOX_OPTIONS", '{"namespace": "arena-sandboxes"}')
    # An environment naming the configured provider gets its options; one naming another provider gets none.
    assert environment_config("ack").kwargs == {"namespace": "arena-sandboxes"}
    assert environment_config("daytona").kwargs == {}


def _ran_trials(monkeypatch, tmp_path, spec):
    """Run one row of `spec` through `HarborTask`, the trial faked: what each trial was configured with."""
    import asyncio

    from harbor.trial.trial import Trial
    from openarena.programs import harbor_program
    from synalinks.src.backend import JsonDataModel

    seen = []

    class Ran:
        async def run(self):
            raise RuntimeError("no sandbox in tests")

    async def create(config):
        seen.append({"config": config, "task_toml": (Path(config.task.path) / "task.toml").read_text()})
        return Ran()

    monkeypatch.setattr(Trial, "create", staticmethod(create))
    monkeypatch.setattr(harbor_program, "ready", lambda name: True)
    task = harbor_program.HarborTask(model="openai/gpt-4o", spec=spec, work_dir=str(tmp_path / "work"))
    row = JsonDataModel(json={"question": "What is 2+2?", "path": str(tmp_path / "source")}, schema={"type": "object", "properties": {}})
    assert asyncio.run(task.call(row)) is None  # the faked trial fails: the row is unanswered
    return seen


def test_a_run_uses_its_environments_sandbox_everywhere(monkeypatch, tmp_path):
    import asyncio

    from openarena.programs import harbor_program

    monkeypatch.setenv("ARENA_HARBOR_SANDBOX", "e2b")
    monkeypatch.setenv("ARENA_HARBOR_SANDBOX_OPTIONS", '{"template": "arena"}')
    entry = EnvironmentSpec(docker_image="python:3.12-slim", sandbox="docker").model_dump(mode="json")
    spec = {"agent": "terminus-2", "dataset": "d", "environment": "py", "environment_entry": entry, "mcp_servers": {}}
    asked = []
    real = harbor_program.sandbox
    monkeypatch.setattr(harbor_program, "sandbox", lambda name=None: asked.append(name) or real(name))
    [trial] = _ran_trials(monkeypatch, tmp_path, spec)
    assert trial["config"].environment.type == "docker"
    assert trial["config"].environment.kwargs == {}  # the options are E2B's, the configured provider
    # The task was written for the local Docker: the CPUs and memory the entry leaves blank get its defaults.
    assert {"cpus", "memory_mb"} <= set(tomllib.loads(trial["task_toml"])["environment"])
    assert asked and set(asked) == {"docker"}
    # And the cell's pre-check asks the environment's provider too.
    asked.clear()
    with pytest.raises(ValueError, match="no sandbox 'nope'"):
        asyncio.run(harbor_program.build_harbor("openai/gpt-4o", None, {"environment_entry": {**entry, "sandbox": "nope"}}, None))
    assert asked == ["nope"]


def test_harbor_tasks_keep_the_configured_sandbox(monkeypatch, tmp_path):
    monkeypatch.setenv("ARENA_HARBOR_SANDBOX", "e2b")
    monkeypatch.setenv("ARENA_HARBOR_SANDBOX_OPTIONS", '{"template": "arena"}')
    source = tmp_path / "source"
    (source / "environment").mkdir(parents=True)
    (source / "task.toml").write_text('[environment]\ndocker_image = "python:3.12-slim"\n')
    (source / "instruction.md").write_text("Say hi.\n")
    [trial] = _ran_trials(monkeypatch, tmp_path, {"agent": "terminus-2", "dataset": "d", "tasks": True})
    assert trial["config"].environment.type == "e2b"
    assert trial["config"].environment.kwargs == {"template": "arena"}


def test_what_the_framework_is_told_about_a_model_is_never_passed_to_a_harbor_agent():
    params = {"temperature": 0.0, "reasoning_effort": "high", "supports_reasoning": True, "supports_vision": True, "supports_audio": True}
    assert agent_kwargs("terminus-2", params) == {"temperature": 0.0, "reasoning_effort": "high"}
