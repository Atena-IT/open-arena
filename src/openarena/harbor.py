# License Apache 2.0: (c) 2026 Atena-Reply

"""An environment entry, written the way a Harbor task holds it, and what the arena knows of Harbor.

A Harbor task is a folder: `task.toml`, `instruction.md`, `environment/`,
`tests/` (run by Harbor's verifier after the agent, writing
`/logs/verifier/reward.txt` or `reward.json`), `solution/` (run by the oracle
agent), and optionally `steps/`. A workspace's own tasks are such folders
(`type: harbor_tasks` datasets run them, scored by their tests).
An `environments.<name>:` entry is the sandbox part of a task generated from rows, `task.toml`'s
`[environment]` section (with its `healthcheck` and `mcp_servers`) and,
when the entry builds its image, `environment/Dockerfile`. `task_files`
returns both, for a task to be assembled around them.

An MCP server is passed as its connection (`McpServerSpec.connection()`);
Harbor reads `stdio`, `sse` and `streamable-http`, so a `websocket` server
is refused. Harbor's server entry is only a name, a transport and a command
or URL: a connection that needs `headers` (a key) or a stdio `env` / `cwd`
would reach the agent without them, so it is refused too. A stdio command
runs inside the sandbox, so the image must hold it.

Every trial runs Harbor's verifier. A task folder's own tests score it; a
task generated from rows gets the arena's default tests, which collect the
agent's answer for the dataset's reward to score on the arena's side. What
the agent did is read back from its ATIF trajectory (`trajectory_messages`,
`final_answer`) into the prediction, `HarborTrajectory`. `agents()` says where each agent runs and which
provider it calls, `agent_problem` refuses a model or a network it cannot
work with, and `sandbox_capabilities` / `environment_problems` an entry a
sandbox provider cannot start. `environment_config` is what a trial runs on:
the provider, with its own settings (`ARENA_HARBOR_SANDBOX_OPTIONS`).
"""

from __future__ import annotations

import asyncio
import json
import os
import posixpath
import shlex
import threading
import tomllib
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from openarena.config.entries import ClockSpec, EnvironmentSpec, HealthcheckSpec, TaskClock
from openarena.config.errors import Invalid

# Our transport name -> Harbor's.
HARBOR_TRANSPORTS = {"stdio": "stdio", "sse": "sse", "streamable_http": "streamable-http"}


def _value(value: Any) -> str:
    """A TOML value: JSON's strings, numbers and arrays are TOML's too."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{_key(k)} = {_value(v)}" for k, v in value.items()) + " }" if value else "{}"
    if isinstance(value, list):
        return "[" + ", ".join(_value(v) for v in value) + "]"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return json.dumps(value)


def _key(key: str) -> str:
    return key if key.replace("_", "").replace("-", "").isalnum() else json.dumps(key)


def _table(header: str, fields: dict[str, Any]) -> str:
    lines = [header] + [f"{_key(k)} = {_value(v)}" for k, v in fields.items() if v is not None]
    return "\n".join(lines)


def mcp_server(name: str, connection: dict[str, Any]) -> dict[str, Any]:
    """One `[[environment.mcp_servers]]` table from an MCP connection."""
    transport = connection.get("transport")
    if transport not in HARBOR_TRANSPORTS:
        raise Invalid(
            f"Harbor cannot reach MCP server {name!r} over {transport!r}",
            f"serve it over {', '.join(HARBOR_TRANSPORTS)}, or remove it from `mcp_servers:`",
        )
    lost = [field for field in ("headers", "env", "cwd") if connection.get(field)]
    if lost:
        raise Invalid(
            f"Harbor cannot hand MCP server {name!r} its {', '.join(f'`{f}`' for f in lost)}: a task's server is only a command or a URL",
            "move a stdio server's variables to the environment's `env:`, serve an HTTP one without headers, or remove it from `mcp_servers:`",
        )
    server: dict[str, Any] = {"name": name, "transport": HARBOR_TRANSPORTS[transport]}
    if transport == "stdio":
        server["command"] = connection.get("command")
        server["args"] = list(connection.get("args") or [])
    else:
        server["url"] = connection.get("url")
    return server


# Where an environment's virtual clock lives in the sandbox: one fixed path, so every process finds it,
# whoever started it (an agent's MCP client may not pass its variables on to the servers it starts).
CLOCK_FILE = "/run/arena/clock.json"


def clock_arm_command(clock: ClockSpec, path: str = CLOCK_FILE) -> str:
    """The POSIX shell command that writes the clock to `path` once: `t0` and `rate` from the entry, `anchor` the real time now.

    Written beside and moved into place, so a reader never sees half a file; a file already
    there is left alone, so a healthcheck retry does not move the anchor. It fails when `path`
    cannot be written, and the healthcheck with it, rather than letting a trial run on real time.
    """
    head = '{"t0": %d, "rate": %s,' % (int(clock.start.timestamp()), json.dumps(clock.rate))
    line = head + ' "anchor": %s, "ahead": 0}\\n'
    folder, target = (shlex.quote(p) for p in (posixpath.dirname(path) or ".", path))
    # Each run writes its own file beside (`$$`, its process id), so runs that overlap never share one.
    tmp = shlex.quote(f"{path}.tmp") + ".$$"
    # Kept only when it holds this entry's start and rate: a retry keeps its anchor, a file left by anything else is replaced.
    return f'mkdir -p {folder} && {{ grep -qsF {shlex.quote(head)} {target} || {{ printf {shlex.quote(line)} "$(date +%s)" > {tmp} && chmod 644 {tmp} && mv {tmp} {target}; }}; }}'


def virtual_now(clock: dict[str, float], real_now: float) -> float:
    """The virtual time, in epoch seconds, of a clock file read at `real_now`: AgentEnv's formula."""
    return clock["t0"] + (real_now - clock["anchor"]) * clock["rate"] + clock.get("ahead", 0)


def task_clock(folder: Path) -> TaskClock | None:
    """The clock a Harbor task folder declares in `[metadata.arena.clock]`, checked against the whole task; None without one.

    Raises `Invalid` naming the task. Step times must name its `[[steps]]`, in the order they run,
    and need its agent to run as root, the user that armed the clock: each step's `setup.sh`,
    which moves the clock, runs as that step's agent.
    """
    config = tomllib.loads((folder / "task.toml").read_text(encoding="utf-8"))
    declared = ((config.get("metadata") or {}).get("arena") or {}).get("clock")
    if declared is None:
        return None

    def refuse(message: str, fix: str | None = None) -> Invalid:
        return Invalid(f"task {folder.name!r}: {message}", fix)

    try:
        clock = TaskClock.model_validate(declared)
    except ValidationError as e:
        said = "; ".join(f"{'.'.join(map(str, err['loc'])) or 'clock'}: {err['msg']}" for err in e.errors())
        raise refuse(f"`[metadata.arena.clock]`: {said}") from e
    steps = config.get("steps") or []
    order = [step.get("name") for step in steps]
    unknown = [name for name in clock.steps if name not in order]
    if unknown:
        raise refuse(f"step {unknown[0]!r} in `[metadata.arena.clock.steps]` is not one of its steps ({', '.join(map(str, order)) or 'none'})", "name one of the task's `[[steps]]`")
    timed = [(name, clock.steps[name]) for name in order if name in clock.steps]
    for (before, at), (after, then) in zip(timed, timed[1:]):
        if then <= at:
            raise refuse(f"step {after!r} begins at or before step {before!r}: step times follow the order the steps run", f"give {after!r} a time after {before!r}'s")
    users = [(config.get("agent") or {}).get("user")] + [(step.get("agent") or {}).get("user") for step in steps]
    others = [user for user in users if user not in (None, "root", 0, "0")]
    if clock.steps and others:
        raise refuse(f"step times need the agent to run as root, not as user {others[0]!r}: each step's setup.sh, which moves the clock, runs as the step's agent", "remove `user` from `[agent]` and the steps, or remove `[metadata.arena.clock.steps]`")
    return clock


def environment_toml(spec: EnvironmentSpec, servers: dict[str, dict[str, Any]] | None = None) -> str:
    """The `[environment]` section of `task.toml`, with its sub-tables.

    `servers` maps each of `spec.mcp_servers` to its connection; a name it
    lacks is refused, so a task never names a server it cannot reach.
    """
    servers = servers or {}
    missing = [n for n in spec.mcp_servers if n not in servers]
    if missing:
        raise Invalid(f"no MCP server named {', '.join(map(repr, missing))}", "register it on the MCP tools page, or remove it from `mcp_servers:`")
    fields = {
        "build_timeout_sec": spec.build_timeout_sec,
        "docker_image": spec.docker_image,
        "os": spec.os,
        "cpus": spec.cpus,
        "memory_mb": spec.memory_mb,
        "storage_mb": spec.storage_mb,
        "gpus": spec.gpus,
        "gpu_types": spec.gpu_types or None,
        "network_mode": spec.network_mode,
        "allowed_hosts": spec.allowed_hosts or None,
        "workdir": spec.workdir,
        "env": spec.env or None,
    }
    healthcheck = spec.healthcheck
    if spec.clock is not None:
        arm = clock_arm_command(spec.clock)
        # Theirs runs in a subshell after the arm, so an `||` of theirs cannot pass a healthcheck whose arm failed,
        # and closes on a line of its own, so a trailing `# comment` of theirs cannot swallow the `)`.
        command = f"{arm} && ({healthcheck.command}\n)" if healthcheck is not None else arm
        healthcheck = (healthcheck or HealthcheckSpec(command=arm)).model_copy(update={"command": command})
    parts = [_table("[environment]", fields)]
    if healthcheck is not None:
        parts.append(_table("[environment.healthcheck]", healthcheck.model_dump()))
    for name in spec.mcp_servers:
        parts.append(_table("[[environment.mcp_servers]]", mcp_server(name, servers[name])))
    return "\n\n".join(parts) + "\n"


def task_files(spec: EnvironmentSpec, servers: dict[str, dict[str, Any]] | None = None) -> dict[str, str]:
    """`{path in the task folder: content}`: the `task.toml` section, and `environment/Dockerfile` when one is built."""
    files = {"task.toml": environment_toml(spec, servers)}
    if spec.dockerfile:
        files["environment/Dockerfile"] = spec.dockerfile.rstrip("\n") + "\n"
    return files


# -- Harbor's agents -------------------------------------------------------------

# Providers served from the arena's own machine by default: a sandbox cannot reach them.
LOCAL_PROVIDERS = ("ollama", "ollama_chat", "lm_studio", "hosted_vllm", "vllm", "sglang")


def agents() -> list[dict[str, Any]]:
    """Each agent a `harbor:` block may name: `{name, runs, provider, params}`.

    `runs` is `host` (it calls the model from the arena, typing into the
    sandbox) or `sandbox` (it is installed in the sandbox and calls the
    model from there); `provider` the one provider it calls, or None for
    any; `params` the model settings it takes of `temperature`,
    `reasoning_effort` and `max_tokens`.
    """
    from harbor.agents.factory import AgentFactory
    from harbor.agents.installed.base import BaseInstalledAgent
    from harbor.models.agent.name import AgentName

    from openarena.config.entries import HARBOR_AGENTS

    out = []
    for name in HARBOR_AGENTS:
        cls = AgentFactory.get_agent_class(AgentName(name))
        fields = set(cls.options_model.model_fields) if cls.options_model else set()
        params = [p for p in ("temperature", "reasoning_effort", "max_tokens") if p in fields or ("llm_call_kwargs" in fields and p == "max_tokens")]
        connection = cls.MODEL_CONNECTION
        out.append({
            "name": name,
            "runs": "sandbox" if issubclass(cls, BaseInstalledAgent) else "host",
            "provider": connection.default_provider if connection else None,
            "params": params,
        })
    return out


def model_provider(model_id: str) -> str | None:
    """The provider of a litellm id as Harbor names it (`gemini` is `google`), or None when litellm cannot route it."""
    import litellm
    from harbor.agents.model_connection import _PROVIDER_ALIASES

    try:
        provider = litellm.get_llm_provider(model_id)[1]
    except Exception:  # noqa: BLE001 - an id litellm cannot route has no provider to compare
        return None
    return _PROVIDER_ALIASES.get(provider, provider)


def agent_problem(agent: str, model_id: str, environment: EnvironmentSpec | None) -> str | None:
    """Why `agent` cannot work with `model_id` in `environment` (None: a task folder's own, not known here), as one sentence, or None.

    An agent bound to a provider calls only its models; one installed in the
    sandbox calls its model from there, so it needs the network, and a model
    served from the arena's machine is out of its reach.
    """
    info = next((a for a in agents() if a["name"] == agent), None)
    if info is None:
        return None
    provider = model_provider(model_id)
    if info["provider"] and provider and provider != info["provider"]:
        return f"Harbor agent {agent!r} calls {info['provider']} models only, and {model_id!r} is a {provider} model"
    if info["runs"] == "sandbox":
        if environment is not None and environment.network_mode == "no-network":
            return f"Harbor agent {agent!r} runs inside the sandbox and calls its model from there, which a no-network environment forbids"
        if (provider or "") in LOCAL_PROVIDERS or model_id.split("/", 1)[0] in LOCAL_PROVIDERS:
            return f"Harbor agent {agent!r} runs inside the sandbox, which cannot reach {model_id!r} on the arena's machine"
    return None


# -- the local Docker ----------------------------------------------------------------

# The provider that runs on the arena's own machine, beside the backend, the browser and often a model server.
LOCAL_SANDBOX = "docker"
# What a task there gets when its entry asks for nothing: Docker would give it the whole machine,
# and a few containers at once (with an Ollama model loading) freeze it.
LOCAL_CPUS = 2
LOCAL_MEMORY_MB = 2048
# Trials there at once, environment checks included.
LOCAL_TRIALS = 1

_local_gate: tuple[int, threading.BoundedSemaphore] | None = None
_local_gate_lock = threading.Lock()


def _setting(name: str, default: int) -> int:
    """A positive integer from the environment, else `default`."""
    try:
        value = int(os.environ.get(name, "").strip())
    except ValueError:
        return default
    return value if value >= 1 else default


def local_limits(spec: EnvironmentSpec, sandbox: str) -> EnvironmentSpec:
    """`spec` with the local Docker's default CPUs and memory where it leaves them blank; unchanged on any other provider.

    `ARENA_HARBOR_LOCAL_CPUS` and `ARENA_HARBOR_LOCAL_MEMORY_MB` change the
    defaults; what the entry itself asks for is kept.
    """
    if sandbox != LOCAL_SANDBOX:
        return spec
    cpus = min(_setting("ARENA_HARBOR_LOCAL_CPUS", LOCAL_CPUS), os.cpu_count() or 1)
    memory = _setting("ARENA_HARBOR_LOCAL_MEMORY_MB", LOCAL_MEMORY_MB)
    return spec.model_copy(update={"cpus": spec.cpus or cpus, "memory_mb": spec.memory_mb or memory})


def _gate() -> threading.BoundedSemaphore:
    global _local_gate
    size = _setting("ARENA_HARBOR_LOCAL_TRIALS", LOCAL_TRIALS)
    with _local_gate_lock:
        if _local_gate is None or _local_gate[0] != size:
            _local_gate = (size, threading.BoundedSemaphore(size))
        return _local_gate[1]


@asynccontextmanager
async def local_slot(sandbox: str):
    """Hold one of the local Docker's trial slots (`ARENA_HARBOR_LOCAL_TRIALS`, one by default); free on other providers.

    A thread semaphore, polled: sweeps and environment checks run on event
    loops of their own threads, and a cancelled wait never holds a slot.
    """
    if sandbox != LOCAL_SANDBOX:
        yield
        return
    gate = _gate()
    while not gate.acquire(blocking=False):
        await asyncio.sleep(0.5)
    try:
        yield
    finally:
        gate.release()


# -- sandbox providers -------------------------------------------------------------

# The configured provider's own settings: a JSON object, handed to its Harbor environment.
SANDBOX_OPTIONS = "ARENA_HARBOR_SANDBOX_OPTIONS"


def sandbox_options() -> dict[str, Any]:
    """`ARENA_HARBOR_SANDBOX_OPTIONS`: the configured provider's settings, as Harbor's environment takes them; `{}` when unset.

    A JSON object of the keyword arguments Harbor passes the provider's
    environment class (what `harbor run --ek` sets): the namespace, registry
    and node selector of the Kubernetes environment (`ack`), for instance.
    Which keys a provider takes is Harbor's to say. A value that is not a JSON
    object raises `ValueError` naming the variable: every trial would
    otherwise fail on it, and each failed row scores 0.
    """
    raw = (os.environ.get(SANDBOX_OPTIONS) or "").strip()
    if not raw:
        return {}
    try:
        options = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"{SANDBOX_OPTIONS} is not JSON ({e}); it holds a JSON object of the provider's settings") from None
    if not isinstance(options, dict):
        raise ValueError(f"{SANDBOX_OPTIONS} must be a JSON object of the provider's settings, not {type(options).__name__}")
    return options


def environment_config(sandbox: str):
    """Harbor's `EnvironmentConfig` for a trial on provider `sandbox`: with `sandbox_options()` when it is the configured one.

    The local Docker a trial falls back to, when the configured provider
    cannot start, gets none: they are that other provider's settings.
    """
    from harbor.models.trial.config import EnvironmentConfig

    from openarena.programs.harbor_program import configured_sandbox

    return EnvironmentConfig(type=sandbox, kwargs=sandbox_options() if sandbox == configured_sandbox() else {})


def sandbox_capabilities(name: str) -> dict[str, Any]:
    """What sandbox provider `name` can give: `{installed, gpus, no_network, allowlist}`.

    `installed` is False when Harbor lacks the provider's extra; each other
    flag is True, False, or None when the provider decides per sandbox (Docker's
    GPUs depend on the host).
    """
    from harbor.environments.factory import _load_environment_class
    from harbor.models.environment_type import EnvironmentType

    try:
        cls = _load_environment_class(EnvironmentType(name))
    except ImportError:
        return {"installed": False, "gpus": None, "no_network": None, "allowlist": None}
    try:
        caps = object.__new__(cls).capabilities  # a class constant for most providers
    except Exception:  # noqa: BLE001 - one that reads its instance decides per sandbox
        return {"installed": True, "gpus": None, "no_network": None, "allowlist": None}
    return {"installed": True, "gpus": caps.gpus, "no_network": caps.disable_internet, "allowlist": caps.network_allowlist}


def environment_problems(spec: EnvironmentSpec, sandbox: str) -> list[str]:
    """What of `spec` sandbox provider `sandbox` cannot give, one sentence each: every trial would fail on it."""
    caps = sandbox_capabilities(sandbox)
    problems = []
    if not caps["installed"]:
        problems.append(f"Harbor is installed without the {sandbox!r} sandbox provider")
    if spec.gpus and caps["gpus"] is False:
        problems.append(f"{sandbox} sandboxes have no GPUs, and the environment asks for {spec.gpus}")
    if spec.network_mode == "no-network" and caps["no_network"] is False:
        problems.append(f"{sandbox} sandboxes cannot be cut off the network")
    if spec.network_mode == "allowlist" and caps["allowlist"] is False:
        problems.append(f"{sandbox} sandboxes cannot restrict the network to an allowlist")
    return problems


# -- what the agent did --------------------------------------------------------------


def _text(message: Any) -> str:
    """An ATIF `message` or observation `content`: a string, or content parts of which the text ones count."""
    if message is None:
        return ""
    if isinstance(message, str):
        return message
    return "\n".join(part.get("text") or "" for part in message if isinstance(part, dict) and part.get("type") == "text")


def trajectory_messages(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    """An ATIF trajectory's steps as chat messages: an agent step's tool calls, then one `tool` message per observation result."""
    roles = {"system": "system", "user": "user", "agent": "assistant"}
    messages: list[dict[str, Any]] = []
    for step in trajectory.get("steps") or []:
        if step.get("is_copied_context"):
            continue
        message: dict[str, Any] = {"role": roles.get(step.get("source"), "user"), "content": _text(step.get("message"))}
        calls = step.get("tool_calls") or []
        if calls:
            message["tool_calls"] = [
                {"id": c.get("tool_call_id"), "type": "function", "function": {"name": c.get("function_name"), "arguments": c.get("arguments") or {}}}
                for c in calls
            ]
        messages.append(message)
        for result in (step.get("observation") or {}).get("results") or []:
            messages.append({"role": "tool", "tool_call_id": result.get("source_call_id"), "content": _text(result.get("content"))})
    return messages


def final_answer(messages: list[dict[str, Any]]) -> str:
    """The agent's last word: its last message with text."""
    return next((m["content"] for m in reversed(messages) if m["role"] == "assistant" and m["content"].strip()), "")


def find_trajectory(logs: Path) -> dict[str, Any]:
    """The agent's ATIF trajectory: `trajectory.json` in its logs, else the first one below them; `FileNotFoundError` without."""
    path = logs / "trajectory.json"
    if not path.is_file():
        path = next(iter(sorted(logs.rglob("trajectory.json"))), None) if logs.is_dir() else None
    if path is None:
        raise FileNotFoundError(f"the agent left no trajectory.json under {logs}")
    return json.loads(path.read_text())


# -- trying an environment -------------------------------------------------------------

PROBE_SH = """#!/bin/sh
# The arena's environment check: what the sandbox holds, one `name: value` line each.
echo "os: $( (. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME") || uname -s)"
echo "user: $(id -un 2>/dev/null || id -u)"
echo "workdir: $(pwd)"
for tool in python3 tmux git curl; do
  if command -v "$tool" >/dev/null 2>&1; then
    if [ "$tool" = tmux ]; then version=$(tmux -V 2>&1); else version=$("$tool" --version 2>&1 | head -n 1); fi
    echo "$tool: $version"
  else
    echo "$tool: missing"
  fi
done
"""
PROBE_SH += f'if [ -f {CLOCK_FILE} ]; then echo "clock: $(date +%s) $(tr -d \'\\n\' < {CLOCK_FILE})"; fi\n'
# The facts a check reports, by the name `PROBE_SH` prints each under.
PROBE_FACTS = ("os", "user", "workdir", "python3", "tmux", "git", "curl", "clock")


def probe_facts(output: str) -> dict[str, str]:
    """`PROBE_SH`'s `name: value` lines as facts, `clock` turned into the virtual time it read (or `unreadable`, never real time)."""
    facts = {}
    for line in output.splitlines():
        key, sep, value = line.partition(": ")
        if sep and key in PROBE_FACTS:
            facts[key] = value.strip()
    if "clock" in facts:
        try:
            read_at, _, raw = facts["clock"].partition(" ")
            clock = json.loads(raw)
            at = datetime.fromtimestamp(virtual_now(clock, float(read_at)), timezone.utc)
            facts["clock"] = f"{at:%Y-%m-%dT%H:%M:%SZ} (rate {clock['rate']:g})"
        except (ValueError, TypeError, KeyError, OverflowError, OSError):
            facts["clock"] = "unreadable"
    return facts


# What a check reports beside the probe's facts, by tool missing.
PROBE_NOTES = {
    "tmux": "tmux is missing: terminus-2 installs it at the start of every task, which needs a package manager and the network, and fails without; install it in the Dockerfile.",
    "python3": "python3 is missing: an agent that runs Python (most do) will fail; start from an image that has it.",
}


def probe_notes(facts: dict[str, str]) -> list[str]:
    """What of the probe's facts will trouble an agent, one sentence each."""
    notes = [note for tool, note in PROBE_NOTES.items() if facts.get(tool) == "missing"]
    if facts.get("clock") == "unreadable":
        notes.append(f"the clock at {CLOCK_FILE} cannot be read: something in the image or its healthcheck changed it, and every server that reads it will fail.")
    return notes


async def probe(spec: EnvironmentSpec, servers: dict[str, dict[str, Any]], context: Path | None, work: Path, sandbox: str) -> dict[str, Any]:
    """Start `spec` the way a task would, look inside, and stop it: `{ok, sandbox, seconds, facts, notes, error}`.

    One Harbor trial with the `oracle` agent, whose solution is `PROBE_SH`: the
    image is built or pulled, the healthcheck waited for and the network policy
    applied exactly as for an evaluation's rows, and the sandbox torn down.
    `context` is the environment's build context (its `environment/` folder).
    """
    import shutil
    import time
    import uuid

    from harbor.models.trial.config import AgentConfig, TaskConfig, TrialConfig, VerifierConfig
    from harbor.trial.trial import Trial

    problems = environment_problems(spec, sandbox)
    if problems:
        return {"ok": False, "sandbox": sandbox, "seconds": 0.0, "facts": {}, "notes": [], "error": "; ".join(problems)}
    name = f"check__{uuid.uuid4().hex[:8]}"
    task = work / "tasks" / name
    began = time.monotonic()
    try:
        for path, text in task_files(local_limits(spec, sandbox), servers).items():
            (task / path).parent.mkdir(parents=True, exist_ok=True)
            (task / path).write_text(text)
        (task / "instruction.md").write_text("The arena's environment check.\n")
        if context is not None and context.is_dir():
            shutil.copytree(context, task / "environment", dirs_exist_ok=True)
        (task / "environment").mkdir(exist_ok=True)
        (task / "solution").mkdir()
        (task / "solution" / "solve.sh").write_text(PROBE_SH)
        (task / "solution" / "solve.sh").chmod(0o755)
        # Harbor wants a task's tests even with the verifier off: one that runs nothing.
        (task / "tests").mkdir()
        (task / "tests" / "test.sh").write_text("#!/bin/sh\nexit 0\n")
        (task / "tests" / "test.sh").chmod(0o755)
        config = TrialConfig(
            task=TaskConfig(path=task),
            trial_name=name,
            trials_dir=work / "trials",
            agent=AgentConfig(name="oracle"),
            environment=environment_config(sandbox),
            verifier=VerifierConfig(disable=True),
        )
        async with local_slot(sandbox):
            result = await (await Trial.create(config)).run()
        out =work / "trials" / name / "agent" / "oracle.txt"
        facts = probe_facts(out.read_text() if out.is_file() else "")
        info = result.exception_info
        error = None
        if info is not None:
            said = " ".join((info.exception_message or "").split())
            error = f"{info.exception_type}: {said[:600]}" if said else info.exception_type
        elif not facts:
            code_file = work / "trials" / name / "agent" / "exit-code.txt"
            code = code_file.read_text().strip() if code_file.is_file() else ""
            if spec.workdir and not spec.dockerfile:
                # Every command starts in the working directory: one the prebuilt image lacks fails them all, agents' included.
                error = f"nothing could run in the sandbox (exit {code or '?'}): the image likely has no {spec.workdir}; create it in a Dockerfile (WORKDIR {spec.workdir}), or clear the working directory"
            else:
                error = f"the sandbox started, but nothing could run in it (exit {code or '?'})"
        notes = probe_notes(facts)
        return {"ok": error is None, "sandbox": sandbox, "seconds": round(time.monotonic() - began, 1), "facts": facts, "notes": notes, "error": error}
    except Exception as e:  # noqa: BLE001 - whatever stopped the check is its answer
        return {"ok": False, "sandbox": sandbox, "seconds": round(time.monotonic() - began, 1), "facts": {}, "notes": [], "error": f"{type(e).__name__}: {e}"}
    finally:
        shutil.rmtree(work / "tasks" / name, ignore_errors=True)
        shutil.rmtree(work / "trials" / name, ignore_errors=True)
