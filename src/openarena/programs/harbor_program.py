# License Apache 2.0: (c) 2026 Atena-Reply

"""Harbor program: each row a Harbor task, done by an agent the model under test drives in a sandbox.

Used for a dataset that declares a `harbor:` block, which takes its tasks
from one of two places:

- **Task folders** (`type: harbor_tasks`): each row is a Harbor task as
  Harbor defines it, `task.toml`, `instruction.md`, `environment/`, `tests/`,
  `solution/`, run as it is with Harbor's verifier on. The task's `tests/`
  score it; what they wrote (`reward.txt` / `reward.json`) comes back as the
  prediction's `rewards`, which the `harbor_verifier` reward reads.
- **Rows over an environment**: each row's question (its `input_template`,
  `{"question": ...}`), after the block's `instructions`, becomes the
  `instruction.md` of a task generated from the environment the block names
  (`openarena.harbor`, the environment's folder copied beside the run's
  config). Its `tests/` are the arena's default (`DEFAULT_TEST_SH`): the
  verifier collects the agent's answer into `/logs/verifier/` and the
  dataset's reward, serialized beside it (`tests/rewards.yaml`), scores
  it on the arena's side, Mirage sandbox and all, with no second sandbox.

Harbor runs one trial of each task: it starts the sandbox (E2B unless
`ARENA_HARBOR_SANDBOX` says otherwise) and lets the block's agent work with
the model the sweep's axis names. The prediction is Harbor's own ATIF
trajectory, as a framework data model (`HarborTrajectory`), with the
`answer` (what the agent wrote to `answer_file`, else its last message) and
the verifier's `rewards`. The arena's rewards read `answer` unless a reward
names masks of its own.

The model entry's params go to the agent where its options take them
(`agent_kwargs`): `temperature`, `reasoning_effort`, and, for an agent that
passes litellm call kwargs, `max_tokens`. A row whose trial left no
trajectory (the sandbox would not start, the agent crashed before its first
step) is logged with what Harbor said and predicts nothing, which scores 0:
one flaky sandbox does not cost the cell. A cell none of whose rows got an
answer fails, though (`NoAnswerGuard`): that is the sandbox, the image or a
key failing, not the model, and a 0 would rank it as if it were.

Each row's trial keeps its logs under the run's state, `harbor/trials/<trial>/`,
and a line in `harbor/trials.jsonl` (`TRIALS_LOG`) says which dataset, model
and question it was, what came of it and why: what the app's trial viewer
reads. The task folder is removed once its trial is done.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import synalinks
import yaml
from harbor.models.trajectories.trajectory import Trajectory
from pydantic import Field
from synalinks.src.backend import JsonDataModel, SymbolicDataModel
from synalinks.src.modules.module import Module

from openarena.config.entries import EnvironmentSpec
from openarena.harbor import (
    LOCAL_SANDBOX,
    environment_config,
    environment_toml,
    final_answer,
    find_trajectory,
    local_limits,
    local_slot,
    sandbox_options,
    trajectory_messages,
)
from openarena.programs.utils import safe_name

class HarborTrajectory(Trajectory, synalinks.DataModel):
    """What an agent did in a Harbor trial: Harbor's ATIF trajectory, as a framework data model.

    Harbor's own `Trajectory` (its `agent`, `steps`, `final_metrics`, ...),
    plus what the arena reads off the trial: the `answer`, and the scores the
    task's verifier gave.
    """

    answer: str = Field(default="", description="What the agent wrote to the answer file, else its last message")
    rewards: dict[str, float] | None = Field(
        default=None, description="What the task's verifier wrote (reward.txt as `reward`, or reward.json's keys); none when no verifier scored"
    )


OUTPUT_SCHEMA = HarborTrajectory.get_schema()
DEFAULT_SANDBOX = "e2b"
# One line per row's trial, beside the trials' folders.
TRIALS_LOG = "trials.jsonl"
# How much of Harbor's error a row's log line quotes; the trial's own logs keep all of it.
MAX_REASON = 400
# How much of the question a trial's line keeps.
MAX_QUESTION = 2000
# The arena's "no thinking" is `disable`; Harbor's agents call it `none`.
_EFFORTS = {"disable": "none"}

log = logging.getLogger("uvicorn.error")



def configured_sandbox() -> str:
    """The provider asked for: `ARENA_HARBOR_SANDBOX` (a Harbor environment type: e2b, docker, daytona, modal, ...), E2B by default.

    A name Harbor does not know raises `ValueError` with the ones it does:
    every trial would fail on it, and each failed row scores 0, so a typo
    would pass for every model failing every task.
    """
    from harbor.models.environment_type import EnvironmentType

    name = (os.environ.get("ARENA_HARBOR_SANDBOX") or DEFAULT_SANDBOX).strip().lower()
    known = [t.value for t in EnvironmentType]
    if name not in known:
        raise ValueError(f"Harbor has no sandbox {name!r}; set ARENA_HARBOR_SANDBOX to one of: {', '.join(known)}")
    return name


def ready(name: str) -> bool:
    """Whether provider `name` can start a sandbox: Harbor's own pre-flight (its credentials, Docker's daemon) passes."""
    from harbor.environments.factory import EnvironmentFactory
    from harbor.models.environment_type import EnvironmentType

    try:
        EnvironmentFactory.run_preflight(EnvironmentType(name))
    except (SystemExit, Exception):  # noqa: BLE001 - a provider's pre-flight exits, or raises, when it cannot start
        return False
    return True


def sandbox() -> str:
    """Where trials run: the configured provider, or the local Docker when that one lacks its credentials.

    A cloud provider without its key (E2B with no `E2B_API_KEY`) is no reason
    to run nothing: the tasks run on this machine's Docker instead, which
    needs none. `sandbox_issue` says when neither can start.
    """
    name = configured_sandbox()
    if name != LOCAL_SANDBOX and not ready(name):
        return LOCAL_SANDBOX
    return name


def sandbox_note() -> str | None:
    """One sentence when the tasks do not run where configured (the provider's key is missing), else None."""
    name = configured_sandbox()
    used = sandbox()
    return f"{name} has no credentials set, so tasks run on the local Docker instead" if used != name else None


def sandbox_issue() -> str | None:
    """Why no task can start at all, as one sentence, or None: an unknown provider or its options, or none with credentials and no Docker.

    The options are checked whichever provider ends up running: a malformed
    `ARENA_HARBOR_SANDBOX_OPTIONS` is a mistake to fix, not one the Docker
    fallback (which gets none) should hide.
    """
    try:
        name = configured_sandbox()
        sandbox_options()
    except ValueError as e:
        return str(e)
    used = sandbox()
    if ready(used):
        return None
    if used != name:
        return f"{name} has no credentials set, and the local Docker it falls back to is not available to the backend either"
    return f"the {name} sandbox provider cannot start: Harbor's pre-flight check refused it"


def docker_is_snap() -> bool:
    """Whether `docker` is the snap package, which reads neither /tmp nor a hidden folder of the home directory."""
    found = shutil.which("docker")
    return bool(found) and (found.startswith("/snap/") or os.path.realpath(found).startswith("/snap/") or os.path.realpath(found).endswith("/usr/bin/snap"))


def work_root(default: Path) -> Path:
    """Where Harbor's task folders and trials go: `ARENA_HARBOR_WORKDIR`, else `default`, unless a snap Docker cannot reach it.

    A snap Docker cannot read the run's state (under a hidden `.arena`) nor
    Harbor's temporary files in /tmp, so its tasks, trials and temporary
    files go under a visible `~/arena-harbor/` instead, one folder per
    `default` so runs stay apart.
    """
    if os.environ.get("ARENA_HARBOR_WORKDIR", "").strip():
        root = Path(os.environ["ARENA_HARBOR_WORKDIR"].strip()).expanduser()
    elif sandbox() == LOCAL_SANDBOX and docker_is_snap() and any(part.startswith(".") for part in Path(default).resolve().parts):
        root = Path.home() / "arena-harbor"
    else:
        return Path(default)
    tmp = root / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(tmp)  # Harbor writes its compose files with `tempfile`; Docker must read them.
    return root / hashlib.sha1(str(Path(default).resolve()).encode()).hexdigest()[:10]


# Where an `answer_file` lands among the trial's artifacts.
ANSWER_ARTIFACT = "answer.txt"


def instruction(row: dict[str, Any], spec: dict[str, Any]) -> str:
    """What the agent is told: the block's `instructions`, the row's question, and where to write its answer when it is asked to."""
    question = str(row.get("question") or "").strip()
    if not question:
        raise ValueError("the row has no question: the agent would be told nothing")
    parts = [str(spec.get("instructions") or "").strip(), question]
    if spec.get("answer_file"):
        parts.append(f"When you are done, write your final answer, and nothing else, to the file {spec['answer_file']}.")
    return "\n\n".join(p for p in parts if p)


# A task folder scored by the arena's reward system: `tests/test.sh` collects the agent's answer into
# `/logs/verifier/answer.txt`, and `tests/rewards.yaml`, written like the workspace's own rewards.yaml
# (`rewards: {<name>: {type, config, direction}}`, models by their models.yaml key), says which rewards of the
# catalog score it, on the arena's side after the trial, against `tests/expected.txt` when the task has one. Each
# reward is one named score of the trial, as reward.json's keys are; the first is also `reward`.
REWARDS_FILE = "rewards.yaml"
EXPECTED = "expected.txt"

# The default `tests/` of a task generated from rows: Harbor's verifier collects the agent's answer, the arena scores it.
# The answer file, when the dataset names one, is copied where the trial's verifier logs are downloaded from; the
# number written is only that something was collected: the score is the dataset's reward, on the arena's side.
DEFAULT_TEST_SH = """#!/bin/sh
# The arena's default test: collect the agent's answer; the arena scores it with tests/rewards.yaml.
mkdir -p /logs/verifier
if [ -n "$ARENA_ANSWER_FILE" ] && [ -f "$ARENA_ANSWER_FILE" ]; then
  cp "$ARENA_ANSWER_FILE" /logs/verifier/answer.txt
  echo '{"answer_collected": 1}' > /logs/verifier/reward.json
else
  echo '{"answer_collected": 0}' > /logs/verifier/reward.json
fi
"""
COLLECTED_ANSWER = "answer.txt"


def task_toml(spec: dict[str, Any], provider: str) -> str:
    """A row's `task.toml`: the environment's section (with `provider`'s defaults, `local_limits`), the agent's timeout, the answer file to collect."""
    environment = EnvironmentSpec.model_validate(
        {k: v for k, v in spec["environment_entry"].items() if k in EnvironmentSpec.model_fields}
    )
    toml = environment_toml(local_limits(environment, provider), spec.get("mcp_servers") or {})
    if spec.get("timeout_sec"):
        toml += f"\n[agent]\ntimeout_sec = {float(spec['timeout_sec'])}\n"
    if spec.get("answer_file"):
        # Collected from the sandbox once the agent is done, into the trial's artifacts.
        toml += f"\n[[artifacts]]\nsource = {json.dumps(spec['answer_file'])}\ndestination = {json.dumps(ANSWER_ARTIFACT)}\n"
    # The default test collects the answer file; the verifier's own env tells it where.
    toml += f"\n[verifier.env]\nARENA_ANSWER_FILE = {json.dumps(spec.get('answer_file') or '')}\n"
    return toml


def write_task(folder: Path, row: dict[str, Any], spec: dict[str, Any], contexts: Path | None) -> Path:
    """The row as a Harbor task folder under `folder`: task.toml, instruction.md and the environment's build context."""
    told = instruction(row, spec)
    folder.mkdir(parents=True, exist_ok=True)
    environment = EnvironmentSpec.model_validate(
        {k: v for k, v in spec["environment_entry"].items() if k in EnvironmentSpec.model_fields}
    )
    (folder / "task.toml").write_text(task_toml(spec, sandbox()))
    (folder / "instruction.md").write_text(told + "\n")
    context = folder / "environment"
    source = contexts / spec["environment"] / "environment" if contexts else None
    if source is not None and source.is_dir():
        shutil.copytree(source, context, dirs_exist_ok=True)
    # Harbor wants the folder even for a prebuilt image.
    context.mkdir(exist_ok=True)
    if environment.dockerfile:
        (context / "Dockerfile").write_text(environment.dockerfile.rstrip("\n") + "\n")
    # The arena's default tests: collect the answer; the reward that scores it, serialized beside them.
    tests = folder / "tests"
    tests.mkdir(exist_ok=True)
    (tests / "test.sh").write_text(DEFAULT_TEST_SH)
    (tests / "test.sh").chmod(0o755)
    if spec.get("reward"):
        # Plain data (the config's lists may be wrappers YAML cannot write).
        block = json.loads(json.dumps(spec["reward"], default=list))
        kind = block.pop("name")
        (tests / REWARDS_FILE).write_text(
            "# The dataset's reward, which scores the collected answer on the arena's side (models as resolved for this run).\n"
            + yaml.safe_dump({"rewards": {spec.get("reward_name") or kind: {"type": kind, "config": block}}}, sort_keys=False, allow_unicode=True)
        )
    return folder


def task_reward(task: Path, resolve) -> dict[str, Any] | None:
    """A task folder's arena rewards: `{rewards: {name: spec}, expected}` from `tests/rewards.yaml` (and `tests/expected.txt`), or None.

    The file is written like the workspace's rewards.yaml, each entry read with
    the registry's own model (`RewardEntry`). `resolve(kwargs, where)` turns the
    models an entry names (models.yaml keys) into ids, refusing an unknown one;
    what is not a reward of the catalog is a `ValueError` saying what to fix.
    """
    from openarena.config.entries import RewardEntry
    from openarena.rewards import _REWARD_TYPES

    path = task / "tests" / REWARDS_FILE
    if not path.is_file():
        return None
    where = f"task {task.name!r}: tests/{REWARDS_FILE}"
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"{where} is not valid YAML: {e}") from None
    entries = data.get("rewards") if isinstance(data, dict) else None
    if not isinstance(entries, dict) or not entries:
        raise ValueError(f"{where} lists the rewards under `rewards:`, as the workspace's rewards.yaml does: `rewards: {{judge: {{type: lm_as_judge, config: {{...}}}}}}`")
    rewards = {}
    for name, raw in entries.items():
        try:
            entry = RewardEntry.model_validate(raw)
        except Exception as e:  # noqa: BLE001 - pydantic's ValidationError: the registry's own refusal
            raise ValueError(f"{where}: reward {name!r}: {str(e).splitlines()[-1].strip()}") from None
        if entry.type not in _REWARD_TYPES:
            raise ValueError(f"{where}: reward {name!r}: no reward {entry.type!r} in the catalog (lm_as_judge, exact_match, ...)")
        rewards[str(name)] = {"name": entry.type, **resolve(entry.config, f"tasks.{task.name}.tests.rewards.{name}.config")}
    expected = task / "tests" / EXPECTED
    return {"rewards": rewards, "expected": expected.read_text(errors="replace").strip() if expected.is_file() else None}


_ANSWER_SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"], "additionalProperties": False}


async def score_answer(scoring: dict[str, Any], answer: str) -> dict[str, float]:
    """Each of the task's arena rewards on the agent's `answer`, against `scoring["expected"]` when there is one: `{name: score}`, the first also as `reward`."""
    from openarena.rewards import get

    y_pred = JsonDataModel(json={"answer": answer}, schema=_ANSWER_SCHEMA)
    y_true = JsonDataModel(json={"answer": scoring["expected"]}, schema=_ANSWER_SCHEMA) if scoring.get("expected") is not None else None
    scores = {name: float(await get(spec)(y_true, y_pred) or 0.0) for name, spec in scoring["rewards"].items()}
    return {"reward": next(iter(scores.values())), **scores} if "reward" not in scores else scores


def copy_task(source: Path, folder: Path) -> Path:
    """A Harbor task folder copied for one trial, so the trial never writes into the source."""
    if not (source / "task.toml").is_file():
        raise ValueError(f"{source} is not a Harbor task: it has no task.toml")
    shutil.copytree(source, folder, dirs_exist_ok=True)
    return folder


def failure(result) -> str | None:
    """What Harbor said went wrong in a trial, as one line, or None."""
    info = result.exception_info
    if info is None:
        return None
    # A failed command quotes its whole command line first; what it printed says why, so the command goes.
    said = re.sub(r"Command: .*?(?=\. Return code|$)", "", info.exception_message or "", flags=re.S)
    said = re.sub(r"\s*(Stderr|Stdout): None\.?", "", said)
    said = " ".join(said.split())
    said = said if len(said) <= MAX_REASON else said[: MAX_REASON - 1] + "…"
    return f"{info.exception_type}" + (f": {said}" if said else "")


def verifier_rewards(result) -> dict[str, float] | None:
    """What the trial's verifier wrote, `{name: score}`, or None when no verifier scored it."""
    verified = getattr(result, "verifier_result", None)
    rewards = getattr(verified, "rewards", None) if verified is not None else None
    return {str(k): float(v) for k, v in rewards.items()} if rewards else None


def prediction(trial_dir: Path, result, answer_file: str | None = None) -> dict[str, Any]:
    """What the agent did: its ATIF trajectory from the downloaded logs, as a `HarborTrajectory`; `ValueError` saying why when there is none.

    With an `answer_file`, the answer is what the agent wrote there (the
    verifier's collected copy, else the artifact; empty when it wrote
    nothing); else its last message. `rewards` are the verifier's.
    """
    try:
        raw = find_trajectory(trial_dir / "agent")
        messages = trajectory_messages(raw)
    except (OSError, ValueError) as e:
        said = failure(result)
        raise ValueError("the Harbor trial left no trajectory" + (f": {said}" if said else f" ({e})")) from None
    collected = trial_dir / "verifier" / COLLECTED_ANSWER
    if collected.is_file():
        answer = collected.read_text(errors="replace").strip()
    elif answer_file:
        written = trial_dir / "artifacts" / ANSWER_ARTIFACT
        answer = written.read_text(errors="replace").strip() if written.is_file() else ""
    else:
        answer = final_answer(messages)
    try:
        trajectory = HarborTrajectory.model_validate({**raw, "answer": answer, "rewards": verifier_rewards(result)})
    except Exception as e:  # noqa: BLE001 - pydantic's ValidationError: a trajectory Harbor's own schema refuses
        raise ValueError(f"the agent's trajectory is not a valid ATIF trajectory: {str(e).splitlines()[0]}") from None
    return trajectory.model_dump(mode="json", exclude_none=True)


def agent_kwargs(agent: str, params: dict[str, Any] | None) -> dict[str, Any]:
    """The model entry's `params` as the options Harbor's `agent` takes; what it does not take is logged and left out."""
    from harbor.agents.factory import AgentFactory
    from harbor.models.agent.name import AgentName

    params = {k: _EFFORTS.get(v, v) if k == "reasoning_effort" else v for k, v in (params or {}).items() if v is not None}
    if not params:
        return {}
    options = AgentFactory.get_agent_class(AgentName(agent)).options_model
    fields = set(options.model_fields) if options else set()
    kwargs: dict[str, Any] = {}
    for key, value in params.items():
        if key in fields:
            kwargs[key] = value
        elif "llm_call_kwargs" in fields:
            kwargs.setdefault("llm_call_kwargs", {})[key] = value
    # A value the agent refuses (an effort level it does not know) is dropped, not the trial.
    refused = set(params) - set(kwargs) - (set(kwargs.get("llm_call_kwargs") or {}))
    if options is not None and kwargs:
        try:
            options.model_validate(kwargs)
        except Exception as e:  # noqa: BLE001 - pydantic's ValidationError, without importing it
            bad = {str(err["loc"][0]) for err in getattr(e, "errors", lambda: [])() if err.get("loc")}
            refused |= bad
            kwargs = {k: v for k, v in kwargs.items() if k not in bad}
    if refused:
        log.warning("Harbor agent %r does not take the model's %s; running without.", agent, ", ".join(sorted(refused)))
    return kwargs


async def run_trial(task: Path, trials: Path, name: str, agent: str, model: str | None, timeout_sec: float | None = None, kwargs: dict[str, Any] | None = None):
    """One Harbor trial of `task`, named `name` under `trials`, its verifier running the task's `tests/`; its `TrialResult`."""
    from harbor.models.trial.config import AgentConfig, TaskConfig, TrialConfig, VerifierConfig
    from harbor.trial.trial import Trial

    provider = sandbox()
    config = TrialConfig(
        task=TaskConfig(path=task),
        trial_name=name,
        trials_dir=trials,
        agent=AgentConfig(name=agent, model_name=model, override_timeout_sec=timeout_sec, kwargs=dict(kwargs or {})),
        environment=environment_config(provider),
        verifier=VerifierConfig(),
    )
    # On the local Docker, one trial at a time (`local_slot`): a batch's rows would otherwise start a container each.
    async with local_slot(provider):
        return await (await Trial.create(config)).run()


def model_calls(trajectory: dict[str, Any]) -> int:
    """The model calls an ATIF trajectory records: each agent step's `llm_call_count`, one when it does not say."""
    return sum(
        int(step.get("llm_call_count") or 1)
        for step in trajectory.get("steps") or []
        if step.get("source") == "agent" and not step.get("is_copied_context")
    )


def _seconds(timing) -> float:
    start, end = getattr(timing, "started_at", None), getattr(timing, "finished_at", None)
    return max((end - start).total_seconds(), 0.0) if start and end else 0.0


def _book(ledger, result, trial_dir: Path | None = None) -> None:
    """Add a trial's usage, as Harbor's agent reported it, to the ledger model's inference counters.

    Tokens and cost as the agent counted them. Calls are the trajectory's
    model calls, and their time the agent's whole run: a Harbor cell's
    latency is the agent's seconds per model call, the commands it ran in
    between included, which is what a task costs in waiting.
    """
    if ledger is None or result is None:
        return
    ctx = getattr(result, "agent_result", None)
    prompt = (ctx.n_input_tokens or 0) if ctx else 0
    completion = (ctx.n_output_tokens or 0) if ctx else 0
    try:
        calls = model_calls(find_trajectory(trial_dir / "agent")) if trial_dir is not None else 0
    except (OSError, ValueError):
        calls = 0
    for suffix, value in (
        ("prompt_tokens", prompt), ("completion_tokens", completion), ("tokens", prompt + completion),
        ("cost", (ctx.cost_usd or 0.0) if ctx else 0.0), ("calls", calls),
        ("elapsed_s", _seconds(getattr(result, "agent_execution", None)) if calls else 0.0),
    ):
        attr = f"inference_cumulated_{suffix}"
        setattr(ledger, attr, getattr(ledger, attr, 0) + value)


def _usage(result) -> dict[str, Any]:
    ctx = getattr(result, "agent_result", None)
    if ctx is None:
        return {}
    usage = {"input_tokens": ctx.n_input_tokens, "output_tokens": ctx.n_output_tokens, "cost_usd": ctx.cost_usd}
    return {k: v for k, v in usage.items() if v is not None}


class HarborTask(Module):
    """Ask each row of an agent the model drives in a sandbox; what the agent did is the output.

    Args:
        model (str): The litellm id the agent calls.
        spec (dict): `Config.harbor_spec`: the block, the environment's entry, its MCP connections, the dataset's name.
        contexts (str | None): Where the environments' folders were copied for the run.
        work_dir (str): Where task folders, trials and `trials.jsonl` are written.
        params (dict | None): The model entry's params, handed to the agent where it takes them.
        model_name (str | None): The model's registry name, for the trials' log.
    """

    def __init__(self, model, spec, contexts=None, work_dir="harbor", params=None, model_name=None, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.model_name = model_name or model
        self.params = dict(params or {})
        self.agent_kwargs = agent_kwargs(spec.get("agent") or "terminus-2", self.params)
        self.spec = dict(spec)
        self.contexts = str(contexts) if contexts else None
        self.work_dir = str(work_dir)
        # The rows tried, by outcome, and what the last unanswered one said: `NoAnswerGuard` reads them.
        self.rows_answered = 0
        self.rows_unanswered = 0
        self.last_failure: str | None = None
        # The agent calls the model itself, not through the framework, so the cell's token and cost metrics would read
        # nothing. This model is never called: each trial's usage is added to its counters (`_book`), which those
        # metrics sum like any program's.
        self.language_model = synalinks.LanguageModel(model=model)
        self.built = True

    async def compute_output_spec(self, inputs, training=False):
        return SymbolicDataModel(schema=OUTPUT_SCHEMA, name=self.name)

    def _record(self, line: dict[str, Any]) -> None:
        path = Path(self.work_dir) / TRIALS_LOG
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(line, default=str) + "\n")

    async def call(self, inputs, training=False):
        if inputs is None:
            return None
        work = Path(self.work_dir)
        # Tasks and trials where the sandbox can reach them; the trials' log stays in the run's state.
        space = work_root(work)
        row = inputs.get_json()
        # A task folder's row names it: its name says which task in the trial's name too.
        label = row.get("task") if self.spec.get("tasks") else self.spec.get("dataset")
        name = f"{safe_name(label or 'harbor')}__{safe_name(self.model_name)}__{uuid.uuid4().hex[:8]}"
        task = space / "tasks" / name
        line: dict[str, Any] = {
            "trial": name,
            "dir": str(space / "trials" / name),
            "dataset": self.spec.get("dataset"),
            "model": self.model_name,
            "agent": self.spec.get("agent") or "terminus-2",
            "question": str(row.get("question") or "")[:MAX_QUESTION],
            **({"task": row.get("task")} if self.spec.get("tasks") else {}),
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        began = time.monotonic()
        result = None
        try:
            if self.spec.get("tasks"):
                copy_task(Path(str(row.get("path") or "")), task)
            else:
                write_task(task, row, self.spec, Path(self.contexts) if self.contexts else None)
            result = await run_trial(
                task, space / "trials", name, line["agent"], self.model, self.spec.get("timeout_sec"), self.agent_kwargs
            )
            _book(self.language_model, result, space / "trials" / name)
            predicted = prediction(space / "trials" / name, result, self.spec.get("answer_file"))
            # A task scored by the arena's reward system: its tests collected the answer, its reward scores it here.
            scoring = (self.spec.get("task_rewards") or {}).get(str(row.get("task"))) if self.spec.get("tasks") else None
            if scoring is not None:
                try:
                    scores = await score_answer(scoring, predicted.get("answer") or "")
                except Exception as e:  # noqa: BLE001 - a reward that cannot score leaves the row at 0, and says why
                    scores, line["error"] = {"reward": 0.0}, f"the task's rewards.yaml could not score: {e}"
                predicted["rewards"] = {**(predicted.get("rewards") or {}), **scores}
        except Exception as e:  # noqa: BLE001 - a row that could not be answered scores 0; the cell goes on
            log.warning("Harbor task for %s answered nothing: %s", self.model, e)
            self.rows_unanswered += 1
            self.last_failure = str(e)
            self._record({**line, "status": "failed", "error": str(e), "seconds": round(time.monotonic() - began, 1), **(_usage(result) if result else {})})
            return None
        finally:
            shutil.rmtree(task, ignore_errors=True)
        self.rows_answered += 1
        # A trial the agent did not finish (its time ran out) still answered: what it did is scored, and the line says why it stopped.
        said = failure(result)
        if self.spec.get("answer_file") and not predicted.get("answer"):
            said = "; ".join(filter(None, [said, f"the agent wrote nothing to {self.spec['answer_file']}"]))
        self._record({
            **line, "status": "answered", "answer": predicted["answer"][:MAX_QUESTION], "steps": len(predicted.get("steps") or []),
            **({"rewards": predicted["rewards"]} if predicted.get("rewards") else {}),
            "seconds": round(time.monotonic() - began, 1), **({"error": said} if said else {}), **_usage(result),
        })
        return JsonDataModel(json=predicted, schema=OUTPUT_SCHEMA, name=self.name)

    def get_config(self):
        return {
            "model": self.model, "spec": self.spec, "contexts": self.contexts, "work_dir": self.work_dir,
            "params": self.params, "model_name": self.model_name, "name": self.name,
        }


class NoAnswerGuard(synalinks.callbacks.Callback):
    """Fail a Harbor cell none of whose rows got an answer, with what the last one said.

    Every row failing is the sandbox, the image or a key, not the model under
    test; scored 0 it would sit on the leaderboard as the model's result.
    Raised at the end of the cell's evaluation, it fails the cell's trial.
    """

    def __init__(self, tasks: list[HarborTask]):
        super().__init__()
        self.tasks = tasks

    def on_test_end(self, logs=None):
        from keras_tuner.errors import FailedTrialError

        answered = sum(t.rows_answered for t in self.tasks)
        unanswered = sum(t.rows_unanswered for t in self.tasks)
        if unanswered and not answered:
            last = next((t.last_failure for t in self.tasks if t.last_failure), None)
            raise FailedTrialError(f"none of the {unanswered} Harbor task(s) got an answer; the last: {last}")


def no_answer_guard(program) -> NoAnswerGuard | None:
    """The guard for `program`'s Harbor tasks, or None when it runs none."""
    tasks = [m for m in program._flatten_modules() if isinstance(m, HarborTask)]
    return NoAnswerGuard(tasks) if tasks else None


async def build_harbor(model_id: str, dataset, spec: dict, reward, metrics=None, contexts=None, work_dir="harbor", lm_kwargs=None, model_name=None):
    """Build (and compile) a program that asks each row of a Harbor agent driven by `model_id` with its entry's `lm_kwargs`."""
    sandbox()  # an unknown sandbox fails the cell, not each of its rows
    sandbox_options()  # and so do malformed options, whichever provider the rows run on
    inputs = synalinks.Input(
        schema=dataset.input_schema,
        data_model=dataset.input_data_model if dataset.input_schema is None else None,
    )
    module = HarborTask(
        model=model_id, spec=spec, contexts=contexts, work_dir=work_dir, params=lm_kwargs, model_name=model_name,
        name=f"harbor_{safe_name(model_id)}",
    )
    outputs = await module(inputs)
    program = synalinks.Program(inputs=inputs, outputs=outputs, name=f"harbor_{safe_name(model_id)}")
    program.compile(reward=reward, metrics=metrics or None)
    return program
