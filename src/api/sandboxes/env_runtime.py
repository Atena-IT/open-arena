# License Apache 2.0: (c) 2026 Athena-Reply
"""``src.api.sandboxes.env_runtime`` — Env-package runtimes for P2-4.

Implements two strategies for executing a pinned eval-environment package
inside a per-task sandbox and returning a Rubric reward in [0, 1]:

* :class:`PrimeVerifiersRuntime` — handles Prime-Intellect *verifiers*
  packages whose repository root contains ``env.py`` + ``pyproject.toml``.
  The runtime calls ``pip install -e .`` then executes
  ``python -c "import env; r = env.load_environment(); …"`` and extracts
  the scalar reward from the rollout output.

* :class:`HarborTaskRuntime` — handles Harbor-style tasks whose repository
  root contains a ``task.toml`` file (and optionally a ``tests/`` directory).
  The runtime runs ``python -m pytest tests/ …`` (or a custom entry-point
  from ``task.toml``) and maps the exit code + optional score line to a
  reward in [0, 1].

Both strategies share a lightweight :class:`SandboxSession` protocol so they
are unit-testable with a :class:`FakeSandboxSession` that returns canned
output without touching any real sandbox.

Integration
-----------
The fan-out path in :meth:`~src.api.service.ArenaAPIService._run_per_task_fan_out`
calls :func:`execute_env_package` when the resolved environment carries a
``local_path`` (i.e. a pinned external package — either ``github_repo`` or
``prime_environment_hub`` kind).  The function returns a
:class:`~src.api.ports.sandbox_provider.TaskResult` whose ``rows`` list
contains exactly one row with ``metric="reward", direction="max",
value=<reward_in_0_1>``.  This row is assembled into a
:class:`~open_arena_core.models.SubjectResult` identically to the inline
dataset+verifier per-task path (P2-2).

Warm OCI image
--------------
When ``policy.image`` is set, the runtime passes the image reference to the
sandbox bootstrap step instead of pulling a fresh image.  The caller (the
E2B provider or any future runtime) is expected to honour the hint; this
module only surfaces the field and documents it for the sandbox layer.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from open_arena_core import models as api
from src.api.ports.sandbox_provider import TaskResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SandboxSession protocol — the minimal surface a runtime needs from a sandbox
# ---------------------------------------------------------------------------


@runtime_checkable
class SandboxSession(Protocol):
    """Minimal session interface required by env-package runtimes.

    Implementations must expose:

    * :meth:`run_command` — execute a shell command and return
      ``(exit_code, stdout, stderr)``.
    * :meth:`write_file` — upload bytes to a remote path inside the sandbox.

    The E2B session exposes a compatible surface via the ``_E2BClient``
    helper in :mod:`src.api.sandboxes.e2b_provider`.  The
    :class:`FakeSandboxSession` below satisfies the protocol for unit tests.
    """

    def run_command(
        self,
        cmd: str,
        *,
        workdir: str | None = None,
    ) -> tuple[int, str, str]:
        """Run *cmd* and return ``(exit_code, stdout, stderr)``."""
        ...

    def write_file(self, remote_path: str, content: bytes) -> None:
        """Write *content* to *remote_path* inside the sandbox."""
        ...


# ---------------------------------------------------------------------------
# FakeSandboxSession — for unit tests only
# ---------------------------------------------------------------------------


class FakeSandboxSession:
    """In-process fake that implements :class:`SandboxSession` for tests.

    Args:
        responses: Ordered list of ``(exit_code, stdout, stderr)`` tuples
            returned by successive :meth:`run_command` calls.  When the
            list is exhausted all further calls return ``(0, "", "")``.
        written: Mutable list that accumulates ``(remote_path, content)``
            pairs recorded by :meth:`write_file`.
    """

    def __init__(
        self,
        responses: list[tuple[int, str, str]] | None = None,
        *,
        written: list[tuple[str, bytes]] | None = None,
    ) -> None:
        self._responses: list[tuple[int, str, str]] = list(responses or [])
        self._idx = 0
        self.written: list[tuple[str, bytes]] = written if written is not None else []
        self.commands: list[str] = []

    def run_command(
        self,
        cmd: str,
        *,
        workdir: str | None = None,
    ) -> tuple[int, str, str]:
        """Return the next canned response, or ``(0, "", "")`` when exhausted."""
        self.commands.append(cmd)
        if self._idx < len(self._responses):
            resp = self._responses[self._idx]
            self._idx += 1
            return resp
        return (0, "", "")

    def write_file(self, remote_path: str, content: bytes) -> None:
        """Record the write for later inspection in tests."""
        self.written.append((remote_path, content))


# ---------------------------------------------------------------------------
# Package shape detection
# ---------------------------------------------------------------------------

_PRIME_MARKERS = frozenset({"env.py", "pyproject.toml"})
_HARBOR_MARKERS = frozenset({"task.toml"})


def detect_package_shape(local_path: str) -> str:
    """Return ``"prime"`` or ``"harbor"`` based on the files present in *local_path*.

    Detection rules (in priority order):

    * ``env.py`` + ``pyproject.toml`` → ``"prime"``
    * ``task.toml`` → ``"harbor"``

    Falls back to ``"prime"`` when neither set of markers is present, so the
    caller can attempt execution and fail with a meaningful error.

    Args:
        local_path: Filesystem path to the checked-out/cached environment tree.

    Returns:
        ``"prime"`` or ``"harbor"``.
    """
    root = Path(local_path)
    present = {p.name for p in root.iterdir() if p.is_file()} if root.is_dir() else set()

    if _PRIME_MARKERS.issubset(present):
        logger.debug("env_runtime: detected Prime verifiers package at %s", local_path)
        return "prime"

    if _HARBOR_MARKERS.issubset(present):
        logger.debug("env_runtime: detected Harbor task at %s", local_path)
        return "harbor"

    # Fallback — attempt Prime path
    logger.warning(
        "env_runtime: could not detect package shape at %s (found: %s); "
        "defaulting to 'prime'.",
        local_path,
        sorted(present),
    )
    return "prime"


# ---------------------------------------------------------------------------
# Reward extraction helpers
# ---------------------------------------------------------------------------

def _extract_reward_from_output(stdout: str) -> float | None:
    """Scan *stdout* for a reward line and return the parsed float, or ``None``.

    Recognised patterns (case-insensitive, in priority order):

    1. ``reward: <float>``  (e.g. ``reward: 0.75``)
    2. ``score: <float>``
    3. ``accuracy: <float>``
    4. Last bare float on any line.
    """
    import re

    patterns = [
        r"(?i)reward\s*[:=]\s*([0-9]*\.?[0-9]+)",
        r"(?i)score\s*[:=]\s*([0-9]*\.?[0-9]+)",
        r"(?i)accuracy\s*[:=]\s*([0-9]*\.?[0-9]+)",
    ]
    for pattern in patterns:
        m = re.search(pattern, stdout)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue

    # Last bare float
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        try:
            v = float(line)
            return v
        except ValueError:
            continue

    return None


def _clamp_reward(value: float) -> float:
    """Clamp *value* to [0.0, 1.0]."""
    return max(0.0, min(1.0, value))


# ---------------------------------------------------------------------------
# Prime-Intellect verifiers runtime
# ---------------------------------------------------------------------------

_PRIME_REMOTE_DIR = "/arena/prime_env"
_PRIME_RUNNER_SCRIPT = """\
import sys
try:
    import env as _env_mod
    env_def = _env_mod.load_environment()

    # Extract reward from several common Prime verifier return shapes:
    # 1. dict with "reward" key (most common documented shape)
    # 2. object attribute .reward
    # 3. callable rollout object — call .run_rollout() then check above
    # 4. bare float / int
    reward = None
    if isinstance(env_def, dict):
        if "reward" in env_def:
            reward = float(env_def["reward"])
        elif "score" in env_def:
            reward = float(env_def["score"])
    elif hasattr(env_def, "reward"):
        reward = float(env_def.reward)
    elif hasattr(env_def, "score"):
        reward = float(env_def.score)
    elif hasattr(env_def, "run_rollout"):
        rollout = env_def.run_rollout()
        if isinstance(rollout, dict):
            reward = float(rollout.get("reward", rollout.get("score", 0.0)))
        elif hasattr(rollout, "reward"):
            reward = float(rollout.reward)
        elif hasattr(rollout, "score"):
            reward = float(rollout.score)
        elif rollout is not None:
            try:
                reward = float(rollout)
            except (TypeError, ValueError):
                reward = 0.0
    else:
        try:
            reward = float(env_def)
        except (TypeError, ValueError):
            reward = 0.0

    if reward is None:
        reward = 0.0
    print(f"reward: {reward}")
    sys.exit(0)
except Exception as exc:
    print(f"ERROR: {exc}", file=sys.stderr)
    sys.exit(1)
"""


class PrimeVerifiersRuntime:
    """Execute a Prime-Intellect verifiers package inside a :class:`SandboxSession`.

    Lifecycle (steps run inside the sandbox via :meth:`SandboxSession.run_command`):

    1. ``mkdir -p <remote_dir>``
    2. Upload the local package tree (tar archive) to the sandbox.
    3. ``pip install --quiet -e <remote_dir>`` to install the package.
    4. Upload a small Python runner that calls ``env.load_environment()`` and
       prints ``reward: <float>`` to stdout.
    5. Run the runner and parse the reward from stdout.

    When ``policy.image`` is set, the image reference is logged; the actual
    image selection is the responsibility of the sandbox provider (e.g. E2B
    template selection).

    Args:
        session: A :class:`SandboxSession`-compatible object.
        policy: Optional :class:`~open_arena_core.models.SandboxPolicy`.
    """

    def __init__(
        self,
        session: SandboxSession,
        policy: api.SandboxPolicy | None = None,
    ) -> None:
        self._session = session
        self._policy = policy

    def run(self, local_path: str, *, dataset_name: str, model_key: str) -> TaskResult:
        """Install and run the Prime verifiers package; return a :class:`TaskResult`.

        Args:
            local_path: Filesystem path to the checked-out package directory
                (must contain ``env.py`` + ``pyproject.toml``).
            dataset_name: The environment id string used as the ``dataset``
                field in the result row.
            model_key: The model runtime id string used as the ``model``
                field in the result row.

        Returns:
            A :class:`TaskResult` with one row:
            ``{"dataset": dataset_name, "model": model_key, "metric": "reward",
            "value": <reward>, "direction": "max"}``.
        """
        if self._policy and self._policy.image:
            logger.info(
                "PrimeVerifiersRuntime: warm OCI image hint: %s (forwarded to sandbox provider).",
                self._policy.image,
            )

        timeout_s = int((self._policy.limits or {}).get("timeout_seconds", 300)) if self._policy else 300

        remote_dir = _PRIME_REMOTE_DIR
        exit_code, _, stderr = self._session.run_command(f"mkdir -p {remote_dir}")
        if exit_code != 0:
            logger.warning("PrimeVerifiersRuntime: mkdir failed: %s", stderr)

        # Upload package as tar archive
        pkg_archive = _tar_local_path(local_path)
        remote_archive = f"{remote_dir}/package.tar.gz"
        self._session.write_file(remote_archive, pkg_archive)

        # Extract
        exit_code, _, stderr = self._session.run_command(
            f"tar -xzf {remote_archive} -C {remote_dir} --strip-components=1",
            workdir=remote_dir,
        )
        if exit_code != 0:
            # Extraction failed — try running without strip (flat layout)
            exit_code, _, stderr = self._session.run_command(
                f"tar -xzf {remote_archive} -C {remote_dir}",
                workdir=remote_dir,
            )

        # Install the package
        exit_code, _, stderr = self._session.run_command(
            f"pip install --quiet -e {remote_dir}",
        )
        if exit_code != 0:
            logger.warning(
                "PrimeVerifiersRuntime: pip install returned %d: %s",
                exit_code, stderr,
            )

        # Upload runner script
        runner_remote = f"{remote_dir}/_oa_runner.py"
        self._session.write_file(runner_remote, _PRIME_RUNNER_SCRIPT.encode())

        # Run runner
        exit_code, stdout, stderr = self._session.run_command(
            f"python {runner_remote}",
            workdir=remote_dir,
        )

        reward = _extract_reward_from_output(stdout)
        if reward is None:
            if exit_code == 0:
                reward = 0.0
            else:
                logger.warning(
                    "PrimeVerifiersRuntime: runner exited %d, no reward in stdout=%r stderr=%r; "
                    "reward=0.0",
                    exit_code, stdout, stderr,
                )
                reward = 0.0
        else:
            reward = _clamp_reward(reward)

        logger.info(
            "PrimeVerifiersRuntime: reward=%.4f for dataset=%r model=%r",
            reward, dataset_name, model_key,
        )

        row: dict[str, Any] = {
            "dataset": dataset_name,
            "model": model_key,
            "metric": "reward",
            "value": reward,
            "direction": "max",
        }
        meta: dict[str, Any] = {
            "runtime": "prime_verifiers",
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": stderr,
        }
        return TaskResult(rows=[row], scratch_tag="", meta=meta)


# ---------------------------------------------------------------------------
# Harbor-style task runtime
# ---------------------------------------------------------------------------

_HARBOR_REMOTE_DIR = "/arena/harbor_task"
_HARBOR_DEFAULT_TEST_DIR = "tests"


class HarborTaskRuntime:
    """Execute a Harbor-style task package inside a :class:`SandboxSession`.

    A Harbor task is identified by a ``task.toml`` file in the repository
    root.  The runtime reads the optional ``[run]`` section for a custom
    entry-point command; falls back to ``python -m pytest tests/ -v`` when
    no entry-point is configured.

    Lifecycle:

    1. Upload the local package tree (tar archive) to the sandbox.
    2. Extract.
    3. Install dependencies listed in ``task.toml``'s ``[dependencies]``
       section (if any) via ``pip install``.
    4. Run the entry-point or the default pytest command.
    5. Map exit code → reward (0 = success → 1.0, non-zero → 0.0).
       If the output contains a ``score:`` or ``reward:`` line, that value
       is used instead of the binary exit-code mapping.

    Args:
        session: A :class:`SandboxSession`-compatible object.
        policy: Optional :class:`~open_arena_core.models.SandboxPolicy`.
    """

    def __init__(
        self,
        session: SandboxSession,
        policy: api.SandboxPolicy | None = None,
    ) -> None:
        self._session = session
        self._policy = policy

    def run(self, local_path: str, *, dataset_name: str, model_key: str) -> TaskResult:
        """Upload, build, and run the Harbor task; return a :class:`TaskResult`.

        Args:
            local_path: Filesystem path to the checked-out task directory
                (must contain ``task.toml``).
            dataset_name: Used as the ``dataset`` field in the result row.
            model_key: Used as the ``model`` field in the result row.

        Returns:
            A :class:`TaskResult` with one reward row.
        """
        if self._policy and self._policy.image:
            logger.info(
                "HarborTaskRuntime: warm OCI image hint: %s (forwarded to sandbox provider).",
                self._policy.image,
            )

        remote_dir = _HARBOR_REMOTE_DIR
        self._session.run_command(f"mkdir -p {remote_dir}")

        # Upload + extract
        pkg_archive = _tar_local_path(local_path)
        remote_archive = f"{remote_dir}/package.tar.gz"
        self._session.write_file(remote_archive, pkg_archive)

        exit_code, _, _ = self._session.run_command(
            f"tar -xzf {remote_archive} -C {remote_dir} --strip-components=1",
            workdir=remote_dir,
        )
        if exit_code != 0:
            self._session.run_command(
                f"tar -xzf {remote_archive} -C {remote_dir}",
                workdir=remote_dir,
            )

        # Read task.toml for custom entry-point and dependencies
        task_toml = _read_task_toml(local_path)
        deps: list[str] = task_toml.get("dependencies", {}).get("packages", [])
        entry_cmd: str = task_toml.get("run", {}).get("command", "")

        # Install declared dependencies
        if deps:
            pip_cmd = "pip install --quiet " + " ".join(deps)
            exit_code, _, stderr = self._session.run_command(pip_cmd)
            if exit_code != 0:
                logger.warning(
                    "HarborTaskRuntime: dependency install returned %d: %s",
                    exit_code, stderr,
                )

        # Choose run command
        if entry_cmd:
            run_cmd = entry_cmd
        else:
            run_cmd = f"python -m pytest {_HARBOR_DEFAULT_TEST_DIR}/ -v --tb=short"

        exit_code, stdout, stderr = self._session.run_command(
            run_cmd,
            workdir=remote_dir,
        )

        # Try to extract a reward from output first
        reward = _extract_reward_from_output(stdout + "\n" + stderr)
        if reward is not None:
            reward = _clamp_reward(reward)
        else:
            # Binary mapping: exit 0 → 1.0 (all tests passed), else → 0.0
            reward = 1.0 if exit_code == 0 else 0.0

        logger.info(
            "HarborTaskRuntime: reward=%.4f (exit=%d) for dataset=%r model=%r",
            reward, exit_code, dataset_name, model_key,
        )

        row: dict[str, Any] = {
            "dataset": dataset_name,
            "model": model_key,
            "metric": "reward",
            "value": reward,
            "direction": "max",
        }
        meta: dict[str, Any] = {
            "runtime": "harbor_task",
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": stderr,
        }
        return TaskResult(rows=[row], scratch_tag="", meta=meta)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def execute_env_package(
    local_path: str,
    session: SandboxSession,
    *,
    dataset_name: str,
    model_key: str,
    policy: api.SandboxPolicy | None = None,
    scratch_tag: str = "",
) -> TaskResult:
    """Detect the env-package shape and execute it; return a :class:`TaskResult`.

    This is the single entry-point called by the per-task fan-out path in
    :meth:`~src.api.service.ArenaAPIService._run_per_task_fan_out` when the
    resolved environment has a ``local_path`` (pinned external package).

    Detection:

    * Files at *local_path* contain ``env.py`` + ``pyproject.toml``
      → :class:`PrimeVerifiersRuntime`
    * Files at *local_path* contain ``task.toml``
      → :class:`HarborTaskRuntime`

    Args:
        local_path: Local filesystem path to the pinned package tree.
        session: A :class:`SandboxSession`-compatible object.
        dataset_name: Environment id used as the row ``dataset`` key.
        model_key: Model runtime id used as the row ``model`` key.
        policy: Optional :class:`~open_arena_core.models.SandboxPolicy`.
            ``policy.image`` signals a warm OCI image reference.
        scratch_tag: Opaque label for traceability; included in the
            returned :class:`~src.api.ports.sandbox_provider.TaskResult`.

    Returns:
        A :class:`TaskResult` whose ``rows`` list contains exactly one
        dict with keys ``dataset``, ``model``, ``metric="reward"``,
        ``value`` in ``[0, 1]``, ``direction="max"``.
    """
    shape = detect_package_shape(local_path)

    if shape == "harbor":
        runtime: PrimeVerifiersRuntime | HarborTaskRuntime = HarborTaskRuntime(
            session=session, policy=policy
        )
    else:
        runtime = PrimeVerifiersRuntime(session=session, policy=policy)

    result = runtime.run(local_path, dataset_name=dataset_name, model_key=model_key)
    # Propagate scratch_tag from the fan-out caller
    return TaskResult(rows=result.rows, scratch_tag=scratch_tag, meta=result.meta)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _tar_local_path(local_path: str) -> bytes:
    """Create an in-memory tar.gz of *local_path* and return the bytes.

    Returns an empty tar.gz archive when *local_path* does not exist (e.g.
    in tests where ``local_path`` is synthesised from Hub metadata and the
    directory has not been created).  A :func:`logging.warning` is emitted
    in that case so the issue is visible in logs before the subsequent sandbox
    extraction step fails.

    The archive is created with ``arcname=.`` so extraction with
    ``tar -xzf archive.tar.gz -C dest --strip-components=1`` replicates the
    source tree one level below *dest*.
    """
    import io
    import tarfile

    root = Path(local_path)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        if root.is_dir():
            tar.add(str(root), arcname=".")
        else:
            logger.warning(
                "_tar_local_path: local_path %r does not exist or is not a directory; "
                "producing an empty archive — pip install will likely fail inside the sandbox.",
                local_path,
            )
    buf.seek(0)
    return buf.read()


def _read_task_toml(local_path: str) -> dict[str, Any]:
    """Parse ``task.toml`` from *local_path* and return its contents as a dict.

    Returns an empty dict when the file is absent or unparseable.
    """
    task_toml_path = Path(local_path) / "task.toml"
    if not task_toml_path.exists():
        return {}
    try:
        import tomllib  # Python ≥ 3.11 stdlib
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            logger.warning(
                "HarborTaskRuntime: cannot parse task.toml — "
                "neither 'tomllib' (stdlib, Python ≥ 3.11) nor 'tomli' is available."
            )
            return {}
    try:
        return tomllib.loads(task_toml_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("HarborTaskRuntime: failed to parse task.toml: %s", exc)
        return {}
