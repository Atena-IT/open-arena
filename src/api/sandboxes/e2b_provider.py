# License Apache 2.0: (c) 2026 Athena-Reply
"""E2B-compatible SandboxProvider adapter (WS6, issue #40).

This module implements :class:`E2BSandboxProvider`, which executes an Open
Arena evaluation sweep inside an E2B-compatible sandbox.

**SDK isolation** — all calls into the E2B Python SDK are channelled through
the private :class:`_E2BClient` helper.  When the CubeSandbox backend becomes
available, only :class:`_E2BClient` needs updating; the lifecycle logic in
:meth:`E2BSandboxProvider.run` stays the same.

Environment variables
---------------------
``E2B_API_KEY``
    Required API key for the E2B / CubeSandbox service.  A missing or empty
    value raises :class:`E2BSandboxConfigError` when ``run()`` is called.

Usage
-----
Select this adapter by setting ``OPEN_ARENA_SANDBOX=e2b`` and installing the
optional dependency group::

    uv sync --extra e2b

Then pass the adapter to the registry (handled automatically by
:func:`~src.api.registry.build_adapters`) or instantiate directly::

    provider = E2BSandboxProvider()
    result = provider.run(config_path, policy=policy)
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

from open_arena_core import models as api
from src.api.ports.sandbox_provider import SandboxProvider, TaskResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class E2BSandboxError(RuntimeError):
    """Raised when the E2B sandbox lifecycle fails."""


class E2BSandboxConfigError(ValueError):
    """Raised when required E2B configuration (e.g. API key) is missing."""


# ---------------------------------------------------------------------------
# E2B SDK thin wrapper — swap this class to change the backend host
# ---------------------------------------------------------------------------

class _E2BClient:
    """Thin wrapper around the E2B Python SDK.

    All SDK imports are deferred to :meth:`__init__` so the module can be
    imported even when the ``e2b`` extra is not installed (the import error
    is raised only when an instance is constructed).

    To swap in the CubeSandbox backend, replace this class with one that
    speaks the CubeSandbox API while exposing the same public methods:
    :meth:`create`, :meth:`upload_file`, :meth:`run_command`, and
    :meth:`kill`.
    """

    # Default sandbox template — can be overridden via bootstrap["template"]
    _DEFAULT_TEMPLATE = "base"

    def __init__(self, api_key: str, *, timeout: int = 300, template: str | None = None) -> None:
        try:
            from e2b import Sandbox  # type: ignore[import-untyped]
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "The 'e2b' package is required for E2BSandboxProvider.  "
                "Install it with: uv sync --extra e2b"
            ) from exc

        self._Sandbox = Sandbox
        self._api_key = api_key
        self._timeout = timeout
        self._template = template or self._DEFAULT_TEMPLATE
        self._sandbox: Any = None  # populated by create()

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def create(self) -> None:
        """Create and start the sandbox instance."""
        logger.debug("Creating E2B sandbox (template=%r, timeout=%ds)", self._template, self._timeout)
        self._sandbox = self._Sandbox(
            template=self._template,
            api_key=self._api_key,
            timeout=self._timeout,
        )
        logger.info("E2B sandbox created: id=%s", getattr(self._sandbox, "sandbox_id", "<unknown>"))

    def upload_file(self, local_path: Path, remote_path: str) -> None:
        """Upload a local file into the sandbox filesystem."""
        if self._sandbox is None:
            raise E2BSandboxError("Sandbox not created; call create() first.")
        with local_path.open("rb") as fh:
            self._sandbox.files.write(remote_path, fh)
        logger.debug("Uploaded %s -> sandbox:%s", local_path, remote_path)

    def run_command(self, cmd: str, *, workdir: str | None = None) -> tuple[int, str, str]:
        """Run *cmd* inside the sandbox.

        Returns
        -------
        tuple[int, str, str]
            ``(exit_code, stdout, stderr)``
        """
        if self._sandbox is None:
            raise E2BSandboxError("Sandbox not created; call create() first.")
        kwargs: dict[str, Any] = {}
        if workdir:
            kwargs["cwd"] = workdir
        result = self._sandbox.commands.run(cmd, **kwargs)
        exit_code = getattr(result, "exit_code", 0)
        stdout = getattr(result, "stdout", "") or ""
        stderr = getattr(result, "stderr", "") or ""
        return int(exit_code), stdout, stderr

    def kill(self) -> None:
        """Terminate and clean up the sandbox."""
        if self._sandbox is None:
            return
        try:
            self._sandbox.kill()
            logger.info("E2B sandbox killed.")
        except Exception as exc:  # pragma: no cover — best-effort teardown
            logger.warning("E2B sandbox kill raised an error (ignored): %s", exc)
        finally:
            self._sandbox = None


# ---------------------------------------------------------------------------
# _E2BClientSession — SandboxSession adapter wrapping _E2BClient
# ---------------------------------------------------------------------------


class _E2BClientSession:
    """Adapts :class:`_E2BClient` to the
    :class:`~src.api.sandboxes.env_runtime.SandboxSession` protocol.

    The E2B SDK's ``upload_file`` takes a local filesystem :class:`~pathlib.Path`
    rather than raw bytes, so this adapter writes the bytes to a temporary file
    first and then calls ``client.upload_file(tmp_path, remote_path)``.

    Args:
        client: An already-created (``create()``-called) :class:`_E2BClient`
            instance owned by the caller.  This adapter does *not* call
            ``create()`` or ``kill()`` — the lifecycle is the caller's
            responsibility.
    """

    def __init__(self, client: _E2BClient) -> None:
        self._client = client

    def run_command(
        self,
        cmd: str,
        *,
        workdir: str | None = None,
    ) -> tuple[int, str, str]:
        """Delegate to :meth:`_E2BClient.run_command`."""
        return self._client.run_command(cmd, workdir=workdir)

    def write_file(self, remote_path: str, content: bytes) -> None:
        """Write *content* to *remote_path* inside the E2B sandbox.

        Translates the bytes-based
        :class:`~src.api.sandboxes.env_runtime.SandboxSession` interface to the
        path-based ``_E2BClient.upload_file`` API via a temporary file.
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp:
            tmp_path = Path(tmp.name)
            tmp_path.write_bytes(content)
        try:
            self._client.upload_file(tmp_path, remote_path)
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:  # pragma: no cover — best-effort cleanup
                pass


# ---------------------------------------------------------------------------
# SandboxProvider adapter
# ---------------------------------------------------------------------------

class E2BSandboxProvider(SandboxProvider):
    """Execute an Open Arena sweep inside an E2B-compatible sandbox.

    Lifecycle (in order)
    --------------------
    1. **Create** — instantiate the E2B sandbox; honour
       ``policy.limits["timeout_seconds"]`` (default 300 s) and
       ``policy.bootstrap["template"]``.
    2. **Bootstrap** — run each command listed in
       ``policy.bootstrap["commands"]`` (if any).
    3. **Upload** — copy the YAML config to ``/arena/config.yaml`` inside
       the sandbox.
    4. **Install** — run ``pip install open-arena`` (or the package list in
       ``policy.bootstrap["packages"]``) so the ``arena`` CLI is available.
    5. **Execute** — run ``arena --config /arena/config.yaml`` and capture
       stdout/stderr.
    6. **Parse** — extract the JSON result from stdout (the sweep emits a
       JSON object as its last line).
    7. **Teardown** — run each command listed in ``policy.teardown["commands"]``
       (if any).
    8. **Kill** — forcibly terminate the sandbox.

    Policy mapping
    --------------
    ``policy.limits``
        - ``timeout_seconds`` (int, default 300) → E2B sandbox timeout
        - ``memory_mb`` / ``cpu`` are passed through to bootstrap info but
          are not yet directly supported by E2B's public SDK; they are logged
          for forward-compatibility with CubeSandbox.
    ``policy.bootstrap``
        - ``template`` (str) → sandbox image/template identifier
        - ``packages`` (list[str]) → extra pip packages to install
        - ``commands`` (list[str]) → shell commands to run before upload
        - ``env`` (dict[str, str]) → environment variables exported before
          each command
    ``policy.teardown``
        - ``commands`` (list[str]) → shell commands to run after execution
    ``policy.isolation_mode``
        - ``none`` / ``container`` → standard E2B microVM (default)
        - ``vm`` → same; logged as a note because E2B is always VM-isolated
        - ``remote`` → same; the sandbox is inherently remote
    """

    _REMOTE_CONFIG = "/arena/config.yaml"
    _REMOTE_RESULT = "/arena/result.json"
    _DEFAULT_PACKAGE = "open-arena"

    @contextmanager
    def open_session(
        self,
        policy: api.SandboxPolicy | None = None,
    ) -> Generator[_E2BClientSession, None, None]:
        """Open an E2B-backed :class:`_E2BClientSession` for env-package runtimes (P2-4).

        Creates a dedicated E2B sandbox instance, yields an
        :class:`_E2BClientSession` adapter (which satisfies the
        :class:`~src.api.sandboxes.env_runtime.SandboxSession` protocol), then
        kills the sandbox on exit — even if an exception is raised inside the
        ``with`` block.

        Args:
            policy: Optional :class:`~open_arena_core.models.SandboxPolicy`.
                ``policy.limits["timeout_seconds"]`` controls the sandbox
                lifetime (default 300 s); ``policy.bootstrap["template"]``
                selects the E2B template / warm OCI image.

        Yields:
            An :class:`_E2BClientSession` backed by a live E2B microVM.

        Raises:
            E2BSandboxConfigError: When ``E2B_API_KEY`` is missing/empty.
            E2BSandboxError: When sandbox creation fails.
        """
        api_key = os.getenv("E2B_API_KEY", "").strip()
        if not api_key:
            raise E2BSandboxConfigError(
                "E2B_API_KEY environment variable is not set.  "
                "Set it to your E2B / CubeSandbox API key, or use "
                "LocalSandboxProvider (OPEN_ARENA_SANDBOX=local) for "
                "in-process execution."
            )

        limits = (policy.limits or {}) if policy else {}
        bootstrap = (policy.bootstrap or {}) if policy else {}
        timeout: int = int(limits.get("timeout_seconds", 300))
        template: str | None = bootstrap.get("template") or (policy.image if policy else None)

        client = _E2BClient(api_key, timeout=timeout, template=template)
        client.create()
        try:
            yield _E2BClientSession(client)
        finally:
            client.kill()

    def run(
        self,
        config_path: Path,
        *,
        policy: api.SandboxPolicy | None = None,
    ) -> dict[str, Any]:
        """Execute the sweep described by *config_path* inside an E2B sandbox.

        Args:
            config_path: Path to the pre-rendered YAML run config.
            policy: Optional :class:`~src.api.models.SandboxPolicy`.
                When ``None`` or ``policy.enabled is False``, falls back to a
                clear error rather than silently running in-process (the
                ``LocalSandboxProvider`` handles that case).

        Returns:
            Result dict with at minimum a ``"rows"`` key, matching the shape
            produced by :class:`~src.api.ports.sandbox_provider.LocalSandboxProvider`.

        Raises:
            E2BSandboxConfigError: When ``E2B_API_KEY`` is missing/empty, or
                when the policy explicitly disables sandbox execution.
            E2BSandboxError: When any sandbox lifecycle step fails.
        """
        # ------------------------------------------------------------------ #
        # 0. Validate configuration                                           #
        # ------------------------------------------------------------------ #
        if policy is not None and policy.enabled is False:
            raise E2BSandboxConfigError(
                "E2BSandboxProvider requires an enabled SandboxPolicy "
                "(policy.enabled=False was passed).  "
                "Use LocalSandboxProvider for in-process execution."
            )

        api_key = os.getenv("E2B_API_KEY", "").strip()
        if not api_key:
            raise E2BSandboxConfigError(
                "E2B_API_KEY environment variable is not set.  "
                "Set it to your E2B / CubeSandbox API key, or use "
                "LocalSandboxProvider (OPEN_ARENA_SANDBOX=local) for "
                "in-process execution."
            )

        # ------------------------------------------------------------------ #
        # 1. Resolve policy knobs                                             #
        # ------------------------------------------------------------------ #
        limits = (policy.limits or {}) if policy else {}
        bootstrap = (policy.bootstrap or {}) if policy else {}
        teardown_cfg = (policy.teardown or {}) if policy else {}
        isolation_mode = policy.isolation_mode if policy else None

        timeout: int = int(limits.get("timeout_seconds", 300))
        template: str | None = bootstrap.get("template")
        extra_packages: list[str] = list(bootstrap.get("packages") or [])
        bootstrap_commands: list[str] = list(bootstrap.get("commands") or [])
        bootstrap_env: dict[str, str] = dict(bootstrap.get("env") or {})
        teardown_commands: list[str] = list(teardown_cfg.get("commands") or [])

        if isolation_mode in (api.IsolationMode.vm, api.IsolationMode.remote):
            logger.info(
                "isolation_mode=%r — E2B sandboxes are always VM-isolated and remote; "
                "no extra configuration needed.",
                isolation_mode,
            )

        if limits.get("memory_mb") or limits.get("cpu"):
            logger.info(
                "policy.limits memory_mb=%r / cpu=%r noted; not yet forwarded "
                "to the E2B SDK (reserved for CubeSandbox resource hints).",
                limits.get("memory_mb"),
                limits.get("cpu"),
            )

        # ------------------------------------------------------------------ #
        # 2. Create sandbox                                                   #
        # ------------------------------------------------------------------ #
        client = _E2BClient(api_key, timeout=timeout, template=template)
        client.create()

        try:
            # -------------------------------------------------------------- #
            # 3. Bootstrap commands                                           #
            # -------------------------------------------------------------- #
            if bootstrap_env:
                env_exports = " && ".join(
                    f"export {k}={v!r}" for k, v in bootstrap_env.items()
                )
                bootstrap_commands = [f"{env_exports} && {cmd}" for cmd in bootstrap_commands] if bootstrap_commands else [env_exports]

            for cmd in bootstrap_commands:
                logger.debug("Bootstrap: %s", cmd)
                exit_code, stdout, stderr = client.run_command(cmd)
                if exit_code != 0:
                    raise E2BSandboxError(
                        f"Bootstrap command failed (exit {exit_code}): {cmd!r}\n"
                        f"stderr: {stderr}"
                    )

            # -------------------------------------------------------------- #
            # 4. Upload config                                                #
            # -------------------------------------------------------------- #
            _ensure_remote_dir(client, "/arena")
            client.upload_file(config_path, self._REMOTE_CONFIG)

            # -------------------------------------------------------------- #
            # 5. Install dependencies                                         #
            # -------------------------------------------------------------- #
            packages = [self._DEFAULT_PACKAGE] + extra_packages
            pip_cmd = f"pip install --quiet {' '.join(packages)}"
            logger.debug("Installing: %s", pip_cmd)
            exit_code, stdout, stderr = client.run_command(pip_cmd)
            if exit_code != 0:
                raise E2BSandboxError(
                    f"Dependency installation failed (exit {exit_code}):\n{stderr}"
                )

            # -------------------------------------------------------------- #
            # 6. Execute sweep                                                #
            # -------------------------------------------------------------- #
            run_cmd = (
                f"arena --config {self._REMOTE_CONFIG} --output-json {self._REMOTE_RESULT}"
            )
            logger.info("Running sweep: %s", run_cmd)
            exit_code, stdout, stderr = client.run_command(run_cmd, workdir="/arena")

            if stderr:
                logger.debug("Sandbox stderr:\n%s", stderr)

            if exit_code != 0:
                raise E2BSandboxError(
                    f"Sweep execution failed (exit {exit_code}):\n"
                    f"stdout: {stdout}\nstderr: {stderr}"
                )

            # -------------------------------------------------------------- #
            # 7. Parse result                                                 #
            # -------------------------------------------------------------- #
            result = _parse_result(stdout, client, self._REMOTE_RESULT)

            # -------------------------------------------------------------- #
            # 8. Teardown commands                                            #
            # -------------------------------------------------------------- #
            for cmd in teardown_commands:
                logger.debug("Teardown: %s", cmd)
                exit_code, _, stderr = client.run_command(cmd)
                if exit_code != 0:
                    logger.warning(
                        "Teardown command exited with %d: %r\nstderr: %s",
                        exit_code,
                        cmd,
                        stderr,
                    )

        finally:
            # ---------------------------------------------------------------- #
            # 9. Kill sandbox (always, even on error)                         #
            # ---------------------------------------------------------------- #
            client.kill()

        return result

    def run_task(
        self,
        config_path: Path,
        *,
        policy: api.SandboxPolicy | None = None,
        scratch_tag: str = "",
    ) -> TaskResult:
        """Execute a single (model, environment) task via the E2B sandbox.

        P2-2: per-task fan-out path.  This implementation currently delegates
        to :meth:`run` (a single shared E2B sandbox per call), which is
        functionally correct for per-task configs but does **not** provide
        independent concurrent E2B sandbox instances.  True per-call isolation
        (launching a fresh, separate E2B sandbox for each concurrent task) is a
        planned follow-up.

        The *config_path* must describe exactly one (model, environment) pair
        (the caller builds per-task configs via ``_config_for_pending([item])``).

        Args:
            config_path: Path to the per-task YAML config.
            policy: Optional :class:`~open_arena_core.models.SandboxPolicy`.
            scratch_tag: Opaque label for the ephemeral scratch directory;
                included in the returned :class:`TaskResult` for traceability.

        Returns:
            A :class:`TaskResult` wrapping the rows for this single task.
        """
        logger.debug(
            "E2BSandboxProvider.run_task: scratch_tag=%r, policy=%r",
            scratch_tag,
            policy.model_dump(exclude_none=True) if policy else None,
        )
        raw = self.run(config_path, policy=policy)
        return TaskResult(
            rows=raw.get("rows", []),
            scratch_tag=scratch_tag,
            meta=raw.get("meta", {}),
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_remote_dir(client: _E2BClient, path: str) -> None:
    """Create *path* inside the sandbox (mkdir -p)."""
    exit_code, _, stderr = client.run_command(f"mkdir -p {path}")
    if exit_code != 0:
        raise E2BSandboxError(f"Failed to create remote directory {path!r}: {stderr}")


def _parse_result(stdout: str, client: _E2BClient, remote_result_path: str) -> dict[str, Any]:
    """Extract the sweep result dict from either *stdout* or the JSON file.

    Strategy
    --------
    1. Try to read the JSON file written to *remote_result_path* inside the
       sandbox (preferred — structured, unambiguous).
    2. Fall back to scanning *stdout* lines in reverse for a valid JSON object
       (backwards-compat with arenas that print JSON as the last output line).
    3. Return a minimal ``{"rows": [], "meta": {"source": "stdout_raw"}}``
       dict with the raw stdout captured under ``"meta"`` so callers are never
       left with an empty result.
    """
    # Strategy 1: read the dedicated result file
    try:
        exit_code, file_stdout, _ = client.run_command(f"cat {remote_result_path}")
        if exit_code == 0 and file_stdout.strip():
            return json.loads(file_stdout.strip())
    except (json.JSONDecodeError, Exception):
        pass

    # Strategy 2: last JSON line in stdout
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue

    # Strategy 3: graceful fallback — preserve raw output in meta
    logger.warning(
        "Could not parse structured JSON result from sandbox output; "
        "returning raw stdout in meta."
    )
    return {
        "rows": [],
        "meta": {
            "source": "stdout_raw",
            "stdout": stdout,
        },
    }
