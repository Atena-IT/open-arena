# License Apache 2.0: (c) 2026 Athena-Reply
"""Port 5 — SandboxProvider

Executes a pending batch of subjects within a sandboxed (or in-process)
environment and returns the raw sweep result.

The default adapter is :class:`LocalSandboxProvider` which runs
``run_sweep`` in-process, matching the original behavior exactly.

P2-2: per-task fan-out — :meth:`SandboxProvider.run_task` executes a
single (model, environment) task in its own sandbox concurrently.
``run`` (whole-run path) is preserved byte-for-byte.

P2-4: env-package runtimes — :meth:`SandboxProvider.open_session` returns
a :class:`~src.api.sandboxes.env_runtime.SandboxSession`-compatible object
so the env-package runtimes can execute inside the same sandbox without
additional lifecycle overhead.

WS6: E2B-compatible — add an ``E2BSandboxProvider`` (or a generic
``RemoteSandboxProvider``) that ships the YAML config to a remote
execution environment (E2B sandbox, Modal, Fly.io, …) and streams
results back.
"""
from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator

from open_arena_core import models as api


@dataclass
class TaskResult:
    """Result produced by :meth:`SandboxProvider.run_task` for a single task.

    Carries the raw metric rows for one (model, environment) pair and the
    scratch directory tag used to isolate this task's state.  The caller
    (``_run_pending_subjects``) assembles these into :class:`SubjectResult`
    objects identical to those produced by the whole-run path.

    Attributes:
        rows: Per-metric result rows (same schema as ``run_sweep``'s
            ``result["rows"]``).
        scratch_tag: Opaque label identifying the ephemeral scratch
            directory for this task (``{env}.{model}.{trial}`` format).
            Informational only — the scratch dir may have been cleaned up.
        meta: Optional metadata dict forwarded from the sandbox result.
    """

    rows: list[dict[str, Any]] = field(default_factory=list)
    scratch_tag: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


class SandboxProvider(ABC):
    """Port for executing a sweep config in (or outside of) the current process.

    Implementors receive the filesystem path to a pre-rendered YAML config
    and return the raw result dict produced by ``run_sweep``.

    WS6: E2B-compatible — implement ``E2BSandboxProvider`` that uploads
    *config_path* to an E2B sandbox, runs ``arena --config``, and
    streams back the result JSON.
    """

    @abstractmethod
    def run(
        self,
        config_path: Path,
        *,
        policy: api.SandboxPolicy | None = None,
    ) -> dict[str, Any]:
        """Execute the sweep described by *config_path*.

        Args:
            config_path: Path to the pre-rendered YAML run config.
            policy: Optional sandbox isolation policy.  ``None`` means
                run in-process (default behavior).

        Returns:
            The result dict returned by ``run_sweep`` (contains at
            minimum a ``"rows"`` key with per-metric result rows).
        """

    @contextmanager
    def open_session(
        self,
        policy: api.SandboxPolicy | None = None,
    ) -> Generator[Any, None, None]:
        """Open a :class:`~src.api.sandboxes.env_runtime.SandboxSession`-compatible
        context for use by env-package runtimes (P2-4).

        The default implementation yields a :class:`LocalShellSession` that
        executes commands via ``subprocess`` in the current process.  Remote
        sandbox providers (E2B, Modal, …) should override this to yield a
        session backed by the remote execution environment.

        Args:
            policy: Optional :class:`~open_arena_core.models.SandboxPolicy`.
                Forwarded to the session for warm-image and timeout hints.

        Yields:
            A :class:`SandboxSession`-compatible object.
        """
        yield LocalShellSession()

    def run_task(
        self,
        config_path: Path,
        *,
        policy: api.SandboxPolicy | None = None,
        scratch_tag: str = "",
    ) -> TaskResult:
        """Execute a *single* (model, environment) task in its own sandbox.

        P2-2: per-task fan-out path.  The config at *config_path* must
        describe exactly one (model, environment) pair — the caller
        (:meth:`~src.api.service.ArenaAPIService._run_pending_subjects`)
        builds per-task configs using ``_config_for_pending([item])``.

        The default implementation delegates to :meth:`run` so subclasses
        that do not override this method stay correct.  Backends that can
        truly isolate each task (e.g. E2B, Modal) should override this
        to launch a fresh sandbox per call.

        Args:
            config_path: Path to the pre-rendered per-task YAML config.
            policy: Optional sandbox isolation policy.
            scratch_tag: Opaque label for the ephemeral scratch directory
                (``{env}.{model}.{trial}`` format); purely informational.

        Returns:
            A :class:`TaskResult` with the rows for this single task.
        """
        raw = self.run(config_path, policy=policy)
        return TaskResult(
            rows=raw.get("rows", []),
            scratch_tag=scratch_tag,
            meta=raw.get("meta", {}),
        )


class LocalSandboxProvider(SandboxProvider):
    """Default adapter — runs ``run_sweep`` in the current process.

    Reproduces the original ``_run_pending_subjects`` execution path
    verbatim: the YAML config is passed to ``run_sweep`` which runs
    synchronously via ``_run_async``.

    P2-2: :meth:`run_task` is inherited from :class:`SandboxProvider`
    and calls :meth:`run` with the per-task config, keeping the in-process
    sweep as the execution model.  No extra sandboxing is added for the
    local provider — isolation must come from scratch-dir separation.

    WS6: E2B-compatible — replace or complement this provider with an
    ``E2BSandboxProvider`` by setting ``OPEN_ARENA_SANDBOX=e2b`` in the
    environment and registering the adapter in the registry.
    """

    def run(
        self,
        config_path: Path,
        *,
        policy: api.SandboxPolicy | None = None,  # noqa: ARG002 — reserved for WS6
    ) -> dict[str, Any]:  # noqa: D102
        from src.evaluate import run_sweep

        def _run_async(coro):  # inline helper mirrors service._run_async
            import asyncio

            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(coro)
            coro.close()
            raise RuntimeError(
                "_run_async() cannot be called while an event loop is already running; "
                "await the coroutine instead."
            )

        return _run_async(run_sweep(str(config_path), no_cache=False, verbose=0))


# ---------------------------------------------------------------------------
# LocalShellSession — SandboxSession backed by subprocess (for local runs)
# ---------------------------------------------------------------------------


class LocalShellSession:
    """In-process :class:`~src.api.sandboxes.env_runtime.SandboxSession` that
    executes commands via :mod:`subprocess` and writes files to the local
    filesystem.

    P2-4: used by :class:`~src.api.ports.sandbox_provider.LocalSandboxProvider`
    as the default session returned by :meth:`SandboxProvider.open_session`.
    Remote providers (E2B, Modal) override :meth:`open_session` to yield a
    remote-backed session instead.

    This class satisfies the
    :class:`~src.api.sandboxes.env_runtime.SandboxSession` protocol.
    """

    def run_command(
        self,
        cmd: str,
        *,
        workdir: str | None = None,
    ) -> tuple[int, str, str]:
        """Run *cmd* via :func:`subprocess.run` and return ``(exit_code, stdout, stderr)``."""
        result = subprocess.run(
            cmd,
            shell=True,  # noqa: S602 — local only, commands are trusted
            capture_output=True,
            text=True,
            cwd=workdir,
        )
        return result.returncode, result.stdout, result.stderr

    def write_file(self, remote_path: str, content: bytes) -> None:
        """Write *content* to *remote_path* on the local filesystem."""
        dest = Path(remote_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
