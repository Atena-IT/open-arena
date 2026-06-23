# License Apache 2.0: (c) 2026 Athena-Reply
"""Port 5 — SandboxProvider

Executes a pending batch of subjects within a sandboxed (or in-process)
environment and returns the raw sweep result.

The default adapter is :class:`LocalSandboxProvider` which runs
``run_sweep`` in-process, matching the original behavior exactly.

WS6: E2B-compatible — add an ``E2BSandboxProvider`` (or a generic
``RemoteSandboxProvider``) that ships the YAML config to a remote
execution environment (E2B sandbox, Modal, Fly.io, …) and streams
results back.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from open_arena_core import models as api


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


class LocalSandboxProvider(SandboxProvider):
    """Default adapter — runs ``run_sweep`` in the current process.

    Reproduces the original ``_run_pending_subjects`` execution path
    verbatim: the YAML config is passed to ``run_sweep`` which runs
    synchronously via ``_run_async``.

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
