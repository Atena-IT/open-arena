# License Apache 2.0: (c) 2026 Athena-Reply
"""Tests for P2-2 — per-task parallel sandboxing (issue #64).

Verifies that:

1. When ``per_task_sandbox=True`` is set in the SandboxPolicy the service fans
   out: each pending subject triggers its own ``run_task`` call (one per task).
2. The concurrency is bounded by ``ArenaAPIService._task_concurrency``
   (configurable via ``OPEN_ARENA_TASK_CONCURRENCY``, default 8).
3. Results assemble into ``SubjectResult`` objects identical in shape to those
   produced by the whole-run path.
4. Heterogeneous sandbox policies (different per subject) are naturally handled:
   each ``run_task`` call receives the policy from its own subject's inline
   environment definition.
5. The pure dataset+verifier path (``per_task_sandbox=False`` / unset) is
   unchanged: ``run`` is called once, ``run_task`` is never called.

All tests use a MOCK SandboxProvider — no real sandbox is launched.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from open_arena_core.models import (
    Direction,
    RunCreate,
    RunStatus,
    SandboxPolicy,
)
from src.api.ports.sandbox_provider import LocalSandboxProvider, SandboxProvider, TaskResult
from src.api.service import ArenaAPIService


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------

_MODEL_PROVIDER = "openai"
_MODEL_NAME = "gpt-4o-mini"
_MODEL_VERSION = "2024-07-18"
_MODEL_KEY = f"{_MODEL_PROVIDER}/{_MODEL_NAME}"


def _make_service(tmp_path):
    """Build an ArenaAPIService wired to a fresh in-memory SQLite store."""
    from src.api.registry import build_adapters
    from src.api.settings import ArenaSettings

    settings = ArenaSettings(db_path=tmp_path / "test.db")
    adapters = build_adapters(settings)
    return ArenaAPIService(adapters=adapters)


def _model_block(name: str = "test-model", provider: str = _MODEL_PROVIDER, model_name: str = _MODEL_NAME) -> dict:
    return {
        "name": name,
        "runtime": {
            "provider": provider,
            "model_name": model_name,
            "model_version": _MODEL_VERSION,
            "temperature": 0.0,
            "max_tokens": 256,
        },
    }


def _env_block(env_name: str = "test-env", *, sandbox: dict | None = None) -> dict:
    env: dict[str, Any] = {
        "inline_definition": {
            "name": env_name,
            "version": "0.1.0",
            "dataset": {
                "provider": "local",
                "source_ref": "examples/minimal_eval/data.jsonl",
            },
            "verifier": {
                "binding_type": "inline",
                "name": "exact-match-verifier",
                "metrics": [
                    {
                        "name": "exact_match",
                        "metric_kind": "exact_match",
                        "weight": 1.0,
                    }
                ],
            },
            "runtime": {
                "supported_modes": ["generator"],
            },
        }
    }
    if sandbox is not None:
        env["inline_definition"]["sandbox"] = sandbox
    return env


def _canned_row(env_id: str, model_key: str = _MODEL_KEY, value: float = 0.80) -> dict:
    return {
        "dataset": env_id,
        "model": model_key,
        "metric": "exact_match",
        "value": value,
        "direction": "max",
        "trajectory": None,
    }


def _wait_for_run(svc, run, timeout: float = 15.0):
    """Poll until the run leaves queued/running state."""
    deadline = time.monotonic() + timeout
    while run.status not in (RunStatus.succeeded, RunStatus.failed):
        run = svc.get_run(run.id)
        if time.monotonic() > deadline:
            break
        time.sleep(0.05)
    return run


# ---------------------------------------------------------------------------
# 1. TaskResult dataclass
# ---------------------------------------------------------------------------

class TestTaskResult:
    """Unit tests for the TaskResult dataclass."""

    def test_task_result_defaults(self):
        result = TaskResult()
        assert result.rows == []
        assert result.scratch_tag == ""
        assert result.meta == {}

    def test_task_result_with_values(self):
        rows = [{"metric": "exact_match", "value": 0.9}]
        result = TaskResult(rows=rows, scratch_tag="env.model.0", meta={"mode": "generator"})
        assert result.rows == rows
        assert result.scratch_tag == "env.model.0"
        assert result.meta == {"mode": "generator"}


# ---------------------------------------------------------------------------
# 2. SandboxProvider.run_task default implementation
# ---------------------------------------------------------------------------

class TestSandboxProviderRunTask:
    """run_task on LocalSandboxProvider delegates to run."""

    def test_run_task_delegates_to_run(self, tmp_path):
        provider = LocalSandboxProvider()
        config_path = tmp_path / "run.yaml"
        config_path.write_text("datasets: {}")

        raw_result = {"rows": [{"metric": "exact_match", "value": 0.5}], "meta": {"mode": "test"}}
        with patch.object(provider, "run", return_value=raw_result) as mock_run:
            result = provider.run_task(config_path, policy=None, scratch_tag="my.tag.0")

        mock_run.assert_called_once_with(config_path, policy=None)
        assert isinstance(result, TaskResult)
        assert result.rows == raw_result["rows"]
        assert result.scratch_tag == "my.tag.0"
        assert result.meta == {"mode": "test"}

    def test_run_task_policy_forwarded(self, tmp_path):
        provider = LocalSandboxProvider()
        config_path = tmp_path / "run.yaml"
        config_path.write_text("datasets: {}")
        policy = SandboxPolicy(enabled=True, per_task_sandbox=True)

        raw_result = {"rows": [], "meta": {}}
        with patch.object(provider, "run", return_value=raw_result) as mock_run:
            provider.run_task(config_path, policy=policy, scratch_tag="tag")

        mock_run.assert_called_once_with(config_path, policy=policy)

    def test_run_task_missing_meta_key_defaults_to_empty(self, tmp_path):
        provider = LocalSandboxProvider()
        config_path = tmp_path / "run.yaml"
        config_path.write_text("datasets: {}")

        with patch.object(provider, "run", return_value={"rows": []}):
            result = provider.run_task(config_path)

        assert result.meta == {}


# ---------------------------------------------------------------------------
# 3. Pure path unchanged — per_task_sandbox=False → run() once, run_task() never
# ---------------------------------------------------------------------------

class TestPurePathUnchanged:
    """The whole-run path (per_task_sandbox False/unset) must be byte-for-byte
    unchanged: sandbox.run is called exactly once; run_task is never called."""

    def test_pure_path_calls_run_not_run_task(self, tmp_path):
        svc = _make_service(tmp_path)
        payload_dict = {
            "mode": "generator",
            "selection": {
                "direct_pairs": [
                    {"model": _model_block(), "environment": _env_block()},
                ]
            },
            "reuse_policy": {"enabled": False},
        }
        payload = RunCreate.model_validate(payload_dict)

        run_call_count = 0
        run_task_call_count = 0

        def _fake_run(config_path, *, policy=None):
            nonlocal run_call_count
            run_call_count += 1
            import yaml
            config = yaml.safe_load(config_path.read_text())
            env_id = next(iter(config["datasets"]))
            return {"rows": [_canned_row(env_id)], "meta": {}}

        def _fake_run_task(config_path, *, policy=None, scratch_tag=""):
            nonlocal run_task_call_count
            run_task_call_count += 1
            return TaskResult(rows=[], scratch_tag=scratch_tag)

        svc._sandbox.run = _fake_run  # type: ignore[method-assign]
        svc._sandbox.run_task = _fake_run_task  # type: ignore[method-assign]

        run = svc.create_run(payload)
        run = _wait_for_run(svc, run)

        assert run.status == RunStatus.succeeded, f"Run failed: {getattr(run, 'error', None)}"
        assert run_call_count == 1, "whole-run path must call run() exactly once"
        assert run_task_call_count == 0, "whole-run path must NEVER call run_task()"

    def test_pure_path_with_sandbox_policy_disabled(self, tmp_path):
        """per_task_sandbox=False with an explicit policy → still whole-run."""
        svc = _make_service(tmp_path)
        payload_dict = {
            "mode": "generator",
            "selection": {
                "direct_pairs": [
                    {
                        "model": _model_block(),
                        "environment": _env_block(
                            sandbox={"enabled": True, "isolation_mode": "container", "per_task_sandbox": False}
                        ),
                    },
                ]
            },
            "reuse_policy": {"enabled": False},
        }
        payload = RunCreate.model_validate(payload_dict)

        run_call_count = 0
        run_task_call_count = 0

        def _fake_run(config_path, *, policy=None):
            nonlocal run_call_count
            run_call_count += 1
            import yaml
            config = yaml.safe_load(config_path.read_text())
            env_id = next(iter(config["datasets"]))
            return {"rows": [_canned_row(env_id)], "meta": {}}

        def _fake_run_task(config_path, *, policy=None, scratch_tag=""):
            nonlocal run_task_call_count
            run_task_call_count += 1
            return TaskResult(rows=[], scratch_tag=scratch_tag)

        svc._sandbox.run = _fake_run  # type: ignore[method-assign]
        svc._sandbox.run_task = _fake_run_task  # type: ignore[method-assign]

        run = svc.create_run(payload)
        run = _wait_for_run(svc, run)

        assert run.status == RunStatus.succeeded, f"Run failed: {getattr(run, 'error', None)}"
        assert run_task_call_count == 0, "per_task_sandbox=False must NOT fan out"
        assert run_call_count == 1


# ---------------------------------------------------------------------------
# 4. Per-task fan-out — each task gets its own run_task() call
# ---------------------------------------------------------------------------

class TestPerTaskFanOut:
    """When per_task_sandbox=True, each pending subject triggers run_task()."""

    def _payload_with_per_task(self, n_models: int = 2) -> dict:
        sandbox_cfg = {"enabled": True, "per_task_sandbox": True}
        pairs = [
            {
                "model": _model_block(name=f"model-{i}", model_name=f"gpt-4o-mini-{i}"),
                "environment": _env_block(env_name=f"env-{i}", sandbox=sandbox_cfg),
            }
            for i in range(n_models)
        ]
        return {
            "mode": "generator",
            "selection": {"direct_pairs": pairs},
            "reuse_policy": {"enabled": False},
        }

    def test_run_task_called_once_per_pending_subject(self, tmp_path):
        """run_task must be invoked exactly N times for N pending subjects."""
        svc = _make_service(tmp_path)
        n = 3
        payload = RunCreate.model_validate(self._payload_with_per_task(n))

        run_task_calls: list[dict] = []
        run_call_count = 0
        lock = threading.Lock()

        def _fake_run_task(config_path, *, policy=None, scratch_tag=""):
            import yaml
            config = yaml.safe_load(config_path.read_text())
            # Single task config: one dataset, one model.
            env_id = next(iter(config["datasets"]))
            model_key = config["experiments"]["language_models"][0]
            with lock:
                run_task_calls.append({"env_id": env_id, "model_key": model_key, "scratch_tag": scratch_tag})
            return TaskResult(rows=[_canned_row(env_id, model_key)], scratch_tag=scratch_tag)

        def _fake_run(config_path, *, policy=None):
            nonlocal run_call_count
            run_call_count += 1
            return {"rows": [], "meta": {}}

        svc._sandbox.run_task = _fake_run_task  # type: ignore[method-assign]
        svc._sandbox.run = _fake_run  # type: ignore[method-assign]

        run = svc.create_run(payload)
        run = _wait_for_run(svc, run)

        assert run.status == RunStatus.succeeded, f"Run failed: {getattr(run, 'error', None)}"
        assert run_call_count == 0, "fan-out path must NOT call run() (whole-run)"
        assert len(run_task_calls) == n, f"expected {n} run_task calls, got {len(run_task_calls)}"

    def test_each_task_gets_unique_scratch_tag(self, tmp_path):
        """Each run_task call must receive a distinct scratch_tag."""
        svc = _make_service(tmp_path)
        n = 4
        payload = RunCreate.model_validate(self._payload_with_per_task(n))

        scratch_tags: list[str] = []
        lock = threading.Lock()

        def _fake_run_task(config_path, *, policy=None, scratch_tag=""):
            import yaml
            config = yaml.safe_load(config_path.read_text())
            env_id = next(iter(config["datasets"]))
            model_key = config["experiments"]["language_models"][0]
            with lock:
                scratch_tags.append(scratch_tag)
            return TaskResult(rows=[_canned_row(env_id, model_key)], scratch_tag=scratch_tag)

        svc._sandbox.run_task = _fake_run_task  # type: ignore[method-assign]
        svc._sandbox.run = MagicMock(return_value={"rows": [], "meta": {}})

        run = svc.create_run(payload)
        run = _wait_for_run(svc, run)

        assert run.status == RunStatus.succeeded, f"Run failed: {getattr(run, 'error', None)}"
        assert len(scratch_tags) == n
        assert len(set(scratch_tags)) == n, "every task must have a unique scratch_tag"

    def test_results_assemble_into_subject_results(self, tmp_path):
        """Fan-out results must produce SubjectResults with correct metrics."""
        svc = _make_service(tmp_path)
        n = 2
        payload = RunCreate.model_validate(self._payload_with_per_task(n))

        def _fake_run_task(config_path, *, policy=None, scratch_tag=""):
            import yaml
            config = yaml.safe_load(config_path.read_text())
            env_id = next(iter(config["datasets"]))
            model_key = config["experiments"]["language_models"][0]
            return TaskResult(
                rows=[_canned_row(env_id, model_key, value=0.95)],
                scratch_tag=scratch_tag,
            )

        svc._sandbox.run_task = _fake_run_task  # type: ignore[method-assign]
        svc._sandbox.run = MagicMock(return_value={"rows": [], "meta": {}})

        run = svc.create_run(payload)
        run = _wait_for_run(svc, run)
        assert run.status == RunStatus.succeeded, f"Run failed: {getattr(run, 'error', None)}"

        result = svc.get_run_result(run.id)
        assert result is not None
        assert len(result.subjects) == n
        for subject in result.subjects:
            assert subject.metrics, "each subject must have metrics"
            assert subject.metrics[0].name == "exact_match"
            assert subject.metrics[0].value == pytest.approx(0.95)
            assert subject.metrics[0].direction == Direction.max
            assert subject.cache_status.value == "miss"

    def test_concurrency_is_bounded_by_max_workers(self, tmp_path):
        """Concurrent tasks must not exceed the per-task concurrency cap at any instant."""
        svc = _make_service(tmp_path)
        n = 6  # more tasks than a low concurrency cap
        # Temporarily lower the per-task fan-out cap for this test
        # (instance-level; backed by OPEN_ARENA_TASK_CONCURRENCY).
        original_cap = svc._task_concurrency
        svc._task_concurrency = 3
        try:
            payload = RunCreate.model_validate(self._payload_with_per_task(n))

            peak_concurrent = [0]
            current_concurrent = [0]
            lock = threading.Lock()

            def _fake_run_task(config_path, *, policy=None, scratch_tag=""):
                import yaml
                config = yaml.safe_load(config_path.read_text())
                env_id = next(iter(config["datasets"]))
                model_key = config["experiments"]["language_models"][0]
                with lock:
                    current_concurrent[0] += 1
                    if current_concurrent[0] > peak_concurrent[0]:
                        peak_concurrent[0] = current_concurrent[0]
                time.sleep(0.05)  # hold the slot briefly
                with lock:
                    current_concurrent[0] -= 1
                return TaskResult(
                    rows=[_canned_row(env_id, model_key)],
                    scratch_tag=scratch_tag,
                )

            svc._sandbox.run_task = _fake_run_task  # type: ignore[method-assign]
            svc._sandbox.run = MagicMock(return_value={"rows": [], "meta": {}})

            run = svc.create_run(payload)
            run = _wait_for_run(svc, run, timeout=30.0)
            assert run.status == RunStatus.succeeded, f"Run failed: {getattr(run, 'error', None)}"
            assert peak_concurrent[0] <= 3, (
                f"Concurrency exceeded cap: peak={peak_concurrent[0]} > cap=3"
            )
        finally:
            svc._task_concurrency = original_cap

    def test_task_concurrency_configurable_via_env(self, tmp_path, monkeypatch):
        """OPEN_ARENA_TASK_CONCURRENCY controls the per-task fan-out cap."""
        monkeypatch.setenv("OPEN_ARENA_TASK_CONCURRENCY", "5")
        svc = _make_service(tmp_path)
        assert svc._task_concurrency == 5


# ---------------------------------------------------------------------------
# 5. Heterogeneous policies — each task gets its own policy
# ---------------------------------------------------------------------------

class TestHeterogeneousPolicies:
    """Heterogeneous sandbox policies are naturally handled by per-task fan-out."""

    def test_each_task_receives_its_own_policy(self, tmp_path):
        """When subjects have different sandbox policies, each run_task call
        receives the policy from its own environment's inline definition."""
        svc = _make_service(tmp_path)

        sandbox_a = {"enabled": True, "per_task_sandbox": True, "isolation_mode": "container"}
        sandbox_b = {"enabled": True, "per_task_sandbox": True, "isolation_mode": "vm"}

        payload_dict = {
            "mode": "generator",
            "selection": {
                "direct_pairs": [
                    {"model": _model_block(name="m1", model_name="model-a"), "environment": _env_block("env-a", sandbox=sandbox_a)},
                    {"model": _model_block(name="m2", model_name="model-b"), "environment": _env_block("env-b", sandbox=sandbox_b)},
                ]
            },
            "reuse_policy": {"enabled": False},
        }
        payload = RunCreate.model_validate(payload_dict)

        received: list[dict] = []
        lock = threading.Lock()

        def _fake_run_task(config_path, *, policy=None, scratch_tag=""):
            import yaml
            config = yaml.safe_load(config_path.read_text())
            env_id = next(iter(config["datasets"]))
            model_key = config["experiments"]["language_models"][0]
            with lock:
                received.append({
                    "scratch_tag": scratch_tag,
                    "isolation_mode": str(policy.isolation_mode) if policy else None,
                })
            return TaskResult(rows=[_canned_row(env_id, model_key)], scratch_tag=scratch_tag)

        svc._sandbox.run_task = _fake_run_task  # type: ignore[method-assign]
        svc._sandbox.run = MagicMock(return_value={"rows": [], "meta": {}})

        run = svc.create_run(payload)
        run = _wait_for_run(svc, run)
        assert run.status == RunStatus.succeeded, f"Run failed: {getattr(run, 'error', None)}"

        assert len(received) == 2
        isolation_modes = {r["isolation_mode"] for r in received}
        assert "container" in isolation_modes, "env-a policy (container) must be forwarded"
        assert "vm" in isolation_modes, "env-b policy (vm) must be forwarded"
