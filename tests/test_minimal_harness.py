# License Apache 2.0: (c) 2026 Athena-Reply
"""Minimal-harness integration test (P2 quick-wins, issue #62).

Drives the in-process ``ArenaAPIService`` end-to-end for an inline
{dataset + exact_match verifier} environment without hitting any real LLM.

The engine boundary (``LocalSandboxProvider.run``) is patched to return a
canned ``{"meta": ..., "rows": [...]}`` payload.  The test exercises:

* inline environment → service → sandbox.run → result-parsing wiring
* SandboxPolicy plumbing (the policy collected from the inline definition is
  forwarded to the SandboxProvider)
* ``RunResult`` is produced and persisted with valid metrics
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from open_arena_core.models import (
    DatasetBinding,
    Direction,
    EnvironmentRuntimePolicy,
    InlineEnvironmentDefinition,
    MetricDefinition,
    RunCreate,
    RunStatus,
    VerifierSuiteBinding,
    VerifierSuiteInline,
)
from src.api.service import ArenaAPIService
from src.api.stores.sqlite import SQLiteStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MODEL_PROVIDER = "openai"
_MODEL_NAME = "gpt-4o-mini"
_MODEL_VERSION = "2024-07-18"
_MODEL_KEY = f"{_MODEL_PROVIDER}/{_MODEL_NAME}"  # matches _model_runtime_id()


def _make_service(tmp_path):
    """Build an ArenaAPIService wired to a fresh in-memory SQLite store."""
    from src.api.registry import build_adapters, AdapterSet
    from src.api.settings import ArenaSettings

    settings = ArenaSettings(db_path=tmp_path / "test.db")
    adapters = build_adapters(settings)
    return ArenaAPIService(adapters=adapters)


def _make_run_create_payload() -> dict:
    """Return the minimal run-request dict that exercises the inline-env path."""
    return {
        "mode": "generator",
        "selection": {
            "direct_pairs": [
                {
                    "model": {
                        "name": "minimal-test-model",
                        "runtime": {
                            "provider": _MODEL_PROVIDER,
                            "model_name": _MODEL_NAME,
                            "model_version": _MODEL_VERSION,
                            "temperature": 0.0,
                            "max_tokens": 256,
                        },
                    },
                    "environment": {
                        "inline_definition": {
                            "name": "minimal-qa",
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
                    },
                }
            ]
        },
        "reuse_policy": {"enabled": False},
    }


# ---------------------------------------------------------------------------
# Test: inline environment → service → sandbox.run → RunResult with metrics
# ---------------------------------------------------------------------------

def _canned_row(env_id: str, value: float = 0.75) -> dict:
    return {
        "dataset": env_id,
        "model": _MODEL_KEY,
        "metric": "exact_match",
        "value": value,
        "direction": "max",
        "trajectory": None,
    }


def _wait_for_run(svc, run, timeout: float = 10.0):
    """Poll until the run leaves queued/running state."""
    deadline = time.monotonic() + timeout
    while run.status not in (RunStatus.succeeded, RunStatus.failed):
        run = svc.get_run(run.id)
        if time.monotonic() > deadline:
            break
        time.sleep(0.05)
    return run


class TestMinimalHarness:
    """End-to-end harness test using a mocked engine boundary.

    The SandboxProvider is replaced by patching ``LocalSandboxProvider.run``
    at the *instance* level so the patch stays alive for the background thread.
    """

    def _install_mock_sandbox(self, svc, side_effect):
        """Replace sandbox.run with *side_effect* in-place on the live instance."""
        svc._sandbox.run = side_effect  # type: ignore[method-assign]

    def test_inline_env_produces_run_result(self, tmp_path):
        """Submitting a run with an inline env produces a RunResult with metrics."""
        svc = _make_service(tmp_path)
        payload = RunCreate.model_validate(_make_run_create_payload())

        def _fake_sandbox_run(config_path, *, policy=None):
            import yaml

            config = yaml.safe_load(config_path.read_text())
            env_ids = list(config.get("datasets", {}).keys())
            assert env_ids, "sandbox config must have at least one dataset entry"
            env_id = env_ids[0]
            return {"meta": {"mode": "generator"}, "rows": [_canned_row(env_id)]}

        self._install_mock_sandbox(svc, _fake_sandbox_run)
        run = svc.create_run(payload)
        run = _wait_for_run(svc, run)

        assert run.status == RunStatus.succeeded, (
            f"Run failed: {getattr(run, 'error', None)}"
        )

        result = svc.get_run_result(run.id)
        assert result is not None
        assert result.run_id == run.id
        assert len(result.subjects) == 1

        subject = result.subjects[0]
        assert subject.metrics, "expected at least one metric in result"
        assert subject.metrics[0].name == "exact_match"
        assert subject.metrics[0].value == pytest.approx(0.75)
        assert subject.metrics[0].direction == Direction.max

    def test_sandbox_policy_forwarded_when_set(self, tmp_path):
        """SandboxPolicy on the inline_definition is forwarded to sandbox.run."""
        svc = _make_service(tmp_path)

        # Build a payload with an explicit sandbox policy.
        raw = _make_run_create_payload()
        raw["selection"]["direct_pairs"][0]["environment"]["inline_definition"][
            "sandbox"
        ] = {
            "enabled": True,
            "isolation_mode": "container",
        }
        payload = RunCreate.model_validate(raw)

        received_policies: list = []

        def _capture_policy(config_path, *, policy=None):
            import yaml

            config = yaml.safe_load(config_path.read_text())
            env_ids = list(config.get("datasets", {}).keys())
            env_id = env_ids[0]
            received_policies.append(policy)
            return {"meta": {}, "rows": [_canned_row(env_id, value=1.0)]}

        self._install_mock_sandbox(svc, _capture_policy)
        run = svc.create_run(payload)
        run = _wait_for_run(svc, run)

        assert run.status == RunStatus.succeeded, (
            f"Run failed: {getattr(run, 'error', None)}"
        )
        assert received_policies, "sandbox.run must have been called"
        policy = received_policies[0]
        assert policy is not None, "policy must be forwarded (not None) when set"
        assert str(policy.isolation_mode) == "container"

    def test_no_sandbox_policy_passes_none(self, tmp_path):
        """When no sandbox policy is set, sandbox.run receives policy=None."""
        svc = _make_service(tmp_path)
        payload = RunCreate.model_validate(_make_run_create_payload())

        received_policies: list = []

        def _capture_policy(config_path, *, policy=None):
            import yaml

            config = yaml.safe_load(config_path.read_text())
            env_ids = list(config.get("datasets", {}).keys())
            env_id = env_ids[0]
            received_policies.append(policy)
            return {"meta": {}, "rows": [_canned_row(env_id, value=0.5)]}

        self._install_mock_sandbox(svc, _capture_policy)
        run = svc.create_run(payload)
        run = _wait_for_run(svc, run)

        assert run.status == RunStatus.succeeded, (
            f"Run failed: {getattr(run, 'error', None)}"
        )
        assert received_policies
        assert received_policies[0] is None, "policy must be None when not set"
