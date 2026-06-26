# License Apache 2.0: (c) 2026 Athena-Reply
"""Tests for P2-4 — verifiers / Harbor task runtime (issue #66).

Verifies that:

1. :func:`~src.api.sandboxes.env_runtime.detect_package_shape` correctly
   identifies Prime vs Harbor packages from their directory contents.
2. :class:`~src.api.sandboxes.env_runtime.PrimeVerifiersRuntime` executes the
   package and extracts a reward from canned ``run_command`` output.
3. :class:`~src.api.sandboxes.env_runtime.HarborTaskRuntime` runs the test
   suite, maps exit code / score line → reward.
4. :func:`~src.api.sandboxes.env_runtime.execute_env_package` dispatches to
   the correct runtime and returns a :class:`~src.api.ports.sandbox_provider.TaskResult`.
5. **Mocked end-to-end**: a pinned env (both Prime package and Harbor task)
   drives the full pipeline → reward in [0, 1] → ``RunResult.subjects[].metrics``
   persisted via a ``ResultsSink``.  The sandbox session is mocked via
   :class:`~src.api.sandboxes.env_runtime.FakeSandboxSession`; no real
   E2B / Docker calls are made.

All tests use MOCKED sandbox sessions — no real sandbox is launched.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from open_arena_core.models import (
    AggregateMetric,
    CacheStatus,
    Direction,
    EnvironmentSource,
    EnvironmentSourceKind,
    MetricResult,
    RunCreate,
    RunSelection,
    RunSelection1,
    RunStatus,
    SandboxPolicy,
    SubjectResult,
)
from src.api.ports.sandbox_provider import TaskResult
from src.api.sandboxes.env_runtime import (
    FakeSandboxSession,
    HarborTaskRuntime,
    PrimeVerifiersRuntime,
    SandboxSession,
    _clamp_reward,
    _extract_reward_from_output,
    detect_package_shape,
    execute_env_package,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MODEL_PROVIDER = "openai"
_MODEL_NAME = "gpt-4o-mini"
_MODEL_VERSION = "2024-07-18"
_MODEL_KEY = f"{_MODEL_PROVIDER}/{_MODEL_NAME}"
_DATASET_NAME = "env-abc-123"


def _make_service(tmp_path: Path):
    """Build an ArenaAPIService wired to a fresh in-memory SQLite store."""
    from src.api.registry import build_adapters
    from src.api.settings import ArenaSettings

    settings = ArenaSettings(db_path=tmp_path / "test.db")
    adapters = build_adapters(settings)
    from src.api.service import ArenaAPIService
    return ArenaAPIService(adapters=adapters)


def _model_block(name: str = "test-model") -> dict:
    return {
        "name": name,
        "runtime": {
            "provider": _MODEL_PROVIDER,
            "model_name": _MODEL_NAME,
            "model_version": _MODEL_VERSION,
            "temperature": 0.0,
            "max_tokens": 256,
        },
    }


def _env_block_external(
    env_name: str = "test-env",
    kind: str = "github_repo",
    *,
    sandbox: dict | None = None,
) -> dict:
    """Build a direct-pair environment block for an external (github_repo / prime_hub) env."""
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
# 1. Package shape detection
# ---------------------------------------------------------------------------


class TestDetectPackageShape:
    """detect_package_shape returns the correct shape from directory contents."""

    def test_prime_shape_detected(self, tmp_path):
        (tmp_path / "env.py").write_text("def load_environment(): return {}")
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\nversion='0.1'")
        assert detect_package_shape(str(tmp_path)) == "prime"

    def test_harbor_shape_detected(self, tmp_path):
        (tmp_path / "task.toml").write_text("[run]\ncommand='python -m pytest tests/'")
        (tmp_path / "tests").mkdir()
        assert detect_package_shape(str(tmp_path)) == "harbor"

    def test_prime_takes_priority_over_harbor_when_both_present(self, tmp_path):
        (tmp_path / "env.py").write_text("")
        (tmp_path / "pyproject.toml").write_text("")
        (tmp_path / "task.toml").write_text("")
        assert detect_package_shape(str(tmp_path)) == "prime"

    def test_fallback_to_prime_when_no_markers(self, tmp_path):
        (tmp_path / "README.md").write_text("hello")
        assert detect_package_shape(str(tmp_path)) == "prime"

    def test_non_existent_path_returns_prime(self, tmp_path):
        result = detect_package_shape(str(tmp_path / "nonexistent"))
        assert result == "prime"


# ---------------------------------------------------------------------------
# 2. Reward extraction helpers
# ---------------------------------------------------------------------------


class TestExtractReward:
    def test_reward_keyword(self):
        assert _extract_reward_from_output("reward: 0.75") == pytest.approx(0.75)

    def test_score_keyword(self):
        assert _extract_reward_from_output("score: 0.5") == pytest.approx(0.5)

    def test_accuracy_keyword(self):
        assert _extract_reward_from_output("accuracy: 1.0") == pytest.approx(1.0)

    def test_bare_float_fallback(self):
        assert _extract_reward_from_output("some noise\n0.33\nmore noise") == pytest.approx(0.33)

    def test_returns_none_for_no_float(self):
        assert _extract_reward_from_output("no numbers here") is None

    def test_case_insensitive(self):
        assert _extract_reward_from_output("REWARD: 0.88") == pytest.approx(0.88)

    def test_clamp_above_one(self):
        assert _clamp_reward(1.5) == pytest.approx(1.0)

    def test_clamp_below_zero(self):
        assert _clamp_reward(-0.1) == pytest.approx(0.0)

    def test_clamp_within_range(self):
        assert _clamp_reward(0.42) == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# 3. FakeSandboxSession
# ---------------------------------------------------------------------------


class TestFakeSandboxSession:
    """FakeSandboxSession correctly satisfies the SandboxSession protocol."""

    def test_satisfies_protocol(self):
        session = FakeSandboxSession()
        assert isinstance(session, SandboxSession)

    def test_returns_canned_responses_in_order(self):
        session = FakeSandboxSession(
            responses=[
                (0, "first", ""),
                (1, "", "error"),
            ]
        )
        assert session.run_command("cmd1") == (0, "first", "")
        assert session.run_command("cmd2") == (1, "", "error")

    def test_returns_zero_after_responses_exhausted(self):
        session = FakeSandboxSession(responses=[(0, "only", "")])
        session.run_command("x")
        exit_code, stdout, stderr = session.run_command("x")
        assert exit_code == 0
        assert stdout == ""

    def test_write_file_records_writes(self):
        session = FakeSandboxSession()
        session.write_file("/tmp/foo.txt", b"hello")
        assert ("/tmp/foo.txt", b"hello") in session.written

    def test_commands_logged(self):
        session = FakeSandboxSession(responses=[(0, "", "")])
        session.run_command("echo hi")
        assert "echo hi" in session.commands

    def test_workdir_accepted(self):
        session = FakeSandboxSession(responses=[(0, "ok", "")])
        code, out, _ = session.run_command("pwd", workdir="/tmp")
        assert code == 0


# ---------------------------------------------------------------------------
# 4. PrimeVerifiersRuntime with mocked session
# ---------------------------------------------------------------------------


class TestPrimeVerifiersRuntime:
    """PrimeVerifiersRuntime extracts reward from canned session output."""

    def _make_session_with_reward(self, reward: float) -> FakeSandboxSession:
        """Return a fake session whose runner script outputs a reward line."""
        return FakeSandboxSession(
            responses=[
                (0, "", ""),           # mkdir
                (0, "", ""),           # tar extract (strip)
                (0, "", ""),           # pip install
                (0, f"reward: {reward}", ""),  # python runner
            ]
        )

    def test_reward_extracted_from_output(self, tmp_path):
        session = self._make_session_with_reward(0.75)
        runtime = PrimeVerifiersRuntime(session=session)
        result = runtime.run(str(tmp_path), dataset_name=_DATASET_NAME, model_key=_MODEL_KEY)

        assert len(result.rows) == 1
        row = result.rows[0]
        assert row["metric"] == "reward"
        assert row["value"] == pytest.approx(0.75)
        assert row["direction"] == "max"
        assert row["dataset"] == _DATASET_NAME
        assert row["model"] == _MODEL_KEY

    def test_reward_clamped_to_zero_on_no_output(self, tmp_path):
        session = FakeSandboxSession(
            responses=[
                (0, "", ""),    # mkdir
                (0, "", ""),    # tar
                (0, "", ""),    # pip
                (1, "", "ERR"), # runner fails, no reward line
            ]
        )
        runtime = PrimeVerifiersRuntime(session=session)
        result = runtime.run(str(tmp_path), dataset_name=_DATASET_NAME, model_key=_MODEL_KEY)
        assert result.rows[0]["value"] == pytest.approx(0.0)

    def test_reward_clamped_above_one(self, tmp_path):
        session = FakeSandboxSession(
            responses=[
                (0, "", ""),
                (0, "", ""),
                (0, "", ""),
                (0, "reward: 2.5", ""),
            ]
        )
        runtime = PrimeVerifiersRuntime(session=session)
        result = runtime.run(str(tmp_path), dataset_name=_DATASET_NAME, model_key=_MODEL_KEY)
        assert result.rows[0]["value"] == pytest.approx(1.0)

    def test_meta_contains_runtime_key(self, tmp_path):
        session = self._make_session_with_reward(0.5)
        runtime = PrimeVerifiersRuntime(session=session)
        result = runtime.run(str(tmp_path), dataset_name=_DATASET_NAME, model_key=_MODEL_KEY)
        assert result.meta.get("runtime") == "prime_verifiers"

    def test_warm_image_logged(self, tmp_path, caplog):
        import logging
        session = self._make_session_with_reward(0.8)
        policy = SandboxPolicy(enabled=True, image="ghcr.io/org/eval@sha256:abc")
        runtime = PrimeVerifiersRuntime(session=session, policy=policy)
        with caplog.at_level(logging.INFO, logger="src.api.sandboxes.env_runtime"):
            result = runtime.run(str(tmp_path), dataset_name=_DATASET_NAME, model_key=_MODEL_KEY)
        assert "ghcr.io/org/eval@sha256:abc" in caplog.text


# ---------------------------------------------------------------------------
# 5. HarborTaskRuntime with mocked session
# ---------------------------------------------------------------------------


class TestHarborTaskRuntime:
    """HarborTaskRuntime maps exit code / score line → reward."""

    def _prime_responses(
        self, exit_code: int = 0, stdout: str = "", stderr: str = ""
    ) -> list[tuple[int, str, str]]:
        """Standard response sequence for Harbor runtime lifecycle."""
        return [
            (0, "", ""),    # mkdir
            (0, "", ""),    # tar extract
            (exit_code, stdout, stderr),  # run command (no deps)
        ]

    def test_exit_zero_gives_reward_one(self, tmp_path):
        session = FakeSandboxSession(responses=self._prime_responses(0))
        runtime = HarborTaskRuntime(session=session)
        result = runtime.run(str(tmp_path), dataset_name=_DATASET_NAME, model_key=_MODEL_KEY)
        assert result.rows[0]["value"] == pytest.approx(1.0)

    def test_exit_nonzero_gives_reward_zero(self, tmp_path):
        session = FakeSandboxSession(responses=self._prime_responses(1, "", "FAIL"))
        runtime = HarborTaskRuntime(session=session)
        result = runtime.run(str(tmp_path), dataset_name=_DATASET_NAME, model_key=_MODEL_KEY)
        assert result.rows[0]["value"] == pytest.approx(0.0)

    def test_score_line_overrides_exit_code(self, tmp_path):
        # exit=1 but stdout contains a score
        session = FakeSandboxSession(
            responses=self._prime_responses(1, "score: 0.6", "some error")
        )
        runtime = HarborTaskRuntime(session=session)
        result = runtime.run(str(tmp_path), dataset_name=_DATASET_NAME, model_key=_MODEL_KEY)
        assert result.rows[0]["value"] == pytest.approx(0.6)

    def test_custom_entry_point_from_task_toml(self, tmp_path):
        """When task.toml has [run] command, that command is used."""
        (tmp_path / "task.toml").write_text(
            "[run]\ncommand = 'python run_eval.py'\n"
        )
        session = FakeSandboxSession(
            responses=[
                (0, "", ""),    # mkdir
                (0, "", ""),    # tar
                (0, "reward: 0.9", ""),  # custom command
            ]
        )
        runtime = HarborTaskRuntime(session=session)
        result = runtime.run(str(tmp_path), dataset_name=_DATASET_NAME, model_key=_MODEL_KEY)
        # Verify the custom command was issued
        assert any("python run_eval.py" in cmd for cmd in session.commands)
        assert result.rows[0]["value"] == pytest.approx(0.9)

    def test_meta_contains_runtime_key(self, tmp_path):
        session = FakeSandboxSession(responses=self._prime_responses(0))
        runtime = HarborTaskRuntime(session=session)
        result = runtime.run(str(tmp_path), dataset_name=_DATASET_NAME, model_key=_MODEL_KEY)
        assert result.meta.get("runtime") == "harbor_task"

    def test_dependencies_installed_from_task_toml(self, tmp_path):
        (tmp_path / "task.toml").write_text(
            "[dependencies]\npackages = ['numpy', 'scipy']\n"
            "[run]\ncommand = 'python eval.py'\n"
        )
        session = FakeSandboxSession(
            responses=[
                (0, "", ""),    # mkdir
                (0, "", ""),    # tar
                (0, "", ""),    # pip install numpy scipy
                (0, "reward: 0.7", ""),  # custom command
            ]
        )
        runtime = HarborTaskRuntime(session=session)
        result = runtime.run(str(tmp_path), dataset_name=_DATASET_NAME, model_key=_MODEL_KEY)
        pip_cmds = [c for c in session.commands if c.startswith("pip install")]
        assert pip_cmds, "pip install must be called when dependencies are declared"
        assert "numpy" in pip_cmds[0]
        assert result.rows[0]["value"] == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# 6. execute_env_package dispatcher
# ---------------------------------------------------------------------------


class TestExecuteEnvPackage:
    """execute_env_package dispatches to the correct runtime by package shape."""

    def test_dispatches_to_prime(self, tmp_path):
        (tmp_path / "env.py").write_text("")
        (tmp_path / "pyproject.toml").write_text("")
        session = FakeSandboxSession(
            responses=[
                (0, "", ""),
                (0, "", ""),
                (0, "", ""),
                (0, "reward: 0.8", ""),
            ]
        )
        result = execute_env_package(
            str(tmp_path),
            session,
            dataset_name=_DATASET_NAME,
            model_key=_MODEL_KEY,
            scratch_tag="tag-0",
        )
        assert result.meta["runtime"] == "prime_verifiers"
        assert result.rows[0]["value"] == pytest.approx(0.8)
        assert result.scratch_tag == "tag-0"

    def test_dispatches_to_harbor(self, tmp_path):
        (tmp_path / "task.toml").write_text("[run]\ncommand='true'\n")
        session = FakeSandboxSession(
            responses=[
                (0, "", ""),  # mkdir
                (0, "", ""),  # tar
                (0, "reward: 0.55", ""),  # true
            ]
        )
        result = execute_env_package(
            str(tmp_path),
            session,
            dataset_name=_DATASET_NAME,
            model_key=_MODEL_KEY,
        )
        assert result.meta["runtime"] == "harbor_task"
        assert result.rows[0]["value"] == pytest.approx(0.55)

    def test_task_result_row_schema(self, tmp_path):
        session = FakeSandboxSession(
            responses=[(0, "", ""), (0, "", ""), (0, "", ""), (0, "reward: 0.5", "")]
        )
        result = execute_env_package(
            str(tmp_path), session, dataset_name="ds", model_key="openai/gpt-4"
        )
        row = result.rows[0]
        assert row["metric"] == "reward"
        assert row["direction"] == "max"
        assert 0.0 <= row["value"] <= 1.0
        assert row["dataset"] == "ds"
        assert row["model"] == "openai/gpt-4"

    def test_reward_in_0_1_range(self, tmp_path):
        session = FakeSandboxSession(
            responses=[(0, "", ""), (0, "", ""), (0, "", ""), (0, "reward: 0.9999", "")]
        )
        result = execute_env_package(
            str(tmp_path), session, dataset_name="ds", model_key="m"
        )
        assert 0.0 <= result.rows[0]["value"] <= 1.0


# ---------------------------------------------------------------------------
# 7. Mocked end-to-end: pinned env → reward → RunResult via ResultsSink
# ---------------------------------------------------------------------------


class TestMockedE2EPinnedEnv:
    """Hard-gate: pinned env (Prime AND Harbor) → reward in [0, 1] →
    ``RunResult.subjects[].metrics``, persisted via a ``ResultsSink``.

    The sandbox session is mocked via ``FakeSandboxSession`` — no real
    E2B / Docker calls are made.  The environment backend is mocked to
    return a ``ResolvedEnvironment`` with ``local_path`` set, simulating
    a pinned external package (github_repo / prime_environment_hub).
    """

    def _make_run_payload(self, env_name: str, kind: str = "github_repo") -> dict:
        sandbox_cfg = {"enabled": True, "per_task_sandbox": True}
        return {
            "mode": "generator",
            "selection": {
                "direct_pairs": [
                    {
                        "model": _model_block(),
                        "environment": _env_block_external(
                            env_name=env_name,
                            kind=kind,
                            sandbox=sandbox_cfg,
                        ),
                    }
                ]
            },
            "reuse_policy": {"enabled": False},
        }

    def _mock_resolved_env(self, local_path: str, kind: str = "github_repo"):
        """Build a ResolvedEnvironment mock with local_path set."""
        from src.api.ports.environment_backend import ResolvedEnvironment
        from open_arena_core import models as api

        inline_def = api.InlineEnvironmentDefinition(
            name="mock-env",
            version="0.1.0",
            dataset=api.DatasetBinding(provider="local", source_ref=str(local_path)),
            verifier=api.VerifierSuiteBinding(
                root=api.VerifierSuiteInline(
                    binding_type="inline",
                    name="mock-verifier",
                    metrics=[
                        api.MetricDefinition(
                            name="accuracy",
                            metric_kind="exact_match",
                            weight=1.0,
                        )
                    ],
                )
            ),
            runtime=api.EnvironmentRuntimePolicy(),
        )
        return ResolvedEnvironment(
            definition=inline_def,
            commit_sha="abc123def456",
            content_hash="sha256:deadbeef",
            local_path=str(local_path),
        )

    def _prime_responses(self, reward: float) -> list[tuple[int, str, str]]:
        return [
            (0, "", ""),            # mkdir
            (0, "", ""),            # tar extract
            (0, "", ""),            # pip install
            (0, f"reward: {reward}", ""),  # python runner
        ]

    def _harbor_responses(self, reward: float) -> list[tuple[int, str, str]]:
        return [
            (0, "", ""),                    # mkdir
            (0, "", ""),                    # tar extract
            (0, f"score: {reward}", ""),    # pytest / entry-point
        ]

    # ------------------------------------------------------------------
    # 7a. Prime package end-to-end
    # ------------------------------------------------------------------

    def test_prime_package_end_to_end_via_service_run_one(self, tmp_path):
        """Service-level: pinned Prime env routed via open_session → reward → RunResult.

        Patches ``_env_backend.resolve`` to return a ``ResolvedEnvironment``
        with ``local_path`` set and ``source.kind=github_repo``, then
        patches ``open_session`` to yield a ``FakeSandboxSession`` with canned
        Prime rollout output.  Verifies that the full pipeline produces a
        ``RunResult`` with ``metric="reward"`` persisted via ``ResultsSink``.
        """
        from src.api.ports.environment_backend import ResolvedEnvironment
        from src.api.service import PendingSubject
        from open_arena_core import models as api

        svc = _make_service(tmp_path)

        pkg_dir = tmp_path / "prime_pkg"
        pkg_dir.mkdir()
        (pkg_dir / "env.py").write_text("def load_environment(): return {}")
        (pkg_dir / "pyproject.toml").write_text("[project]\nname='t'\nversion='0.1'\n")

        # Build resolved env with local_path set (simulates pinned github_repo)
        resolved = self._mock_resolved_env(str(pkg_dir), kind="github_repo")

        # Mock env_backend.resolve to return the resolved env with local_path
        svc._env_backend = MagicMock()
        svc._env_backend.resolve.return_value = resolved

        # Override open_session to return a FakeSandboxSession with Prime output
        fake_session = FakeSandboxSession(responses=self._prime_responses(0.85))

        @contextmanager
        def _mock_open_session(policy=None):
            yield fake_session

        svc._sandbox.open_session = _mock_open_session  # type: ignore[method-assign]

        # Build a minimal PendingSubject with a github_repo source kind
        now = __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
        model_def = api.ModelDefinition(
            id=uuid4(),
            name="test-model",
            runtime=api.ModelExecutionConfig(
                provider="openai",
                model_name="gpt-4o-mini",
                model_version="2024-07-18",
            ),
            created_at=now,
            updated_at=now,
        )
        env_def = api.InlineEnvironmentDefinition(
            name="prime-env",
            version="0.1.0",
            dataset=api.DatasetBinding(provider="local", source_ref="data"),
            verifier=api.VerifierSuiteBinding(
                root=api.VerifierSuiteInline(
                    binding_type="inline",
                    name="v",
                    metrics=[
                        api.MetricDefinition(
                            name="accuracy",
                            metric_kind="exact_match",
                            weight=1.0,
                        )
                    ],
                )
            ),
            runtime=api.EnvironmentRuntimePolicy(supported_modes=["generator"]),
            sandbox=api.SandboxPolicy(enabled=True, per_task_sandbox=True),
        )
        environment = api.Environment(
            id=uuid4(),
            source=api.EnvironmentSource(
                kind=api.EnvironmentSourceKind.github_repo,
                name="prime-env",
                version="0.1.0",
                uri="https://github.com/org/prime-env",
            ),
            inline_definition=env_def,
            created_at=now,
            updated_at=now,
        )

        pending = [
            PendingSubject(
                model=model_def,
                environment=environment,
                fingerprint="fp-prime-001",
            )
        ]

        # Call _run_per_task_fan_out directly
        subjects = svc._run_per_task_fan_out(
            api.RunMode.generator,
            None,
            pending,
        )

        assert len(subjects) == 1
        subject = subjects[0]
        assert subject.metrics, "subject must have metrics"

        reward_metric = next((m for m in subject.metrics if m.name == "reward"), None)
        assert reward_metric is not None, "reward metric must be present"
        assert 0.0 <= reward_metric.value <= 1.0
        assert reward_metric.value == pytest.approx(0.85)
        assert reward_metric.direction == api.Direction.max

        # Persist via ResultsSink and verify storage
        from src.api.ports.results_sink import StoreResultsSink

        run_id = uuid4()
        lb_id = uuid4()
        run = api.Run(
            id=run_id,
            mode=api.RunMode.generator,
            selection=api.RunSelection(root=api.RunSelection1(leaderboard_id=lb_id)),
            status=api.RunStatus.succeeded,
            cache_status=api.CacheStatus.miss,
            created_at=now,
        )
        run_result = api.RunResult(
            run_id=run_id,
            mode=api.RunMode.generator,
            subjects=subjects,
            aggregates=[
                api.AggregateMetric(
                    name="reward",
                    value=reward_metric.value,
                    aggregation="weighted_mean",
                )
            ],
        )

        sink = StoreResultsSink(store=svc.store)
        svc.store.save_run(run)
        sink.write(run, run_result)

        stored = svc.store.get_run_result(run_id)
        assert stored is not None
        assert len(stored.subjects) == 1
        stored_reward = next(
            m for m in stored.subjects[0].metrics if m.name == "reward"
        )
        assert stored_reward.value == pytest.approx(0.85)
        assert stored_reward.direction == api.Direction.max

    def test_prime_package_e2e_via_direct_env_runtime_call(self, tmp_path):
        """Direct test of execute_env_package integration via TaskResult assembly.

        This is the true hard-gate test: a FakeSandboxSession whose run_command
        returns canned rollout output / verifier exit+score → TaskResult.rows →
        SubjectResult.metrics → RunResult persisted via ResultsSink.
        """
        from src.api.ports.results_sink import StoreResultsSink
        from open_arena_core import models as api

        svc = _make_service(tmp_path)

        pkg_dir = tmp_path / "prime_pkg"
        pkg_dir.mkdir()
        (pkg_dir / "env.py").write_text("def load_environment(): return {}")
        (pkg_dir / "pyproject.toml").write_text("[project]\nname='t'\nversion='0.1'\n")

        # Directly test execute_env_package returns the right TaskResult
        fake_session = FakeSandboxSession(
            responses=[
                (0, "", ""),
                (0, "", ""),
                (0, "", ""),
                (0, "reward: 0.77", ""),
            ]
        )
        task_result = execute_env_package(
            str(pkg_dir),
            fake_session,
            dataset_name="my-dataset",
            model_key="openai/gpt-4o",
            scratch_tag="my-dataset.openai/gpt-4o.0",
        )

        assert len(task_result.rows) == 1
        row = task_result.rows[0]
        assert row["metric"] == "reward"
        assert row["value"] == pytest.approx(0.77)
        assert row["direction"] == "max"
        assert row["dataset"] == "my-dataset"
        assert row["model"] == "openai/gpt-4o"
        assert task_result.scratch_tag == "my-dataset.openai/gpt-4o.0"

        # Now assemble into SubjectResult + RunResult + persist via ResultsSink
        now = __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
        model_def = api.ModelDefinition(
            id=uuid4(),
            name="test-model",
            runtime=api.ModelExecutionConfig(
                provider="openai",
                model_name="gpt-4o",
                model_version="2024-07-18",
            ),
            created_at=now,
            updated_at=now,
        )
        inline_def = api.InlineEnvironmentDefinition(
            name="test-env",
            version="0.1.0",
            dataset=api.DatasetBinding(provider="local", source_ref="data.jsonl"),
            verifier=api.VerifierSuiteBinding(
                root=api.VerifierSuiteInline(
                    binding_type="inline",
                    name="verifier",
                    metrics=[
                        api.MetricDefinition(
                            name="accuracy",
                            metric_kind="exact_match",
                            weight=1.0,
                        )
                    ],
                )
            ),
            runtime=api.EnvironmentRuntimePolicy(),
        )
        env_id = uuid4()
        environment = api.Environment(
            id=env_id,
            source=api.EnvironmentSource(
                kind=api.EnvironmentSourceKind.inline,
                name="test-env",
                version="0.1.0",
            ),
            inline_definition=inline_def,
            created_at=now,
            updated_at=now,
        )

        # Build SubjectResult from TaskResult rows (mirrors service._run_per_task_fan_out logic)
        metrics = [
            api.MetricResult(
                name=row["metric"],
                value=float(row["value"]),
                direction=api.Direction(row["direction"]),
            )
            for row in task_result.rows
            if row["value"] is not None
        ]
        subject = api.SubjectResult(
            model=model_def,
            environment=environment,
            metrics=metrics,
            cache_status=api.CacheStatus.miss,
            trajectory_summary=None,
            run_fingerprint="fp-abc123",
        )

        # Verify reward is in [0, 1] and direction is max
        reward_metric = next(m for m in subject.metrics if m.name == "reward")
        assert 0.0 <= reward_metric.value <= 1.0
        assert reward_metric.direction == api.Direction.max

        # Build RunResult and persist via ResultsSink
        run_id = uuid4()
        lb_id = uuid4()
        run = api.Run(
            id=run_id,
            mode=api.RunMode.generator,
            selection=api.RunSelection(root=api.RunSelection1(leaderboard_id=lb_id)),
            status=api.RunStatus.succeeded,
            cache_status=api.CacheStatus.miss,
            created_at=now,
        )
        run_result = api.RunResult(
            run_id=run_id,
            mode=api.RunMode.generator,
            subjects=[subject],
            aggregates=[
                api.AggregateMetric(
                    name="reward",
                    value=reward_metric.value,
                    aggregation="weighted_mean",
                )
            ],
        )

        # Persist via StoreResultsSink (the default ResultsSink)
        sink = StoreResultsSink(store=svc.store)
        svc.store.save_run(run)
        sink.write(run, run_result)

        # Verify the result is stored and retrievable
        stored = svc.store.get_run_result(run_id)
        assert stored is not None
        assert len(stored.subjects) == 1
        stored_subject = stored.subjects[0]
        stored_reward = next(m for m in stored_subject.metrics if m.name == "reward")
        assert stored_reward.value == pytest.approx(0.77)
        assert stored_reward.direction == api.Direction.max

    # ------------------------------------------------------------------
    # 7b. Harbor task end-to-end
    # ------------------------------------------------------------------

    def test_harbor_task_e2e_via_direct_env_runtime_call(self, tmp_path):
        """Direct test of HarborTaskRuntime integration: canned session →
        TaskResult.rows → SubjectResult → RunResult → ResultsSink.
        """
        from src.api.ports.results_sink import StoreResultsSink
        from open_arena_core import models as api

        svc = _make_service(tmp_path)

        pkg_dir = tmp_path / "harbor_task"
        pkg_dir.mkdir()
        (pkg_dir / "task.toml").write_text("[run]\ncommand='python -m pytest tests/'\n")
        tests_dir = pkg_dir / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_eval.py").write_text(
            "def test_score():\n    assert True\n"
        )

        # Canned responses for HarborTaskRuntime lifecycle
        fake_session = FakeSandboxSession(
            responses=[
                (0, "", ""),                 # mkdir
                (0, "", ""),                 # tar
                (0, "score: 0.92", ""),      # pytest returns a score line
            ]
        )
        task_result = execute_env_package(
            str(pkg_dir),
            fake_session,
            dataset_name="harbor-dataset",
            model_key="openai/gpt-4o-mini",
            scratch_tag="harbor-dataset.openai/gpt-4o-mini.0",
        )

        assert task_result.meta["runtime"] == "harbor_task"
        row = task_result.rows[0]
        assert row["metric"] == "reward"
        assert row["value"] == pytest.approx(0.92)
        assert 0.0 <= row["value"] <= 1.0
        assert row["direction"] == "max"

        # Assemble → SubjectResult → RunResult → persist via ResultsSink
        now = __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
        model_def = api.ModelDefinition(
            id=uuid4(),
            name="test-model",
            runtime=api.ModelExecutionConfig(
                provider="openai",
                model_name="gpt-4o-mini",
                model_version="2024-07-18",
            ),
            created_at=now,
            updated_at=now,
        )
        env_id = uuid4()
        environment = api.Environment(
            id=env_id,
            source=api.EnvironmentSource(
                kind=api.EnvironmentSourceKind.inline,
                name="harbor-env",
                version="0.1.0",
            ),
            inline_definition=api.InlineEnvironmentDefinition(
                name="harbor-env",
                version="0.1.0",
                dataset=api.DatasetBinding(provider="local", source_ref="data"),
                verifier=api.VerifierSuiteBinding(
                    root=api.VerifierSuiteInline(
                        binding_type="inline",
                        name="v",
                        metrics=[
                            api.MetricDefinition(
                                name="accuracy",
                                metric_kind="exact_match",
                                weight=1.0,
                            )
                        ],
                    )
                ),
                runtime=api.EnvironmentRuntimePolicy(),
            ),
            created_at=now,
            updated_at=now,
        )

        metrics = [
            api.MetricResult(
                name=r["metric"],
                value=float(r["value"]),
                direction=api.Direction(r["direction"]),
            )
            for r in task_result.rows
            if r["value"] is not None
        ]
        subject = api.SubjectResult(
            model=model_def,
            environment=environment,
            metrics=metrics,
            cache_status=api.CacheStatus.miss,
            trajectory_summary=None,
            run_fingerprint="fp-harbor-001",
        )

        reward_m = next(m for m in subject.metrics if m.name == "reward")
        assert 0.0 <= reward_m.value <= 1.0
        assert reward_m.direction == api.Direction.max

        run_id = uuid4()
        lb_id = uuid4()
        run = api.Run(
            id=run_id,
            mode=api.RunMode.generator,
            selection=api.RunSelection(root=api.RunSelection1(leaderboard_id=lb_id)),
            status=api.RunStatus.succeeded,
            cache_status=api.CacheStatus.miss,
            created_at=now,
        )
        run_result = api.RunResult(
            run_id=run_id,
            mode=api.RunMode.generator,
            subjects=[subject],
            aggregates=[
                api.AggregateMetric(
                    name="reward",
                    value=reward_m.value,
                    aggregation="weighted_mean",
                )
            ],
        )

        sink = StoreResultsSink(store=svc.store)
        svc.store.save_run(run)
        sink.write(run, run_result)

        stored = svc.store.get_run_result(run_id)
        assert stored is not None
        stored_reward = next(m for m in stored.subjects[0].metrics if m.name == "reward")
        assert stored_reward.value == pytest.approx(0.92)
        assert stored_reward.direction == api.Direction.max

    # ------------------------------------------------------------------
    # 7c. Service-level integration: open_session hook
    # ------------------------------------------------------------------

    def test_sandbox_open_session_returns_session(self, tmp_path):
        """SandboxProvider.open_session() yields a SandboxSession-compatible object."""
        from src.api.ports.sandbox_provider import LocalSandboxProvider
        from src.api.sandboxes.env_runtime import SandboxSession

        provider = LocalSandboxProvider()
        with provider.open_session() as session:
            assert isinstance(session, SandboxSession)

    def test_is_pinned_external_false_for_inline_env(self, tmp_path):
        """Inline envs do NOT route to env-package runtime (inline path unchanged)."""
        svc = _make_service(tmp_path)
        from open_arena_core.models import EnvironmentSourceKind

        # Build a minimal inline environment
        from open_arena_core import models as api

        now = __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
        env = api.Environment(
            id=uuid4(),
            source=api.EnvironmentSource(
                kind=EnvironmentSourceKind.inline,
                name="inline-env",
                version="0.1.0",
            ),
            inline_definition=api.InlineEnvironmentDefinition(
                name="inline-env",
                version="0.1.0",
                dataset=api.DatasetBinding(provider="local", source_ref="data"),
                verifier=api.VerifierSuiteBinding(
                    root=api.VerifierSuiteInline(
                        binding_type="inline",
                        name="v",
                        metrics=[
                            api.MetricDefinition(
                                name="accuracy",
                                metric_kind="exact_match",
                                weight=1.0,
                            )
                        ],
                    )
                ),
                runtime=api.EnvironmentRuntimePolicy(),
            ),
            created_at=now,
            updated_at=now,
        )

        # Inline environments resolve without local_path → is_pinned_external = False
        from src.api.ports.environment_backend import InlineEnvironmentBackend
        backend = InlineEnvironmentBackend()
        resolved = backend.resolve(env)
        assert resolved.local_path is None
        # is_pinned_external check
        is_external = (
            resolved.local_path is not None
            and env.source.kind in (
                EnvironmentSourceKind.github_repo,
                EnvironmentSourceKind.prime_environment_hub,
            )
        )
        assert not is_external, "inline env must NOT be routed to env-package runtime"
