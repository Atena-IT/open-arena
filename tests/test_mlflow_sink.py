# License Apache 2.0: (c) 2026 Athena-Reply
"""Tests for :class:`~src.api.sinks.mlflow_sink.MlflowResultsSink` (WS5 / issue #39).

All MLflow calls are mocked so the test suite does not require a running
tracking server (or even the ``mlflow`` extra installed).

Test coverage
-------------
* Experiment naming: ``{org}/{project}``, ``open-arena/{project}``, and the
  ``open-arena/default`` fallback.
* Metrics logged per :class:`~src.api.models.MetricResult`.
* Params logged (provider, name, version, environment, mode, temperature,
  max_tokens).
* Tags set when ``unity_catalog.full_name``, ``storage.uri``, and
  ``artifact.sha256`` are present in metadata; not set when absent.
* ``open_arena.run_id`` and ``open_arena.run_fingerprint`` tags.
* Composed store write always runs (even when MLflow raises).
* Graceful degradation: if MLflow raises, the sink logs a warning but does
  NOT propagate the exception.
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

from src.api import models as api


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _now() -> datetime:
    return datetime.now(UTC)


def _make_model(
    name: str = "test-model",
    provider: str = "openai",
    model_name: str = "gpt-4o",
    model_version: str = "1.0",
    temperature: float | None = 0.7,
    max_tokens: int | None = 512,
    metadata: dict[str, Any] | None = None,
) -> api.ModelDefinition:
    return api.ModelDefinition(
        id=uuid.uuid4(),
        name=name,
        runtime=api.ModelExecutionConfig(
            provider=provider,
            model_name=model_name,
            model_version=model_version,
            temperature=temperature,
            max_tokens=max_tokens,
        ),
        metadata=metadata,
        created_at=_now(),
        updated_at=_now(),
    )


def _make_environment(
    name: str = "test-env",
    version: str = "1.0",
    metadata: dict[str, Any] | None = None,
) -> api.Environment:
    return api.Environment(
        id=uuid.uuid4(),
        source=api.EnvironmentSource(
            kind=api.EnvironmentSourceKind.inline,
            name=name,
            version=version,
            metadata=metadata,
        ),
        metadata=metadata,
        created_at=_now(),
        updated_at=_now(),
    )


def _make_subject(
    model: api.ModelDefinition | None = None,
    environment: api.Environment | None = None,
    metrics: list[api.MetricResult] | None = None,
    run_fingerprint: str | None = "fp-abc123",
) -> api.SubjectResult:
    return api.SubjectResult(
        model=model or _make_model(),
        environment=environment or _make_environment(),
        metrics=metrics or [
            api.MetricResult(name="accuracy", value=0.95, direction="max"),
            api.MetricResult(name="latency_ms", value=120.0, direction="min"),
        ],
        cache_status=api.CacheStatus.miss,
        run_fingerprint=run_fingerprint,
    )


def _make_run(
    labels: dict[str, str] | None = None,
    mode: api.RunMode = api.RunMode.generator,
) -> api.Run:
    leaderboard_id = uuid.uuid4()
    return api.Run(
        id=uuid.uuid4(),
        mode=mode,
        selection=api.RunSelection(
            root=api.RunSelection1(leaderboard_id=leaderboard_id)
        ),
        status=api.RunStatus.succeeded,
        cache_status=api.CacheStatus.miss,
        labels=labels,
        created_at=_now(),
    )


def _make_run_result(
    run: api.Run,
    subjects: list[api.SubjectResult] | None = None,
) -> api.RunResult:
    return api.RunResult(
        run_id=run.id,
        mode=run.mode,
        subjects=subjects or [_make_subject()],
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_sink(store_mock=None):
    """Return an MlflowResultsSink with a mocked store."""
    from src.api.sinks.mlflow_sink import MlflowResultsSink

    if store_mock is None:
        store_mock = MagicMock()
    return MlflowResultsSink(store=store_mock, tracking_uri="http://localhost:5000"), store_mock


# ---------------------------------------------------------------------------
# Experiment naming tests
# ---------------------------------------------------------------------------

class TestExperimentNaming:
    """Verify the {org}/{project} experiment naming logic."""

    def test_org_and_project_labels(self):
        from src.api.sinks.mlflow_sink import _experiment_name

        run = _make_run(labels={"org": "acme", "project": "chat-eval"})
        assert _experiment_name(run) == "acme/chat-eval"

    def test_project_only_label(self):
        from src.api.sinks.mlflow_sink import _experiment_name

        run = _make_run(labels={"project": "chat-eval"})
        assert _experiment_name(run) == "open-arena/chat-eval"

    def test_no_labels_fallback(self):
        from src.api.sinks.mlflow_sink import _experiment_name

        run = _make_run(labels=None)
        assert _experiment_name(run) == "open-arena/default"

    def test_empty_labels_fallback(self):
        from src.api.sinks.mlflow_sink import _experiment_name

        run = _make_run(labels={})
        assert _experiment_name(run) == "open-arena/default"

    def test_org_without_project(self):
        from src.api.sinks.mlflow_sink import _experiment_name

        run = _make_run(labels={"org": "acme"})
        assert _experiment_name(run) == "open-arena/default"


# ---------------------------------------------------------------------------
# Metrics logging tests
# ---------------------------------------------------------------------------

class TestMetricsLogging:
    """Assert all MetricResult values are forwarded to mlflow.log_metric."""

    def test_metrics_logged_per_subject(self):
        sink, store_mock = _build_sink()
        run = _make_run(labels={"org": "acme", "project": "eval"})
        subject = _make_subject(
            metrics=[
                api.MetricResult(name="f1", value=0.88, direction="max"),
                api.MetricResult(name="bleu", value=0.72, direction="max"),
            ]
        )
        result = _make_run_result(run, subjects=[subject])

        mlflow_mock = _make_mlflow_mock()

        with patch.dict("sys.modules", {"mlflow": mlflow_mock}):
            sink.write(run, result)

        logged_metrics = {
            c.args[0]: c.args[1]
            for c in mlflow_mock.log_metric.call_args_list
        }
        assert logged_metrics["f1"] == pytest.approx(0.88)
        assert logged_metrics["bleu"] == pytest.approx(0.72)

    def test_multiple_subjects_each_get_own_run(self):
        sink, store_mock = _build_sink()
        run = _make_run(labels={"org": "acme", "project": "eval"})
        subjects = [_make_subject(), _make_subject()]
        result = _make_run_result(run, subjects=subjects)

        mlflow_mock = _make_mlflow_mock()

        with patch.dict("sys.modules", {"mlflow": mlflow_mock}):
            sink.write(run, result)

        # start_run must have been called once per subject
        assert mlflow_mock.start_run.call_count == len(subjects)


# ---------------------------------------------------------------------------
# Params logging tests
# ---------------------------------------------------------------------------

class TestParamsLogging:
    """Assert model / env params are forwarded to mlflow.log_params."""

    def test_core_params_logged(self):
        sink, store_mock = _build_sink()
        run = _make_run()
        subject = _make_subject(
            model=_make_model(
                provider="anthropic",
                model_name="claude-3-5-sonnet",
                model_version="20241022",
                temperature=0.5,
                max_tokens=1024,
            ),
            environment=_make_environment(name="math-env", version="2.0"),
        )
        result = _make_run_result(run, subjects=[subject])

        mlflow_mock = _make_mlflow_mock()
        with patch.dict("sys.modules", {"mlflow": mlflow_mock}):
            sink.write(run, result)

        logged_params = mlflow_mock.log_params.call_args[0][0]
        assert logged_params["model.provider"] == "anthropic"
        assert logged_params["model.name"] == "claude-3-5-sonnet"
        assert logged_params["model.version"] == "20241022"
        assert logged_params["environment.name"] == "math-env"
        assert logged_params["environment.version"] == "2.0"
        assert logged_params["run.mode"] == "generator"
        assert logged_params["model.temperature"] == "0.5"
        assert logged_params["model.max_tokens"] == "1024"

    def test_optional_params_omitted_when_none(self):
        sink, store_mock = _build_sink()
        run = _make_run()
        subject = _make_subject(
            model=_make_model(temperature=None, max_tokens=None)
        )
        result = _make_run_result(run, subjects=[subject])

        mlflow_mock = _make_mlflow_mock()
        with patch.dict("sys.modules", {"mlflow": mlflow_mock}):
            sink.write(run, result)

        logged_params = mlflow_mock.log_params.call_args[0][0]
        assert "model.temperature" not in logged_params
        assert "model.max_tokens" not in logged_params


# ---------------------------------------------------------------------------
# Tags tests
# ---------------------------------------------------------------------------

class TestTagsLogging:
    """Assert UC / storage / SHA256 tags are only set when values are present."""

    def test_uc_storage_sha_tags_from_environment_metadata(self):
        sink, store_mock = _build_sink()
        run = _make_run()
        env_metadata = {
            "unity_catalog.full_name": "main.arena.math_v2",
            "storage.uri": "s3://my-bucket/datasets/math/v2",
            "artifact.sha256": "deadbeef1234567890",
        }
        subject = _make_subject(environment=_make_environment(metadata=env_metadata))
        result = _make_run_result(run, subjects=[subject])

        mlflow_mock = _make_mlflow_mock()
        with patch.dict("sys.modules", {"mlflow": mlflow_mock}):
            sink.write(run, result)

        tags_used = mlflow_mock.start_run.call_args.kwargs.get("tags", {})
        assert tags_used["unity_catalog.full_name"] == "main.arena.math_v2"
        assert tags_used["storage.uri"] == "s3://my-bucket/datasets/math/v2"
        assert tags_used["artifact.sha256"] == "deadbeef1234567890"

    def test_uc_storage_sha_tags_from_model_metadata(self):
        """Tags can also be sourced from model metadata."""
        sink, store_mock = _build_sink()
        run = _make_run()
        model_metadata = {
            "unity_catalog.full_name": "main.models.gpt4o",
            "storage.uri": "dbfs:/models/gpt4o",
            "artifact.sha256": "cafebabe",
        }
        subject = _make_subject(model=_make_model(metadata=model_metadata))
        result = _make_run_result(run, subjects=[subject])

        mlflow_mock = _make_mlflow_mock()
        with patch.dict("sys.modules", {"mlflow": mlflow_mock}):
            sink.write(run, result)

        tags_used = mlflow_mock.start_run.call_args.kwargs.get("tags", {})
        assert tags_used["unity_catalog.full_name"] == "main.models.gpt4o"
        assert tags_used["storage.uri"] == "dbfs:/models/gpt4o"
        assert tags_used["artifact.sha256"] == "cafebabe"

    def test_uc_tags_absent_when_no_metadata(self):
        sink, store_mock = _build_sink()
        run = _make_run()
        subject = _make_subject()  # no metadata on model or env
        result = _make_run_result(run, subjects=[subject])

        mlflow_mock = _make_mlflow_mock()
        with patch.dict("sys.modules", {"mlflow": mlflow_mock}):
            sink.write(run, result)

        tags_used = mlflow_mock.start_run.call_args.kwargs.get("tags", {})
        assert "unity_catalog.full_name" not in tags_used
        assert "storage.uri" not in tags_used
        assert "artifact.sha256" not in tags_used

    def test_run_id_tag_always_present(self):
        sink, store_mock = _build_sink()
        run = _make_run()
        subject = _make_subject()
        result = _make_run_result(run, subjects=[subject])

        mlflow_mock = _make_mlflow_mock()
        with patch.dict("sys.modules", {"mlflow": mlflow_mock}):
            sink.write(run, result)

        tags_used = mlflow_mock.start_run.call_args.kwargs.get("tags", {})
        assert tags_used["open_arena.run_id"] == str(run.id)

    def test_fingerprint_tag_present_when_set(self):
        sink, store_mock = _build_sink()
        run = _make_run()
        subject = _make_subject(run_fingerprint="fp-xyz")
        result = _make_run_result(run, subjects=[subject])

        mlflow_mock = _make_mlflow_mock()
        with patch.dict("sys.modules", {"mlflow": mlflow_mock}):
            sink.write(run, result)

        tags_used = mlflow_mock.start_run.call_args.kwargs.get("tags", {})
        assert tags_used["open_arena.run_fingerprint"] == "fp-xyz"

    def test_fingerprint_tag_absent_when_none(self):
        sink, store_mock = _build_sink()
        run = _make_run()
        subject = _make_subject(run_fingerprint=None)
        result = _make_run_result(run, subjects=[subject])

        mlflow_mock = _make_mlflow_mock()
        with patch.dict("sys.modules", {"mlflow": mlflow_mock}):
            sink.write(run, result)

        tags_used = mlflow_mock.start_run.call_args.kwargs.get("tags", {})
        assert "open_arena.run_fingerprint" not in tags_used


# ---------------------------------------------------------------------------
# Composed store write
# ---------------------------------------------------------------------------

class TestStoreComposition:
    """The store write must always happen regardless of MLflow outcome."""

    def test_store_sink_write_called_on_success(self):
        sink, store_mock = _build_sink()
        run = _make_run()
        result = _make_run_result(run)

        mlflow_mock = _make_mlflow_mock()
        with patch.dict("sys.modules", {"mlflow": mlflow_mock}):
            sink.write(run, result)

        store_mock.save_run_result.assert_called_once_with(result)

    def test_store_sink_write_called_when_mlflow_raises(self):
        sink, store_mock = _build_sink()
        run = _make_run()
        result = _make_run_result(run)

        mlflow_mock = _make_mlflow_mock()
        mlflow_mock.get_experiment_by_name.side_effect = RuntimeError("connection refused")

        with patch.dict("sys.modules", {"mlflow": mlflow_mock}):
            # Must not raise
            sink.write(run, result)

        # Store write still happened
        store_mock.save_run_result.assert_called_once_with(result)


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------

class TestGracefulDegradation:
    """MLflow errors must be swallowed; the sink should not re-raise them."""

    def test_mlflow_connection_error_does_not_propagate(self):
        sink, store_mock = _build_sink()
        run = _make_run()
        result = _make_run_result(run)

        mlflow_mock = _make_mlflow_mock()
        mlflow_mock.start_run.side_effect = ConnectionError("MLflow server unreachable")

        with patch.dict("sys.modules", {"mlflow": mlflow_mock}):
            # Should not raise
            sink.write(run, result)

    def test_mlflow_auth_error_does_not_propagate(self):
        sink, store_mock = _build_sink()
        run = _make_run()
        result = _make_run_result(run)

        mlflow_mock = _make_mlflow_mock()
        mlflow_mock.set_tracking_uri.side_effect = PermissionError("auth failed")

        with patch.dict("sys.modules", {"mlflow": mlflow_mock}):
            sink.write(run, result)

    def test_warning_emitted_on_mlflow_error(self, caplog):
        import logging

        sink, store_mock = _build_sink()
        run = _make_run()
        result = _make_run_result(run)

        mlflow_mock = _make_mlflow_mock()
        mlflow_mock.get_experiment_by_name.side_effect = TimeoutError("timeout")

        with patch.dict("sys.modules", {"mlflow": mlflow_mock}):
            with caplog.at_level(logging.WARNING, logger="open-arena.mlflow_sink"):
                sink.write(run, result)

        assert any("MLflow logging failed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------

class TestRegistryWiring:
    """Verify _build_results_sink returns MlflowResultsSink for settings.results_sink='mlflow'."""

    def test_registry_returns_mlflow_sink_when_configured(self, tmp_path):
        from src.api.registry import _build_results_sink
        from src.api.settings import ArenaSettings
        from src.api.stores.sqlite import SQLiteStore
        from src.api.sinks.mlflow_sink import MlflowResultsSink

        store = SQLiteStore(path=tmp_path / "test.db")
        settings = ArenaSettings(results_sink="mlflow")
        sink = _build_results_sink(store, settings)
        assert isinstance(sink, MlflowResultsSink)

    def test_registry_returns_store_sink_by_default(self, tmp_path):
        from src.api.registry import _build_results_sink
        from src.api.settings import ArenaSettings
        from src.api.stores.sqlite import SQLiteStore
        from src.api.ports.results_sink import StoreResultsSink

        store = SQLiteStore(path=tmp_path / "test.db")
        settings = ArenaSettings(results_sink="store")
        sink = _build_results_sink(store, settings)
        assert isinstance(sink, StoreResultsSink)


# ---------------------------------------------------------------------------
# Internal helper: build a usable mlflow mock
# ---------------------------------------------------------------------------

def _make_mlflow_mock() -> MagicMock:
    """Return a MagicMock that mimics the mlflow module interface."""
    mlflow_mock = MagicMock(name="mlflow")

    # Experiment helpers
    mlflow_mock.get_experiment_by_name.return_value = None  # experiment not found → create
    mlflow_mock.create_experiment.return_value = "1"

    # Context manager for start_run
    run_ctx = MagicMock()
    mlflow_mock.start_run.return_value.__enter__ = lambda s: run_ctx
    mlflow_mock.start_run.return_value.__exit__ = MagicMock(return_value=False)

    return mlflow_mock
