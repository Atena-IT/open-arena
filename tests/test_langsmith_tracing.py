# License Apache 2.0: (c) 2026 Atena-Reply

"""LangSmith gets a copy of the traces, over MLflow's OTLP dual export, only when switched on, each trace in its experiment's project."""

import importlib.util
import os

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import mlflow
import mlflow.tracing.config
import mlflow.tracing.provider
import pytest

from openarena.programs import observability

OTLP = (
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
    "OTEL_EXPORTER_OTLP_TRACES_HEADERS",
    "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL",
    "OTEL_EXPORTER_OTLP_TRACES_TIMEOUT",
    "MLFLOW_TRACE_ENABLE_OTLP_DUAL_EXPORT",
    "MLFLOW_ENABLE_OTEL_GENAI_SEMCONV",
)


def _clean(monkeypatch):
    """No LangSmith or OTLP variable, a fresh tracing config, and the engine's LangSmith state empty; all undone after."""
    for name in (
        "ARENA_LANGSMITH_TRACING",
        "LANGSMITH_API_KEY",
        "LANGSMITH_ENDPOINT",
        *OTLP,
    ):
        monkeypatch.setenv(name, "")  # so the delete below is undone after the test
        monkeypatch.delenv(name)
    monkeypatch.setattr(
        mlflow.tracing.config,
        "_MLFLOW_TRACING_CONFIG",
        mlflow.tracing.config.TracingConfig(),
    )
    monkeypatch.setattr(observability, "_langsmith_previous", {})
    monkeypatch.setattr(observability, "_experiment_names", {})


@pytest.fixture
def resets(monkeypatch):
    """MLflow tracing on (no server to ask), no LangSmith setting, and every provider reset counted."""
    _clean(monkeypatch)
    monkeypatch.setenv("ARENA_MLFLOW_TRACING", "1")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///unused.db")
    monkeypatch.setattr(
        observability.synalinks, "enable_observability", lambda **kw: None
    )
    counted = []
    monkeypatch.setattr(mlflow.tracing.provider, "reset", lambda: counted.append(1))
    return counted


def _switch_on(monkeypatch, key="k"):
    monkeypatch.setenv("ARENA_LANGSMITH_TRACING", "1")
    monkeypatch.setenv("LANGSMITH_API_KEY", key)


def _routing():
    return (
        observability._langsmith_project
        in mlflow.tracing.config.get_config().span_processors
    )


def test_off_by_default(resets):
    assert observability.langsmith_wanted() is False
    assert observability.enable_observability("exp") is True
    assert not any(name in os.environ for name in OTLP)
    assert not _routing()


def test_key_alone_does_not_export(resets, monkeypatch):
    """`LANGSMITH_API_KEY` is also a dataset source's key: holding it must not send traces anywhere."""
    monkeypatch.setenv("LANGSMITH_API_KEY", "k")
    assert observability.langsmith_wanted() is False
    observability.enable_observability("exp")
    assert not any(name in os.environ for name in OTLP)


def test_env_points_at_langsmith_with_the_key_only(resets, monkeypatch):
    """No project header: it would apply to every span the process exports, the assistant's and a run's alike."""
    _switch_on(monkeypatch)
    assert observability.langsmith_wanted() is True
    env = observability.langsmith_env()
    assert (
        env["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"]
        == "https://api.smith.langchain.com/otel/v1/traces"
    )
    assert env["OTEL_EXPORTER_OTLP_TRACES_HEADERS"] == "x-api-key=k"
    assert env["OTEL_EXPORTER_OTLP_TRACES_PROTOCOL"] == "http/protobuf"
    assert env["MLFLOW_TRACE_ENABLE_OTLP_DUAL_EXPORT"] == "true"
    assert env["MLFLOW_ENABLE_OTEL_GENAI_SEMCONV"] == "true"
    monkeypatch.setenv("LANGSMITH_ENDPOINT", "https://eu.api.smith.langchain.com/")
    assert (
        observability.langsmith_env()["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"]
        == "https://eu.api.smith.langchain.com/otel/v1/traces"
    )


def test_exporter_timeout_is_short(resets, monkeypatch):
    """A LangSmith that does not answer costs a rebuild this many seconds at most, not the exporter's 10 with retries."""
    _switch_on(monkeypatch)
    assert observability.langsmith_env()["OTEL_EXPORTER_OTLP_TRACES_TIMEOUT"] == "5"


def test_runs_do_not_rebuild_the_provider(resets, monkeypatch):
    """The backend enables a run's experiment, then the base one again at its first cell: neither may drop open spans."""
    _switch_on(monkeypatch)
    observability.enable_observability("a")
    observability.enable_observability("b")
    observability.enable_observability()
    assert len(resets) == 1
    assert _routing()


def test_routing_outlives_a_provider_reset(monkeypatch):
    """MLflow's provider reset also empties its tracing config (a new tracking URI resets it too): the router must be there after it."""
    _clean(monkeypatch)
    monkeypatch.setenv("ARENA_MLFLOW_TRACING", "1")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///unused.db")
    monkeypatch.setattr(
        observability.synalinks, "enable_observability", lambda **kw: None
    )
    _switch_on(monkeypatch)
    previous_uri = mlflow.get_tracking_uri()
    try:
        observability.enable_observability("a")
        assert _routing()
        # A synalinks hook sets the URI on its first call: no change, no reset.
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        assert _routing()
        mlflow.tracing.provider.reset()  # what a changed tracking URI does
        observability.enable_observability("b")
        assert _routing()
    finally:
        observability._apply_langsmith(False)
        mlflow.set_tracking_uri(previous_uri)


def test_switching_off_gives_back_the_deployments_vars(resets, monkeypatch):
    """An OTLP collector the deployment set itself is LangSmith's while it is on, and the deployment's again after."""
    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://collector:4318/v1/traces"
    )
    _switch_on(monkeypatch)
    observability.enable_observability("a")
    assert os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"].startswith(
        "https://api.smith.langchain.com"
    )
    monkeypatch.setenv("ARENA_LANGSMITH_TRACING", "0")
    observability.enable_observability("b")
    assert (
        os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"]
        == "http://collector:4318/v1/traces"
    )
    assert not any(
        name in os.environ
        for name in OTLP
        if name != "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"
    )
    assert not _routing()
    assert len(resets) == 2


def test_off_leaves_foreign_otlp_vars(resets, monkeypatch):
    """A deployment's own OTLP collector is not the engine's to touch while LangSmith is off."""
    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://collector:4318/v1/traces"
    )
    observability.enable_observability("a")
    assert (
        os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"]
        == "http://collector:4318/v1/traces"
    )
    assert resets == []


def test_rides_on_mlflow(resets, monkeypatch):
    """LangSmith gets a copy of what MLflow traces: with MLflow tracing off there is nothing to copy."""
    _switch_on(monkeypatch)
    monkeypatch.setenv("ARENA_MLFLOW_TRACING", "0")
    assert observability.enable_observability("a") is False
    assert not any(name in os.environ for name in OTLP)


def test_a_missing_otlp_exporter_keeps_mlflow_tracing(resets, monkeypatch):
    """Without the OTLP exporter MLflow cannot build its tracer provider, and every span, MLflow's own too, is dropped."""
    _switch_on(monkeypatch)
    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a: (
            None if name.startswith("opentelemetry.exporter.otlp") else real(name, *a)
        ),
    )
    assert observability.langsmith_wanted() is False
    assert observability.enable_observability("a") is True
    assert not any(name in os.environ for name in OTLP)


def test_each_trace_is_filed_under_its_experiment(monkeypatch, tmp_path):
    """A run and the assistant trace at once into different experiments: each span carries its own project."""
    _clean(monkeypatch)
    previous_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path.as_posix()}/mlflow.db")
    try:
        client = mlflow.MlflowClient()
        run, base = (
            client.create_experiment("synalinks_traces/eval-smoke/1"),
            client.create_experiment("synalinks_traces"),
        )
        root = mlflow.start_span_no_context("program", experiment_id=run)
        child = mlflow.start_span_no_context("model", parent_span=root)
        other = mlflow.start_span_no_context("assistant", experiment_id=base)
        for span in (child, root, other):
            observability._langsmith_project(span)
            span.end()
        assert (
            root._span.attributes["langsmith.trace.session_name"]
            == "synalinks_traces/eval-smoke/1"
        )
        assert (
            child._span.attributes["langsmith.trace.session_name"]
            == "synalinks_traces/eval-smoke/1"
        )
        assert (
            other._span.attributes["langsmith.trace.session_name"] == "synalinks_traces"
        )
    finally:
        mlflow.flush_trace_async_logging()  # into the temporary store, not ./mlruns
        mlflow.set_tracking_uri(previous_uri)
        mlflow.tracing.provider.reset()
