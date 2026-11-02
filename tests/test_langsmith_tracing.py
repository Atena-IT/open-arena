# License Apache 2.0: (c) 2026 Atena-Reply

"""LangSmith gets a copy of a run's traces, over MLflow's OTLP dual export, only when switched on."""

import os

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

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


@pytest.fixture
def resets(monkeypatch):
    """MLflow tracing on (no server to ask), no LangSmith setting, and every provider reset counted."""
    for name in (
        "ARENA_LANGSMITH_TRACING",
        "LANGSMITH_API_KEY",
        "LANGSMITH_ENDPOINT",
        *OTLP,
    ):
        monkeypatch.setenv(name, "")  # so the delete below is undone after the test
        monkeypatch.delenv(name)
    monkeypatch.setenv("ARENA_MLFLOW_TRACING", "1")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///unused.db")
    monkeypatch.setattr(
        observability.synalinks, "enable_observability", lambda **kw: None
    )
    monkeypatch.setattr(observability, "_langsmith_applied", {})
    counted = []
    monkeypatch.setattr(mlflow.tracing.provider, "reset", lambda: counted.append(1))
    return counted


def _switch_on(monkeypatch, key="k"):
    monkeypatch.setenv("ARENA_LANGSMITH_TRACING", "1")
    monkeypatch.setenv("LANGSMITH_API_KEY", key)


def test_off_by_default(resets):
    assert observability.langsmith_wanted() is False
    assert observability.enable_observability("exp") is True
    assert not any(name in os.environ for name in OTLP)


def test_key_alone_does_not_export(resets, monkeypatch):
    """`LANGSMITH_API_KEY` is also a dataset source's key: holding it must not send traces anywhere."""
    monkeypatch.setenv("LANGSMITH_API_KEY", "k")
    assert observability.langsmith_wanted() is False
    observability.enable_observability("exp")
    assert not any(name in os.environ for name in OTLP)


def test_env_names_project_and_key(resets, monkeypatch):
    _switch_on(monkeypatch)
    assert observability.langsmith_wanted() is True
    env = observability.langsmith_env("synalinks_traces/eval-smoke/1")
    assert (
        env["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"]
        == "https://api.smith.langchain.com/otel/v1/traces"
    )
    assert (
        env["OTEL_EXPORTER_OTLP_TRACES_HEADERS"]
        == "x-api-key=k,Langsmith-Project=synalinks_traces/eval-smoke/1"
    )
    assert env["OTEL_EXPORTER_OTLP_TRACES_PROTOCOL"] == "http/protobuf"
    assert env["MLFLOW_TRACE_ENABLE_OTLP_DUAL_EXPORT"] == "true"
    assert env["MLFLOW_ENABLE_OTEL_GENAI_SEMCONV"] == "true"
    monkeypatch.setenv("LANGSMITH_ENDPOINT", "https://eu.api.smith.langchain.com/")
    assert (
        observability.langsmith_env("p")["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"]
        == "https://eu.api.smith.langchain.com/otel/v1/traces"
    )


def test_exporter_timeout_is_short(resets, monkeypatch):
    """A LangSmith that does not answer costs a run this many seconds at most, not the exporter's 10 with retries."""
    _switch_on(monkeypatch)
    assert observability.langsmith_env("p")["OTEL_EXPORTER_OTLP_TRACES_TIMEOUT"] == "5"


def test_each_run_gets_its_project(resets, monkeypatch):
    """The exporter reads its headers when the tracer provider is built: a new project needs a new provider, once."""
    _switch_on(monkeypatch)
    observability.enable_observability("a")
    observability.enable_observability("b")
    assert os.environ["OTEL_EXPORTER_OTLP_TRACES_HEADERS"].endswith(
        "Langsmith-Project=b"
    )
    assert len(resets) == 2
    observability.enable_observability("b")
    assert (
        len(resets) == 2
    )  # same project: the provider, and the spans it holds, are left alone


def test_switching_off_removes_otlp_vars(resets, monkeypatch):
    _switch_on(monkeypatch)
    observability.enable_observability("a")
    monkeypatch.setenv("ARENA_LANGSMITH_TRACING", "0")
    observability.enable_observability("b")
    assert not any(name in os.environ for name in OTLP)
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
