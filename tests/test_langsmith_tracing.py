# License Apache 2.0: (c) 2026 Atena-Reply

"""LangSmith gets a copy of the traces, over MLflow's OTLP dual export, only when switched on, each trace in its experiment's project."""

import importlib.util
import os
from types import SimpleNamespace

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import mlflow
import mlflow.tracing.config
import mlflow.tracing.provider
import mlflow.tracing.utils.processor
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


def _server(created=None):
    """A stand-in MLflow client whose experiments exist under `id-<name>`; names it was asked to create go in `created`."""

    def client(tracking_uri=None):
        return SimpleNamespace(
            get_experiment_by_name=lambda name: SimpleNamespace(
                experiment_id=f"id-{name}"
            ),
            create_experiment=lambda name: (created or []).append(name),
        )

    return client


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
    monkeypatch.setattr(observability, "_projects", {})
    monkeypatch.setattr(observability, "_projects_uri", None)


def _traced(monkeypatch):
    """MLflow tracing on, against a server that is never asked anything over the network."""
    _clean(monkeypatch)
    monkeypatch.setenv("ARENA_MLFLOW_TRACING", "1")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///unused.db")
    monkeypatch.setattr(
        observability.synalinks, "enable_observability", lambda **kw: None
    )
    monkeypatch.setattr(mlflow, "MlflowClient", _server())


@pytest.fixture
def resets(monkeypatch):
    """`_traced`, and every provider rebuild counted instead of run."""
    _traced(monkeypatch)
    counted = []
    monkeypatch.setattr(mlflow.tracing.provider, "reset", lambda: counted.append(1))
    return counted


def _switch_on(monkeypatch, key="k"):
    """LangSmith switched on, the OTLP exporter installed or not (only the `langsmith` extra brings it)."""
    monkeypatch.setenv("ARENA_LANGSMITH_TRACING", "1")
    monkeypatch.setenv("LANGSMITH_API_KEY", key)
    monkeypatch.setattr(observability, "_otlp_exporter_installed", lambda: True)


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


def test_enabling_learns_each_experiments_project(resets, monkeypatch):
    """The project a span goes to is looked up when tracing is enabled, the server just checked, never when a span ends."""
    _switch_on(monkeypatch)
    observability.enable_observability("synalinks_traces/eval-smoke/1")
    observability.enable_observability()
    assert observability._projects == {
        "id-synalinks_traces/eval-smoke/1": "synalinks_traces/eval-smoke/1",
        "id-synalinks_traces": "synalinks_traces",
    }


def test_a_new_tracking_server_forgets_the_old_ids(monkeypatch):
    """Ids are per server: after an admin points the arena at another one, its '2' is not the old server's '2'."""
    _clean(monkeypatch)
    monkeypatch.setattr(mlflow, "MlflowClient", _server())
    observability._remember_project("http://a", "run-x")
    observability._remember_project("http://b", "run-y")
    assert observability._projects == {"id-run-y": "run-y"}


def test_a_project_that_cannot_be_learned_is_said(monkeypatch, caplog):
    """A server that fails the lookup costs that experiment its project, with a line in the log, never the run."""
    _clean(monkeypatch)

    def down(tracking_uri=None):
        raise ConnectionError("refused")

    monkeypatch.setattr(mlflow, "MlflowClient", down)
    observability._remember_project("http://a", "run-x")
    assert observability._projects == {}
    assert "run-x" in caplog.text


def test_routing_outlives_a_provider_reset(monkeypatch):
    """MLflow's provider reset also empties its tracing config, and a tracking URI spelt another way resets it: the router stays."""
    _traced(monkeypatch)
    _switch_on(monkeypatch)
    try:
        observability.enable_observability("a")
        assert _routing()
        # A synalinks hook sets the same URI on its first call: no change, no reset.
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        assert _routing()
        # A caller spelling the same server another way resets MLflow's config.
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"] + "/")
        assert _routing()
        mlflow.tracing.provider.reset()
        assert _routing()
    finally:
        observability._apply_langsmith(False)


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


def test_switching_off_holds_while_mlflow_is_down(resets, monkeypatch):
    """Programs already built keep tracing on their own: an opt-out must stop the copy even when MLflow does not answer."""
    _switch_on(monkeypatch)
    observability.enable_observability("a")
    monkeypatch.setenv("ARENA_LANGSMITH_TRACING", "0")
    monkeypatch.setattr(
        observability, "mlflow_reachable", lambda uri=None, timeout=3.0: "down"
    )
    assert observability.enable_observability("b") is False
    assert not any(name in os.environ for name in OTLP)
    assert not _routing()


def test_a_synalinks_that_loses_open_spans_is_said(resets, monkeypatch, caplog):
    """A synalinks whose Monitor rebuilds MLflow's tracer provider on each module's first call drops LangSmith's copy of the spans still open."""
    import synalinks.src.hooks.monitor as monitor

    _switch_on(monkeypatch)
    monkeypatch.delattr(monitor, "experiment_id_of", raising=False)
    observability.enable_observability("a")
    assert "SynaLinks/synalinks#150" in caplog.text


def test_rides_on_mlflow(resets, monkeypatch):
    """LangSmith gets a copy of what MLflow traces: with MLflow tracing off there is nothing to copy."""
    _switch_on(monkeypatch)
    monkeypatch.setenv("ARENA_MLFLOW_TRACING", "0")
    assert observability.enable_observability("a") is False
    assert not any(name in os.environ for name in OTLP)


@pytest.mark.parametrize("parent_installed", [True, False])
def test_a_missing_otlp_exporter_keeps_mlflow_tracing(
    resets, monkeypatch, parent_installed
):
    """Without the OTLP exporter MLflow cannot build its tracer provider, and every span, MLflow's own too, is dropped."""
    monkeypatch.setenv("ARENA_LANGSMITH_TRACING", "1")
    monkeypatch.setenv("LANGSMITH_API_KEY", "k")
    real = importlib.util.find_spec

    def find_spec(name, *a):
        if name.startswith("opentelemetry.exporter"):
            if parent_installed:
                return None
            # Without the extra, `opentelemetry.exporter` is missing too, and
            # asking for a module under it raises instead of answering None.
            raise ModuleNotFoundError(f"No module named {name!r}")
        return real(name, *a)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    assert observability.langsmith_wanted() is False
    assert observability.enable_observability("a") is True
    assert not any(name in os.environ for name in OTLP)


def _span_processor_warnings(monkeypatch):
    """What MLflow says when a span processor fails: it swallows the error and only logs it."""
    said = []
    monkeypatch.setattr(
        mlflow.tracing.utils.processor._logger,
        "warning",
        lambda message, *a, **kw: said.append(message),
    )
    return said


def _real_store(monkeypatch, tmp_path):
    """MLflow tracing into a temporary store, each trace written as its root span ends.

    Batched or queued, a trace could be written after the test, once MLflow's
    URI is back to its default, into ./mlruns.
    """
    monkeypatch.setenv("MLFLOW_ENABLE_ASYNC_TRACE_LOGGING", "false")
    monkeypatch.setenv("MLFLOW_USE_BATCH_SPAN_PROCESSOR", "false")
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path.as_posix()}/mlflow.db")
    mlflow.tracing.provider.reset()


def test_each_trace_is_filed_under_its_experiment(monkeypatch, tmp_path):
    """A run and the assistant trace at once into different experiments: each span carries its own project, through MLflow's own processor path."""
    _clean(monkeypatch)
    _real_store(monkeypatch, tmp_path)
    uri = mlflow.get_tracking_uri()
    try:
        observability._remember_project(uri, "synalinks_traces/eval-smoke/1")
        observability._remember_project(uri, "synalinks_traces")
        client = mlflow.MlflowClient()
        run = client.get_experiment_by_name(
            "synalinks_traces/eval-smoke/1"
        ).experiment_id
        base = client.get_experiment_by_name("synalinks_traces").experiment_id
        mlflow.tracing.configure(span_processors=[observability._langsmith_project])
        warned = _span_processor_warnings(monkeypatch)
        root = mlflow.start_span_no_context("program", experiment_id=run)
        child = mlflow.start_span_no_context("model", parent_span=root)
        other = mlflow.start_span_no_context("assistant", experiment_id=base)
        for span in (child, root, other):
            span.end()
        # On the OpenTelemetry span as a plain string: MLflow's setter would store JSON quotes.
        project = "langsmith.trace.session_name"
        assert root._span.attributes[project] == "synalinks_traces/eval-smoke/1"
        assert child._span.attributes[project] == "synalinks_traces/eval-smoke/1"
        assert other._span.attributes[project] == "synalinks_traces"
        assert warned == []
    finally:
        mlflow.tracing.provider.reset()


def test_a_span_never_waits_on_the_server(monkeypatch, tmp_path):
    """A span of an experiment never learned ends without a project, and without calling MLflow."""
    _clean(monkeypatch)
    _real_store(monkeypatch, tmp_path)
    try:
        unseen = mlflow.MlflowClient().create_experiment("unseen")

        def no_call(*a, **kw):
            pytest.fail("a span's end called the MLflow server")

        monkeypatch.setattr(mlflow, "MlflowClient", no_call)
        span = mlflow.start_span_no_context("program", experiment_id=unseen)
        observability._langsmith_project(span)
        span.end()
        assert "langsmith.trace.session_name" not in span._span.attributes
    finally:
        mlflow.tracing.provider.reset()
