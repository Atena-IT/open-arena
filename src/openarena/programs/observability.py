# License Apache 2.0: (c) 2026 Atena-Reply

"""MLflow tracing for every program this backend builds or loads.

`enable_observability()` is process-wide: once it has run, every
module constructed afterwards carries a `Monitor` hook (one MLflow span per
module call) and every `fit` / `evaluate` attaches the `Monitor` callback
(one MLflow run with metrics). It therefore has to run BEFORE a program is
built or loaded — see `main.lifespan`, the `build` verb, and
`EvaluationManager._run_blocking`, which all call `enable_observability()`
below.

Configuration comes from the environment, matching docker-compose:

    ARENA_MLFLOW_TRACING     on by default; "0" opts out
    MLFLOW_TRACKING_URI      tracking server (unset = observability off)
    MLFLOW_EXPERIMENT_NAME   experiment for traces (default: synalinks_traces)
    ARENA_LANGSMITH_TRACING  off by default; "1" sends LangSmith a copy
    LANGSMITH_API_KEY        the key LangSmith's copy is sent with
    LANGSMITH_ENDPOINT       LangSmith's API (default: api.smith.langchain.com)

LangSmith gets a copy of everything MLflow traces, a run's spans and the
assistant's turns alike, over MLflow's OTLP dual export in OpenTelemetry's
GenAI conventions: spans, tokens and model, not the cost, which LangSmith
prices on its own (it reads no cost attribute over OTLP). So it rides on
MLflow: no MLflow tracing, no copy. Each span lands in the LangSmith project
named like its trace's MLflow experiment, so a run's traces sit in a project
of their own.

On by default: a run's MLflow experiment is where its spend shows, per call
and per cell. Spans are exported asynchronously (MLflow's own default), but
opening a cell's run and logging its metrics still wait on the server, and
MLflow's HTTP client gives a wedged server 120 s × 7 retries by default: that
once turned a half-second build into four minutes. So unless the deployment
says otherwise, a request here gets the HTTP budget in `MLFLOW_DEFAULTS`, and
a slow server costs a cell seconds, not minutes.

The spend itself rides on the `LanguageModel` span (`mlflow.llm.cost`), and a
self-hosted server prices a trace only from the spans it stores. MLflow's
batch span processor, its default, held that one span back until its trace
had already been sent, then dropped it: tokens showed, cost read $0. So
`MLFLOW_DEFAULTS` also hands each span to the async queue as it ends.
"""

import importlib.util
import logging
import os
from urllib.parse import quote

import mlflow
import mlflow.tracing.config
import mlflow.tracing.provider
import synalinks

log = logging.getLogger("uvicorn.error")

DEFAULT_EXPERIMENT = "synalinks_traces"

# MLflow's client settings when the deployment sets none.
MLFLOW_DEFAULTS = {
    # The HTTP budget: seconds per request, and retries.
    "MLFLOW_HTTP_REQUEST_TIMEOUT": "10",
    "MLFLOW_HTTP_REQUEST_MAX_RETRIES": "2",
    # No batching: a batched LanguageModel span outlives its trace and is
    # dropped, and with it the run's cost (module docstring).
    "MLFLOW_USE_BATCH_SPAN_PROCESSOR": "false",
}


LANGSMITH_ENDPOINT = "https://api.smith.langchain.com"

# The span attribute LangSmith files a span's project by.
LANGSMITH_PROJECT_ATTRIBUTE = "langsmith.trace.session_name"

# Seconds an export to LangSmith may take. Switching LangSmith on or off
# rebuilds MLflow's tracer provider, whose shutdown flushes what it holds:
# against a LangSmith that does not answer, the exporter's own 10 s and
# retries would stall whoever switched it.
LANGSMITH_EXPORT_TIMEOUT = "5"

# What the deployment had in the variables LangSmith takes over (None: unset),
# given back when LangSmith is switched off. Empty while LangSmith is off.
_langsmith_previous: dict[str, str | None] = {}

# The project of each experiment enabled so far, by experiment id, learned
# when tracing is enabled so a span's end never calls the server. Ids are per
# tracking server: `_projects_uri` is the one they come from.
_projects: dict[str, str] = {}
_projects_uri: str | None = None


def _enabled() -> bool:
    """Tracing is on unless `ARENA_MLFLOW_TRACING` says no."""
    return os.environ.get("ARENA_MLFLOW_TRACING", "").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def tracing_wanted() -> bool:
    """Is tracing asked for: not opted out, with a tracking URI to send to."""
    return _enabled() and bool(tracking_uri())


def tracking_uri() -> str | None:
    """The tracking server, or `None` when tracing is off.

    Every MLflow touch in the backend — the process-wide hooks, the sweep's
    per-run experiment, the `Monitor` callback, the settings snapshot — asks
    this one function, so opting out here is opting out everywhere.
    """
    if not _enabled():
        return None
    return os.environ.get("MLFLOW_TRACKING_URI") or None


def base_experiment_name() -> str:
    return os.environ.get("MLFLOW_EXPERIMENT_NAME") or DEFAULT_EXPERIMENT


def _otlp_exporter_installed() -> bool:
    """Is OpenTelemetry's OTLP HTTP exporter installed (the `langsmith` extra brings it)."""
    try:
        return (
            importlib.util.find_spec("opentelemetry.exporter.otlp.proto.http")
            is not None
        )
    except ModuleNotFoundError:  # `opentelemetry.exporter` itself is missing
        return False


def langsmith_wanted() -> bool:
    """Is LangSmith asked for: switched on, with a key to send to it.

    The key alone is not enough: `LANGSMITH_API_KEY` is also what a LangSmith
    dataset source reads.
    """
    switched_on = os.environ.get("ARENA_LANGSMITH_TRACING", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if not (switched_on and os.environ.get("LANGSMITH_API_KEY")):
        return False
    if not _otlp_exporter_installed():
        # MLflow could not build its tracer provider, and would drop its own spans too.
        log.warning(
            "LangSmith tracing stays off: install openarena[langsmith] for its OTLP exporter."
        )
        return False
    return True


def langsmith_env() -> dict[str, str]:
    """The environment that points MLflow's OTLP dual export at LangSmith.

    No project header: it would hold for every span the process exports, a
    run's and the assistant's alike. Each span names its own project instead
    (`_langsmith_project`).
    """
    endpoint = (os.environ.get("LANGSMITH_ENDPOINT") or LANGSMITH_ENDPOINT).rstrip("/")
    # OpenTelemetry percent-decodes header values.
    key = quote(os.environ.get("LANGSMITH_API_KEY", ""), safe="")
    return {
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": f"{endpoint}/otel/v1/traces",
        "OTEL_EXPORTER_OTLP_TRACES_HEADERS": f"x-api-key={key}",
        "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL": "http/protobuf",
        "OTEL_EXPORTER_OTLP_TRACES_TIMEOUT": LANGSMITH_EXPORT_TIMEOUT,
        "MLFLOW_TRACE_ENABLE_OTLP_DUAL_EXPORT": "true",
        "MLFLOW_ENABLE_OTEL_GENAI_SEMCONV": "true",
    }


def _remember_project(uri: str, name: str) -> None:
    """Learn the id of experiment `name` on `uri`, creating it, as the project its spans go to.

    Called when tracing is enabled, its server just checked: a span's end then
    only reads `_projects`. A lookup that fails costs that experiment its
    project (LangSmith's default one), with a log line, never the run.
    """
    global _projects_uri
    if uri != _projects_uri:
        _projects.clear()
        _projects_uri = uri
    if name in _projects.values():
        return
    try:
        client = mlflow.MlflowClient(tracking_uri=uri)
        experiment = client.get_experiment_by_name(name)
        if experiment is None:
            try:
                client.create_experiment(name)
            except mlflow.exceptions.MlflowException:
                pass  # created by another thread in between: the lookup below finds it
            experiment = client.get_experiment_by_name(name)
        _projects[experiment.experiment_id] = name
    # Any failure: tracing is a convenience, not the run.
    except Exception as e:  # noqa: BLE001
        log.warning(
            "LangSmith: no project for the experiment %s (%s); its traces go to LangSmith's default project.",
            name,
            e,
        )


def _langsmith_project(span) -> None:
    """MLflow span processor: file `span` under the project named like its trace's experiment."""
    from mlflow.tracing.trace_manager import InMemoryTraceManager

    with InMemoryTraceManager.get_instance().get_trace(span.trace_id) as trace:
        experiment_id = trace.info.experiment_id if trace else None
    if name := _projects.get(experiment_id):
        # On the OpenTelemetry span itself: MLflow's own setter stores every
        # value as JSON, and LangSmith would read the quotes into the name.
        span._span.set_attribute(LANGSMITH_PROJECT_ATTRIBUTE, name)


def _reset_config_keeping_the_router() -> None:
    """MLflow's `reset_config`, then the router back while LangSmith is on.

    MLflow's provider reset empties its tracing config, and any caller setting
    the tracking URI spelt another way runs it: the router would be gone until
    the next `enable_observability`, and the spans in between would land in
    LangSmith's default project.
    """
    mlflow.tracing.config.reset_config()
    if _langsmith_previous:
        mlflow.tracing.configure(span_processors=[_langsmith_project])


# shortcut: wraps the name MLflow's provider `reset()` calls (MLflow 3.17); if a later MLflow moves it, the router
# test fails and the router is back only at the next `enable_observability`.
mlflow.tracing.provider.reset_config = _reset_config_keeping_the_router


def _apply_langsmith(on: bool) -> None:
    """Send LangSmith its copy, or stop sending it.

    MLflow builds its OTLP exporter with its tracer provider, so a change takes
    a new provider. Rebuilding it ends the export of the spans still open, so
    it happens only when the switch, the key or the endpoint changed, never
    because another experiment is traced. The router is checked on every call:
    MLflow's provider reset (a new tracking URI's too) empties its tracing
    config, the router with it.
    """
    global _langsmith_previous
    wanted = langsmith_env() if on else {}
    changed = False
    if on and not (
        _langsmith_previous and all(os.environ.get(k) == v for k, v in wanted.items())
    ):
        if not _langsmith_previous:
            _langsmith_previous = {name: os.environ.get(name) for name in wanted}
        os.environ.update(wanted)
        changed = True
    elif not on and _langsmith_previous:
        for name, value in _langsmith_previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        _langsmith_previous = {}
        changed = True
    if changed:
        mlflow.tracing.provider.reset()
    processors = mlflow.tracing.config.get_config().span_processors
    if on != (_langsmith_project in processors):
        others = [p for p in processors if p is not _langsmith_project]
        mlflow.tracing.configure(
            span_processors=[*others, _langsmith_project] if on else others
        )


def enable_observability(experiment_name: str | None = None) -> bool:
    """Turn on MLflow tracing when `MLFLOW_TRACKING_URI` is set and its server answers.

    Returns `True` when tracing is on. Safe to call repeatedly: a later call
    only changes the experiment used by modules built *after* it, so an
    evaluation run can trace into its own experiment without touching the
    already-loaded serving program.

    A server that does not answer leaves tracing off, with a warning: the
    hooks are process-wide and synchronous, so turning them on against a dead
    server would make every module call built afterwards (the assistant's,
    a run's) wait on connection retries. Every caller gets this check, the
    backend's startup included; a later call tries again.
    """
    uri = tracking_uri()
    if not uri:
        _switch_off()
        return False
    unreachable = mlflow_reachable(uri)
    if unreachable:
        log.warning(
            "MLflow tracing stays off: %s. Start the server, or fix MLFLOW_TRACKING_URI (Settings → Observability).",
            unreachable,
        )
        return False
    for name, value in MLFLOW_DEFAULTS.items():
        os.environ.setdefault(name, value)
    # Set here, before the LangSmith router: a hook setting a new URI later
    # would reset MLflow's tracing config, the router with it.
    mlflow.set_tracking_uri(uri)
    experiment_name = experiment_name or base_experiment_name()
    langsmith = langsmith_wanted()
    _apply_langsmith(langsmith)
    if langsmith:
        _remember_project(uri, experiment_name)
    synalinks.enable_observability(tracking_uri=uri, experiment_name=experiment_name)
    return True


def _switch_off() -> None:
    """Undo `enable_observability()` for whatever is built from now on.

    The framework has a switch on and none off: once a run has traced, every program built after it
    would keep the MLflow hook, even after an admin turned tracing off in Settings. Its flag is the
    one thing a new module and a new `evaluate` read, so clearing it is enough; this is the gap to
    upstream as `disable_observability()`.
    """
    from synalinks.src.backend import config

    config._ENABLE_OBSERVABILITY = False
    _apply_langsmith(False)


def mlflow_reachable(uri: str | None = None, timeout: float = 3.0) -> str | None:
    """Why the MLflow server cannot be used right now, or None when it answers.

    A run asks this before turning tracing on: exporting to a server that is
    down would otherwise fail the run at its first cell (the experiment
    cannot be created) — and tracing is a convenience, not the evaluation.
    `GET /health` is what an MLflow server answers with `OK`.
    """
    uri = uri or tracking_uri()
    if not uri:
        return "MLFLOW_TRACKING_URI is not set"
    if not uri.lower().startswith(("http://", "https://")):
        return None  # a sqlite:/// or file store needs no server
    import httpx

    try:
        response = httpx.get(f"{uri.rstrip('/')}/health", timeout=timeout)
    except Exception as e:  # noqa: BLE001 — any failure means "not now"
        return f"nothing answered at {uri} ({type(e).__name__})"
    if response.status_code >= 400:
        return f"{uri} answered {response.status_code} to /health"
    return None


def run_experiment_name(run_id: str, scope: str | None = None) -> str:
    """Experiment that collects one evaluation run's traces and metrics.

    The whole id: a run is `<evaluation>/<number>`, so a prefix of it would
    send every launch of an evaluation into one experiment. `scope` names
    the organisation and workspace the run belongs to, so two workspaces'
    `smoke/1` never share one either.
    """
    parts = [base_experiment_name(), *([scope] if scope else []), f"eval-{run_id}"]
    return "/".join(parts)
