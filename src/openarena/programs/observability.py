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

import logging
import os

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
        log.warning("MLflow tracing stays off: %s. Start the server, or fix MLFLOW_TRACKING_URI (Settings → Observability).", unreachable)
        return False
    for name, value in MLFLOW_DEFAULTS.items():
        os.environ.setdefault(name, value)
    synalinks.enable_observability(
        tracking_uri=uri,
        experiment_name=experiment_name or base_experiment_name(),
    )
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
