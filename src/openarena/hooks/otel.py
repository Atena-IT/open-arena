# License Apache 2.0: (c) 2026 Atena-Reply

"""OpenTelemetry spans of every module call, sent to the trace destinations.

MLflow keeps its own tracing (`programs/observability.py`, synalinks'
`Monitor`), untouched. Beside it, `OtelHook` writes each module call as a span
in OpenTelemetry's GenAI conventions (`gen_ai.*`) and sends it to any number of
destinations at once, each a small entry: an OTLP endpoint, its headers, and
the span attributes it files a run by. LangSmith and Datadog are the two today;
another backend that reads OTLP is one more entry.

The hook reads no environment. Whoever starts the work hands it its
destinations (`use_tracing`), so a later step can give each organisation its
own; for now `config_from_env` reads the deployment's:

    ARENA_TRACE_DESTINATIONS  comma list, `langsmith`, `datadog` (empty: MLflow only)
    LANGSMITH_API_KEY         LangSmith's key, LANGSMITH_ENDPOINT its API
    DD_API_KEY                Datadog's API key (the intake refuses application keys)
    DD_SITE                   Datadog's site, e.g. datadoghq.eu (default datadoghq.com)
    ARENA_TRACE_SERVICE       `service.name`, Datadog's ml_app (default open-arena)
    ARENA_TRACE_CONTENT       "0" keeps prompts and answers out

Never `OTEL_EXPORTER_OTLP_*`: MLflow reads those too and would send its own
spans there instead of to its server.
"""

import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import quote

log = logging.getLogger("uvicorn.error")

LANGSMITH_ENDPOINT = "https://api.smith.langchain.com"
DATADOG_SITE = "datadoghq.com"
DEFAULT_SERVICE = "open-arena"


def _off(value: str | None) -> bool:
    return (value or "").strip().lower() in ("0", "false", "no", "off")


def _headers(**values: str) -> tuple[tuple[str, str], ...]:
    # OpenTelemetry percent-decodes header values.
    return tuple(
        (name.replace("_", "-"), quote(value, safe=""))
        for name, value in values.items()
    )


@dataclass(frozen=True)
class Destination:
    """Where spans go: an OTLP traces endpoint, its headers, and the attributes a run is filed by."""

    name: str
    endpoint: str
    headers: tuple[tuple[str, str], ...]
    run_attributes: Callable[[str], dict]


def _langsmith_run(run: str) -> dict:
    # The project a span lands in: a run's spans in a project of their own.
    return {"langsmith.trace.session_name": run}


def langsmith(api_key: str, endpoint: str = LANGSMITH_ENDPOINT) -> Destination:
    return Destination(
        "langsmith",
        f"{endpoint.rstrip('/')}/otel/v1/traces",
        _headers(x_api_key=api_key),
        _langsmith_run,
    )


def _datadog_run(run: str) -> dict:
    # A tag, and the span's metadata (kept whole, without the tags' 256-character limit).
    return {"arena.run": run, "_dd.ml_obs.metadata": json.dumps({"run": run})}


def datadog(api_key: str, site: str = DATADOG_SITE) -> Destination:
    return Destination(
        "datadog",
        f"https://otlp.{site.strip().strip('/')}/v1/traces",
        _headers(dd_api_key=api_key, dd_otlp_source="llmobs"),
        _datadog_run,
    )


@dataclass(frozen=True)
class TraceConfig:
    """What is traced and how: the destinations, the service name they see, whether content goes."""

    destinations: tuple[Destination, ...] = ()
    service: str = DEFAULT_SERVICE
    content: bool = True


def config_from_env() -> TraceConfig:
    """The deployment's trace destinations; one named without its key is skipped with a warning."""
    destinations = []
    for name in (
        n.strip().lower()
        for n in os.environ.get("ARENA_TRACE_DESTINATIONS", "").split(",")
    ):
        if not name:
            continue
        if name == "langsmith" and (key := os.environ.get("LANGSMITH_API_KEY")):
            destinations.append(
                langsmith(
                    key, os.environ.get("LANGSMITH_ENDPOINT") or LANGSMITH_ENDPOINT
                )
            )
        elif name == "datadog" and (key := os.environ.get("DD_API_KEY")):
            destinations.append(datadog(key, os.environ.get("DD_SITE") or DATADOG_SITE))
        elif name in ("langsmith", "datadog"):
            needs = "LANGSMITH_API_KEY" if name == "langsmith" else "DD_API_KEY"
            log.warning("Trace destination %s is skipped: %s is not set.", name, needs)
        else:
            log.warning(
                "Trace destination %s is unknown and skipped (langsmith, datadog).",
                name,
            )
    return TraceConfig(
        destinations=tuple(destinations),
        service=os.environ.get("ARENA_TRACE_SERVICE") or DEFAULT_SERVICE,
        content=not _off(os.environ.get("ARENA_TRACE_CONTENT")),
    )
