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
    LANGSMITH_AGENT_ID        an agent-based LangSmith workspace's agent, with
    LANGSMITH_AGENT_ENVIRONMENT  its environment (local, development, staging, production)
    DD_API_KEY                Datadog's API key (the intake refuses application keys)
    DD_SITE                   Datadog's site, e.g. datadoghq.eu (default datadoghq.com)
    ARENA_TRACE_SERVICE       `service.name`, Datadog's ml_app (default open-arena)
    ARENA_TRACE_CONTENT       "0" keeps prompts and answers out

Never `OTEL_EXPORTER_OTLP_*`: MLflow reads those too and would send its own
spans there instead of to its server.
"""

import contextvars
import json
import logging
import os
import re
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass

import synalinks
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from opentelemetry.trace import SpanKind, Status, StatusCode
from synalinks.src import tree
from synalinks.src.backend import any_symbolic_data_models
from synalinks.src.hooks.monitor import (
    _chat_messages_of,
    _chat_output_of,
    _chat_tools_of,
)


log = logging.getLogger("uvicorn.error")

LANGSMITH_ENDPOINT = "https://api.smith.langchain.com"
# What LangSmith accepts in its agent headers; anything else is a 400.
LANGSMITH_AGENT_ID = re.compile(r"[a-z][a-z0-9-]{0,62}")
LANGSMITH_ENVIRONMENTS = ("local", "development", "staging", "production")
DATADOG_SITE = "datadoghq.com"
DEFAULT_SERVICE = "open-arena"


def _off(value: str | None) -> bool:
    return (value or "").strip().lower() in ("0", "false", "no", "off")


def _headers(**values: str) -> tuple[tuple[str, str], ...]:
    # As given: the exporter sends a dict of headers verbatim.
    return tuple((name.replace("_", "-"), value) for name, value in values.items())


@dataclass(frozen=True)
class Destination:
    """Where spans go: an OTLP traces endpoint, its headers, and the attributes a run is filed by."""

    name: str
    endpoint: str
    headers: tuple[tuple[str, str], ...]
    run_attributes: Callable[[str], dict]


def _langsmith_run(run: str) -> dict:
    # The project a span lands in: a run's spans in a project of their own; the
    # run as metadata too.
    return {"langsmith.trace.session_name": run, "langsmith.metadata.run": run}


def _langsmith_agent_run(run: str) -> dict:
    # An agent-based workspace files spans by its agent headers, never a
    # project (LangSmith refuses both at once): the run as metadata, and as the
    # thread its Threads tab groups a run's traces by (on every span, as it asks).
    # The thread id is a segment of LangSmith's thread URL, whose gateway refuses a slash (%2F)
    # and so the thread will not open: anything but a name's own characters becomes a colon.
    return {
        "langsmith.metadata.run": run,
        "langsmith.metadata.thread_id": re.sub(r"[^A-Za-z0-9_.:-]", ":", run),
    }


def langsmith(
    api_key: str,
    endpoint: str = LANGSMITH_ENDPOINT,
    agent_id: str | None = None,
    environment: str | None = None,
) -> Destination:
    """LangSmith; an agent-based workspace also needs its agent and environment, sent as headers."""
    headers = {"x_api_key": api_key}
    if agent_id or environment:
        if not (agent_id and environment):
            raise ValueError(
                "LangSmith needs both the agent id and its environment, or neither"
            )
        if not LANGSMITH_AGENT_ID.fullmatch(agent_id):
            raise ValueError(
                f"LangSmith agent id {agent_id!r}: 1 to 63 lowercase letters, digits or hyphens, starting with a letter"
            )
        if environment not in LANGSMITH_ENVIRONMENTS:
            raise ValueError(
                f"LangSmith environment {environment!r}: one of {', '.join(LANGSMITH_ENVIRONMENTS)}"
            )
        headers |= {
            "Langsmith_Agent_Id": agent_id,
            "Langsmith_Agent_Environment": environment,
        }
    return Destination(
        "langsmith",
        f"{endpoint.rstrip('/')}/otel/v1/traces",
        _headers(**headers),
        _langsmith_agent_run if agent_id else _langsmith_run,
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
            try:
                destinations.append(
                    langsmith(
                        key,
                        os.environ.get("LANGSMITH_ENDPOINT") or LANGSMITH_ENDPOINT,
                        os.environ.get("LANGSMITH_AGENT_ID") or None,
                        os.environ.get("LANGSMITH_AGENT_ENVIRONMENT") or None,
                    )
                )
            except ValueError as e:
                log.warning("Trace destination langsmith is skipped: %s.", e)
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


# -- providers: one per destination list -----------------------------------------

# Seconds an export may take: a destination that does not answer costs its own
# queue, never the run (the export runs on the processor's thread).
EXPORT_TIMEOUT = 5

# One private provider per (destinations, service), never the global one: MLflow keeps its own.
_providers: dict[tuple, TracerProvider] = {}


def _exporter(destination: Destination) -> SpanExporter:
    return OTLPSpanExporter(
        endpoint=destination.endpoint,
        headers=dict(destination.headers),
        timeout=EXPORT_TIMEOUT,
    )


def provider_for(config: TraceConfig) -> TracerProvider:
    """The provider sending to `config`'s destinations, built once per destination list."""
    key = (config.destinations, config.service)
    if (provider := _providers.get(key)) is None:
        provider = TracerProvider(
            resource=Resource.create({"service.name": config.service})
        )
        for destination in config.destinations:
            provider.add_span_processor(BatchSpanProcessor(_exporter(destination)))
        provider = _providers.setdefault(key, provider)
    return provider


# -- what is traced now ------------------------------------------------------------


@dataclass(frozen=True)
class _Active:
    tracer: trace.Tracer
    config: TraceConfig
    run: str | None


_ACTIVE: contextvars.ContextVar[_Active | None] = contextvars.ContextVar(
    "openarena_tracing", default=None
)


@contextmanager
def use_tracing(config: TraceConfig, *, run: str | None = None):
    """Trace the module calls made inside, to `config`'s destinations, filed under `run`.

    A `ContextVar`: tasks started inside (`asyncio.run`, `gather`) see it too,
    and two runs in one process each keep their own. No destination: nothing.
    """
    if not config.destinations:
        yield
        return
    tracer = provider_for(config).get_tracer("openarena")
    token = _ACTIVE.set(_Active(tracer, config, run))
    try:
        yield
    finally:
        _ACTIVE.reset(token)


# -- the hook ------------------------------------------------------------------------

# Open spans by call id, off the hook: synalinks deepcopies tracked state that reaches hooks.
_SPANS: dict[str, tuple] = {}
_warned = False


def _warn_once(what: str, e: Exception) -> None:
    global _warned
    if not _warned:
        _warned = True
        log.warning(
            "Trace destinations: %s failed (%s); calls go on untraced.", what, e
        )


def _describe(module) -> tuple[str, str, SpanKind, dict]:
    """The GenAI operation of a module call: its name, the span's name and kind, its attributes."""
    name = module.name or type(module).__name__
    models = (
        synalinks.LanguageModel,
        synalinks.DecisionModel,
        synalinks.EmbeddingModel,
    )
    if isinstance(module, models):
        model = getattr(module, "model", "") or ""
        op = "embeddings" if isinstance(module, synalinks.EmbeddingModel) else "chat"
        attributes = {"gen_ai.request.model": model, "gen_ai.response.model": model}
        if "/" in model:
            provider = model.split("/")[0]
            # `gen_ai.system` too: LangSmith reads the provider there.
            attributes |= {"gen_ai.provider.name": provider, "gen_ai.system": provider}
        return op, f"{op} {model}", SpanKind.CLIENT, attributes
    if isinstance(module, synalinks.Tool):
        op, attributes = "execute_tool", {"gen_ai.tool.name": name}
    elif isinstance(module, synalinks.FunctionCallingAgent) or (
        type(module).__name__ == "A2aAgent"
    ):
        op, attributes = "invoke_agent", {"gen_ai.agent.name": name}
    else:
        op, attributes = "invoke_workflow", {"gen_ai.workflow.name": name}
    return op, f"{op} {name}", SpanKind.INTERNAL, attributes


def _parts(message: dict) -> dict:
    """An OpenAI-style chat message in the GenAI conventions' shape: a role and its parts."""
    parts = []
    if message.get("role") == "tool":
        parts.append(
            {
                "type": "tool_call_response",
                "id": message.get("tool_call_id"),
                "response": message.get("content"),
            }
        )
    elif isinstance(message.get("content"), list):
        # A multimodal message: its text, and only the kind of each image or clip,
        # whose inline data would make the span megabytes (Datadog's intake caps a request at 15 MiB).
        for item in message["content"]:
            kind = item.get("type") if isinstance(item, dict) else None
            if kind == "text":
                parts.append({"type": "text", "content": item.get("text")})
            elif kind:
                parts.append({"type": kind})
    elif message.get("content"):
        parts.append({"type": "text", "content": message["content"]})
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        parts.append(
            {
                "type": "tool_call",
                "id": call.get("id"),
                "name": function.get("name"),
                "arguments": function.get("arguments"),
            }
        )
    return {"role": message.get("role"), "parts": parts}


def _usage(module) -> dict | None:
    """The usage of the model call that just ran in this task, as synalinks keeps it."""
    if isinstance(module, synalinks.LanguageModel):
        from synalinks.src.modules.language_models.language_model import (
            current_call_usage,
        )
    elif isinstance(module, synalinks.DecisionModel):
        from synalinks.src.modules.decision_models.decision_model import (
            current_call_usage,
        )
    else:
        return None
    return current_call_usage() or {}


# synalinks' usage keys, by the GenAI attribute each one fills.
_USAGE = (
    ("gen_ai.usage.input_tokens", "input_tokens"),
    ("gen_ai.usage.output_tokens", "output_tokens"),
    ("gen_ai.usage.cache_read.input_tokens", "cache_read_input_tokens"),
    # Both spellings: `cache_write` on the conventions' main, `cache_creation` in v1.41.
    ("gen_ai.usage.cache_write.input_tokens", "cache_creation_input_tokens"),
    ("gen_ai.usage.cache_creation.input_tokens", "cache_creation_input_tokens"),
    ("gen_ai.usage.reasoning.output_tokens", "reasoning_tokens"),
)


def _usage_attributes(usage: dict) -> dict:
    attributes = {
        name: usage[key]
        for name, key in _USAGE
        if isinstance(usage.get(key), (int, float))
    }
    if usage.get("finish_reason"):
        attributes["gen_ai.response.finish_reasons"] = [str(usage["finish_reason"])]
    return attributes | _cost_attributes(usage.get("cost"))


def _cost_attributes(cost) -> dict:
    # The arena's own price (litellm's, USD): Datadog shows it over its own estimate.
    if isinstance(cost, (int, float)) and cost > 0:
        return {
            "gen_ai.usage.cost.amount": float(cost),
            "gen_ai.usage.cost.currency": "USD",
        }
    return {}


def _json_leaves(data) -> list:
    leaves = tree.flatten(data) if data is not None else []
    return [d.get_json() for d in leaves if d is not None and hasattr(d, "get_json")]


def _content_at_begin(op: str, inputs, kwargs) -> dict:
    """The prompt side of a call: messages, system instructions, tools, a tool's arguments."""
    if op == "execute_tool":
        arguments = {k: v for k, v in (kwargs or {}).items() if k != "training"}
        return {
            "gen_ai.tool.call.arguments": json.dumps(
                arguments or _json_leaves(inputs), default=str
            )
        }
    if op != "chat":
        return {}
    messages = _chat_messages_of(inputs) or []
    attributes = {
        "gen_ai.input.messages": json.dumps(
            [_parts(m) for m in messages if m.get("role") != "system"]
        )
    }
    if system := [m.get("content") for m in messages if m.get("role") == "system"]:
        attributes["gen_ai.system_instructions"] = json.dumps(
            [{"type": "text", "content": text} for text in system]
        )
    if tools := _chat_tools_of(kwargs or {}):
        attributes["gen_ai.tool.definitions"] = json.dumps(tools)
    return attributes


def _content_at_end(op: str, outputs, usage) -> dict:
    """The answer side of a call: the model's message, or a tool's result."""
    if op == "execute_tool":
        return {
            "gen_ai.tool.call.result": json.dumps(_json_leaves(outputs), default=str)
        }
    if op != "chat":
        return {}
    choices = _chat_output_of(_json_leaves(outputs), usage) or []
    messages = [
        _parts(c["message"]) | {"finish_reason": c.get("finish_reason")}
        for c in choices
    ]
    return {"gen_ai.output.messages": json.dumps(messages)}


class OtelHook(synalinks.hooks.Hook):
    """One span per module call, in the GenAI conventions, to the destinations `use_tracing` names.

    Parented by OpenTelemetry's own context (attached at a call's begin,
    detached at its end, both in the calling task), not by synalinks'
    `parent_call_id`. Does nothing outside `use_tracing`, and never raises: a
    span that cannot be written leaves the call as it is.
    """

    def on_call_begin(self, call_id, parent_call_id=None, inputs=None, kwargs=None):
        active = _ACTIVE.get()
        if active is None or any_symbolic_data_models(inputs):
            return
        try:
            op, name, kind, attributes = _describe(self.module)
            attributes["gen_ai.operation.name"] = op
            if active.run:
                for destination in active.config.destinations:
                    attributes |= destination.run_attributes(active.run)
            if active.config.content:
                attributes |= _content_at_begin(op, inputs, kwargs)
            span = active.tracer.start_span(name, kind=kind, attributes=attributes)
            token = otel_context.attach(trace.set_span_in_context(span))
            _SPANS[call_id] = (span, token, op, active.config.content)
        # Tracing never breaks the call.
        except Exception as e:  # noqa: BLE001
            _warn_once("starting a span", e)

    def on_call_end(self, call_id, parent_call_id=None, outputs=None, exception=None):
        entry = _SPANS.pop(call_id, None)
        if entry is None:
            return
        span, token, op, content = entry
        try:
            usage = _usage(self.module)
            if usage:
                span.set_attributes(_usage_attributes(usage))
            elif op == "embeddings":
                cost = getattr(self.module, "last_call_cost", None)
                span.set_attributes(_cost_attributes(cost))
            # `exception` is the message, which may be empty.
            error = exception if exception is not None else (usage or {}).get("error")
            if error is not None:
                span.set_status(Status(StatusCode.ERROR, str(error)))
                # synalinks hands a hook the message, not the exception's class.
                span.set_attribute("error.type", "_OTHER")
            elif content:
                span.set_attributes(_content_at_end(op, outputs, usage))
        except Exception as e:  # noqa: BLE001
            _warn_once("ending a span", e)
        finally:
            span.end()
            otel_context.detach(token)


def attach(program) -> None:
    """Give every module of `program` an `OtelHook`, once (as `evaluate._Tally` attaches its own)."""
    for module in {id(m): m for m in program._flatten_modules()}.values():
        if not any(isinstance(h, OtelHook) for h in module._hooks.hooks):
            hook = OtelHook()
            hook.set_module(module)
            module.add_hook(hook)
