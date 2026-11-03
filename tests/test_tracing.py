# License Apache 2.0: (c) 2026 Atena-Reply

"""Trace destinations: the entries read from the environment, the hook's spans, a traced run.

Spans are caught by an in-memory exporter put where each destination's OTLP
exporter would be (`otel._exporter`), so nothing leaves the process.
"""

import asyncio
import copy
import json
import logging
import os
import time

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import litellm
import mlflow
import mlflow.tracing.config
import mlflow.tracing.provider
import pytest
import synalinks
import yaml
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode
from test_sweep import _config as _sweep_config

from openarena.evaluate import run_sweep
from openarena.hooks import otel
from openarena.programs import observability
from openarena.hooks.otel import (
    OtelHook,
    TraceConfig,
    attach,
    config_from_env,
    datadog,
    langsmith,
    provider_for,
    use_tracing,
)

TRACE_VARS = (
    "ARENA_TRACE_DESTINATIONS",
    "LANGSMITH_API_KEY",
    "LANGSMITH_ENDPOINT",
    "LANGSMITH_AGENT_ID",
    "LANGSMITH_AGENT_ENVIRONMENT",
    "DD_API_KEY",
    "DD_SITE",
    "ARENA_TRACE_SERVICE",
    "ARENA_TRACE_CONTENT",
)


@pytest.fixture(autouse=True)
def _no_trace_env(monkeypatch):
    for name in TRACE_VARS:
        monkeypatch.delenv(name, raising=False)


# -- destinations ----------------------------------------------------------------


def test_nothing_named_is_nothing_sent():
    assert config_from_env().destinations == ()


def test_langsmith_entry(monkeypatch):
    entry = langsmith("k/ey")
    assert entry.endpoint == "https://api.smith.langchain.com/otel/v1/traces"
    # As given: the exporter sends a dict of headers verbatim (only its env var is percent-decoded).
    assert entry.headers == (("x-api-key", "k/ey"),)
    assert entry.run_attributes("ws/eval-smoke/1") == {
        "langsmith.trace.session_name": "ws/eval-smoke/1",
        "langsmith.metadata.run": "ws/eval-smoke/1",
    }

    monkeypatch.setenv("ARENA_TRACE_DESTINATIONS", "langsmith")
    monkeypatch.setenv("LANGSMITH_API_KEY", "k")
    monkeypatch.setenv("LANGSMITH_ENDPOINT", "https://eu.api.smith.langchain.com/")
    (entry,) = config_from_env().destinations
    assert entry.endpoint == "https://eu.api.smith.langchain.com/otel/v1/traces"


def test_langsmith_agent_headers(monkeypatch):
    """An agent-based LangSmith workspace takes traces only for a registered agent, named by two headers.

    Span attributes do not route them there, so the run rides along as metadata.
    """
    entry = langsmith("k", agent_id="open-arena", environment="staging")
    assert ("Langsmith-Agent-Id", "open-arena") in entry.headers
    assert ("Langsmith-Agent-Environment", "staging") in entry.headers
    assert ("x-api-key", "k") in entry.headers

    monkeypatch.setenv("ARENA_TRACE_DESTINATIONS", "langsmith")
    monkeypatch.setenv("LANGSMITH_API_KEY", "k")
    monkeypatch.setenv("LANGSMITH_AGENT_ID", "open-arena")
    monkeypatch.setenv("LANGSMITH_AGENT_ENVIRONMENT", "staging")
    (entry,) = config_from_env().destinations
    assert ("Langsmith-Agent-Id", "open-arena") in entry.headers


def test_langsmith_agent_headers_go_in_pairs_and_follow_its_rules():
    """LangSmith answers 400 to one header without the other, and to an id or environment it does not accept."""
    with pytest.raises(ValueError, match="both"):
        langsmith("k", agent_id="open-arena")
    with pytest.raises(ValueError, match="both"):
        langsmith("k", environment="staging")
    with pytest.raises(ValueError, match="agent id"):
        langsmith("k", agent_id="Open Arena", environment="staging")
    with pytest.raises(ValueError, match="environment"):
        langsmith("k", agent_id="open-arena", environment="prod")


def test_a_half_set_langsmith_agent_skips_langsmith(monkeypatch, caplog):
    monkeypatch.setenv("ARENA_TRACE_DESTINATIONS", "langsmith")
    monkeypatch.setenv("LANGSMITH_API_KEY", "k")
    monkeypatch.setenv("LANGSMITH_AGENT_ID", "open-arena")
    with caplog.at_level(logging.WARNING):
        assert config_from_env().destinations == ()
    assert "both" in caplog.text


def test_datadog_entry_per_site():
    assert (
        datadog("k", "datadoghq.eu").endpoint == "https://otlp.datadoghq.eu/v1/traces"
    )
    assert (
        datadog("k", "us3.datadoghq.com").endpoint
        == "https://otlp.us3.datadoghq.com/v1/traces"
    )
    entry = datadog("k")
    assert entry.endpoint == "https://otlp.datadoghq.com/v1/traces"
    assert ("dd-api-key", "k") in entry.headers
    assert ("dd-otlp-source", "llmobs") in entry.headers
    attributes = entry.run_attributes("r")
    assert attributes["arena.run"] == "r"
    assert json.loads(attributes["_dd.ml_obs.metadata"]) == {"run": "r"}


def test_both_at_once(monkeypatch):
    monkeypatch.setenv("ARENA_TRACE_DESTINATIONS", "langsmith, datadog")
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls")
    monkeypatch.setenv("DD_API_KEY", "dd")
    assert [d.name for d in config_from_env().destinations] == ["langsmith", "datadog"]


def test_a_destination_without_its_key_is_skipped(monkeypatch, caplog):
    monkeypatch.setenv("ARENA_TRACE_DESTINATIONS", "datadog,langsmith,nowhere")
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls")
    with caplog.at_level(logging.WARNING):
        config = config_from_env()
    assert [d.name for d in config.destinations] == ["langsmith"]
    assert "DD_API_KEY" in caplog.text
    assert "nowhere" in caplog.text


def test_service_and_content_switch(monkeypatch):
    assert config_from_env() == TraceConfig()
    monkeypatch.setenv("ARENA_TRACE_SERVICE", "open-arena-staging")
    monkeypatch.setenv("ARENA_TRACE_CONTENT", "0")
    config = config_from_env()
    assert config.service == "open-arena-staging"
    assert config.content is False


def test_the_module_is_importable_without_a_destination():
    assert otel.TraceConfig().destinations == ()


# -- the hook's spans --------------------------------------------------------------

RUN = "ws/eval-smoke/1"


class Query(synalinks.DataModel):
    question: str


class Answer(synalinks.DataModel):
    answer: str


class LanguageModel(synalinks.LanguageModel):
    """`openai/gpt-4o-mini` answered by litellm's mock: no provider, but litellm's usage and price.

    Named like the framework's own, which MLflow's `Monitor` types spans by.
    """

    async def _acompletion(self, formatted_messages, **kwargs):
        kwargs.pop("timeout", None)
        return await litellm.acompletion(
            model=self.model,
            messages=formatted_messages,
            mock_response='{"answer": "4"}',
            **kwargs,
        )


@pytest.fixture
def exported(monkeypatch):
    """Spans every destination would have sent, caught in memory."""
    exporter = InMemorySpanExporter()
    monkeypatch.setattr(otel, "_exporter", lambda destination: exporter)
    monkeypatch.setattr(otel, "_providers", {})
    return exporter


def _config(**kwargs):
    return TraceConfig(destinations=(langsmith("k"),), **kwargs)


def _program():
    async def build():
        x0 = synalinks.Input(data_model=Query)
        x1 = await synalinks.Generator(
            data_model=Answer, language_model=LanguageModel(model="openai/gpt-4o-mini")
        )(x0)
        return synalinks.Program(inputs=x0, outputs=x1, name="qa")

    program = asyncio.run(build())
    attach(program)
    return program


def _ask(program, config, *questions):
    async def ask():
        with use_tracing(config, run=RUN):
            return await asyncio.gather(
                *(program(Query(question=q)) for q in questions)
            )

    answers = asyncio.run(ask())
    provider_for(config).force_flush()
    return answers


def _chat_spans(exporter):
    return [
        s
        for s in exporter.get_finished_spans()
        if s.attributes.get("gen_ai.operation.name") == "chat"
    ]


def test_a_program_call_is_one_trace_with_a_chat_span(exported):
    _ask(_program(), _config(), "2 + 2?")
    spans = exported.get_finished_spans()
    assert len({s.context.trace_id for s in spans}) == 1
    (root,) = [s for s in spans if s.parent is None]
    assert root.name == "invoke_workflow qa"
    (chat,) = _chat_spans(exported)
    assert chat.name == "chat openai/gpt-4o-mini"
    assert chat.kind == SpanKind.CLIENT
    a = chat.attributes
    assert a["gen_ai.provider.name"] == "openai"
    assert a["gen_ai.request.model"] == "openai/gpt-4o-mini"
    assert a["gen_ai.usage.input_tokens"] > 0
    assert a["gen_ai.usage.cost.amount"] > 0
    assert a["gen_ai.usage.cost.currency"] == "USD"
    assert "gen_ai.input.messages" in a
    assert json.loads(a["gen_ai.output.messages"])[0]["role"] == "assistant"
    assert all(s.attributes["langsmith.trace.session_name"] == RUN for s in spans)


def test_outside_use_tracing_nothing_is_sent(exported):
    program = _program()
    asyncio.run(program(Query(question="2 + 2?")))
    assert exported.get_finished_spans() == ()


def test_content_off_keeps_messages_out(exported):
    _ask(_program(), _config(content=False), "2 + 2?")
    (chat,) = _chat_spans(exported)
    assert "gen_ai.input.messages" not in chat.attributes
    assert "gen_ai.output.messages" not in chat.attributes
    assert chat.attributes["gen_ai.usage.input_tokens"] > 0


def test_concurrent_calls_get_their_own_traces(exported):
    _ask(_program(), _config(), "a?", "b?", "c?")
    spans = exported.get_finished_spans()
    by_id = {s.context.span_id: s for s in spans}
    chats = _chat_spans(exported)
    assert len(chats) == 3
    assert len({s.context.trace_id for s in chats}) == 3
    for chat in chats:
        parent = by_id[chat.parent.span_id]
        assert parent.context.trace_id == chat.context.trace_id


def test_symbolic_calls_make_no_span(exported):
    async def build():
        with use_tracing(_config(), run=RUN):
            generator = synalinks.Generator(
                data_model=Answer,
                language_model=LanguageModel(model="openai/gpt-4o-mini"),
            )
            hook = OtelHook()
            hook.set_module(generator)
            generator.add_hook(hook)
            await generator(synalinks.Input(data_model=Query))

    asyncio.run(build())
    provider_for(_config()).force_flush()
    assert exported.get_finished_spans() == ()


class Boom(synalinks.Module):
    async def call(self, inputs, training=False):
        raise ValueError("boom")

    async def compute_output_spec(self, inputs, training=False):
        return inputs


def test_a_failed_call_is_an_error_span(exported):
    boom = Boom(name="boom")
    hook = OtelHook()
    hook.set_module(boom)
    boom.add_hook(hook)

    async def call():
        with use_tracing(_config(), run=RUN):
            await boom(Query(question="?"))

    with pytest.raises(ValueError, match="boom"):
        asyncio.run(call())
    provider_for(_config()).force_flush()
    (span,) = exported.get_finished_spans()
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes["error.type"] == "_OTHER"


def test_a_broken_hook_never_breaks_the_call(exported, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("broken")

    monkeypatch.setattr(otel, "_describe", broken)
    (answer,) = _ask(_program(), _config(), "2 + 2?")
    assert answer.get_json() == {"answer": "4"}


def test_a_broken_exporter_never_breaks_the_call(monkeypatch):
    class Broken(InMemorySpanExporter):
        def export(self, spans):
            raise RuntimeError("down")

    monkeypatch.setattr(otel, "_exporter", lambda destination: Broken())
    monkeypatch.setattr(otel, "_providers", {})
    (answer,) = _ask(_program(), _config(), "2 + 2?")
    assert answer.get_json() == {"answer": "4"}


def test_attach_adds_one_hook():
    program = _program()
    attach(program)
    for module in program._flatten_modules():
        assert sum(isinstance(h, OtelHook) for h in module._hooks.hooks) == 1


def test_a_traced_program_still_deep_copies(exported):
    program = _program()
    _ask(program, _config(), "2 + 2?")
    copy.deepcopy(program)


def test_one_provider_per_destination_list(exported):
    assert provider_for(_config()) is provider_for(_config())
    assert provider_for(_config()) is not provider_for(
        TraceConfig(destinations=(datadog("k"),))
    )


# -- a traced run --------------------------------------------------------------------


@pytest.fixture
def mocked_models(monkeypatch):
    """Every model of a sweep answered by litellm's mock, priced as the real one."""
    monkeypatch.setattr(
        synalinks.LanguageModel, "_acompletion", LanguageModel._acompletion
    )


def _sweep(tmp_path, config=None, config_trace=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = _sweep_config(tmp_path)
    if config:
        raw = yaml.safe_load(path.read_text())
        config(raw)
        path.write_text(yaml.safe_dump(raw))

    def run():
        return asyncio.run(
            run_sweep(str(path), state_dir=tmp_path / "state", no_cache=True)
        )

    if config_trace is None:
        return run()
    with use_tracing(config_trace, run=RUN):
        result = run()
    provider_for(config_trace).force_flush()
    return result


def _scores(result):
    return sorted((r["model"], r["metric"], r["value"]) for r in result["rows"])


def test_a_sweep_inside_use_tracing_sends_every_row(tmp_path, exported, mocked_models):
    untraced = _sweep(tmp_path / "plain")
    traced = _sweep(tmp_path / "traced", config_trace=_config())
    assert _scores(traced) == _scores(untraced)
    spans = exported.get_finished_spans()
    roots = {s.context.trace_id for s in spans if s.parent is None}
    assert len(roots) >= 6  # 3 rows x 2 models, at least
    assert {
        s.attributes.get("gen_ai.request.model") for s in _chat_spans(exported)
    } == {
        "openai/gpt-4o-mini",
        "openai/gpt-4o",
    }
    assert all(s.attributes["langsmith.trace.session_name"] == RUN for s in spans)


def test_a_judge_reward_is_traced(tmp_path, exported, mocked_models):
    def judged(raw):
        raw["models"]["judge"] = {"model": "openai/gpt-4.1-mini"}
        raw["rewards"]["graded"] = {
            "type": "lm_as_judge",
            "config": {"language_model": "judge"},
        }
        raw["datasets"]["qa"]["reward"] = "graded"

    _sweep(tmp_path, config=judged, config_trace=_config())
    models = {s.attributes.get("gen_ai.request.model") for s in _chat_spans(exported)}
    assert "openai/gpt-4.1-mini" in models


def test_a_dead_destination_costs_the_run_nothing(tmp_path, monkeypatch, mocked_models):
    monkeypatch.setattr(otel, "_providers", {})
    nowhere = TraceConfig(
        destinations=(
            otel.Destination(
                "nowhere", "http://127.0.0.1:9/v1/traces", (), lambda run: {}
            ),
        )
    )
    assert isinstance(otel._exporter(nowhere.destinations[0]), OTLPSpanExporter)
    start = time.perf_counter()
    untraced = _sweep(tmp_path / "plain")
    plain = time.perf_counter() - start
    start = time.perf_counter()
    # No flush: what is timed is the run, the export waits on its own thread.
    (tmp_path / "traced").mkdir()
    path = _sweep_config(tmp_path / "traced")
    with use_tracing(nowhere, run=RUN):
        traced = asyncio.run(
            run_sweep(str(path), state_dir=tmp_path / "traced" / "state", no_cache=True)
        )
    assert time.perf_counter() - start < plain + otel.EXPORT_TIMEOUT
    assert _scores(traced) == _scores(untraced)


# -- MLflow keeps its own path -----------------------------------------------------


def test_mlflow_is_untouched_by_trace_destinations(monkeypatch):
    """Destinations named, even the old LangSmith switch on: MLflow exports to its server alone."""
    for name in (
        "MLFLOW_TRACE_ENABLE_OTLP_DUAL_EXPORT",
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        "OTEL_EXPORTER_OTLP_TRACES_HEADERS",
    ):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    monkeypatch.setenv("ARENA_TRACE_DESTINATIONS", "langsmith,datadog")
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls")
    monkeypatch.setenv("DD_API_KEY", "dd")
    monkeypatch.setenv("ARENA_LANGSMITH_TRACING", "1")
    monkeypatch.setenv("ARENA_MLFLOW_TRACING", "1")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///unused.db")
    monkeypatch.setattr(
        mlflow.tracing.config,
        "_MLFLOW_TRACING_CONFIG",
        mlflow.tracing.config.TracingConfig(),
    )
    monkeypatch.setattr(
        observability.synalinks, "enable_observability", lambda **kw: None
    )
    monkeypatch.setattr(mlflow.tracing.provider, "reset", lambda: None)
    monkeypatch.setattr(mlflow, "MlflowClient", lambda tracking_uri=None: None)

    assert observability.enable_observability()
    assert mlflow.tracing.config.get_config().span_processors == []
    assert not os.environ.get("MLFLOW_TRACE_ENABLE_OTLP_DUAL_EXPORT")
    assert not os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")


class Silent(synalinks.Module):
    async def call(self, inputs, training=False):
        raise ValueError()

    async def compute_output_spec(self, inputs, training=False):
        return inputs


def test_an_exception_without_a_message_is_still_an_error(exported):
    silent = Silent(name="silent")
    hook = OtelHook()
    hook.set_module(silent)
    silent.add_hook(hook)

    async def call():
        with use_tracing(_config(), run=RUN):
            await silent(Query(question="?"))

    with pytest.raises(ValueError):
        asyncio.run(call())
    provider_for(_config()).force_flush()
    (span,) = exported.get_finished_spans()
    assert span.status.status_code == StatusCode.ERROR


def test_inline_images_and_audio_stay_out_of_the_span():
    image = "data:image/png;base64," + "A" * 10_000
    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "What is this?"},
            {"type": "image_url", "image_url": {"url": image}},
            {
                "type": "input_audio",
                "input_audio": {"data": "B" * 10_000, "format": "wav"},
            },
        ],
    }
    parts = otel._parts(message)["parts"]
    assert parts[0] == {"type": "text", "content": "What is this?"}
    assert [p["type"] for p in parts[1:]] == ["image_url", "input_audio"]
    assert "AAAA" not in json.dumps(parts) and "BBBB" not in json.dumps(parts)
