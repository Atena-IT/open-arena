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

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import litellm
import pytest
import synalinks
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode

from openarena.hooks import otel
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
    assert entry.headers == (("x-api-key", "k%2Fey"),)
    assert entry.run_attributes("ws/eval-smoke/1") == {
        "langsmith.trace.session_name": "ws/eval-smoke/1"
    }

    monkeypatch.setenv("ARENA_TRACE_DESTINATIONS", "langsmith")
    monkeypatch.setenv("LANGSMITH_API_KEY", "k")
    monkeypatch.setenv("LANGSMITH_ENDPOINT", "https://eu.api.smith.langchain.com/")
    (entry,) = config_from_env().destinations
    assert entry.endpoint == "https://eu.api.smith.langchain.com/otel/v1/traces"


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
            model=self.model, messages=formatted_messages, mock_response='{"answer": "4"}', **kwargs
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
            return await asyncio.gather(*(program(Query(question=q)) for q in questions))

    answers = asyncio.run(ask())
    provider_for(config).force_flush()
    return answers


def _chat_spans(exporter):
    return [s for s in exporter.get_finished_spans() if s.attributes.get("gen_ai.operation.name") == "chat"]


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
                data_model=Answer, language_model=LanguageModel(model="openai/gpt-4o-mini")
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
    assert provider_for(_config()) is not provider_for(TraceConfig(destinations=(datadog("k"),)))
