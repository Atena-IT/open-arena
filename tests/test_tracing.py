# License Apache 2.0: (c) 2026 Atena-Reply

"""Trace destinations: the entries read from the environment, the hook's spans, a traced run.

Spans are caught by an in-memory exporter put where each destination's OTLP
exporter would be (`otel._exporter`), so nothing leaves the process.
"""

import json
import logging

import pytest

from openarena.hooks import otel
from openarena.hooks.otel import TraceConfig, config_from_env, datadog, langsmith

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
