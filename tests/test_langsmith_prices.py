# License Apache 2.0: (c) 2026 Atena-Reply

"""LangSmith's price table kept in step with the arena's prices of a run's models."""

import pytest

from openarena.hooks import langsmith_prices, otel
from openarena.lookups.prices import Price

API = "https://eu.api.smith.langchain.com/api/v1/model-price-map"
MODEL = "openrouter/openai/gpt-oss-120b"


class FakeLangSmith:
    """The price table's API: a few built-in rows, and the ones written to it."""

    def __init__(self, rows=None):
        self.rows = list(rows or [])
        self.calls = []

    def __call__(self, method, url, key, body=None):
        self.calls.append((method, url.removeprefix(API), key, body))
        if method == "GET":
            return [r for r in self.rows if "gpt-oss" in r["name"]]
        if method == "POST":
            self.rows.append(body | {"id": f"row-{len(self.rows)}"})
            return self.rows[-1]
        if method == "PUT":
            row_id = url.rsplit("/", 1)[1]
            self.rows = [
                body | {"id": row_id} if r["id"] == row_id else r for r in self.rows
            ]
            return body
        raise AssertionError(method)


@pytest.fixture
def langsmith(monkeypatch):
    fake = FakeLangSmith(
        [
            {
                "id": "groq",
                "name": "GPT OSS 120B",
                "prompt_cost": 1.5e-07,
                "completion_cost": 6e-07,
            }
        ]
    )
    monkeypatch.setattr(langsmith_prices, "_call", fake)
    monkeypatch.setattr(langsmith_prices, "_written", {})
    monkeypatch.setattr(
        langsmith_prices, "model_price", lambda model: Price(1e-07, 5e-07, "openrouter")
    )
    return fake


def _config(*destinations):
    return otel.TraceConfig(destinations=destinations)


def test_a_model_missing_from_the_table_gets_a_row(langsmith):
    langsmith_prices.price_models(
        _config(otel.langsmith("k", "https://eu.api.smith.langchain.com")),
        [MODEL, MODEL],
    )
    (post,) = [c for c in langsmith.calls if c[0] == "POST"]
    assert post[2] == "k"
    # LangSmith wants a price per token, and matches the name a span carries: the arena's model id.
    assert post[3] == {
        "name": "arena: openrouter/openai/gpt-oss-120b",
        "match_pattern": r"^openrouter/openai/gpt\-oss\-120b$",
        "prompt_cost": 1e-07,
        "completion_cost": 5e-07,
    }


def test_a_row_is_written_once_and_rewritten_when_the_price_moves(
    langsmith, monkeypatch
):
    config = _config(otel.langsmith("k", "https://eu.api.smith.langchain.com"))
    langsmith_prices.price_models(config, [MODEL])
    # Prices that did not move: LangSmith is not asked again.
    langsmith_prices.price_models(config, [MODEL])
    assert [c[0] for c in langsmith.calls] == ["GET", "POST"]
    monkeypatch.setattr(
        langsmith_prices, "model_price", lambda model: Price(2e-07, 5e-07, "openrouter")
    )
    langsmith_prices.price_models(config, [MODEL])
    assert langsmith.calls[-1][0] == "PUT"
    assert langsmith.calls[-1][1] == "/row-1"
    assert len([r for r in langsmith.rows if r["name"].startswith("arena: ")]) == 1


def test_a_search_that_misses_the_row_never_writes_it_twice(langsmith, monkeypatch):
    def missing(method, url, key, body=None):
        return [] if method == "GET" else langsmith(method, url, key, body)

    monkeypatch.setattr(langsmith_prices, "_call", missing)
    config = _config(otel.langsmith("k", "https://eu.api.smith.langchain.com"))
    langsmith_prices.price_models(config, [MODEL])
    langsmith_prices.price_models(config, [MODEL])
    assert [c[0] for c in langsmith.calls].count("POST") == 1


def test_a_self_hosted_langsmith_keeps_one_api_prefix():
    assert langsmith_prices._table(
        "https://eu.api.smith.langchain.com/otel/v1/traces"
    ) == ("https://eu.api.smith.langchain.com/api/v1/model-price-map")
    assert langsmith_prices._table("https://ls.example.com/api/v1/otel/v1/traces") == (
        "https://ls.example.com/api/v1/model-price-map"
    )


def test_an_unknown_or_local_price_writes_nothing(langsmith, monkeypatch):
    config = _config(otel.langsmith("k", "https://eu.api.smith.langchain.com"))
    monkeypatch.setattr(
        langsmith_prices, "model_price", lambda model: Price(None, 5e-07, "openrouter")
    )
    langsmith_prices.price_models(config, [MODEL])
    monkeypatch.setattr(
        langsmith_prices, "model_price", lambda model: Price(0.0, 0.0, "local")
    )
    langsmith_prices.price_models(config, ["ollama/qwen3"])
    assert langsmith.calls == []


def test_only_langsmith_is_priced_and_a_failure_never_stops_the_run(
    monkeypatch, caplog
):
    calls = []

    def broken(*args, **kwargs):
        calls.append(args)
        raise OSError("LangSmith is down")

    monkeypatch.setattr(langsmith_prices, "_call", broken)
    monkeypatch.setattr(langsmith_prices, "_written", {})
    monkeypatch.setattr(
        langsmith_prices, "model_price", lambda model: Price(1e-07, 5e-07, "openrouter")
    )
    langsmith_prices.price_models(_config(otel.datadog("dd")), [MODEL])
    langsmith_prices.price_models(
        _config(otel.langsmith("k", "https://eu.api.smith.langchain.com")),
        [MODEL, "x/y"],
    )
    assert "LangSmith is down" in caplog.text
    # The first failure ends the turn: one timeout and one line, however many models.
    assert len(calls) == 1
    assert caplog.text.count("LangSmith prices") == 1
