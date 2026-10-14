# License Apache 2.0: (c) 2026 Atena-Reply

"""A model's price per token, from the first source that knows it. The network and both tables are stubbed."""

import os

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import litellm
import pytest

from openarena.lookups import eol as eol_module
from openarena.lookups.eol import litellm_model_info
from openarena.lookups.prices import Price, ensure_priced, model_price

ROW = {
    "input_cost_per_token": 3e-6,
    "output_cost_per_token": 15e-6,
    "litellm_provider": "anthropic",
}
QWEN = {
    "id": "qwen/qwen3.8-27b",
    "pricing": {"prompt": "0.0000002", "completion": "0.0000006"},
}
OPUS = {
    "id": "anthropic/claude-opus-5.5",
    "pricing": {"prompt": "0.000005", "completion": "0.000025"},
}


@pytest.fixture(autouse=True)
def sources(monkeypatch):
    """litellm's live table, its shipped copy and OpenRouter's list, empty unless a test fills them."""
    live: dict = {}
    bundled: dict = {}
    listed: dict = {"data": []}
    monkeypatch.setattr(litellm, "model_cost", live)
    monkeypatch.setattr(eol_module, "bundled_table", lambda: bundled)
    monkeypatch.setattr(eol_module, "_filled_table", None)
    monkeypatch.setattr(eol_module, "_from_bundled", frozenset())
    monkeypatch.setattr(eol_module, "_fetch_openrouter", lambda: listed)
    monkeypatch.setattr(eol_module, "_openrouter_cache", None)
    return live, bundled, listed


def test_a_server_of_your_own_is_free():
    assert model_price("ollama/qwen3") == Price(0.0, 0.0, "local")
    assert model_price("vllm/llama-4") == Price(0.0, 0.0, "local")


def test_litellms_live_table_answers_first(sources):
    live, bundled, listed = sources
    live["anthropic/claude-sonnet-5"] = ROW
    bundled["anthropic/claude-sonnet-5"] = {**ROW, "input_cost_per_token": 1.0}
    listed["data"] = [
        {
            "id": "anthropic/claude-sonnet-5",
            "pricing": {"prompt": "1", "completion": "1"},
        }
    ]
    assert model_price("anthropic/claude-sonnet-5") == Price(3e-6, 15e-6, "litellm")


def test_rows_only_the_shipped_table_has_fill_the_live_one(sources):
    live, bundled, _ = sources
    bundled["openrouter/anthropic/claude-haiku-4.5"] = {
        **ROW,
        "litellm_provider": "openrouter",
        "max_input_tokens": 200_000,
    }
    assert model_price("openrouter/anthropic/claude-haiku-4.5") == Price(
        3e-6, 15e-6, "litellm-bundled"
    )
    # Every reader gets the row, litellm's own cost of a call included; the old table is left as it was.
    assert (
        litellm.model_cost["openrouter/anthropic/claude-haiku-4.5"]["max_input_tokens"]
        == 200_000
    )
    assert (
        litellm_model_info("openrouter/anthropic/claude-haiku-4.5")["max_input_tokens"]
        == 200_000
    )
    assert "openrouter/anthropic/claude-haiku-4.5" not in live


def test_openrouter_prices_its_own_ids_and_the_vendors_same_model(sources):
    _, _, listed = sources
    listed["data"] = [
        QWEN,
        OPUS,
        {
            "id": "google/gemini-3.1-pro",
            "pricing": {"prompt": "0.000002", "completion": "0.000012"},
        },
    ]
    assert model_price("openrouter/qwen/qwen3.8-27b") == Price(2e-7, 6e-7, "openrouter")
    # The vendor's own id, spelt its own way: the same model, the price OpenRouter passes through.
    assert model_price("anthropic/claude-opus-5-5") == Price(5e-6, 25e-6, "openrouter")
    assert model_price("gemini/gemini-3.1-pro") == Price(2e-6, 12e-6, "openrouter")
    # A provider OpenRouter does not stand for is not guessed at.
    assert model_price("together_ai/qwen/qwen3.8-27b") is None


def test_a_complete_price_wins_over_a_half_known_one(sources):
    _, bundled, listed = sources
    bundled["anthropic/claude-opus-5-5"] = {
        "output_cost_per_token": 25e-6,
        "litellm_provider": "anthropic",
    }
    assert model_price("anthropic/claude-opus-5-5") == Price(
        None, 25e-6, "litellm-bundled"
    )
    eol_module._openrouter_cache = None
    listed["data"] = [OPUS]
    assert model_price("anthropic/claude-opus-5-5") == Price(5e-6, 25e-6, "openrouter")


def test_unknown_is_not_free(sources):
    _, _, listed = sources
    # OpenRouter's auto router has no fixed price.
    listed["data"] = [
        {"id": "openrouter/auto", "pricing": {"prompt": "-1", "completion": "-1"}}
    ]
    assert model_price("openrouter/openrouter/auto") is None
    assert model_price("openai/not-a-model-anyone-sells") is None


def test_offline_the_tables_still_answer(monkeypatch, sources):
    _, bundled, _ = sources
    monkeypatch.setattr(eol_module, "_fetch_openrouter", lambda: None)
    bundled["anthropic/claude-sonnet-5"] = ROW
    assert model_price("anthropic/claude-sonnet-5") == Price(
        3e-6, 15e-6, "litellm-bundled"
    )
    assert model_price("openrouter/qwen/qwen3.8-27b") is None


def test_a_price_only_openrouter_has_is_given_to_litellm_for_a_run(sources):
    live, _, listed = sources
    listed["data"] = [OPUS]
    ensure_priced("anthropic/claude-opus-5-5")
    row = litellm.model_cost["anthropic/claude-opus-5-5"]
    assert (row["input_cost_per_token"], row["output_cost_per_token"], row["mode"]) == (
        5e-6,
        25e-6,
        "chat",
    )
    # A new table, not the old one changed: a request reading the old one is not disturbed.
    assert "anthropic/claude-opus-5-5" not in live
    # Still shown as OpenRouter's, and the row knows nothing else, so it does not hide the facts.
    assert model_price("anthropic/claude-opus-5-5").source == "openrouter"
    assert litellm_model_info("anthropic/claude-opus-5-5") is None
    # A price that moved is picked up by the next run.
    eol_module._openrouter_cache = None
    listed["data"] = [
        {**OPUS, "pricing": {"prompt": "0.000004", "completion": "0.000025"}}
    ]
    ensure_priced("anthropic/claude-opus-5-5")
    assert litellm.model_cost["anthropic/claude-opus-5-5"][
        "input_cost_per_token"
    ] == pytest.approx(4e-6)


def test_the_prices_join_a_row_litellm_has_without_them(sources):
    live, _, listed = sources
    live["anthropic/claude-opus-5-5"] = {
        "litellm_provider": "anthropic",
        "mode": "chat",
        "max_input_tokens": 1_000_000,
    }
    listed["data"] = [OPUS]
    ensure_priced("anthropic/claude-opus-5-5")
    row = litellm.model_cost["anthropic/claude-opus-5-5"]
    assert row["max_input_tokens"] == 1_000_000 and row["input_cost_per_token"] == 5e-6
    assert (
        litellm_model_info("anthropic/claude-opus-5-5")["max_input_tokens"] == 1_000_000
    )


def test_an_aliased_provider_is_priced_under_litellms_own_key(sources):
    _, _, listed = sources
    listed["data"] = [
        {
            "id": "cohere/command-x",
            "pricing": {"prompt": "0.000001", "completion": "0.000002"},
        }
    ]
    ensure_priced("cohere/command-x")
    assert (
        litellm.model_cost["cohere_chat/command-x"]["litellm_provider"] == "cohere_chat"
    )


def test_left_alone_what_litellm_prices_is_free_or_reports_itself(sources):
    live, _, listed = sources
    live["anthropic/claude-sonnet-5"] = ROW
    listed["data"] = [QWEN]
    ensure_priced("anthropic/claude-sonnet-5")
    ensure_priced("ollama/qwen3")
    # OpenRouter's answers carry their own cost, which litellm reads.
    ensure_priced("openrouter/qwen/qwen3.8-27b")
    assert litellm.model_cost is live


def test_a_half_known_price_is_not_given_to_litellm(sources):
    _, bundled, _ = sources
    bundled["anthropic/claude-sonnet-5"] = {
        "output_cost_per_token": 15e-6,
        "litellm_provider": "anthropic",
    }
    ensure_priced("anthropic/claude-sonnet-5")
    assert "input_cost_per_token" not in litellm.model_cost["anthropic/claude-sonnet-5"]


def test_a_pricing_failure_never_stops_a_run(monkeypatch):
    def broken():
        raise RuntimeError("openrouter answered nonsense")

    monkeypatch.setattr(eol_module, "_fetch_openrouter", broken)
    assert ensure_priced("anthropic/claude-opus-5-5") is None


def test_a_stale_catalogue_answers_at_once_and_is_refreshed_behind(monkeypatch, sources):
    import threading

    _, _, listed = sources
    listed["data"] = [QWEN]
    assert model_price("openrouter/qwen/qwen3.8-27b").input == pytest.approx(2e-7)
    # A day later OpenRouter is slow: the request is answered from what was known, not kept waiting.
    gate = threading.Event()

    def slow():
        gate.wait(5)
        return {"data": [{**QWEN, "pricing": {"prompt": "0.0000003", "completion": "0.0000006"}}]}

    monkeypatch.setattr(eol_module, "_fetch_openrouter", slow)
    at, _ttl, *index = eol_module._openrouter_cache
    eol_module._openrouter_cache = (at - 2 * eol_module._TTL_S, eol_module._TTL_S, *index)
    assert model_price("openrouter/qwen/qwen3.8-27b").input == pytest.approx(2e-7)
    gate.set()
    for _ in range(50):
        if not eol_module._openrouter_refreshing:
            break
        threading.Event().wait(0.05)
    assert model_price("openrouter/qwen/qwen3.8-27b").input == pytest.approx(3e-7)
