# License Apache 2.0: (c) 2026 Atena-Reply

"""OpenRouter's endpoints of a model, read once a day. The network is stubbed."""

import json
import threading
from pathlib import Path

import pytest

from openarena.lookups import endpoints as endpoints_module
from openarena.lookups.endpoints import endpoints_url, openrouter_endpoints

LLAMA = "meta-llama/llama-3.1-8b-instruct"
PAYLOAD = json.loads(
    (Path(__file__).parent / "data" / "openrouter_endpoints.json").read_text(encoding="utf-8")
)


@pytest.fixture(autouse=True)
def fetches(monkeypatch):
    """The slugs fetched, in order; each fetch answers with the llama payload unless a test says otherwise."""
    asked: list[str] = []

    def fetch(slug):
        asked.append(slug)
        return PAYLOAD

    monkeypatch.setattr(endpoints_module, "_fetch_endpoints", fetch)
    monkeypatch.setattr(endpoints_module, "_cache", {})
    monkeypatch.setattr(endpoints_module, "_refreshing", set())
    return asked


def _wait_for_refresh(slug):
    for _ in range(100):
        if slug not in endpoints_module._refreshing:
            return
        threading.Event().wait(0.05)
    raise AssertionError("the refresh never ended")


def test_a_slug_outside_openrouters_form_is_never_fetched(fetches):
    for bad in ("../x", "./x", "a/..", "a/b?c=1", "a/b#c", "a/b/c", "A/b", "a", "", "a/b c", "a/b%2f"):
        assert openrouter_endpoints(bad) is None
    assert fetches == []


def test_endpoints_are_read_once_a_day(fetches):
    first = openrouter_endpoints(LLAMA)
    second = openrouter_endpoints(LLAMA)
    assert [e["tag"] for e in first] == [e["tag"] for e in PAYLOAD["data"]["endpoints"]]
    assert second == first
    assert fetches == [LLAMA]


def test_a_stale_entry_is_served_while_it_refreshes(monkeypatch, fetches):
    openrouter_endpoints(LLAMA)
    gate = threading.Event()
    cheaper = json.loads(json.dumps(PAYLOAD))
    cheaper["data"]["endpoints"] = cheaper["data"]["endpoints"][:1]

    def slow(slug):
        gate.wait(5)
        return cheaper

    monkeypatch.setattr(endpoints_module, "_fetch_endpoints", slow)
    at, ttl, rows = endpoints_module._cache[LLAMA]
    endpoints_module._cache[LLAMA] = (at - 2 * endpoints_module._TTL_S, ttl, rows)
    # Answered at once from what was known, not kept waiting on OpenRouter.
    assert len(openrouter_endpoints(LLAMA)) == len(PAYLOAD["data"]["endpoints"])
    gate.set()
    _wait_for_refresh(LLAMA)
    assert len(openrouter_endpoints(LLAMA)) == 1


def test_offline_keeps_the_last_list_and_asks_again_sooner(monkeypatch):
    openrouter_endpoints(LLAMA)
    monkeypatch.setattr(endpoints_module, "_fetch_endpoints", lambda slug: None)
    at, ttl, rows = endpoints_module._cache[LLAMA]
    endpoints_module._cache[LLAMA] = (at - 2 * endpoints_module._TTL_S, ttl, rows)
    openrouter_endpoints(LLAMA)
    _wait_for_refresh(LLAMA)
    assert openrouter_endpoints(LLAMA) == rows
    assert endpoints_module._cache[LLAMA][1] == endpoints_module._RETRY_S


def test_offline_before_any_read_is_none(monkeypatch):
    monkeypatch.setattr(endpoints_module, "_fetch_endpoints", lambda slug: None)
    assert openrouter_endpoints(LLAMA) is None


def test_a_failed_first_read_is_not_asked_again_within_the_hour(monkeypatch, fetches):
    def offline(slug):
        fetches.append(slug)
        return None

    monkeypatch.setattr(endpoints_module, "_fetch_endpoints", offline)
    assert openrouter_endpoints(LLAMA) is None
    assert openrouter_endpoints(LLAMA) is None
    assert fetches == [LLAMA]
    assert endpoints_module._cache[LLAMA][1] == endpoints_module._RETRY_S


def test_an_answer_without_endpoints_is_an_empty_list(monkeypatch):
    monkeypatch.setattr(endpoints_module, "_fetch_endpoints", lambda slug: {"data": {"endpoints": []}})
    assert openrouter_endpoints(LLAMA) == []


def test_the_slug_goes_into_a_fixed_host():
    assert endpoints_url(LLAMA + ":free") == (
        "https://openrouter.ai/api/v1/models/meta-llama/llama-3.1-8b-instruct:free/endpoints"
    )
