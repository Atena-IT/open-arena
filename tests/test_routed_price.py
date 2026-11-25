# License Apache 2.0: (c) 2026 Atena-Reply

"""An OpenRouter model's price range over the hosts its evaluation calls can reach, and why a kind of call has none.

The endpoints are stubbed from a real read of llama-3.1-8b (`tests/data/openrouter_endpoints.json`):
DeepInfra (structured answers, no tools), Groq and Bedrock (tools), CoreWeave (both, and
`structured_outputs`, but ruled out of structured calls by the framework).
"""

import asyncio
import copy
import json
from pathlib import Path
from typing import get_args

import pytest

from openarena.lookups import endpoints as endpoints_module
from openarena.lookups import prices as prices_module
from openarena.lookups.prices import Reason, RoutedPrice, routed_price

LLAMA = "openrouter/meta-llama/llama-3.1-8b-instruct"
ENDPOINTS = json.loads(
    (Path(__file__).parent / "data" / "openrouter_endpoints.json").read_text(encoding="utf-8")
)["data"]["endpoints"]


def _endpoint(tag, prompt, completion, params, max_completion=100_000, service_tier=None):
    return {
        "tag": tag,
        "provider_name": tag,
        "pricing": {"prompt": str(prompt), "completion": str(completion)},
        "supported_parameters": list(params),
        "max_completion_tokens": max_completion,
        "service_tier": service_tier,
    }


STRUCTURED = ("response_format", "structured_outputs", "max_tokens", "temperature")
TOOLS = ("tools", "tool_choice", "max_tokens", "temperature")
BOTH = STRUCTURED + ("tools", "tool_choice")


@pytest.fixture
def listed(monkeypatch):
    """What OpenRouter lists for any model id asked; the slugs asked, in order."""
    state = {"endpoints": copy.deepcopy(ENDPOINTS), "asked": []}

    def endpoints(slug):
        state["asked"].append(slug)
        return state["endpoints"]

    monkeypatch.setattr(prices_module, "openrouter_endpoints", endpoints)
    return state


def _price(model_id=LLAMA, params=None) -> RoutedPrice | None:
    return asyncio.run(routed_price(model_id, params))


def test_llamas_structured_calls_have_no_host_left_and_its_tool_calls_have_three(listed):
    price = _price()
    # CoreWeave is the only host with `structured_outputs`, and the framework rules it out of structured calls.
    assert price.kinds == {"structured": "platform_exclusion", "tools": None}
    # The range is the tool hosts': Groq ($0.05/$0.08), Bedrock and CoreWeave ($0.22/$0.22).
    assert (price.input_min, price.input_max) == pytest.approx((5e-8, 2.2e-7))
    assert (price.output_min, price.output_max) == pytest.approx((8e-8, 2.2e-7))
    assert listed["asked"] == ["meta-llama/llama-3.1-8b-instruct"]


def test_a_model_both_kinds_reach_gives_the_union_range(listed):
    listed["endpoints"] = [
        _endpoint("alpha", 1e-7, 2e-7, STRUCTURED),
        _endpoint("beta/fp8", 3e-7, 6e-7, TOOLS),
    ]
    price = _price()
    assert price.kinds == {"structured": None, "tools": None}
    assert (price.input_min, price.input_max, price.output_min, price.output_max) == pytest.approx(
        (1e-7, 3e-7, 2e-7, 6e-7)
    )


def test_hosts_without_structured_answers_give_that_reason(listed):
    listed["endpoints"] = [_endpoint("beta", 3e-7, 6e-7, TOOLS)]
    assert _price().kinds == {"structured": "structured_output", "tools": None}


def test_hosts_without_tools_give_that_reason(listed):
    listed["endpoints"] = [_endpoint("alpha", 1e-7, 2e-7, STRUCTURED)]
    assert _price().kinds == {"structured": None, "tools": "tools"}


def test_a_host_that_takes_tools_but_not_tool_choice_cannot_serve_an_agent(listed):
    listed["endpoints"] = [_endpoint("alpha", 1e-7, 2e-7, STRUCTURED + ("tools",))]
    assert _price().kinds == {"structured": None, "tools": "tools"}


def test_a_reasoning_level_no_host_takes(listed):
    listed["endpoints"] = [_endpoint("alpha", 1e-7, 2e-7, BOTH)]
    price = _price(params={"reasoning_effort": "high", "supports_reasoning": True})
    assert price.kinds == {"structured": "reasoning", "tools": "reasoning"}
    assert price.input_min is None and price.output_max is None


def test_a_temperature_no_host_takes(listed):
    listed["endpoints"] = [_endpoint("alpha", 1e-7, 2e-7, ("response_format", "structured_outputs", "tools", "tool_choice"))]
    assert _price(params={"temperature": 0.2}).kinds == {"structured": "temperature", "tools": "temperature"}


def test_max_tokens_above_every_hosts_limit(listed):
    listed["endpoints"] = [_endpoint("alpha", 1e-7, 2e-7, BOTH, max_completion=8192)]
    assert _price(params={"max_tokens": 16_000}).kinds == {"structured": "max_tokens", "tools": "max_tokens"}
    assert _price(params={"max_tokens": 4_000}).kinds == {"structured": None, "tools": None}


def test_an_unknown_param_no_host_takes(listed):
    listed["endpoints"] = [_endpoint("alpha", 1e-7, 2e-7, BOTH)]
    assert _price(params={"top_p": 0.5}).kinds == {"structured": "parameter", "tools": "parameter"}


def test_flex_or_priority_hosts_only(listed):
    listed["endpoints"] = [
        _endpoint("alpha", 1e-7, 2e-7, BOTH, service_tier="flex"),
        _endpoint("beta", 1e-7, 2e-7, BOTH, service_tier="priority"),
    ]
    price = _price()
    assert price.kinds == {"structured": "platform_exclusion", "tools": "platform_exclusion"}
    assert price.input_min is None


def test_a_host_the_model_entry_rules_out(listed):
    listed["endpoints"] = [_endpoint("alpha", 1e-7, 2e-7, BOTH), _endpoint("beta/fp8", 9e-7, 9e-7, BOTH)]
    price = _price(params={"provider": {"ignore": ["Beta"]}})
    assert price.kinds == {"structured": None, "tools": None}
    assert price.input_max == pytest.approx(1e-7)
    only = _price(params={"provider": {"only": ["gamma"]}})
    assert only.kinds == {"structured": "platform_exclusion", "tools": "platform_exclusion"}


def test_a_negative_price_is_left_out_of_the_range(listed):
    listed["endpoints"] = [_endpoint("alpha", 1e-7, 2e-7, BOTH), _endpoint("auto", -1, -1, BOTH)]
    price = _price()
    assert price.kinds == {"structured": None, "tools": None}
    assert (price.input_min, price.input_max) == pytest.approx((1e-7, 1e-7))


def test_openrouter_unreachable_before_any_read(listed):
    listed["endpoints"] = None
    price = _price()
    assert price == RoutedPrice(None, None, None, None, {"structured": "price_unavailable", "tools": "price_unavailable"})


def test_an_id_outside_openrouters_form_is_unavailable_and_never_fetched(monkeypatch):
    fetched = []
    monkeypatch.setattr(endpoints_module, "_fetch_endpoints", lambda slug: fetched.append(slug))
    monkeypatch.setattr(endpoints_module, "_cache", {})
    price = _price("openrouter/../admin")
    assert price.kinds == {"structured": "price_unavailable", "tools": "price_unavailable"}
    assert fetched == []


def test_the_endpoints_are_read_off_the_event_loop(listed, monkeypatch):
    import threading

    readers = []

    def endpoints(slug):
        readers.append(threading.get_ident())
        return listed["endpoints"]

    monkeypatch.setattr(prices_module, "openrouter_endpoints", endpoints)

    async def main():
        return threading.get_ident(), await routed_price(LLAMA)

    loop_thread, _ = asyncio.run(main())
    assert readers and readers[0] != loop_thread


def test_another_providers_id_has_no_routed_price(listed):
    assert _price("openai/gpt-4o-mini") is None
    assert listed["asked"] == []


def test_every_reason_is_one_of_the_codes(listed):
    codes = set(get_args(Reason)) | {None}
    cases = [
        ([_endpoint("alpha", 1e-7, 2e-7, BOTH)], {"reasoning_effort": "high", "supports_reasoning": True}),
        ([_endpoint("alpha", 1e-7, 2e-7, ())], {"temperature": 0.1, "max_tokens": 10, "top_p": 0.1}),
        (None, None),
        (copy.deepcopy(ENDPOINTS), None),
    ]
    for endpoints, params in cases:
        listed["endpoints"] = endpoints
        assert set(_price(params=params).kinds.values()) <= codes
