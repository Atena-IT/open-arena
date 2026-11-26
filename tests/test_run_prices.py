# License Apache 2.0: (c) 2026 Atena-Reply

"""What each model paid per token in a run, estimated from its billed calls. No network: litellm answers with its mock."""

import asyncio
import os

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import litellm
import numpy as np
import pytest
import synalinks
from synalinks.src.modules.language_models.language_model import _CURRENT_CALL_USAGE

from openarena import run_prices
from openarena.lookups.prices import Price
from openarena.run_prices import (
    MAX_COSINE,
    MIN_CALLS,
    MIN_R2,
    BilledCall,
    RunPrices,
    estimate_prices,
    judge_keys,
)


def _calls(n, price_in, price_out, kind="structured", seed=0):
    rng = np.random.default_rng(seed)
    ins = rng.integers(200, 4000, n)
    outs = rng.integers(10, 800, n)
    return [BilledCall(int(i), int(o), float(i * price_in + o * price_out), kind) for i, o in zip(ins, outs)]


# -- the estimate --------------------------------------------------------------


def test_one_provider_is_found_exactly():
    a, b = estimate_prices(_calls(35, 2.2e-7, 2.2e-7))
    assert a == pytest.approx(2.2e-7, rel=1e-9) and b == pytest.approx(2.2e-7, rel=1e-9)


def test_two_providers_mixed_give_a_price_between_theirs():
    calls = _calls(20, 1e-7, 2e-7, seed=1) + _calls(20, 1.2e-7, 2.4e-7, seed=2)
    a, b = estimate_prices(calls)
    assert 1e-7 <= a <= 1.2e-7 and 2e-7 <= b <= 2.4e-7


def test_proportional_calls_give_none():
    # Every answer as long as its question: input and output cannot be told apart.
    calls = [BilledCall(n, 2 * n, n * 3e-7, "structured") for n in range(100, 100 + 10 * MIN_CALLS, 10)]
    assert estimate_prices(calls) is None


def test_too_few_calls_give_none():
    assert estimate_prices(_calls(MIN_CALLS - 1, 2.2e-7, 2.2e-7)) is None
    assert estimate_prices([]) is None


def test_a_negative_fit_gives_none():
    # Costs that fall as the answers grow: no price per output token can be negative.
    calls = [BilledCall(1000, o, 1e-4 - o * 1e-7, "structured") for o in range(10, 10 + 20 * MIN_CALLS, 20)]
    assert estimate_prices(calls) is None


def test_a_poor_fit_gives_none():
    rng = np.random.default_rng(3)
    calls = [BilledCall(c.input_tokens, c.output_tokens, float(rng.uniform(0, 1e-3)), c.kind) for c in _calls(40, 1e-7, 2e-7)]
    assert estimate_prices(calls) is None


def test_the_thresholds_are_the_spec_values():
    assert (MIN_CALLS, MAX_COSINE, MIN_R2) == (10, 0.99, 0.85)


# -- what the run records ------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_usage():
    token = _CURRENT_CALL_USAGE.set(None)
    yield
    _CURRENT_CALL_USAGE.reset(token)


def _usage(**usage):
    _CURRENT_CALL_USAGE.set(usage or None)


def _hook_of(model):
    return model._hooks.hooks[-1]


def _end(model, **kwargs):
    """Each of the model's hooks told its call ended, as `Module.__call__` tells them."""
    for hook in model._hooks.hooks:
        hook.on_call_end("1", outputs=kwargs.get("outputs", object()), exception=kwargs.get("exception"))


def test_calls_are_kept_by_entry_not_by_model_id():
    prices = RunPrices({})
    fast = synalinks.LanguageModel(model="openrouter/acme/m", max_tokens=100)
    slow = synalinks.LanguageModel(model="openrouter/acme/m")
    with prices.recording():
        prices.watch(fast, "m-short", "structured")
        prices.watch(slow, "m", "tools")
        _usage(input_tokens=100, output_tokens=10, cost=1e-5)
        _hook_of(fast).on_call_end("1", outputs=object())
        _usage(input_tokens=200, output_tokens=20, cost=2e-5)
        _hook_of(slow).on_call_end("2", outputs=object())
    assert prices.calls == {
        "m-short": [BilledCall(100, 10, 1e-5, "structured")],
        "m": [BilledCall(200, 20, 2e-5, "tools")],
    }


def test_cache_hits_failures_and_free_calls_are_not_counted():
    prices = RunPrices({})
    model = synalinks.LanguageModel(model="openrouter/acme/m")
    with prices.recording():
        prices.watch(model, "m", "structured")
        hook = _hook_of(model)
        for usage in (
            {"cache_hit": True},
            {"error": "boom"},
            {"error": "boom", "fallback": True},
            {"input_tokens": 100, "output_tokens": 10, "cost": None},
            {"input_tokens": 100, "output_tokens": 10, "cost": 0.0},
            {"input_tokens": 100, "output_tokens": 10, "cost": float("nan")},
            {"input_tokens": 100, "output_tokens": 10, "cost": float("inf")},
            {"input_tokens": 0, "output_tokens": 0, "cost": 1e-5},
            {},
        ):
            _usage(**usage)
            hook.on_call_end("1", outputs=object())
        _usage(input_tokens=100, output_tokens=10, cost=1e-5)
        hook.on_call_end("2", outputs=None, exception="the call raised")
    assert prices.calls == {}


def test_a_call_outside_any_recording_run_is_not_counted():
    prices = RunPrices({})
    model = synalinks.LanguageModel(model="openrouter/acme/m")
    prices.watch(model, "m", "structured")
    _usage(input_tokens=100, output_tokens=10, cost=1e-5)
    _end(model)
    assert prices.calls == {}


def test_a_model_shared_by_two_runs_counts_for_the_run_that_calls_it_and_keeps_no_hook_after():
    # A process-wide judge model (synalinks' default) is seen by every run of the process.
    model = synalinks.LanguageModel(model="openrouter/acme/judge")
    first, second = RunPrices({}), RunPrices({})
    second.watch(model, "judge", "structured")
    with first.recording():
        first.watch(model, "judge", "structured")
        _usage(input_tokens=100, output_tokens=10, cost=1e-5)
        _end(model)
    assert first.calls == {"judge": [BilledCall(100, 10, 1e-5, "structured")]}
    assert second.calls == {}
    # The run that ended took its hook away; the other run's is left.
    assert [h._prices for h in model._hooks.hooks if type(h).__name__ == "_BilledHook"] == [second]


def test_a_model_watched_twice_records_each_call_once():
    prices = RunPrices({})
    model = synalinks.LanguageModel(model="openrouter/acme/m")
    prices.watch(model, "m", "structured")
    prices.watch(model, "m", "structured")
    assert sum(type(h).__name__ == "_BilledHook" for h in model._hooks.hooks) == 1


def test_both_kinds_in_one_run_are_listed():
    prices = RunPrices({})
    for call in _calls(12, 2.2e-7, 2.2e-7, kind="structured") + _calls(12, 2.2e-7, 2.2e-7, kind="tools", seed=5):
        prices.record("llama", call)
    estimate = prices.estimates()["llama"]
    assert estimate["kinds"] == ["structured", "tools"]
    assert estimate["input"] == pytest.approx(2.2e-7) and estimate["output"] == pytest.approx(2.2e-7)


def test_a_cost_that_is_not_a_number_never_fails_the_run():
    prices = RunPrices({})
    for call in _calls(12, 2.2e-7, 2.2e-7):
        prices.record("odd", BilledCall(call.input_tokens, call.output_tokens, float("nan"), call.kind))
    assert estimate_prices(prices.calls["odd"]) is None
    assert prices.estimates() == {}


def _total(calls):
    return sum(c.cost for c in calls), sum(c.input_tokens for c in calls), sum(c.output_tokens for c in calls)


def _ratio(monkeypatch, ratios):
    """`host_ratios` as the test sets it, by model id: what each host the calls could reach charges out over in."""
    monkeypatch.setattr(
        run_prices, "host_ratios", lambda model, kinds: list(ratios.get(getattr(model, "model", model), []))
    )


def test_a_fit_that_holds_gives_two_prices_not_approximate():
    prices = RunPrices({})
    for call in _calls(12, 2.2e-7, 4.4e-7):
        prices.record("llama", call)
    estimate = prices.estimates()["llama"]
    assert estimate == {"input": pytest.approx(2.2e-7), "output": pytest.approx(4.4e-7), "kinds": ["structured"], "approx": False}


def test_hosts_at_different_prices_still_give_two_prices():
    # Each call billed by one of two hosts, at its own prices: no fit to trust, yet one that weighs them.
    cheap, dear = _calls(20, 1.7e-8, 8.5e-8), _calls(20, 1.5e-7, 9.5e-7, seed=7)
    calls = [c for pair in zip(cheap, dear) for c in pair]
    assert estimate_prices(calls) is None
    prices = RunPrices({})
    for call in calls:
        prices.record("gptoss-high", call)
    estimate = prices.estimates()["gptoss-high"]
    assert estimate["approx"] is True and estimate["kinds"] == ["structured"]
    assert 1.7e-8 <= estimate["input"] <= 1.5e-7 and 8.5e-8 <= estimate["output"] <= 9.5e-7
    # The two prices give back what the run paid, within the fit's spread.
    cost, tokens_in, tokens_out = _total(calls)
    assert estimate["input"] * tokens_in + estimate["output"] * tokens_out == pytest.approx(cost, rel=0.1)


def test_a_few_calls_of_one_host_give_its_prices():
    prices = RunPrices({})
    for call in _calls(MIN_CALLS - 1, 2.2e-7, 4.4e-7):
        prices.record("few", call)
    assert prices.estimates()["few"] == {
        "input": pytest.approx(2.2e-7),
        "output": pytest.approx(4.4e-7),
        "kinds": ["structured"],
        "approx": True,
    }


def test_calls_that_cannot_be_told_apart_are_split_by_the_hosts_ratio(monkeypatch):
    # One call, or input and output always in a fixed ratio: the run's cost is split as the model's hosts price.
    _ratio(monkeypatch, {"openrouter/acme/m": [4.0, 5.0, 6.0]})
    one = [BilledCall(1000, 200, 3e-5, "tools")]
    proportional = [BilledCall(n, 2 * n, n * 3e-7, "structured") for n in range(100, 100 + 10 * MIN_CALLS, 10)]
    prices = RunPrices({})
    for name, calls in (("one", one), ("proportional", proportional)):
        prices.models[name] = "openrouter/acme/m"
        for call in calls:
            prices.record(name, call)
    paid = prices.estimates()
    for name, calls in (("one", one), ("proportional", proportional)):
        cost, tokens_in, tokens_out = _total(calls)
        assert paid[name]["approx"] is True
        assert paid[name]["output"] == pytest.approx(5.0 * paid[name]["input"])
        assert paid[name]["input"] * tokens_in + paid[name]["output"] * tokens_out == pytest.approx(cost)


def test_without_a_ratio_the_cost_is_split_evenly(monkeypatch):
    _ratio(monkeypatch, {})
    prices = RunPrices({})
    prices.models["one"] = "acme/unknown"
    prices.record("one", BilledCall(1000, 200, 3e-5, "structured"))
    paid = prices.estimates()["one"]
    assert paid["input"] == paid["output"] == pytest.approx(3e-5 / 1200) and paid["approx"] is True


def test_what_is_not_a_cost_is_left_out_and_an_entry_with_none_is_absent(monkeypatch):
    _ratio(monkeypatch, {})
    good = BilledCall(1000, 200, 3e-5, "structured")
    odd = BilledCall(1000, 200, float("nan"), "structured")
    prices = RunPrices({})
    prices.record("mixed", good)
    prices.record("mixed", odd)
    prices.record("only-odd", odd)
    paid = prices.estimates()
    assert set(paid) == {"mixed"} and paid["mixed"]["input"] == pytest.approx(3e-5 / 1200)


def test_a_fit_far_from_the_hosts_ratio_gives_way_to_the_split(monkeypatch):
    # Two calls from two hosts: the exact solve puts almost the whole cost on output, input near nothing.
    _ratio(monkeypatch, {"openrouter/acme/m": [5.0, 6.4]})
    calls = [BilledCall(1000, 200, 1000 * 1e-10 + 200 * 1.5e-6, "structured"), BilledCall(2000, 100, 2000 * 1e-10 + 100 * 1.5e-6, "structured")]
    assert run_prices.loose_prices(calls) == (pytest.approx(1e-10), pytest.approx(1.5e-6))
    prices = RunPrices({})
    prices.models["pair"] = "openrouter/acme/m"
    for call in calls:
        prices.record("pair", call)
    paid = prices.estimates()["pair"]
    cost, tokens_in, tokens_out = _total(calls)
    assert paid["approx"] is True
    assert paid["output"] == pytest.approx(5.7 * paid["input"])  # the hosts' median ratio
    assert paid["input"] * tokens_in + paid["output"] * tokens_out == pytest.approx(cost)


def test_a_fit_within_the_hosts_ratio_is_kept(monkeypatch):
    _ratio(monkeypatch, {"openrouter/acme/m": [1.5, 2.5]})
    prices = RunPrices({})
    prices.models["few"] = "openrouter/acme/m"
    for call in _calls(MIN_CALLS - 1, 2.2e-7, 4.4e-7):
        prices.record("few", call)
    paid = prices.estimates()["few"]
    assert (paid["input"], paid["output"]) == (pytest.approx(2.2e-7), pytest.approx(4.4e-7))


def test_a_watched_model_is_known_by_its_id_for_its_entry():
    prices = RunPrices({})
    prices.watch(synalinks.LanguageModel(model="openrouter/acme/m"), "m-entry", "structured")
    assert prices.models == {"m-entry": "openrouter/acme/m"}


ENDPOINTS = [
    # Takes structured output, one price for in and out.
    {"tag": "structured-host", "pricing": {"prompt": "2.2e-7", "completion": "2.2e-7"},
     "supported_parameters": ["response_format", "structured_outputs", "tools", "tool_choice", "max_tokens"]},
    # No structured output: a structured call never reaches it.
    {"tag": "cheap", "pricing": {"prompt": "5e-8", "completion": "8e-8"}, "supported_parameters": ["tools", "tool_choice", "max_tokens"]},
    {"tag": "per-request", "pricing": {"prompt": "-1", "completion": "-1"}, "supported_parameters": ["response_format", "structured_outputs"]},
]


def test_the_ratios_are_of_the_hosts_the_calls_could_reach(monkeypatch):
    monkeypatch.setattr(run_prices, "openrouter_endpoints", lambda slug: ENDPOINTS if slug == "acme/m" else None)
    model = synalinks.LanguageModel(model="openrouter/acme/m")
    # A structured call reaches the one host that takes structured output; a tools call reaches both.
    assert run_prices.host_ratios(model, ["structured"]) == [pytest.approx(1.0)]
    assert run_prices.host_ratios(model, ["tools"]) == [pytest.approx(1.0), pytest.approx(1.6)]
    # Known by its id only: every host with a price per token.
    assert run_prices.host_ratios("openrouter/acme/m", []) == [pytest.approx(1.0), pytest.approx(1.6)]


def test_the_ratio_of_another_provider_is_the_tables(monkeypatch):
    monkeypatch.setattr(run_prices, "model_price", lambda model_id: Price(2.5e-6, 1e-5, "litellm") if model_id == "openai/gpt-4o" else None)
    assert run_prices.host_ratios("openai/gpt-4o", ["structured"]) == [pytest.approx(4.0)]
    assert run_prices.host_ratios("acme/unknown", ["structured"]) == []


# -- judges --------------------------------------------------------------------


def test_judge_keys_name_each_registry_entry_and_leave_out_the_ambiguous_ones():
    keys = judge_keys(
        {
            "judge": "openai/gpt-4o-mini",
            "twin-a": {"model": "openrouter/acme/m", "temperature": 0.0},
            "twin-b": {"model": "openrouter/acme/m", "temperature": 0.0},
            "other": {"model": "openrouter/acme/m", "temperature": 0.5},
        }
    )
    assert sorted(keys.values()) == ["judge", "other"]


def test_two_entries_told_apart_only_by_what_the_model_is_declared_to_take_are_both_judges():
    # The same id and params; one declared as reasoning (a constructor argument, not a default kwarg),
    # so it sends the reasoning level and the other does not: two different requests.
    keys = judge_keys(
        {
            "plain": {"model": "openrouter/acme/m", "reasoning_effort": "high"},
            "thinker": {"model": "openrouter/acme/m", "reasoning_effort": "high", "supports_reasoning": True},
            "seer": {"model": "openrouter/acme/m", "reasoning_effort": "high", "supports_vision": True},
        }
    )
    assert sorted(keys.values()) == ["plain", "seer", "thinker"]


def _judge_program(lm):
    async def build():
        x0 = synalinks.Input(data_model=Query)
        x1 = await synalinks.Generator(data_model=Answer, language_model=lm)(x0)
        return synalinks.Program(inputs=x0, outputs=x1, name="judge")

    return asyncio.run(build())


class Query(synalinks.DataModel):
    question: str


class Answer(synalinks.DataModel):
    answer: str


def test_a_judge_counts_for_its_own_entry():
    from openarena.programs.utils import language_model

    prices = RunPrices(judge_keys({"judge": {"model": "openrouter/acme/judge", "temperature": 0.0}}))
    lm = language_model("openrouter/acme/judge", {"temperature": 0.0})
    with prices.recording():
        prices.watch_judge(_judge_program(lm))
        _usage(input_tokens=100, output_tokens=10, cost=1e-5)
        _hook_of(lm).on_call_end("1", outputs=object())
    assert prices.calls == {"judge": [BilledCall(100, 10, 1e-5, "structured")]}


@pytest.mark.parametrize("named", ["openai/gpt-4o-mini", {"model": "openrouter/acme/judge"}])
def test_a_judge_without_params_counts_for_its_entry(named):
    from openarena.rewards import build_models

    # Built the way a reward builds the judge its config names (`Config._id`, `rewards.build_models`).
    lm = build_models({"language_model": named})["language_model"]
    if isinstance(lm, str):
        lm = synalinks.LanguageModel(model=lm)
    prices = RunPrices(judge_keys({"judge": named}))
    with prices.recording():
        prices.watch_judge(_judge_program(lm))
        _usage(input_tokens=100, output_tokens=10, cost=1e-5)
        _end(lm)
    assert prices.calls == {"judge": [BilledCall(100, 10, 1e-5, "structured")]}


def test_a_judge_outside_the_registry_is_not_recorded():
    prices = RunPrices(judge_keys({"judge": {"model": "openrouter/acme/judge"}}))
    lm = synalinks.LanguageModel(model="openai/gpt-4o-mini")
    prices.watch_judge(_judge_program(lm))
    assert all(type(h).__name__ != "_BilledHook" for h in lm._hooks.hooks)
    assert prices.calls == {}


# -- a real call ---------------------------------------------------------------


class MockedLanguageModel(synalinks.LanguageModel):
    """`openai/gpt-4o-mini` answered by litellm's mock: no provider, but litellm's usage and price."""

    async def _acompletion(self, formatted_messages, **kwargs):
        kwargs.pop("timeout", None)
        return await litellm.acompletion(
            model=self.model, messages=formatted_messages, mock_response='{"answer": "4"}', **kwargs
        )


def test_a_billed_call_of_the_model_under_test_is_recorded():
    lm = MockedLanguageModel(model="openai/gpt-4o-mini")
    program = _judge_program(lm)
    prices = RunPrices({})

    async def run():
        with prices.recording():
            prices.watch_program(program, "mini", "structured")
            await program(Query(question="2 + 2?"))

    asyncio.run(run())
    (call,) = prices.calls["mini"]
    assert call.kind == "structured" and call.input_tokens > 0 and call.cost > 0


def test_a_cells_hooks_come_off_when_its_evaluation_ends():
    lm = MockedLanguageModel(model="openai/gpt-4o-mini")
    program = _judge_program(lm)
    prices = RunPrices({})
    watch = prices.callback("mini", "model")
    watch.set_program(program)

    async def cell():
        with prices.recording():
            watch.on_test_begin()
            await program(Query(question="2 + 2?"))
            watch.on_test_end()
            # Nothing of the finished cell is kept: its models can go with its program.
            assert prices._hooked == []

    asyncio.run(cell())
    assert len(prices.calls["mini"]) == 1
    assert all(type(h).__name__ != "_BilledHook" for h in lm._hooks.hooks)
