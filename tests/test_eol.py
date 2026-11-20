# License Apache 2.0: (c) 2026 Atena-Reply

"""End-of-life dates, and reaching them through an aggregator's id.

The sources are keyed by the vendor's own spelling, so every
`openrouter/...` id used to answer "none announced" even when the vendor had
published a date for exactly that model. These pin the unwrapping that fixed
it, without asking the network and without depending on what a release of
litellm happens to carry: litellm's table is stubbed with the dates below, the
Azure schedule falls back to the snapshot shipped beside the module, and
OpenRouter's model list is stubbed (offline, it adds nothing).
"""

import os
from datetime import date, timedelta

# litellm fetches its pricing table from GitHub at import unless told to use
# the copy it ships with; tests must not depend on the network.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import pytest

from openarena.lookups import eol as eol_module
from openarena.lookups.eol import (
    EOL_HORIZON_DAYS,
    direct_id,
    eol_days,
    model_eol,
    parse_openrouter_models,
)


# What litellm's table says, as far as these tests are concerned: a vendor's announced date for one
# model. The real table changes with every litellm release; the logic under test does not.
LITELLM_DATES = {"anthropic/claude-sonnet-5": "2027-03-01"}

# The real check, kept before the fixture below replaces it in every test.
LIVE_TABLE_FETCHED = eol_module._live_table_fetched


@pytest.fixture(autouse=True)
def litellm_dates(monkeypatch):
    """litellm's table, reduced to `LITELLM_DATES`: no other model has a provider date.

    It stands for a table litellm fetched, whatever `LITELLM_LOCAL_MODEL_COST_MAP` makes the real one.
    """
    monkeypatch.setattr(
        eol_module,
        "litellm_model_info",
        lambda model_id: {"deprecation_date": LITELLM_DATES[model_id]} if model_id in LITELLM_DATES else None,
    )
    monkeypatch.setattr(eol_module, "_live_table_fetched", lambda: True)


@pytest.fixture(autouse=True)
def openrouter_lists(monkeypatch):
    """OpenRouter's model list, as the test sets it; empty (offline) by default."""
    listed: dict = {"data": []}
    monkeypatch.setattr(eol_module, "_fetch_openrouter", lambda: listed)
    monkeypatch.setattr(eol_module, "_openrouter_cache", None)
    return listed


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        # The vendor's own spelling, which is how both sources are keyed.
        ("openrouter/anthropic/claude-sonnet-5", "anthropic/claude-sonnet-5"),
        ("openrouter/openai/gpt-4.1-mini", "openai/gpt-4.1-mini"),
        # OpenRouter says `google`; litellm's table says `gemini`.
        ("openrouter/google/gemini-3.1-flash-lite", "gemini/gemini-3.1-flash-lite"),
        ("openrouter/mistralai/mistral-large", "mistral/mistral-large"),
        # A vendor with no direct prefix of its own is left alone rather than
        # guessed at: `mancer/weaver` is not a thing any source knows.
        ("openrouter/mancer/weaver", None),
        # Not an aggregator's id at all.
        ("anthropic/claude-sonnet-5", None),
        ("ollama/llama3.2", None),
        ("openrouter/no-vendor-segment", None),
    ],
)
def test_an_aggregator_id_unwraps_to_the_vendors_own(model_id, expected):
    assert direct_id(model_id) == expected


def test_a_date_announced_for_a_model_is_found_through_openrouter_too():
    # Same model, same shutdown day, whoever is proxying it.
    direct = model_eol("anthropic/claude-sonnet-5")
    assert direct == ("2027-03-01", "provider")
    assert model_eol("openrouter/anthropic/claude-sonnet-5") == direct


def test_litellms_row_for_an_openrouter_id_does_not_date_it(monkeypatch, openrouter_lists):
    # litellm's shipped copy kept an `openrouter/google/...` row carrying Vertex's shutdown day, while
    # OpenRouter itself lists the model with no expiration and Google's own API id has no date.
    monkeypatch.setitem(LITELLM_DATES, "openrouter/google/gemini-2.5-flash-lite", "2026-10-20")
    openrouter_lists["data"] = [{"id": "google/gemini-2.5-flash-lite", "expiration_date": None}]
    assert model_eol("openrouter/google/gemini-2.5-flash-lite") is None
    assert eol_days("openrouter/google/gemini-2.5-flash-lite", date(2026, 11, 20)) == EOL_HORIZON_DAYS


def test_litellms_row_for_an_openrouter_id_answers_only_when_openrouter_cannot(monkeypatch, openrouter_lists):
    monkeypatch.setitem(LITELLM_DATES, "openrouter/mancer/weaver", "2026-01-01")
    # Offline, or no longer listed: litellm's row is all there is.
    assert model_eol("openrouter/mancer/weaver") == ("2026-01-01", "provider")
    # Listed: OpenRouter says when it stops, or that it does not.
    openrouter_lists["data"] = [{"id": "mancer/weaver", "expiration_date": "2027-06-30"}]
    eol_module._openrouter_cache = None
    assert model_eol("openrouter/mancer/weaver") == ("2027-06-30", "openrouter")
    openrouter_lists["data"] = [{"id": "mancer/weaver", "expiration_date": None}]
    eol_module._openrouter_cache = None
    assert model_eol("openrouter/mancer/weaver") is None


def test_with_both_sources_offline_the_shipped_copys_openrouter_row_dates_nothing(monkeypatch, openrouter_lists):
    # litellm could not download its table and loaded the copy, where the stale date is; OpenRouter is unreachable.
    monkeypatch.setitem(LITELLM_DATES, "openrouter/google/gemini-2.5-flash-lite", "2026-10-20")
    monkeypatch.setattr(eol_module, "_live_table_fetched", lambda: False)
    assert model_eol("openrouter/google/gemini-2.5-flash-lite") is None
    # The vendor's own id still answers from the copy.
    assert model_eol("openrouter/anthropic/claude-sonnet-5") == ("2027-03-01", "provider")


def test_openrouters_date_is_found_whatever_the_ids_punctuation(openrouter_lists):
    # Registered with a dash, listed by OpenRouter with a dot: the same model, and its date.
    openrouter_lists["data"] = [{"id": "anthropic/claude-opus-5.5", "expiration_date": "2026-12-31"}]
    assert model_eol("openrouter/anthropic/claude-opus-5-5") == ("2026-12-31", "openrouter")


def test_the_live_table_counts_as_fetched_only_when_litellm_downloaded_it(monkeypatch):
    from litellm.litellm_core_utils import get_model_cost_map as source

    monkeypatch.setattr(source, "get_model_cost_map_source_info", lambda: {"source": "local"})
    assert LIVE_TABLE_FETCHED() is False
    monkeypatch.setattr(source, "get_model_cost_map_source_info", lambda: {"source": "remote"})
    assert LIVE_TABLE_FETCHED() is True


def test_a_row_only_the_shipped_copy_has_is_filled_in_without_its_date(monkeypatch):
    import litellm

    original = litellm.model_cost
    monkeypatch.setattr(eol_module, "_filled_table", None)
    monkeypatch.setattr(eol_module, "_from_bundled", frozenset())
    monkeypatch.setattr(litellm, "model_cost", {"gemini/kept": {"input_cost_per_token": 1e-6, "deprecation_date": "2027-01-01"}})
    monkeypatch.setattr(
        eol_module,
        "bundled_table",
        lambda: {
            "gemini/kept": {"input_cost_per_token": 9e-6},
            "openrouter/google/dropped": {"input_cost_per_token": 2e-6, "deprecation_date": "2026-10-20"},
        },
    )
    try:
        assert eol_module.fill_from_bundled() == frozenset({"openrouter/google/dropped"})
        # The price is filled in, the date the live table no longer carries is not.
        assert litellm.model_cost["openrouter/google/dropped"] == {"input_cost_per_token": 2e-6}
        # A live row is left as it is, date and all.
        assert litellm.model_cost["gemini/kept"] == {"input_cost_per_token": 1e-6, "deprecation_date": "2027-01-01"}
    finally:
        eol_module.replace_table(original)


def test_azures_schedule_is_reached_through_openrouter_too():
    direct = model_eol("openai/gpt-4.1-mini")
    assert direct is not None and direct[1] == "azure"
    assert model_eol("openrouter/openai/gpt-4.1-mini") == direct


def test_a_model_nobody_has_dated_still_answers_none():
    assert model_eol("openrouter/mancer/weaver") is None
    assert model_eol("ollama/qwen3-embedding:0.6b") is None


def test_time_to_eol_is_the_horizon_when_no_date_is_known():
    # No date means no end of life, not a gap: every model carries the
    # metric, so weighing it never drops the local ones.
    today = date(2026, 9, 16)
    assert eol_days("ollama/qwen3-embedding:0.6b", today) == EOL_HORIZON_DAYS
    assert eol_days("openrouter/mancer/weaver", today) == EOL_HORIZON_DAYS


def test_time_to_eol_counts_down_to_an_announced_date_and_stops_at_the_horizon():
    eol = model_eol("openai/gpt-4.1-mini")
    assert eol is not None
    announced = date.fromisoformat(eol[0])
    assert eol_days("openai/gpt-4.1-mini", announced - timedelta(days=30)) == 30
    assert eol_days("openai/gpt-4.1-mini", announced + timedelta(days=1)) == -1
    # A date beyond the horizon reads as the horizon: the scale stops there.
    far = announced - timedelta(days=EOL_HORIZON_DAYS + 400)
    assert eol_days("openai/gpt-4.1-mini", far) == EOL_HORIZON_DAYS


def test_openrouters_list_keeps_only_the_models_it_has_dated():
    payload = {
        "data": [
            {"id": "bytedance-seed/seed-1.6", "expiration_date": "2026-11-11"},
            {
                "id": "qwen/qwen3-max-thinking",
                "expiration_date": "2026-10-09T00:00:00Z",
            },
            {"id": "anthropic/claude-opus-5.5", "expiration_date": None},
            {"id": "z-ai/glm-4.7"},
            {"id": "odd/model", "expiration_date": "soon"},
        ]
    }
    assert parse_openrouter_models(payload) == {
        "bytedance-seed/seed-1.6": "2026-11-11",
        "qwen/qwen3-max-thinking": "2026-10-09",
    }


def test_openrouter_dates_a_model_nobody_else_has(openrouter_lists):
    # A lab with no direct prefix: litellm and Azure know nothing of it.
    assert model_eol("openrouter/bytedance-seed/seed-1.6") is None
    eol_module._openrouter_cache = None
    openrouter_lists["data"] = [
        {"id": "bytedance-seed/seed-1.6", "expiration_date": "2026-11-11"}
    ]
    assert model_eol("openrouter/bytedance-seed/seed-1.6") == (
        "2026-11-11",
        "openrouter",
    )
    assert eol_days("openrouter/bytedance-seed/seed-1.6", date(2026, 10, 8)) == 34
    # Only OpenRouter's ids: the same model elsewhere is not routed by it.
    assert model_eol("together_ai/bytedance-seed/seed-1.6") is None


def test_the_earlier_of_openrouters_and_the_vendors_date_wins(openrouter_lists):
    vendor = model_eol("openrouter/openai/gpt-4.1-mini")
    assert vendor is not None
    day = date.fromisoformat(vendor[0])
    # OpenRouter stops routing it first: that is the day the id stops working.
    earlier = (day - timedelta(days=10)).isoformat()
    openrouter_lists["data"] = [
        {"id": "openai/gpt-4.1-mini", "expiration_date": earlier}
    ]
    eol_module._openrouter_cache = None
    assert model_eol("openrouter/openai/gpt-4.1-mini") == (earlier, "openrouter")
    # A later OpenRouter date changes nothing: the vendor retires it first.
    later = (day + timedelta(days=10)).isoformat()
    openrouter_lists["data"] = [{"id": "openai/gpt-4.1-mini", "expiration_date": later}]
    eol_module._openrouter_cache = None
    assert model_eol("openrouter/openai/gpt-4.1-mini") == vendor


def test_openrouter_unreachable_leaves_the_other_sources_answering(monkeypatch):
    monkeypatch.setattr(eol_module, "_fetch_openrouter", lambda: None)
    assert model_eol("openrouter/bytedance-seed/seed-1.6") is None
    assert model_eol("openrouter/openai/gpt-4.1-mini") == model_eol(
        "openai/gpt-4.1-mini"
    )
