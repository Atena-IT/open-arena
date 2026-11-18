# License Apache 2.0: (c) 2026 Atena-Reply

"""What a cell's calls cost at list price, read off the counters the framework keeps on each model object."""

import os
from types import SimpleNamespace

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import pytest
import synalinks

from openarena.list_cost import list_usage, price_as_asked, snapshot

SONNET = "anthropic/claude-sonnet-5"
JUDGE = "openai/gpt-4o"
EMBEDDER = "openai/text-embedding-3-small"


def test_each_models_spend_since_the_baseline_is_divided_by_its_own_factor():
    model, judge, embedder = synalinks.LanguageModel(SONNET), synalinks.LanguageModel(JUDGE), synalinks.EmbeddingModel(model=EMBEDDER)
    # Spent before the cell started: not the cell's.
    model.inference_cumulated_cost = 1.0
    models = [model, judge, embedder]
    baseline = snapshot(models)
    model.inference_cumulated_cost += 0.5
    judge.reward_cumulated_cost += 0.3
    embedder.inference_cumulated_cost += 0.1
    embedder.reward_cumulated_cost += 0.06
    assert list_usage(models, baseline, {SONNET: 1.25, JUDGE: 1.5}) == {
        SONNET: {"cost": pytest.approx(0.4)},
        JUDGE: {"reward_cost": pytest.approx(0.2)},
        # Given no factor: its calls were charged at list price.
        EMBEDDER: {"embedding_cost": pytest.approx(0.1), "reward_embedding_cost": pytest.approx(0.06)},
    }


def test_two_objects_of_one_model_add_up():
    first, second = synalinks.LanguageModel(SONNET), synalinks.LanguageModel(SONNET)
    baseline = snapshot([first, second])
    first.inference_cumulated_cost += 0.25
    second.inference_cumulated_cost += 0.25
    second.reward_cumulated_cost += 0.5
    assert list_usage([first, second], baseline, {SONNET: 1.25}) == {
        SONNET: {"cost": pytest.approx(0.4), "reward_cost": pytest.approx(0.4)}
    }


def test_a_model_that_spent_nothing_is_left_out():
    model, judge = synalinks.LanguageModel(SONNET), synalinks.LanguageModel(JUDGE)
    baseline = snapshot([model, judge])
    judge.reward_cumulated_cost += 0.3
    assert list_usage([model, judge], baseline, {SONNET: 1.25}) == {JUDGE: {"reward_cost": pytest.approx(0.3)}}
    assert list_usage([model], snapshot([model]), {SONNET: 1.25}) == {}


def test_two_judges_given_different_factors_are_told_apart_each_divided_by_its_own():
    first, second = synalinks.LanguageModel(SONNET), synalinks.LanguageModel(JUDGE)
    baseline = snapshot([first, second])
    first.reward_cumulated_cost += 0.5
    second.reward_cumulated_cost += 0.3
    assert list_usage([first, second], baseline, {SONNET: 1.25, JUDGE: 1.5}) == {
        SONNET: {"reward_cost": pytest.approx(0.4)},
        JUDGE: {"reward_cost": pytest.approx(0.2)},
    }


def test_a_model_whose_calls_take_no_parameter_cannot_be_held_to_its_factor():
    # Left alone, an answer naming another model would have its call charged at that model's list price.
    grader = SimpleNamespace(model="sglang/a-grader")
    with pytest.raises(ValueError, match="sglang/a-grader"):
        price_as_asked([grader], {"sglang/a-grader": 1.25})
    # A model given no factor is left as it is.
    price_as_asked([grader], {SONNET: 1.25})
