# License Apache 2.0: (c) 2026 Atena-Reply

"""How the engine builds what scores a run: the metrics' default direction, the deterministic rewards' scope to the answer, and the models a judge calls."""

import asyncio

from synalinks.src.backend import JsonDataModel

from openarena.config import Config, MetricEntry
from openarena.evaluate import _instantiate_metrics, _scope_to_answer
from openarena.rewards import get


def _row(json):
    return JsonDataModel(json=json, schema={"type": "object", "properties": {k: {} for k in json}})


def _score(reward_name, target, gold, pred):
    reward = get(_scope_to_answer({"name": reward_name}, target))
    return float(asyncio.run(reward(_row(gold), _row(pred))))


# -- direction ----------------------------------------------------------------


def test_a_measurement_without_direction_is_better_lower():
    for name in ("cost", "reward_cost", "embedding_cost", "input_tokens", "output_tokens", "total_tokens", "avg_latency", "error_rate", "AvgLatency"):
        assert MetricEntry.model_validate(name).direction == "min", name
        assert MetricEntry.model_validate({"class": name}).direction == "min", name


def test_a_score_or_a_rate_without_direction_is_better_higher():
    for name in ("exact_match", "answer_accuracy", "throughput", "tokens_per_second", "cache_hit_rate"):
        assert MetricEntry.model_validate(name).direction == "max", name


def test_a_written_direction_wins():
    assert MetricEntry.model_validate({"class": "cost", "direction": "max"}).direction == "max"
    assert MetricEntry.model_validate({"class": "exact_match", "direction": "min"}).direction == "min"


# -- a deterministic reward scoped to the answer ------------------------------

GOLD_CALL = {"tool_calls": [{"function": {"name": "add", "arguments": {"a": 1, "b": 2}}}]}


def test_a_wrong_tool_call_scores_zero():
    pred = {"tool_calls": [{"function": {"name": "add", "arguments": '{"a": 1, "b": 3}'}}], "content": ""}
    assert _score("tool_calls_accuracy", ["tool_calls"], GOLD_CALL, pred) == 0.0


def test_the_right_tool_call_scores_one():
    pred = {"tool_calls": [{"id": "c1", "function": {"name": "add", "arguments": '{"b": 2, "a": 1}'}}], "content": "sure"}
    assert _score("tool_calls_accuracy", ["tool_calls"], GOLD_CALL, pred) == 1.0


def test_a_nested_object_answer_is_compared():
    gold = {"answer": {"city": "Paris", "country": "France"}}
    right = {"answer": {"city": "Paris", "country": "France"}, "thinking": "the capital"}
    wrong = {"answer": {"city": "Lyon", "country": "France"}, "thinking": "the capital"}
    assert _score("exact_match", ["answer"], gold, right) == 1.0
    assert _score("exact_match", ["answer"], gold, wrong) == 0.0
    assert _score("answer_accuracy", ["answer"], gold, wrong) < _score("answer_accuracy", ["answer"], gold, right)


def test_what_the_program_returns_beside_the_answer_stays_out():
    assert _score("exact_match", ["answer"], {"answer": "4"}, {"answer": "4", "question": "2+2?"}) == 1.0


def test_a_mask_the_user_set_and_a_judge_are_left_as_written():
    assert _scope_to_answer({"name": "exact_match", "out_mask": ["id"]}, ["answer"]) == {"name": "exact_match", "out_mask": ["id"]}
    assert _scope_to_answer({"name": "lm_as_judge"}, ["answer"]) == {"name": "lm_as_judge"}


def test_a_reward_metric_is_scoped_too():
    [(alias, metric, _, _)] = _instantiate_metrics([MetricEntry.model_validate("tool_calls_accuracy")], context="metrics", target=["tool_calls"])
    assert alias == "tool_calls_accuracy"
    assert metric._fn.fields == ["tool_calls"]


# -- the models a judge calls --------------------------------------------------


def _config(**models):
    return Config.model_validate(
        {
            "evaluation": {"language_models": ["under_test"], "datasets": ["ds"], "default_language_model": "judge"},
            "models": {"under_test": {"model": "openai/gpt-4o-mini"}, **models},
            "rewards": {"graded": {"type": "lm_as_judge", "config": {"language_model": "judge"}}},
            "datasets": {
                "ds": {
                    "type": "local",
                    "config": {"path": "x.jsonl", "input_template": '{"q": {{ q | tojson }}}'},
                    "reward": "graded",
                    "metrics": [{"class": "lm_as_judge", "alias": "judged", "language_model": "judge"}],
                }
            },
        }
    )


def _judge_models(reward):
    return [m.language_model for m in reward.program._flatten_modules(include_self=True, recursive=True) if getattr(m, "language_model", None) is not None]


def test_a_judge_is_built_with_its_entry_params():
    cfg = _config(judge={"model": "openai/gpt-4o", "params": {"temperature": 0.0, "max_tokens": 256}})
    lms = _judge_models(get(cfg.reward_spec("ds")))
    assert lms and all(lm.model == "openai/gpt-4o" and lm.default_kwargs == {"temperature": 0.0, "max_tokens": 256} for lm in lms)
    [(_, metric, _, _)] = _instantiate_metrics(cfg.dataset_metrics("ds"), context="datasets.ds.metrics")
    assert all(lm.default_kwargs["temperature"] == 0.0 for lm in _judge_models(metric._fn))
    assert cfg.default_language_model.default_kwargs["temperature"] == 0.0


def test_a_judge_without_params_stays_a_bare_id():
    cfg = _config(judge={"model": "openai/gpt-4o"})
    assert cfg.reward_spec("ds")["language_model"] == "openai/gpt-4o"
    assert cfg.default_language_model == "openai/gpt-4o"


def test_an_embedding_reward_model_gets_its_dimensions():
    reward = get({"name": "cosine_similarity", "embedding_model": {"model": "openai/text-embedding-3-small", "dimensions": 256}})
    assert reward.embedding_model.default_kwargs == {"dimensions": 256}
