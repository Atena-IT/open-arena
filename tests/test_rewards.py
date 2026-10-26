# License Apache 2.0: (c) 2026 Atena-Reply

"""The reward catalog and the engine's own rewards.

What `rewards.get` builds from a spec (a name, a mapping, models with params,
the answer's fields), what the catalog says of each reward (the kwargs an
entry may set, its category, the model it calls), and the deterministic
rewards scored on data models: `tool_calls_accuracy`, `retrieval_recall`,
and the judge panel's vote with its judges faked.
"""

import asyncio
import os

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import pytest
import synalinks
from synalinks.src.backend import JsonDataModel

from openarena.rewards import (
    _REWARD_TYPES,
    ANSWER_FIELDS,
    HIDDEN_KWARGS,
    MASK_KWARGS,
    SELF_SCORED,
    AnswerFields,
    accepted_kwargs,
    build_models,
    category,
    get,
    needs,
)
from openarena.rewards.multi_judge_panel import MultiJudgePanel, MultiJudgePanelProgram
from openarena.rewards.retrieval_recall import RetrievalRecall, recall
from openarena.rewards.tool_calls_accuracy import (
    ToolCallsAccuracy,
    _normalize,
    calls_of,
    score,
)


def _dm(value: dict) -> JsonDataModel:
    return JsonDataModel(json=value, schema={"type": "object", "properties": {k: {} for k in value}})


def _run(reward, gold, pred):
    return float(asyncio.run(reward(_dm(gold), _dm(pred))))


# -- the catalog ---------------------------------------------------------------------


def test_the_catalog_holds_the_engines_rewards_beside_the_frameworks_under_snake_names():
    assert {"exact_match", "lm_as_judge", "cosine_similarity", "rubrics_as_judge"} <= set(_REWARD_TYPES)
    assert _REWARD_TYPES["tool_calls_accuracy"] is ToolCallsAccuracy and _REWARD_TYPES["retrieval_recall"] is RetrievalRecall
    assert _REWARD_TYPES["multi_judge_panel"] is MultiJudgePanel
    assert {"answer_accuracy", "harbor_verifier"} <= set(_REWARD_TYPES)
    # A base class, or a judge that needs a program built in code, is not something YAML can name.
    assert "reward" not in _REWARD_TYPES and "program_as_judge" not in _REWARD_TYPES
    assert SELF_SCORED == ("harbor_verifier",)


def test_each_reward_is_placed_by_how_it_scores_and_what_it_calls():
    assert category("exact_match") == "deterministic" and category("cosine_similarity") == "deterministic"
    assert category("tool_calls_accuracy") == "deterministic" and category("answer_accuracy") == "deterministic"
    assert category("lm_as_judge") == "lm_judge" and category("multi_judge_panel") == "lm_judge"
    assert category("rubrics_as_judge") == "rubrics_judge" and category("agent_as_judge") == "agent_judge"
    assert needs(synalinks.rewards.LMAsJudge) == "language_model" and needs(MultiJudgePanel) == "language_model"
    assert needs(synalinks.rewards.CosineSimilarity) == "embedding_model" and needs(ToolCallsAccuracy) is None
    with pytest.raises(KeyError):
        category("nope")


def test_an_entry_may_set_a_rewards_named_kwargs_less_the_hidden_ones_plus_the_masks():
    judge = accepted_kwargs("lm_as_judge")
    assert {"language_model", "instructions"} <= judge
    assert not (judge & HIDDEN_KWARGS - set(MASK_KWARGS))  # the prompt engineering and the runtime stay hidden
    assert set(MASK_KWARGS) <= judge and set(MASK_KWARGS) <= accepted_kwargs("exact_match")
    assert "name" not in accepted_kwargs("exact_match") and "reduction" in accepted_kwargs("exact_match")


def test_get_builds_a_reward_from_a_name_or_a_mapping_and_names_the_known_ones_on_a_miss():
    assert isinstance(get("exact_match"), synalinks.rewards.ExactMatch)
    judge = get({"name": "lm_as_judge", "language_model": "openai/gpt-4o", "instructions": "Grade it."})
    assert isinstance(judge, synalinks.rewards.LMAsJudge)
    with pytest.raises(KeyError, match="Unknown reward 'exact_mach'. Known: .*exact_match"):
        get("exact_mach")
    with pytest.raises(KeyError, match="Known: "):
        get({"name": "nope"})
    assert get("tool_calls_accuracy", name="calls").name == "calls"  # kwargs on top of the spec's


def test_a_model_given_with_params_is_built_with_them_and_a_bare_id_left_to_the_framework():
    built = build_models({
        "language_model": {"model": "openai/gpt-4o", "temperature": 0.0},
        "embedding_model": {"model": "openai/text-embedding-3-small", "dimensions": 64},
        "panel_language_models": ["ollama/a", {"model": "ollama/b", "max_tokens": 8}],
        "smart_language_model": "openai/gpt-4o",
        "instructions": "x",
    })
    assert isinstance(built["language_model"], synalinks.LanguageModel) and built["language_model"].default_kwargs == {"temperature": 0.0}
    assert isinstance(built["embedding_model"], synalinks.EmbeddingModel) and built["embedding_model"].default_kwargs == {"dimensions": 64}
    assert built["panel_language_models"][0] == "ollama/a" and built["panel_language_models"][1].default_kwargs == {"max_tokens": 8}
    assert built["smart_language_model"] == "openai/gpt-4o" and built["instructions"] == "x"
    routed = build_models({"language_model": {"model": "openrouter/x/y"}})["language_model"]
    assert routed.default_kwargs == {"extra_body": {"provider": {"require_parameters": True}}}


def test_answer_fields_scope_a_reward_to_the_named_top_level_fields_each_kept_whole():
    scoped = get({"name": "exact_match", ANSWER_FIELDS: ["answer"]})
    assert isinstance(scoped, AnswerFields) and scoped.fields == ["answer"] and scoped.name == "exact_match"
    assert isinstance(scoped.reward, synalinks.rewards.ExactMatch) and scoped.rewards == [scoped.reward]
    gold = {"answer": {"city": "Paris", "calls": [{"id": 1}]}}
    assert _run(scoped, gold, {**gold, "thinking": "the capital", "question": "?"}) == 1.0
    assert _run(scoped, gold, {"answer": {"city": "Lyon", "calls": [{"id": 1}]}}) == 0.0
    assert asyncio.run(scoped(_dm(gold), None)) == 0.0  # no prediction
    config = scoped.get_config()
    assert config["fields"] == ["answer"] and AnswerFields.from_config(config).fields == ["answer"]
    assert not isinstance(get({"name": "exact_match", ANSWER_FIELDS: []}), AnswerFields)  # nothing to scope to


# -- tool_calls_accuracy ---------------------------------------------------------------


def test_a_call_in_either_shape_normalizes_to_its_name_and_canonical_arguments():
    wire = {"id": "c1", "type": "function", "function": {"name": "add", "arguments": '{"b": 2, "a": 1}'}}
    short = {"name": "add", "arguments": {"a": 1, "b": 2}}
    assert _normalize(wire) == _normalize(short) == ("add", '{"a": 1, "b": 2}')
    assert _normalize({"name": "noop"}) == ("noop", "{}") and _normalize({"name": "noop", "arguments": "  "}) == ("noop", "{}")
    assert _normalize({"name": "f", "arguments": "not json"}) == ("f", "not json")  # kept as text: it will not match a dict
    assert _normalize("add") is None and _normalize({"function": {"arguments": {}}}) is None and _normalize({"name": ""}) is None
    assert calls_of({"tool_calls": [wire, "junk", short]}) == [("add", '{"a": 1, "b": 2}')] * 2
    assert calls_of([short]) == [("add", '{"a": 1, "b": 2}')] and calls_of({"content": "hi"}) == [] and calls_of(None) == []


def test_the_score_is_the_matched_calls_over_the_longer_side():
    add, mul = ("add", '{"a": 1}'), ("mul", '{"a": 1}')
    assert score([], []) == 1.0 and score([add], []) == 0.0 and score([], [add]) == 0.0
    assert score([add], [add]) == 1.0 and score([add], [mul]) == 0.0
    assert score([add, mul], [add]) == 0.5 and score([add], [add, mul]) == 0.5  # a missed call, an extra one
    assert score([add, add], [add]) == 0.5 and score([add], [add, add]) == 0.5  # each gold call matched once
    reward = ToolCallsAccuracy()
    message = {"role": "assistant", "content": None, "tool_calls": [{"id": "c", "type": "function", "function": {"name": "add", "arguments": '{"a": 1}'}}]}
    assert _run(reward, {"tool_calls": [{"name": "add", "arguments": {"a": 1}}]}, message) == 1.0
    assert _run(reward, {"tool_calls": []}, message) == 0.0 and _run(reward, {"tool_calls": []}, {"content": "no call"}) == 1.0
    assert asyncio.run(reward(_dm({}), None)) == 0.0
    assert ToolCallsAccuracy.from_config(ToolCallsAccuracy(out_mask=["id"]).get_config()).get_config() == {"name": "tool_calls_accuracy", "in_mask": None, "out_mask": ["id"], "in_mask_pattern": None, "out_mask_pattern": None}


# -- retrieval_recall ------------------------------------------------------------------


def test_recall_is_the_share_of_gold_chunks_retrieved_compared_as_trimmed_text():
    assert recall(["a", "b"], ["b ", "c"]) == 0.5 and recall([" a"], ["a"]) == 1.0 and recall([], ["a"]) == 0.0
    assert recall([1, 2], ["1", "3"]) == 0.5
    reward = RetrievalRecall()
    gold = {"chunks": ["a", "b"], "ids": [1, 2], "query": "q"}
    assert _run(reward, gold, {"chunks": ["a"], "ids": [1, 2]}) == 0.75  # each list field, averaged; `query` is no list
    assert _run(reward, gold, {"chunks": ["a"]}) == 0.25  # a list the prediction lacks scores 0
    assert _run(RetrievalRecall(in_mask=["chunks"]), gold, {"chunks": ["a"], "ids": [1, 2]}) == 0.5
    assert _run(RetrievalRecall(out_mask=["ids"]), gold, {"chunks": ["a"], "ids": [1, 2]}) == 0.5
    assert _run(reward, {"query": "q"}, {"chunks": ["a"]}) == 0.0 and asyncio.run(reward(None, _dm({}))) == 0.0
    assert RetrievalRecall.from_config(reward.get_config()).name == "retrieval_recall"


# -- the judge panel -------------------------------------------------------------------


class _Judge:
    """A panelist or the smart judge: answers a fixed verdict, counts its calls."""

    def __init__(self, reward):
        self.reward, self.calls = reward, 0

    async def __call__(self, inputs):
        self.calls += 1
        return None if self.reward is None else {"critique": f"says {self.reward}", "reward": self.reward}


def _panel(*rewards, smart=0.9, threshold=0.2):
    program = MultiJudgePanelProgram(panel_language_models=["ollama/a", "ollama/b"], smart_language_model="openai/gpt-4o", agreement_threshold=threshold)
    program.panel = [_Judge(r) for r in rewards]
    program.smart = _Judge(smart)
    return program


def _judge(program, gold, pred):
    return asyncio.run(program.call([gold and _dm(gold), pred and _dm(pred)]))


def test_an_agreeing_panel_answers_with_the_judge_nearest_the_mean_and_a_split_one_asks_the_smart_judge():
    panel = _panel(0.5, 0.6, 0.7)
    verdict = _judge(panel, {"answer": "4"}, {"answer": "4"})
    assert verdict["reward"] == 0.6 and panel.smart.calls == 0 and [j.calls for j in panel.panel] == [1, 1, 1]
    split = _panel(0.1, 0.9)
    assert _judge(split, {"answer": "4"}, {"answer": "5"})["reward"] == 0.9 and split.smart.calls == 1
    failed = _panel(None, None)
    assert _judge(failed, {"answer": "4"}, {"answer": "4"})["reward"] == 0.9 and failed.smart.calls == 1  # the whole panel failed: the smart judge decides
    dropped = _panel(0.5, None, 0.55)
    assert _judge(dropped, {"answer": "4"}, {"answer": "4"})["reward"] == 0.5 and dropped.smart.calls == 0  # one glitch is dropped
    wide = _panel(0.1, 0.9, threshold=1.0)
    assert _judge(wide, {"answer": "4"}, {"answer": "4"})["reward"] in (0.1, 0.9) and wide.smart.calls == 0
    assert _judge(_panel(0.5), None, {"answer": "4"})["reward"] == 0.5  # no gold: the prediction alone is judged


def test_an_empty_prediction_is_very_bad_without_a_single_call_and_the_inputs_are_checked():
    panel = _panel(0.5)
    verdict = asyncio.run(panel.call([_dm({"answer": "4"}), None]))
    assert float(verdict.get_json()["reward"]) == 0.0 and panel.panel[0].calls == 0 and panel.smart.calls == 0
    with pytest.raises(ValueError, match="list or tuple"):
        asyncio.run(panel.call(_dm({})))
    with pytest.raises(ValueError, match="length of 2"):
        asyncio.run(panel.call([_dm({})]))


def test_the_panel_needs_its_judges_and_keeps_its_settings():
    with pytest.raises(ValueError, match="`panel_language_models` must be a non-empty list"):
        MultiJudgePanel(panel_language_models=[], smart_language_model="openai/gpt-4o")
    with pytest.raises(ValueError, match="`smart_language_model` is required"):
        MultiJudgePanel(panel_language_models=["ollama/a"])
    reward = get({"name": "multi_judge_panel", "panel_language_models": ["ollama/a", {"model": "ollama/b", "temperature": 0.0}], "smart_language_model": "openai/gpt-4o", "agreement_threshold": 0.3, "instructions": "Grade."})
    program = reward.program
    assert isinstance(program, MultiJudgePanelProgram) and program.agreement_threshold == 0.3 and len(program.panel) == 2
    assert [lm.model.split("/")[-1] for lm in program.panel_language_models] == ["a", "b"] and program.panel_language_models[1].default_kwargs == {"temperature": 0.0}
    assert program.smart_language_model.model == "openai/gpt-4o" and program.instructions == "Grade."
    config = program.get_config()
    assert config["agreement_threshold"] == 0.3 and len(config["panel_language_models"]) == 2 and "smart_language_model" in config
    assert {"panel_language_models", "smart_language_model", "agreement_threshold", "instructions"} <= accepted_kwargs("multi_judge_panel")
