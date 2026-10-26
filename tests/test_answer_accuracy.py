# License Apache 2.0: (c) 2026 Atena-Reply

"""`answer_accuracy`: each field graded by the metric its type calls for, then averaged."""

import asyncio
import os

import pytest

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from synalinks.src.backend import JsonDataModel

from openarena.rewards.answer_accuracy import AnswerAccuracy


def _dm(value: dict, schema: dict | None = None) -> JsonDataModel:
    schema = schema or {"type": "object", "properties": {k: {} for k in value}}
    return JsonDataModel(json=value, schema=schema)


def _score(reward, true, pred, schema=None) -> float:
    return asyncio.run(reward(_dm(true, schema), _dm(pred, schema)))


def test_routes_each_field_by_its_schema_type_and_averages():
    schema = {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": ["yes", "no"]},
            "flag": {"type": "boolean"},
            "count": {"type": "integer"},
            "text": {"type": "string"},
        },
    }
    r = AnswerAccuracy(schema=schema)
    gold = {"label": "yes", "flag": False, "count": 3, "text": "the cat sat"}
    assert _score(r, gold, dict(gold), schema) == pytest.approx(1.0, abs=1e-6)  # the metrics compute in float32
    # A correct negative on a boolean still scores that field 1.0; a wrong number 0.0.
    half = _score(r, gold, {"label": "no", "flag": False, "count": 4, "text": "the cat sat"}, schema)
    assert half == pytest.approx(0.5, abs=1e-6)


def test_without_a_schema_the_value_type_decides_and_masks_apply():
    r = AnswerAccuracy(in_mask=["answer"])
    assert _score(r, {"answer": "42", "rationale": "x"}, {"answer": "42", "rationale": "y"}) == pytest.approx(1.0, abs=1e-6)
    assert _score(r, {"answer": "42"}, {"answer": "41"}) == pytest.approx(0.0, abs=1e-6)
    # Multi-token text keeps partial credit rather than collapsing to 0.
    partial = _score(AnswerAccuracy(), {"text": "a b c d"}, {"text": "a b"})
    assert 0.0 < partial < 1.0
    assert _score(AnswerAccuracy(), {}, {"x": 1}) == 0.0


def test_a_list_of_labels_is_categorical_and_a_number_is_exact():
    from openarena.rewards.answer_accuracy import kind_of

    assert kind_of({"type": "array", "items": {"enum": ["a", "b"]}}, None) == "categorical"
    assert kind_of({"const": "x"}, None) == "categorical" and kind_of({"type": "boolean"}, None) == "binary"
    assert kind_of({"type": "number"}, None) == "numeric" and kind_of({"type": "array"}, None) == "text" and kind_of({}, 2.5) == "text"
    # No schema for the field: its value decides.
    assert kind_of(None, True) == "binary" and kind_of("string", 2.5) == "numeric" and kind_of(None, ["a"]) == "text"
    schema = {"type": "object", "properties": {"labels": {"type": "array", "items": {"enum": ["a", "b", "c"]}}, "n": {"type": "integer"}}}
    r = AnswerAccuracy(schema=schema)
    assert _score(r, {"labels": ["a", "b"], "n": 3}, {"labels": ["a", "b"], "n": 3}, schema) == pytest.approx(1.0, abs=1e-6)
    assert _score(r, {"labels": ["a", "b"], "n": 3}, {"labels": ["a", "b"], "n": "3"}, schema) == pytest.approx(0.5, abs=1e-6)  # a number as text is not the number
    assert _score(r, {"n": 3}, {}, schema) == 0.0  # a missing field scores 0


def test_a_nested_answer_is_compared_whole_and_the_out_mask_reaches_inside_it():
    gold = {"tool_calls": [{"id": "gold-1", "function": {"name": "add", "arguments": {"a": 1}}}]}
    pred = {"tool_calls": [{"id": "pred-9", "function": {"name": "add", "arguments": {"a": 1}}}]}
    assert _score(AnswerAccuracy(out_mask=["id"]), gold, pred) == pytest.approx(1.0, abs=1e-6)
    assert _score(AnswerAccuracy(out_mask_pattern="^id$"), gold, pred) == pytest.approx(1.0, abs=1e-6)
    assert _score(AnswerAccuracy(), gold, pred) < 1.0  # the ids differ, and they count without the mask
    nested = {"answer": {"city": "Paris"}, "rationale": "x"}
    assert _score(AnswerAccuracy(), nested, {**nested, "rationale": "y"}) < 1.0  # the rationale counts without a mask
    assert _score(AnswerAccuracy(in_mask=["answer"]), {"answer": "Paris", "rationale": "x"}, {"answer": "Paris", "rationale": "y"}) == pytest.approx(1.0, abs=1e-6)


def test_an_in_mask_keeps_a_nested_answer_whole():
    """The masks are the reward's own, not the base class's: applied at every level, an in-mask would empty a nested answer."""
    nested = {"answer": {"city": "Paris"}, "rationale": "x"}
    assert _score(AnswerAccuracy(in_mask=["answer"]), nested, {**nested, "rationale": "y"}) == pytest.approx(1.0, abs=1e-6)
    assert _score(AnswerAccuracy(in_mask_pattern="^ans"), nested, {**nested, "rationale": "y"}) == pytest.approx(1.0, abs=1e-6)


def test_no_prediction_scores_zero_and_the_reward_round_trips_through_its_config():
    from openarena.rewards.answer_accuracy import bounded, scalar

    r = AnswerAccuracy(schema={"type": "object"}, in_mask=["a"], out_mask=["b"], in_mask_pattern="^a", out_mask_pattern="^b", name="acc")
    assert asyncio.run(r(_dm({"a": 1}), None)) == 0.0 and asyncio.run(r(None, _dm({"a": 1}))) == 0.0
    assert r.get_config() == {"schema": {"type": "object"}, "in_mask": ["a"], "out_mask": ["b"], "in_mask_pattern": "^a", "out_mask_pattern": "^b", "name": "acc"}
    assert AnswerAccuracy.from_config(r.get_config()).get_config() == r.get_config()
    assert bounded(float("nan")) == 0.0 and bounded(1.5) == 1.0 and bounded(-0.1) == 0.0
    assert scalar(None) == 0.0 and scalar([1.0, 0.0]) == 0.5 and scalar([]) == 0.0 and scalar(0.25) == 0.25
