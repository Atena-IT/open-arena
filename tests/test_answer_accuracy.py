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
