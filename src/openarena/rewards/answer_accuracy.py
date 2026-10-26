# License Apache 2.0: (c) 2026 Atena-Reply

"""`answer_accuracy` — every field of the answer graded by the metric its type calls for, then averaged.

The framework's deterministic graders (token-Jaccard `Accuracy`, and the
`BinaryAccuracy` / `CategoricalAccuracy` pair) are metrics, not rewards. This
reward routes each field of the target to one of them, or to exact equality
for a number, and averages the fields' scores:

- a field with a schema (`output_schema`) is routed by what it declares:
  an `enum` or `const` (or an array of them) is categorical, a `boolean` is
  binary, a `number` or `integer` is exact, anything else is text;
- a field without one is routed by its value: a bool is binary, a number is
  exact, the rest is text (token-Jaccard, exact for a one-token label).

The accuracy family rather than F1, so a correct negative (a `False`, an
absent label) scores 1.0; Jaccard-based, so a multi-token or multi-label
field keeps partial credit. A field holding objects (a nested answer, a list
of tool calls) is compared whole.

```yaml
reward:
  name: answer_accuracy
  in_mask: [verdict, label]   # the fields to grade; all of the target's otherwise
  out_mask: [id]              # names left out at every level, a tool call's id say
```

The score is in [0, 1]. Each field is graded by a fresh metric, so concurrent
scoring shares no state.
"""

from __future__ import annotations

import math

import synalinks
from synalinks.src.backend import JsonDataModel
from synalinks.src.backend.common.json_utils import in_mask_json, out_mask_json
from synalinks.src.rewards.reward import Reward

# The grader of each kind of field; `numeric` is exact equality, which no token metric gives.
GRADERS = {
    "categorical": synalinks.metrics.CategoricalAccuracy,
    "binary": synalinks.metrics.BinaryAccuracy,
    "text": synalinks.metrics.Accuracy,
}


def kind_of(declared, value) -> str:
    """`categorical`, `binary`, `numeric` or `text`: what the field's schema says when it has one, else what its value is."""
    if isinstance(declared, dict):
        items = declared.get("items")
        if {"enum", "const"} & declared.keys() or (isinstance(items, dict) and {"enum", "const"} & items.keys()):
            return "categorical"
        return {"boolean": "binary", "number": "numeric", "integer": "numeric"}.get(declared.get("type"), "text")
    if isinstance(value, bool):
        return "binary"
    return "numeric" if isinstance(value, (int, float)) else "text"


def bounded(score: float) -> float:
    """A score in [0, 1]; NaN (nothing to compare) counts as 0."""
    return 0.0 if math.isnan(score) else min(1.0, max(0.0, score))


def scalar(result) -> float:
    """A metric's result as one number: a per-field list is averaged, nothing is 0."""
    if result is None:
        return 0.0
    if isinstance(result, (list, tuple)):
        return sum(float(v) for v in result) / len(result) if result else 0.0
    return float(result)


class AnswerAccuracy(Reward):
    """Grade each field of the answer with the deterministic metric that fits its type, and average.

    Args:
        schema: Optional. The `output_schema` (JSON Schema) each field is
            routed by; a field it does not declare is routed by its value.
        in_mask / in_mask_pattern: The fields to grade, by name or by
            pattern; every field of the target otherwise.
        out_mask / out_mask_pattern: Names left out of the comparison, at
            every level.
        name: The reward instance's name.
    """

    def __init__(self, schema=None, in_mask=None, out_mask=None, in_mask_pattern=None, out_mask_pattern=None, name="answer_accuracy"):
        # The masks are applied here, not by the base class: it masks at every level, which would empty a nested answer.
        super().__init__(name=name)
        self.schema = schema
        self.keep, self.keep_pattern = in_mask, in_mask_pattern
        self.drop, self.drop_pattern = out_mask, out_mask_pattern

    def fields(self, target: dict) -> list[str]:
        """The target's fields to grade: what the in-mask keeps, less what the out-mask drops."""
        chosen = target
        if self.keep or self.keep_pattern:
            chosen = in_mask_json(chosen, mask=self.keep, pattern=self.keep_pattern)
        if self.drop or self.drop_pattern:
            chosen = out_mask_json(chosen, mask=self.drop, pattern=self.drop_pattern)
        return list(chosen)

    def _without_dropped(self, data: dict) -> dict:
        return out_mask_json(data, mask=self.drop, pattern=self.drop_pattern) if self.drop or self.drop_pattern else data

    async def call(self, y_true, y_pred):
        if y_true is None or y_pred is None:
            return 0.0
        target, answer = y_true.get_json() or {}, y_pred.get_json() or {}
        fields = self.fields(target)
        if not fields:
            return 0.0
        declared = (self.schema.get("properties") or {}) if isinstance(self.schema, dict) else {}
        by_kind: dict[str, list[str]] = {}
        for field in fields:
            by_kind.setdefault(kind_of(declared.get(field), target[field]), []).append(field)

        weighted = 0.0
        for kind, names in by_kind.items():
            if kind == "numeric":
                score = sum(1.0 for name in names if name in answer and answer[name] == target[name]) / len(names)
            else:
                score = await self._grade(kind, names, declared, self._without_dropped(target), self._without_dropped(answer))
            weighted += score * len(names)
        return bounded(weighted / len(fields))

    async def _grade(self, kind: str, names: list[str], declared: dict, target: dict, answer: dict) -> float:
        """The kind's grader over the named fields alone, each field's value whole (a metric's own in-mask would hollow out a nested one)."""
        shape = {"type": "object", "properties": {name: declared.get(name) or {} for name in names}}

        def only(data):
            return JsonDataModel(schema=shape, json={name: data[name] for name in names if name in data})

        grader = GRADERS[kind](in_mask=[], average="macro")
        return scalar(await grader(only(target), only(answer)))

    def get_config(self):
        return {
            "schema": self.schema,
            "in_mask": self.keep,
            "out_mask": self.drop,
            "in_mask_pattern": self.keep_pattern,
            "out_mask_pattern": self.drop_pattern,
            "name": self.name,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
