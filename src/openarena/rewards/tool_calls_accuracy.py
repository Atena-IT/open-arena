# License Apache 2.0: (c) 2026 Atena-Reply

"""`tool_calls_accuracy` — did the model call the right functions with the right arguments?

The deterministic reward of a model-with-tools dataset. The prediction is the
model's chat message; its `tool_calls` are what it chose to call. The target
lists the gold calls. Each gold call is matched to one predicted call with the
same function name and the same arguments (JSON-equal, order of keys aside),
and the score is `matched / max(len(gold), len(predicted))` — 1.0 for exactly
the right calls, less for a missed, extra or wrong one, 1.0 when both sides
call nothing.

Both sides accept either shape: the wire form `{"function": {"name",
"arguments"}}` (arguments as a dict or a JSON string) or the short form
`{"name", "arguments"}`.

```yaml
reward:
  name: tool_calls_accuracy
```
"""

import json
from typing import Any

from synalinks.src.rewards.reward import Reward


def _normalize(call: Any) -> tuple[str, str] | None:
    """`(name, canonical arguments JSON)` for a call in either shape, or None for something that is not a call."""
    if not isinstance(call, dict):
        return None
    function = call.get("function") if isinstance(call.get("function"), dict) else call
    name = function.get("name")
    if not isinstance(name, str) or not name:
        return None
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except ValueError:
            return (name, arguments)  # unparsable text stays as it is; it will not match a dict
    return (name, json.dumps(arguments, sort_keys=True))


def calls_of(payload: Any) -> list[tuple[str, str]]:
    """The normalized calls a message or target carries under `tool_calls`; a bare list is accepted too."""
    items = payload.get("tool_calls") if isinstance(payload, dict) else payload
    out = []
    for call in items or []:
        normalized = _normalize(call)
        if normalized is not None:
            out.append(normalized)
    return out


def score(gold: list[tuple[str, str]], predicted: list[tuple[str, str]]) -> float:
    """Matched calls over the longer side; 1.0 when both are empty."""
    if not gold and not predicted:
        return 1.0
    remaining = list(predicted)
    matched = 0
    for call in gold:
        if call in remaining:
            remaining.remove(call)
            matched += 1
    return matched / max(len(gold), len(predicted))


class ToolCallsAccuracy(Reward):
    """How many of the gold tool calls the model made, with the right arguments and nothing extra."""

    def __init__(self, name="tool_calls_accuracy", in_mask=None, out_mask=None, in_mask_pattern=None, out_mask_pattern=None):
        super().__init__(name=name)
        self.in_mask = in_mask
        self.out_mask = out_mask
        self.in_mask_pattern = in_mask_pattern
        self.out_mask_pattern = out_mask_pattern

    async def call(self, y_true, y_pred):
        if y_true is None or y_pred is None:
            return 0.0
        return score(calls_of(y_true.get_json() or {}), calls_of(y_pred.get_json() or {}))

    def get_config(self):
        return {
            "name": self.name,
            "in_mask": self.in_mask,
            "out_mask": self.out_mask,
            "in_mask_pattern": self.in_mask_pattern,
            "out_mask_pattern": self.out_mask_pattern,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
