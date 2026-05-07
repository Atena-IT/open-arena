# License Apache 2.0: (c) 2026 Athena-Reply

"""DeepEval reward.

Wraps any `deepeval.metrics` metric (`GEval`, `AnswerRelevancyMetric`,
`HallucinationMetric`, `FaithfulnessMetric`, `ContextualPrecisionMetric`,
`ToolCorrectnessMetric`, `ToxicityMetric`, `BiasMetric`, …) into a
`synalinks.rewards.Reward` so it can be selected from `config.yaml`
under the snake_case name ``deep_eval``.

Install (optional): ``uv add deepeval``. The import is lazy — the module
itself loads even without DeepEval, but constructing a `DeepEval`
reward will raise `ImportError` until the dep is installed.

Per-slot routing
----------------

Each `LLMTestCase` slot (``input``, ``actual_output``, ``expected_output``,
``context``, ``retrieval_context``, ``tools_called``, ``expected_tools``)
is built independently from a configurable view of `y_true` or `y_pred`.
Each slot config carries its own mask knobs — the same four the base
`Reward` class exposes — so different slots can pull different fields
from the same data model without interfering:

```yaml
- name: deep_eval
  alias: deep_eval_correctness
  metric: GEval
  metric_kwargs:
    name: Correctness
    criteria: |
      Determine whether the actual output matches the expected output
      in meaning, ignoring formatting differences.
    evaluation_params: [input, actual_output, expected_output]
    model: gpt-4o
    threshold: 0.5
  testcase:
    input:           {from: y_pred, in_mask: [messages], extract: chat_user}
    actual_output:   {from: y_pred, in_mask: [content]}
    expected_output: {from: y_true, in_mask: [content]}
```

Slot-config keys (all optional unless noted):

- ``from``: ``y_pred`` or ``y_true`` — which side to read. Defaults are
  conventional per slot (input/actual/tools_called/retrieval_context →
  y_pred; expected_*/context → y_true).
- ``in_mask`` / ``out_mask`` / ``in_mask_pattern`` / ``out_mask_pattern``:
  same semantics as `synalinks.Reward`. At least one must be set.
- ``extract``: ``scalar`` (default), ``list``, or ``chat_user``. Controls
  how the masked sub-model is collapsed into the slot's required Python
  type (string for scalar slots, ``list[str]`` for context-style slots,
  joined-user-content string for chat extraction).
- ``user_role``: only for ``extract: chat_user``. Defaults to ``"user"``.

The `testcase` block is optional. When omitted, defaults that match the
chat-message convention used elsewhere in this repo are applied (input
from `y_pred.messages` via `chat_user`, actual/expected from `content`).

Fail-fast: at construction time, the metric is built and its
`_required_params` introspected; any required slot without a config
*and* without a default raises `ValueError` so the YAML can be fixed
before any rows are processed. Unknown slot names also raise.
"""

import asyncio
import json as _json

from synalinks.src.rewards.reward import Reward


_VALID_FROM = ("y_true", "y_pred")
_VALID_EXTRACT = ("scalar", "list", "chat_user")
_VALID_SLOTS = (
    "input",
    "actual_output",
    "expected_output",
    "context",
    "retrieval_context",
    "tools_called",
    "expected_tools",
)
# Conventional defaults — applied only for slots a metric actually requires
# AND that the user hasn't overridden in YAML. Keeps the common chat-message
# case zero-config while still letting any slot be customised.
_DEFAULT_SLOTS = {
    "input": {"from": "y_pred", "in_mask": ["messages"], "extract": "chat_user"},
    "actual_output": {"from": "y_pred", "in_mask": ["content"]},
    "expected_output": {"from": "y_true", "in_mask": ["content"]},
    "context": {"from": "y_true", "in_mask": ["context"], "extract": "list"},
    "retrieval_context": {
        "from": "y_pred",
        "in_mask": ["retrieval_context"],
        "extract": "list",
    },
    "tools_called": {"from": "y_pred", "in_mask": ["tool_calls"], "extract": "list"},
    "expected_tools": {"from": "y_true", "in_mask": ["tool_calls"], "extract": "list"},
}


class _SlotConfig:
    """Per-slot routing + mask config for one `LLMTestCase` field."""

    __slots__ = (
        "source",
        "in_mask",
        "out_mask",
        "in_mask_pattern",
        "out_mask_pattern",
        "extract",
        "user_role",
    )

    def __init__(self, slot, raw):
        if not isinstance(raw, dict):
            raise ValueError(
                f"testcase slot {slot!r}: config must be a mapping, got "
                f"{type(raw).__name__}"
            )
        unknown = set(raw) - {
            "from",
            "in_mask",
            "out_mask",
            "in_mask_pattern",
            "out_mask_pattern",
            "extract",
            "user_role",
        }
        if unknown:
            raise ValueError(
                f"testcase slot {slot!r}: unknown key(s) {sorted(unknown)}"
            )
        self.source = raw.get("from", "y_pred")
        self.in_mask = raw.get("in_mask")
        self.out_mask = raw.get("out_mask")
        self.in_mask_pattern = raw.get("in_mask_pattern")
        self.out_mask_pattern = raw.get("out_mask_pattern")
        self.extract = raw.get("extract", "scalar")
        self.user_role = raw.get("user_role", "user")

        if self.source not in _VALID_FROM:
            raise ValueError(
                f"testcase slot {slot!r}: `from` must be one of {_VALID_FROM}, "
                f"got {self.source!r}"
            )
        if self.extract not in _VALID_EXTRACT:
            raise ValueError(
                f"testcase slot {slot!r}: `extract` must be one of "
                f"{_VALID_EXTRACT}, got {self.extract!r}"
            )
        if not (
            self.in_mask
            or self.out_mask
            or self.in_mask_pattern
            or self.out_mask_pattern
        ):
            raise ValueError(
                f"testcase slot {slot!r}: at least one of `in_mask`, "
                f"`out_mask`, `in_mask_pattern`, `out_mask_pattern` must be "
                f"set (per-slot masks are how this reward picks data-model fields)"
            )

    def to_dict(self):
        return {
            "from": self.source,
            "in_mask": self.in_mask,
            "out_mask": self.out_mask,
            "in_mask_pattern": self.in_mask_pattern,
            "out_mask_pattern": self.out_mask_pattern,
            "extract": self.extract,
            "user_role": self.user_role,
        }


def _resolve_evaluation_params(values):
    """Map snake_case YAML strings to `LLMTestCaseParams` enum members."""
    from deepeval.test_case import LLMTestCaseParams

    enum_lookup = {p.name: p for p in LLMTestCaseParams}
    resolved = []
    for v in values:
        if isinstance(v, LLMTestCaseParams):
            resolved.append(v)
            continue
        if not isinstance(v, str):
            raise ValueError(
                f"`evaluation_params` entries must be strings or "
                f"LLMTestCaseParams; got {type(v).__name__}: {v!r}"
            )
        try:
            resolved.append(enum_lookup[v.upper()])
        except KeyError as e:
            valid = ", ".join(sorted(p.name.lower() for p in LLMTestCaseParams))
            raise ValueError(
                f"Unknown evaluation_params entry {v!r}. Valid: {valid}"
            ) from e
    return resolved


def _build_metric(metric, metric_kwargs):
    if not isinstance(metric, str):
        return metric

    try:
        from deepeval import metrics as _metrics
    except ImportError as e:
        raise ImportError(
            "DeepEval is not installed. Run `uv add deepeval` to enable "
            "the `deep_eval` reward."
        ) from e

    cls = getattr(_metrics, metric, None)
    if cls is None:
        available = sorted(n for n in dir(_metrics) if not n.startswith("_"))
        raise ValueError(
            f"DeepEval has no metric class named {metric!r}. "
            f"Available: {available}"
        )

    kwargs = dict(metric_kwargs or {})
    if "evaluation_params" in kwargs:
        kwargs["evaluation_params"] = _resolve_evaluation_params(
            kwargs["evaluation_params"]
        )
    return cls(**kwargs)


def _required_slots(metric):
    """Snake-case slot names a metric needs. Robust to GEval (lives on instance
    via `evaluation_params`) and standard metrics (`_required_params` list).
    """
    rp = getattr(metric, "_required_params", None)
    if isinstance(rp, list) and rp:
        return [getattr(p, "value", str(p)) for p in rp]
    eps = getattr(metric, "evaluation_params", None)
    if isinstance(eps, list) and eps:
        return [getattr(p, "value", str(p)) for p in eps]
    # LLMTestCase itself only requires input + actual_output; conservative fallback
    return ["input", "actual_output"]


def _apply_mask(dm, cfg):
    # `recursive=False`: filter only the top-level fields. The default
    # `recursive=True` walks into nested objects (e.g. inside list items),
    # which leaves sibling top-level fields as empty `[{}]` shells — confusing
    # when the intent is "keep only the messages field on this row".
    if cfg.in_mask or cfg.in_mask_pattern:
        dm = dm.in_mask(
            mask=cfg.in_mask, pattern=cfg.in_mask_pattern, recursive=False
        )
    if cfg.out_mask or cfg.out_mask_pattern:
        dm = dm.out_mask(
            mask=cfg.out_mask, pattern=cfg.out_mask_pattern, recursive=False
        )
    return dm


def _stringify(v):
    if isinstance(v, (dict, list)):
        return _json.dumps(v)
    return str(v)


def _extract_value(dm, cfg, slot):
    """Mask `dm`, then collapse the result to the Python type the slot needs."""
    masked = _apply_mask(dm, cfg)
    data = masked.get_json() if hasattr(masked, "get_json") else masked
    if not isinstance(data, dict) or not data:
        raise ValueError(
            f"testcase slot {slot!r}: mask kept no fields from the "
            f"{cfg.source!r} data model. Check that the mask names match the "
            f"dataset's actual fields."
        )

    if cfg.extract == "scalar":
        if len(data) == 1:
            return _stringify(next(iter(data.values())))
        return _json.dumps(data)

    if cfg.extract == "list":
        if len(data) == 1:
            v = next(iter(data.values()))
            if isinstance(v, list):
                return [_stringify(x) for x in v]
            return [_stringify(v)]
        return [_json.dumps({k: v}) for k, v in data.items()]

    # extract == "chat_user"
    if len(data) != 1:
        raise ValueError(
            f"testcase slot {slot!r}: extract=chat_user expects exactly one "
            f"masked field (typically 'messages'), got {sorted(data)}"
        )
    messages = next(iter(data.values()))
    if not isinstance(messages, list):
        raise ValueError(
            f"testcase slot {slot!r}: extract=chat_user expects the masked "
            f"field to be a list of chat messages, got {type(messages).__name__}"
        )
    user_chunks, all_chunks = [], []
    for m in messages:
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
        content = (
            m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
        )
        if content is None:
            continue
        all_chunks.append(_stringify(content))
        if role == cfg.user_role:
            user_chunks.append(_stringify(content))
    chunks = user_chunks or all_chunks
    if not chunks:
        raise ValueError(
            f"testcase slot {slot!r}: chat_user extraction yielded no content "
            f"(no messages with role={cfg.user_role!r} or any content at all)"
        )
    return "\n\n".join(chunks)


class DeepEval(Reward):
    """Wrap a DeepEval metric as a synalinks reward.

    See module docstring for the per-slot routing model and YAML example.

    Args:
        metric: Either a DeepEval metric class name (string, e.g.
            ``"GEval"``) resolved against `deepeval.metrics`, or a
            pre-built metric instance for programmatic use.
        metric_kwargs: kwargs forwarded to the metric constructor when
            `metric` is a string. `evaluation_params` entries are
            auto-resolved from snake_case strings to
            `LLMTestCaseParams`.
        testcase: Mapping of `LLMTestCase` slot name → slot config dict.
            See module docstring for the per-slot key reference. Slots
            the metric doesn't require are silently ignored. Required
            slots without a config inherit a default from the chat-
            message convention; if no default exists for a required
            slot, construction fails.
        name: Reward instance name.
    """

    def __init__(
        self,
        metric="GEval",
        metric_kwargs=None,
        testcase=None,
        name="deep_eval",
    ):
        # Per-slot masks replace the base class's global in/out_mask: each slot
        # routes from y_true or y_pred through its own filter. A global mask
        # would be a second, confusing layer.
        super().__init__(name=name)
        self.metric = metric
        self.metric_kwargs = dict(metric_kwargs or {})
        self._metric_instance = _build_metric(metric, self.metric_kwargs)

        user_slots = dict(testcase or {})
        unknown = set(user_slots) - set(_VALID_SLOTS)
        if unknown:
            raise ValueError(
                f"unknown testcase slot(s) {sorted(unknown)}. "
                f"Valid: {list(_VALID_SLOTS)}"
            )

        required = list(dict.fromkeys(_required_slots(self._metric_instance)))
        # LLMTestCase mandates input + actual_output regardless of metric
        for s in ("input", "actual_output"):
            if s not in required:
                required.append(s)

        self.testcase: dict[str, _SlotConfig] = {}
        missing = []
        for slot in required:
            raw = user_slots.get(slot, _DEFAULT_SLOTS.get(slot))
            if raw is None:
                missing.append(slot)
                continue
            self.testcase[slot] = _SlotConfig(slot, raw)
        if missing:
            metric_name = type(self._metric_instance).__name__
            raise ValueError(
                f"metric {metric_name!r} requires LLMTestCase slot(s) "
                f"{missing}, but no `testcase` config was provided for them "
                f"and no default exists. Add a `testcase: {{<slot>: {{from: ..., "
                f"in_mask: [...]}}}}` block to the reward config."
            )

        # Surface user-configured slots that the metric won't use — quiet error
        # is friendlier than a silent typo.
        extras = set(user_slots) - set(self.testcase)
        if extras:
            metric_name = type(self._metric_instance).__name__
            raise ValueError(
                f"testcase config has slot(s) {sorted(extras)} that metric "
                f"{metric_name!r} does not require (required: {required}). "
                f"Remove them from the YAML."
            )

    async def call(self, y_true, y_pred):
        if not y_pred:
            return 0.0

        from deepeval.test_case import LLMTestCase

        kwargs = {}
        for slot, cfg in self.testcase.items():
            source = y_pred if cfg.source == "y_pred" else y_true
            if source is None:
                raise ValueError(
                    f"testcase slot {slot!r}: required source {cfg.source!r} "
                    f"is None at runtime — dataset may be missing y_true."
                )
            kwargs[slot] = _extract_value(source, cfg, slot)

        test_case = LLMTestCase(**kwargs)

        a_measure = getattr(self._metric_instance, "a_measure", None)
        if a_measure is not None and asyncio.iscoroutinefunction(a_measure):
            await a_measure(test_case)
        else:
            await asyncio.to_thread(self._metric_instance.measure, test_case)

        score = getattr(self._metric_instance, "score", 0.0)
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 0.0
        if score != score:  # NaN
            score = 0.0
        return max(0.0, min(1.0, score))

    def get_config(self):
        return {
            "metric": self.metric if isinstance(self.metric, str) else None,
            "metric_kwargs": self.metric_kwargs,
            "testcase": {slot: cfg.to_dict() for slot, cfg in self.testcase.items()},
            "name": self.name,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
