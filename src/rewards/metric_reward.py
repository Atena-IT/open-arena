"""`metric_reward` — use synalinks deterministic Metrics as the reward.

The deterministic graders in `synalinks.metrics` (token-Jaccard `accuracy`/
`f1_score`, plus the `binary_*` / `categorical_*` families) are Metrics, not
Rewards, so they can't go in a `reward:` block directly. This wraps them as a
`synalinks.rewards.Reward` so the *same* computation can drive the reward **and**
still be tracked under `metrics:` for later task-level aggregation.

Two modes:

- ``metric: auto`` (default) — pick the best metric **per key**, score each key,
  and average. When an ``output_schema`` is given, each field is routed by its
  schema type — ``enum``/``const`` → categorical accuracy, ``boolean`` → binary
  accuracy, ``number``/``integer`` → exact match, anything else → token-Jaccard
  accuracy. Without a schema it falls back to the value's runtime type (booleans
  → binary, numbers → exact, everything else → token-Jaccard, which collapses to
  exact match for single-token labels). The accuracy family is used rather than
  F1 so a correct negative (boolean ``False``, an absent label) scores 1.0; it
  stays Jaccard-based, so multi-token/multi-label fields keep partial credit. A
  multi-field output is graded field-appropriately and per-key scores averaged.
- ``metric: <name>`` — force one metric (e.g. ``categorical_f1_score``) over the
  masked fields.

```yaml
reward:
  name: metric_reward      # metric: auto by default
  in_mask: [verdict, rationale, label]
  schema: {...}            # optional output_schema for schema-aware routing
```

The reward is in [0, 1]. A fresh metric is built per call, so concurrent scoring
(open-arena's `asyncio.gather`) never shares mutable metric state.
"""

import math

import synalinks
from synalinks.src.backend.common.json_utils import in_mask_json, out_mask_json
from synalinks.src.rewards.reward import Reward

# Deterministic metrics selectable as a reward, by their snake_case name (the
# same identifier used under `metrics:`).
_METRIC_CLASSES = {
    "accuracy": synalinks.metrics.Accuracy,
    "binary_accuracy": synalinks.metrics.BinaryAccuracy,
    "categorical_accuracy": synalinks.metrics.CategoricalAccuracy,
    "f1_score": synalinks.metrics.F1Score,
    "binary_f1_score": synalinks.metrics.BinaryF1Score,
    "categorical_f1_score": synalinks.metrics.CategoricalF1Score,
    "precision": synalinks.metrics.Precision,
    "recall": synalinks.metrics.Recall,
    "binary_precision": synalinks.metrics.BinaryPrecision,
    "binary_recall": synalinks.metrics.BinaryRecall,
    "categorical_precision": synalinks.metrics.CategoricalPrecision,
    "categorical_recall": synalinks.metrics.CategoricalRecall,
}

# In `auto` mode, each per-key "kind" maps to the metric used for that group.
# "numeric" is handled by exact equality (no token metric fits a bare number).
# The accuracy family is used (not F1) because per-example it's symmetric: a
# correct negative (e.g. boolean `False`/the absent label) scores 1.0, whereas
# token/set-F1 only counts the positive class and would score it 0.0. The
# synalinks accuracy metrics are still Jaccard-based, so text/multi-label fields
# keep partial credit (|A∩B|/|A∪B|) rather than collapsing to all-or-nothing.
_AUTO_KIND_METRIC = {
    "categorical": "categorical_accuracy",
    "binary": "binary_accuracy",
    "text": "accuracy",
}


def _to_scalar(result):
    """Collapse a metric `result()` (scalar, per-field list, or None) to a float."""
    if result is None:
        return 0.0
    if isinstance(result, (list, tuple)):
        values = [float(v) for v in result]
        return sum(values) / len(values) if values else 0.0
    return float(result)


def _clamp(value):
    """Bound a reward to [0, 1]; map NaN to 0.0 (matches `DeepEval.call`)."""
    if math.isnan(value):
        return 0.0
    return max(0.0, min(1.0, value))


def _schema_kind(node):
    """Per-field metric kind from its JSON-schema node, or None if undecidable."""
    if not isinstance(node, dict):
        return None
    if "enum" in node or "const" in node:
        return "categorical"
    node_type = node.get("type")
    if node_type == "boolean":
        return "binary"
    if node_type in ("number", "integer"):
        return "numeric"
    if node_type == "array":
        items = node.get("items")
        if isinstance(items, dict) and ("enum" in items or "const" in items):
            return "categorical"
    return "text"


def _value_kind(value):
    """Per-field metric kind from the runtime value (schema-free fallback)."""
    if isinstance(value, bool):
        return "binary"
    if isinstance(value, (int, float)):
        return "numeric"
    return "text"  # str / list / dict -> token-Jaccard (exact for one token)


class MetricReward(Reward):
    """Wrap synalinks deterministic Metric(s) as a reward.

    Args:
        metric: ``"auto"`` (per-key best metric, averaged) or a snake_case metric
            name from ``_METRIC_CLASSES`` (one metric over the masked fields).
        schema: Optional ``output_schema`` (JSON Schema) for schema-aware routing
            of fields to categorical/binary/numeric/text in ``auto`` mode.
        in_mask / out_mask / in_mask_pattern / out_mask_pattern: field selection.
        average: per-field reduction for the fixed-metric mode (default
            ``"micro"``); ``auto`` always averages per key.
        metric_kwargs: extra kwargs forwarded to the metric constructor.
        name: reward instance name.
    """

    def __init__(
        self,
        metric="auto",
        schema=None,
        in_mask=None,
        out_mask=None,
        in_mask_pattern=None,
        out_mask_pattern=None,
        average="micro",
        metric_kwargs=None,
        name="metric_reward",
    ):
        super().__init__(name=name)
        if metric != "auto" and metric not in _METRIC_CLASSES:
            raise ValueError(
                f"metric_reward has no metric named {metric!r}. "
                f"Use 'auto' or one of: {sorted(_METRIC_CLASSES)}"
            )
        self.metric = metric
        self.schema = schema
        self.in_mask = in_mask
        self.out_mask = out_mask
        self.in_mask_pattern = in_mask_pattern
        self.out_mask_pattern = out_mask_pattern
        self.average = average
        self.metric_kwargs = dict(metric_kwargs or {})
        if metric != "auto":
            self._build_metric(metric, average)  # validate kwargs up front

    def _build_metric(self, metric, average, in_mask=None):
        return _METRIC_CLASSES[metric](
            in_mask=in_mask if in_mask is not None else self.in_mask,
            out_mask=None if in_mask is not None else self.out_mask,
            in_mask_pattern=None if in_mask is not None else self.in_mask_pattern,
            out_mask_pattern=None if in_mask is not None else self.out_mask_pattern,
            average=average,
            **self.metric_kwargs,
        )

    def _candidate_keys(self, true_json):
        """Keys of the gold to score, after applying the user's in/out masks.

        Mirrors how synalinks Metrics mask fields: in-mask first (list ``in_mask``
        OR'd with regex ``in_mask_pattern``), then out-mask — so pattern-based
        masking configured in YAML reaches ``auto`` mode too.
        """
        masked = true_json
        if self.in_mask or self.in_mask_pattern:
            masked = in_mask_json(
                masked, mask=self.in_mask, pattern=self.in_mask_pattern
            )
        if self.out_mask or self.out_mask_pattern:
            masked = out_mask_json(
                masked, mask=self.out_mask, pattern=self.out_mask_pattern
            )
        return list(masked)

    def _key_kind(self, key, value, props):
        """Route a key to a metric kind: schema first, runtime value as fallback."""
        if props is not None and isinstance(props.get(key), dict):
            kind = _schema_kind(props[key])
            if kind is not None:
                return kind
        return _value_kind(value)

    async def call(self, y_true, y_pred):
        if y_pred is None or y_true is None:
            return 0.0
        if self.metric != "auto":
            metric = self._build_metric(self.metric, self.average)
            return _clamp(_to_scalar(await metric(y_true, y_pred)))

        # auto: group the gold's keys by metric kind (schema-driven, value as
        # fallback), score each group with the type-appropriate metric, then
        # average weighted by key count (= mean of per-key scores).
        true_json = y_true.get_json() or {}
        pred_json = y_pred.get_json() or {}
        keys = self._candidate_keys(true_json)
        if not keys:
            return 0.0
        props = self.schema.get("properties") if isinstance(self.schema, dict) else None

        groups: dict[str, list] = {}
        for k in keys:
            groups.setdefault(self._key_kind(k, true_json[k], props), []).append(k)

        parts = []  # (score, weight)
        for kind, group_keys in groups.items():
            if kind == "numeric":
                hits = sum(
                    1.0 for k in group_keys
                    if k in pred_json and pred_json[k] == true_json[k]
                )
                parts.append((hits / len(group_keys), len(group_keys)))
            else:
                m = self._build_metric(_AUTO_KIND_METRIC[kind], "macro", in_mask=group_keys)
                parts.append((_to_scalar(await m(y_true, y_pred)), len(group_keys)))

        total_w = sum(w for _, w in parts)
        if not total_w:
            return 0.0
        return _clamp(sum(s * w for s, w in parts) / total_w)

    def get_config(self):
        return {
            "metric": self.metric,
            "schema": self.schema,
            "in_mask": self.in_mask,
            "out_mask": self.out_mask,
            "in_mask_pattern": self.in_mask_pattern,
            "out_mask_pattern": self.out_mask_pattern,
            "average": self.average,
            "metric_kwargs": self.metric_kwargs,
            "name": self.name,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
