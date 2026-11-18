# License Apache 2.0: (c) 2026 Atena-Reply

"""What a cell's calls cost at list price, when a run charges some of its models at a factor.

A host may have litellm charge a model's calls at its list price times a factor
(`lookups.prices.ensure_priced`). The framework keeps, on each model object,
what its calls cost, phase by phase, every attempt of a retried call included;
its cost metrics read those counters. Read the same way and divided by each
model's factor, the counters say what the same calls cost at list price. Both
figures come from one place, so they cannot count different calls.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import synalinks
from synalinks.src.metrics.em_metrics import _collect_embedding_models
from synalinks.src.metrics.lm_metrics import _collect_language_models

from openarena.callbacks.otel import _judges

_PHASES = ("inference", "reward")
# The name the framework's metrics give the cost of a call, by the kind of model and the phase it ran in.
_KEYS = {
    (False, "inference"): "cost",
    (False, "reward"): "reward_cost",
    (True, "inference"): "embedding_cost",
    (True, "reward"): "reward_embedding_cost",
}


def models_of(program) -> list:
    """Every model object a program, its reward and its judges call: language, decision and embedding models, each once."""
    found: dict[int, object] = {}
    for part in (program, *_judges(program)):
        for model in (*_collect_language_models(part), *_collect_embedding_models(part)):
            found.setdefault(id(model), model)
    return list(found.values())


def snapshot(models: Sequence) -> list[dict[str, float]]:
    """Each model object's cost counters now, in the order of `models`: what a cell starts from."""
    return [
        {phase: float(getattr(model, f"{phase}_cumulated_cost", 0.0) or 0.0) for phase in _PHASES}
        for model in models
    ]


def list_usage(
    models: Sequence, baseline: Sequence[Mapping[str, float]], price_factors: Mapping[str, float]
) -> dict[str, dict[str, float]]:
    """What each model's calls cost at list price since `baseline`, by model id and by the framework's cost name.

    `baseline` is `snapshot(models)` taken when the cell started. A model given
    no factor was charged at list price, so its figure is what it was charged.
    Objects of one model id add up; a model, or a kind of cost, with nothing
    spent is left out.
    """
    usage: dict[str, dict[str, float]] = {}
    for model, before, now in zip(models, baseline, snapshot(models)):
        embedding = isinstance(model, synalinks.EmbeddingModel)
        factor = price_factors.get(model.model, 1.0)
        for phase in _PHASES:
            spent = now[phase] - before[phase]
            if spent:
                costs = usage.setdefault(model.model, {})
                key = _KEYS[embedding, phase]
                costs[key] = costs.get(key, 0.0) + spent / factor
    return usage
