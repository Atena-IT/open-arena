# License Apache 2.0: (c) 2026 Athena-Reply

"""Verifiers: pairwise / group-relative scoring primitives.

Verifiers compare predictions *against each other* rather than against
an absolute reference. The per-sample score is therefore meaningful
only relative to the batch composition, which makes them a natural fit
for group-relative training signals (e.g. GRPO-style rollout
advantages) but a poor fit for `program.evaluate()`-style sweeps where
an absolute, dataset-aggregable score is expected. Kept separate from
`src.rewards` for that reason — these are not YAML-resolvable as eval
rewards.
"""

from src.verifiers.lm_as_verifier import LMAsVerifier

__all__ = ["LMAsVerifier"]
