# License Apache 2.0: (c) 2026 Athena-Reply

"""LM-as-a-Verifier reward: tournament-style batched judge.

Pairwise round-robin: each prediction plays head-to-head against the
other predictions in the batch; the per-sample reward is the fraction
of matches it won, in [0, 1]. Subclasses `synalinks.BatchReward`
because pairwise comparisons need cross-sample context — a single
(y_true, y_pred) carries no information about its siblings.

Ports "LLM-as-a-Verifier" (Kwok et al., Stanford / UC Berkeley / NVIDIA,
2026; https://github.com/llm-as-a-verifier/llm-as-a-verifier) using
**structured output** for the score letters instead of token logprobs.
The paper extracts the top-k logprobs at the `<score_A>` / `<score_B>`
positions and computes a logprob-weighted expectation over the 20-token
scale; we instead read the single sampled letter through synalinks's
structured-output Generator. The two are equivalent in expectation when
the judge LM is well-calibrated, and structured output works against
any hosted LM (most don't expose top-k logprobs).

Three robustness knobs from the paper:

- **Granularity (G)**: fixed at 20 (`A` best, `T` worst). The paper
  found G=20 the sweet spot; lower values lose discrimination (28%
  ties at G=8 vs. 14% at G=20 on Terminal-Bench).
- **Repeated verification (K)**: `n_repetitions=` runs the same pair
  through the judge K times and averages. Reduces judge variance.
- **Criteria decomposition (C)**: `criteria=[{name, description}, …]`
  splits the rubric into C independent passes (one LM call per
  criterion per pair); per-pair scalars average across criteria before
  the winner is decided. The paper's Terminal-Bench run uses C=3
  (specification adherence, output match, error signals).

Cost per batch: O(N · (N − 1) / 2 · C · K) judge calls for full
round-robin, halved by `pairings_per_sample`. All matches run
concurrently via `asyncio.gather`.

Mitigations baked in:

* **Position bias**: A/B side assignment is randomized per pair (seeded
  via `synalinks.set_seed()`).
* **Identical predictions**: auto-tied without LM calls (common at low
  sampling temperature).
* **Empty predictions / LM glitches**: auto-resolved or dropped from
  the tally — never fabricated.
"""

import asyncio
import json
import random
from typing import Literal

import synalinks
from synalinks.src import ops
from synalinks.src.modules import Generator
from synalinks.src.modules.language_models import get as _get_lm
from synalinks.src.programs import Program
from synalinks.src.rewards.batch_reward import BatchReward
from synalinks.src.saving import serialization_lib
from synalinks.src.utils.naming import to_snake_case


# 20-point letter scale, exactly as in LLM-as-a-Verifier (Sec. 4.1).
# A = 20 (best), T = 1 (worst). Linear φ maps letters to integers; we
# normalize to [0, 1] via (v − 1) / 19 before the per-pair comparison.
_GRANULARITY = 20
_LETTER_TO_VALUE = {
    chr(ord("A") + i): _GRANULARITY - i for i in range(_GRANULARITY)
}
_SCALE_TEXT = (
    "Rate quality on a 20-point letter scale (A best, T worst):\n"
    "  A     = clearly the best; correct, complete, on-spec\n"
    "  B-D   = succeeded with only minor issues\n"
    "  E-G   = above average, mostly correct\n"
    "  H-J   = uncertain, leans toward success\n"
    "  K-M   = uncertain, leans toward failure\n"
    "  N-P   = below average, significant issues\n"
    "  Q-S   = failed with some partial progress\n"
    "  T     = clearly the worst; wrong, incomplete, or off-spec"
)


class PairwiseScores(synalinks.DataModel):
    """Output schema produced by the verifier LM for one pairwise match."""

    reasoning: str = synalinks.Field(
        description=(
            "One or two sentences contrasting the two contestants on the "
            "evaluation criterion. Reach a conclusion before assigning scores."
        ),
    )
    score_A: Literal[
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
        "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    ] = synalinks.Field(
        description=(
            "Letter grade for contestant A on the 20-point scale "
            "(A best, T worst)."
        ),
    )
    score_B: Literal[
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
        "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    ] = synalinks.Field(
        description=(
            "Letter grade for contestant B on the 20-point scale "
            "(A best, T worst)."
        ),
    )


_DEFAULT_INSTRUCTIONS = """\
You are judging two assistant trajectories (A and B), each answering its \
own prompt. The `A_gold_*` / `A_pred_*` fields hold contestant A's gold \
answer and prediction; the `B_gold_*` / `B_pred_*` fields hold \
contestant B's.

For each side, ask: how close did the assistant's prediction come to \
its own gold? Then assign each side a letter grade on the rating scale \
below. The trajectory that did better on this criterion should get the \
higher letter (closer to A). When the two golds happen to be identical \
(rollouts of the same prompt), the comparison reduces to "which \
prediction is closer to the shared gold?".

If no gold-prefixed fields are present, grade each prediction on how \
correct, coherent, and complete it is on its face.\
"""


_DEFAULT_CRITERIA = (
    {
        "name": "overall correctness",
        "description": (
            "Evaluate how closely each prediction matches its own gold "
            "reference. Consider factual correctness, completeness, and "
            "format adherence on equal footing."
        ),
    },
)


class LMAsVerifierProgram(Program):
    """Inner program backing `LMAsVerifier`.

    Holds one `Generator(data_model=PairwiseScores, ...)` per evaluation
    criterion. Each generator's instructions bake in the criterion-
    specific rubric plus the shared 20-point rating scale, so the
    `criterion_idx` argument to `compare()` is just an index into
    `self.comparators`.
    """

    def __init__(
        self,
        language_model=None,
        criteria=None,
        prompt_template=None,
        examples=None,
        instructions=None,
        name=None,
        description=None,
        trainable=True,
    ):
        super().__init__(
            name=name,
            description=description,
            trainable=trainable,
        )
        if language_model is None:
            raise ValueError("`language_model` is required.")

        resolved_criteria = list(criteria) if criteria else list(_DEFAULT_CRITERIA)
        for c in resolved_criteria:
            if not isinstance(c, dict) or "name" not in c or "description" not in c:
                raise ValueError(
                    "`criteria` entries must be dicts with `name` and "
                    f"`description`; got {c!r}."
                )

        language_model = _get_lm(language_model)
        base_instructions = instructions or _DEFAULT_INSTRUCTIONS

        self.comparators = []
        for c in resolved_criteria:
            crit_instructions = (
                f"{base_instructions}\n\n"
                f"Evaluation criterion — {c['name']}:\n"
                f"{c['description']}\n\n"
                f"{_SCALE_TEXT}"
            )
            self.comparators.append(
                Generator(
                    data_model=PairwiseScores,
                    language_model=language_model,
                    prompt_template=prompt_template,
                    examples=examples,
                    instructions=crit_instructions,
                    name=f"comparator_{to_snake_case(c['name'])}_{self.name}",
                )
            )

        self.language_model = language_model
        self.criteria = resolved_criteria
        self.prompt_template = prompt_template
        self.examples = examples
        self.instructions = instructions

    async def compare(self, a_gold, a_pred, b_gold, b_pred, criterion_idx):
        """Run one pairwise judgment for the given criterion.

        Each side's gold is optional — when missing, the prefix block is
        simply omitted from the concatenated input.
        """
        a_pred_p = await ops.prefix(a_pred, prefix="A_pred", name="a_pred")
        b_pred_p = await ops.prefix(b_pred, prefix="B_pred", name="b_pred")
        pair = await ops.concat(a_pred_p, b_pred_p, name="ab_pred")

        if a_gold:
            a_gold_p = await ops.prefix(a_gold, prefix="A_gold", name="a_gold")
            pair = await ops.concat(a_gold_p, pair, name="with_a_gold")
        if b_gold:
            b_gold_p = await ops.prefix(b_gold, prefix="B_gold", name="b_gold")
            pair = await ops.concat(pair, b_gold_p, name="with_b_gold")

        return await self.comparators[criterion_idx](pair)

    async def call(self, inputs):
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 5:
            raise ValueError(
                "`LMAsVerifierProgram` expects "
                "`[a_gold, a_pred, b_gold, b_pred, criterion_idx]`."
            )
        a_gold, a_pred, b_gold, b_pred, criterion_idx = inputs
        return await self.compare(a_gold, a_pred, b_gold, b_pred, criterion_idx)

    def get_config(self):
        config = {
            "criteria": self.criteria,
            "prompt_template": self.prompt_template,
            "examples": self.examples,
            "instructions": self.instructions,
            "name": self.name,
            "description": self.description,
            "trainable": self.trainable,
        }
        lm_config = {
            "language_model": serialization_lib.serialize_synalinks_object(
                self.language_model
            ),
        }
        return {**lm_config, **config}

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        if "language_model" in config:
            config["language_model"] = serialization_lib.deserialize_synalinks_object(
                config["language_model"]
            )
        return cls(**config)


def _pred_key(y_pred_item):
    """Stable string key used to short-circuit identical-prediction matches."""
    if not y_pred_item:
        return None
    if hasattr(y_pred_item, "get_json"):
        return json.dumps(y_pred_item.get_json(), sort_keys=True)
    return json.dumps(y_pred_item, sort_keys=True)


def _letter_to_unit(letter):
    """Linearly map a letter grade in [A..T] to a unit-interval scalar."""
    val = _LETTER_TO_VALUE.get(letter)
    if val is None:
        return None
    return (val - 1) / (_GRANULARITY - 1)


class LMAsVerifier(BatchReward):
    """Pairwise-tournament reward over a batch of predictions.

    Ports "LLM-as-a-Verifier" (Kwok et al., 2026) to synalinks with
    **structured output** for the score letters instead of token logprobs:
    every pair is shown to the verifier LM, which returns a letter grade
    A..T (A best, T worst) for each side via a typed `PairwiseScores`
    schema. The higher mean grade wins; the per-sample reward is the
    fraction of matches won, in [0, 1].

    The algorithm is invariant to whether the batch is a set of grouped
    rollouts (`repeat == batch_size` in the project dataset loader) or
    a mix of independent samples: when two golds in a pair happen to be
    identical, the comparison reduces to "which prediction is closer to
    the shared gold?" — same code path, no special-casing.

    Robustness scaling (from the paper):
      - `criteria` (C): split the rubric into C independent passes
        (one LM call per criterion per pair); per-pair scalars average
        across criteria before the winner is decided.
      - `n_repetitions` (K): repeat each comparison K times and average,
        reducing judge variance.
      - Granularity (G): fixed at 20 letters. Lower G loses tie
        discrimination per the paper.

    Pairings strategy:
      - `pairings_per_sample=None` (default): full round-robin —
        `N * (N - 1) / 2` matches per batch.
      - `pairings_per_sample=k`: each sample plays `min(k, N - 1)` random
        opponents. Use this to bound cost when `batch_size` is large.

    Bias mitigations:
      - **Position bias**: A/B side assignment is randomized per pair
        (seeded by `synalinks.get_seed()`), so the LM's tendency to favor
        whichever option appears first doesn't systematically advantage
        one slot.

    Edge cases:
      - Batch size 0 -> `[]`. Batch size 1 -> `[0.5]` (no opponents).
      - Identical predictions auto-tie without an LM call (common at low
        sampling temperature).
      - Empty / falsy `y_pred[i]` auto-loses every match it participates
        in (a both-empty match is a tie).
      - Individual passes that raise or return an invalid letter are
        dropped from the per-pair averaging; the pair still resolves
        from whichever passes did succeed. A pair where ALL C × K
        passes fail is recorded as a tie.

    Example:

    ```python
    program.compile(
        reward=LMAsVerifier(
            language_model="ollama/llama3.2",
            criteria=[
                {"name": "factual accuracy", "description": "..."},
                {"name": "format adherence", "description": "..."},
            ],
            n_repetitions=2,
        ),
    )
    ```

    Args:
        language_model: The judge model. Accepts a `LanguageModel`, a
            config dict, or a string identifier (e.g. `"ollama/llama3.2"`).
            Required.
        criteria (list[dict]): Optional. List of `{"name", "description"}`
            entries. Defaults to a single "overall correctness" rubric.
        n_repetitions (int): How many times each criterion's comparison
            is repeated per pair, with results averaged. Defaults to 1.
        prompt_template (str): Optional jinja2 prompt template forwarded
            to every per-criterion comparator `Generator`.
        examples (list): Optional examples forwarded to every comparator.
        instructions (str): The base judging-task description forwarded
            to every comparator; each criterion's rubric and the rating
            scale are appended automatically.
        pairings_per_sample (int | None): Number of opponents each
            sample faces. `None` (default) means full round-robin.
            Capped at `N - 1`.
        name (str): Optional. Reward instance name
            (default `"lm_as_verifier"`).
        reduction (str): Optional. Reduction applied by standalone
            `__call__` (the trainer always consumes the unreduced
            per-sample list).
        in_mask (list): Optional. List of fields to keep before judging.
        out_mask (list): Optional. List of fields to drop before judging.
        in_mask_pattern (str): Optional. Regex form of `in_mask`.
        out_mask_pattern (str): Optional. Regex form of `out_mask`.
    """

    def __init__(
        self,
        language_model=None,
        criteria=None,
        n_repetitions=1,
        prompt_template=None,
        examples=None,
        instructions=None,
        pairings_per_sample=None,
        name="lm_as_verifier",
        reduction="mean",
        in_mask=None,
        out_mask=None,
        in_mask_pattern=None,
        out_mask_pattern=None,
    ):
        super().__init__(
            name=name,
            reduction=reduction,
            in_mask=in_mask,
            out_mask=out_mask,
            in_mask_pattern=in_mask_pattern,
            out_mask_pattern=out_mask_pattern,
        )
        if pairings_per_sample is not None and int(pairings_per_sample) < 1:
            raise ValueError(
                "`pairings_per_sample` must be >= 1 or None for full round-robin; "
                f"got {pairings_per_sample}."
            )
        if int(n_repetitions) < 1:
            raise ValueError(
                f"`n_repetitions` must be >= 1; got {n_repetitions}."
            )
        self.program = LMAsVerifierProgram(
            language_model=language_model,
            criteria=criteria,
            prompt_template=prompt_template,
            examples=examples,
            instructions=instructions,
        )
        self.language_model = language_model
        self.criteria = self.program.criteria
        self.n_repetitions = int(n_repetitions)
        self.prompt_template = prompt_template
        self.examples = examples
        self.instructions = instructions
        self.pairings_per_sample = pairings_per_sample

    def _build_pairings(self, n, rng):
        """Return the list of `(i, j)` match-ups for a batch of size `n`."""
        if n < 2:
            return []
        if self.pairings_per_sample is None:
            return [(i, j) for i in range(n) for j in range(i + 1, n)]
        k = min(int(self.pairings_per_sample), n - 1)
        seen = set()
        for i in range(n):
            opponents = [j for j in range(n) if j != i]
            for j in rng.sample(opponents, k):
                a, b = (i, j) if i < j else (j, i)
                seen.add((a, b))
        return sorted(seen)

    async def call(self, y_true, y_pred):
        n = len(y_pred)
        if n == 0:
            return []
        if n == 1:
            return [0.5]

        if y_true is not None and len(y_true) != n:
            raise ValueError(
                f"`y_true` and `y_pred` must have the same length; got "
                f"{len(y_true)} vs {n}."
            )

        rng = random.Random(synalinks.get_seed())
        pairings = self._build_pairings(n, rng)
        pred_cache = {idx: _pred_key(y_pred[idx]) for idx in range(n)}

        n_criteria = len(self.criteria)
        n_reps = self.n_repetitions

        judge_tasks = []
        judge_meta = []  # parallel list of (i, j, swap)
        auto_results = []  # (i, j, "A" | "B" | "tie") settled without an LM call

        for i, j in pairings:
            a_pred_empty = not y_pred[i]
            b_pred_empty = not y_pred[j]
            # Both empty -> tie. One empty -> the other wins.
            if a_pred_empty and b_pred_empty:
                auto_results.append((i, j, "tie"))
                continue
            if a_pred_empty:
                auto_results.append((i, j, "B"))
                continue
            if b_pred_empty:
                auto_results.append((i, j, "A"))
                continue
            # Identical predictions -> tie (skip the C × K LM calls).
            if pred_cache[i] is not None and pred_cache[i] == pred_cache[j]:
                auto_results.append((i, j, "tie"))
                continue

            # Schedule C × K independent judge calls per pair. Each call
            # gets its own A/B swap so position bias averages out across
            # the repetitions even within a single pair.
            for crit_idx in range(n_criteria):
                for _ in range(n_reps):
                    swap = rng.random() < 0.5
                    if swap:
                        a_gold = y_true[j] if y_true is not None else None
                        b_gold = y_true[i] if y_true is not None else None
                        a_pred, b_pred = y_pred[j], y_pred[i]
                    else:
                        a_gold = y_true[i] if y_true is not None else None
                        b_gold = y_true[j] if y_true is not None else None
                        a_pred, b_pred = y_pred[i], y_pred[j]
                    judge_tasks.append(
                        self.program.compare(
                            a_gold=a_gold,
                            a_pred=a_pred,
                            b_gold=b_gold,
                            b_pred=b_pred,
                            criterion_idx=crit_idx,
                        )
                    )
                    judge_meta.append((i, j, swap))

        verdicts = await asyncio.gather(*judge_tasks, return_exceptions=True)

        # Average (s_i, s_j) per pair across all successful C × K passes.
        # `pair_scores[(i, j)]` accumulates un-swapped (sample i's grade,
        # sample j's grade) tuples in the unit interval.
        pair_scores: dict[tuple[int, int], list[tuple[float, float]]] = {}
        for (i, j, swap), verdict in zip(judge_meta, verdicts):
            if isinstance(verdict, Exception) or verdict is None:
                continue
            score_a = verdict.get("score_A") if hasattr(verdict, "get") else None
            score_b = verdict.get("score_B") if hasattr(verdict, "get") else None
            unit_a = _letter_to_unit(score_a)
            unit_b = _letter_to_unit(score_b)
            if unit_a is None or unit_b is None:
                continue
            # Un-swap: if i was presented as B, the LM's `score_A` is j's.
            if swap:
                s_i, s_j = unit_b, unit_a
            else:
                s_i, s_j = unit_a, unit_b
            pair_scores.setdefault((i, j), []).append((s_i, s_j))

        wins = [0.0] * n
        counts = [0] * n

        def _tally(i, j, winner):
            counts[i] += 1
            counts[j] += 1
            if winner == "A":
                wins[i] += 1.0
            elif winner == "B":
                wins[j] += 1.0
            else:  # tie
                wins[i] += 0.5
                wins[j] += 0.5

        # Pairs with at least one successful pass: compare per-pair means.
        # Pairs that have NO successful passes (all C × K failed) fall
        # through to a recorded tie below.
        resolved_pairs = set()
        for (i, j), scores_list in pair_scores.items():
            mean_i = sum(s_i for s_i, _ in scores_list) / len(scores_list)
            mean_j = sum(s_j for _, s_j in scores_list) / len(scores_list)
            if mean_i > mean_j:
                _tally(i, j, "A")
            elif mean_j > mean_i:
                _tally(i, j, "B")
            else:
                _tally(i, j, "tie")
            resolved_pairs.add((i, j))

        for i, j in pairings:
            if (i, j) in resolved_pairs:
                continue
            if any(pair == (i, j) for pair in (p[:2] for p in auto_results)):
                continue
            _tally(i, j, "tie")

        for i, j, winner in auto_results:
            _tally(i, j, winner)

        return [
            (wins[i] / counts[i]) if counts[i] > 0 else 0.5 for i in range(n)
        ]

    def get_config(self):
        config = super().get_config()
        config["language_model"] = serialization_lib.serialize_synalinks_object(
            self.program.language_model
        )
        config.update(
            {
                "criteria": self.criteria,
                "n_repetitions": self.n_repetitions,
                "prompt_template": self.prompt_template,
                "examples": self.examples,
                "instructions": self.instructions,
                "pairings_per_sample": self.pairings_per_sample,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        if "language_model" in config:
            config["language_model"] = serialization_lib.deserialize_synalinks_object(
                config["language_model"]
            )
        return cls(**config)
