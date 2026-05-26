# License Apache 2.0: (c) 2026 Athena-Reply

"""Rubrics-as-judge reward.

Mirrors `synalinks.rewards.LMAsJudge`, but instead of a single judging pass the
prediction is scored against a *list of rubrics* — one criterion each — and the
per-rubric scores are mixed into a single reward.

The inner program is a small DAG:

```
                       [y_true, y_pred]
                              |
              (prefix gold_ + concat, or just y_pred)
                              |
                         judge_input
              +---------------+---------------+      fan-out, one branch per rubric
              v               v               v
        SelfCritique_0  SelfCritique_1 ...  SelfCritique_{N-1}
        (rubrics[0])    (rubrics[1])        (rubrics[N-1])
              |               |               |
        CritiqueWith    CritiqueWith     CritiqueWith
        Reward          Reward           Reward
              +---------------+---------------+
                              v
                     (mix, inline in call)    reads each reward, weights +
                              |               reduces them
                              v
                   RubricsJudgment { critique, reward: float }
```

Each `SelfCritique` gets the same `judge_input` but rubric-specific
`instructions` ("score only how well the prediction satisfies *this* rubric")
and emits a `CritiqueWithReward` (a `critique` plus a `reward` drawn from the
11-bucket `Score` enum). The panel runs concurrently; `call` then reads each
`reward`, applies `rubrics_weights`, reduces per `rubrics_reduction`, and
assembles the result inline.

The result is a `RubricsJudgment` whose `reward` is a plain **float** rather
than a `Score` bucket: a weighted mean lands between buckets, so the enum's
[0, 1] snapping would throw away precision. Every reduction (`mean`/`min`/`max`)
stays within [0, 1] on its own. Its `critique` carries the per-rubric breakdown
so the reasoning behind the aggregate survives in the trajectory.

Use it when "good" is multi-dimensional and you want to weight or gate the
dimensions explicitly — e.g. score an answer on correctness, completeness and
tone, weight correctness highest, and (with `rubrics_reduction="min"`) require
every rubric to pass.
"""

import asyncio

from synalinks.src import ops
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.modules import SelfCritique
from synalinks.src.modules.language_models import get as _get_lm
from synalinks.src.programs import Program
from synalinks.src.rewards.reward_wrappers import ProgramAsJudge
from synalinks.src.saving import serialization_lib

# How the per-rubric scores are mixed (the `rubrics_reduction` param — distinct
# from the base `Reward.reduction` that reduces per-sample rewards over a
# batch). All bounded in [0, 1], ordered `min` <= `mean` <= `max`. Weights are
# always-on with a uniform default: `mean` is therefore a weighted arithmetic
# mean (uniform weights reduce it to the plain mean), while `min`/`max` are
# order statistics that ignore weights (a weight can't sensibly scale a
# worst-/best-case gate).
_VALID_RUBRICS_REDUCTIONS = ("mean", "min", "max")


_DEFAULT_JUDGE_TASK = """\
You are scoring a model's prediction against a SINGLE evaluation rubric.

If gold-prefixed fields (e.g. `gold_content`) appear in the input, those are
the ground truth — compare the un-prefixed prediction fields against them. If
no gold-prefixed fields are present, judge the prediction on its own merits.\
"""

_SCORING_GUIDANCE = """\
Score ONLY how well the prediction satisfies the rubric above; ignore aspects
covered by other rubrics. Return a one-or-two sentence `critique` and a
`reward` of one of 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
where 1.0 means the rubric is fully satisfied and 0.0 means it is not met at
all. Partial credit is encouraged when the rubric is only partially met.\
"""


class RubricsJudgment(DataModel):
    """Output schema produced by `call` (what `ProgramAsJudge` reads).

    Unlike `CritiqueWithReward`, `reward` is an unconstrained float: the mixed
    score (a weighted mean / sum / ...) generally falls between the `Score`
    enum's 11 buckets, and snapping to the nearest bucket would discard
    precision. The `rubrics_reduction` is responsible for keeping it in range.
    """

    critique: str = Field(
        description="Per-rubric breakdown and the reduction that produced the reward",
    )
    reward: float = Field(
        description="The mixed reward across all rubrics (a float between 0.0 and 1.0)",
    )


def _build_rubric_instructions(user_task, rubric):
    """Instructions for the `SelfCritique` scoring a single `rubric`.

    `user_task` (or the default) is the shared preamble describing how to read
    the input (gold vs. judge-on-merits); the specific rubric and the 0.0–1.0
    scoring scale are appended so every panelist scores on the same scale while
    judging a different criterion.
    """
    task = (user_task or _DEFAULT_JUDGE_TASK).strip()
    return f"{task}\n\nRUBRIC:\n{rubric.strip()}\n\n{_SCORING_GUIDANCE}"


def _reduce(scores, weights, reduction):
    """Mix per-rubric `scores` (aligned with `weights`) into one reward.

    Operates only on the rubrics that returned a usable score; the caller drops
    failures before calling this, so `scores` is always non-empty here. All
    branches return a value in [0, 1] when the inputs are.
    """
    if reduction == "mean":
        # Weighted arithmetic mean; uniform weights reduce it to the plain
        # mean. All-zero surviving weights would divide by zero — fall back to
        # the plain mean so a degenerate weighting still yields a sensible
        # score.
        total_w = sum(weights)
        if total_w == 0:
            return sum(scores) / len(scores)
        return sum(w * s for w, s in zip(weights, scores)) / total_w
    if reduction == "min":
        return min(scores)
    if reduction == "max":
        return max(scores)
    # Unreachable: the reduction is validated at construction time.
    raise ValueError(f"Unknown rubrics_reduction {reduction!r}.")


class RubricsAsJudgeProgram(Program):
    """Inner judge program backing `RubricsAsJudge`.

    Builds one `SelfCritique` per rubric. In `call`, prefixes the gold side
    with `gold_`, concats it with the prediction, fans that single input out
    across the panel concurrently, then reads each critique's `reward`, weights
    them, and reduces inline into a single `RubricsJudgment`. When `y_true` is
    missing, the prediction is judged on its own merits.

    Rubrics whose `SelfCritique` fails (returns `None` or no `reward`) are
    dropped from the reduction — like `MultiJudgePanel` drops failed panelists
    — so a transient LLM glitch on one criterion doesn't sink the judgment. If
    every rubric fails, `call` returns `reward=0.0`.

    Example:

    ```python
    program = RubricsAsJudgeProgram(
        language_model="openai/gpt-4o",
        rubrics=[
            "Check that every factual claim in the answer is supported by the "
            "gold answer, with no fabricated or contradicted facts.",
            "Check that the answer addresses every part of the question, not "
            "just the easiest sub-question.",
            "Check that the answer is concise and well-structured, with no "
            "hedging or filler.",
        ],
        rubrics_weights=[0.6, 0.3, 0.1],
        rubrics_reduction="mean",
    )
    judgment = await program([y_true, y_pred])
    ```

    Args:
        language_model: The model scoring every rubric. Accepts a
            `LanguageModel`, a config dict, or a string identifier (e.g.
            `"openai/gpt-4o"`).
        rubrics (list): One instruction per rubric — each string becomes the
            `instructions` of its own SelfCritique, telling it the single
            criterion to score the prediction against. Required, non-empty.
        rubrics_weights (list): Optional. Per-rubric weights aligned with
            `rubrics`; must be the same length and non-negative. Defaults to
            equal weights (`1.0` each). Used only by the `mean` reduction
            (`min`/`max` ignore weights).
        rubrics_reduction (str): How to mix the per-rubric scores — one of
            `"mean"` (default, a weighted arithmetic mean), `"min"`, `"max"`.
        prompt_template (str): The default jinja2 prompt template forwarded to
            every `SelfCritique` (see `Generator`).
        examples (list): The default examples forwarded to every panelist.
        instructions (str): The shared judging preamble spliced before each
            rubric (the rubric text and 0.0–1.0 scale are appended
            automatically). Defaults to a generic "score vs. gold, or on its
            own merits" preamble.
        temperature (float): Sampling temperature for the panelists
            (default 0.0).
        reasoning_effort (str): Forwarded to the panelists (for
            reasoning-capable LMs).
        name (str): Optional. The name of the program.
        description (str): Optional. The description of the program.
        trainable (bool): Whether the program's variables should be trainable.
    """

    def __init__(
        self,
        language_model=None,
        rubrics=None,
        rubrics_weights=None,
        rubrics_reduction="mean",
        prompt_template=None,
        examples=None,
        instructions=None,
        temperature=0.0,
        reasoning_effort=None,
        name=None,
        description=None,
        trainable=True,
    ):
        super().__init__(
            name=name,
            description=description,
            trainable=trainable,
        )
        if not rubrics:
            raise ValueError("`rubrics` must be a non-empty list of criteria.")
        if rubrics_reduction not in _VALID_RUBRICS_REDUCTIONS:
            raise ValueError(
                f"Unknown rubrics_reduction {rubrics_reduction!r}. "
                f"Expected one of {', '.join(_VALID_RUBRICS_REDUCTIONS)}."
            )
        rubrics = list(rubrics)
        if rubrics_weights is None:
            rubrics_weights = [1.0] * len(rubrics)
        else:
            rubrics_weights = [float(w) for w in rubrics_weights]
            if len(rubrics_weights) != len(rubrics):
                raise ValueError(
                    "`rubrics_weights` must have the same length as `rubrics` "
                    f"({len(rubrics_weights)} != {len(rubrics)})."
                )
            if any(w < 0 for w in rubrics_weights):
                raise ValueError("`rubrics_weights` must all be non-negative.")
            if (
                rubrics_reduction == "mean"
                and sum(rubrics_weights) == 0
            ):
                raise ValueError(
                    "`rubrics_weights` must not all be zero when "
                    "`rubrics_reduction` is 'mean'."
                )

        # Resolve string / dict / instance identifiers up front, matching the
        # pattern used inside synalinks (e.g. `ChainOfThought`).
        language_model = _get_lm(language_model)

        # One panelist per rubric — same LM and prompt, rubric-specific
        # instructions, and `return_inputs=False` so each output is just the
        # `CritiqueWithReward` the reduction reads.
        self.critics = [
            SelfCritique(
                language_model=language_model,
                prompt_template=prompt_template,
                examples=examples,
                instructions=_build_rubric_instructions(instructions, rubric),
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                return_inputs=False,
                name=f"rubric_{i}_{self.name}",
            )
            for i, rubric in enumerate(rubrics)
        ]

        self.language_model = language_model
        self.rubrics = rubrics
        self.rubrics_weights = rubrics_weights
        self.rubrics_reduction = rubrics_reduction
        self.prompt_template = prompt_template
        self.examples = examples
        self.instructions = instructions
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort

    async def call(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            raise ValueError("The inputs should be a list or tuple.")
        if len(inputs) != 2:
            raise ValueError("The inputs of the program should have a length of 2.")
        y_true, y_pred = inputs
        if not y_pred:
            # ProgramAsJudge.call reads `result.get("reward", 0.0)` — return a
            # RubricsJudgment (matching the mixer's output schema) so the
            # wrapper's `.get()` call doesn't crash on a bare float.
            return RubricsJudgment(
                critique="empty prediction — nothing to judge",
                reward=0.0,
            )

        if y_true:
            y_true = await ops.prefix(y_true, prefix="gold", name="gold_y_true")
            judge_input = await ops.concat(y_true, y_pred, name="y_true_with_y_pred")
        else:
            judge_input = y_pred

        # Fan the single input out across the panel concurrently, then mix the
        # per-rubric rewards inline.
        critique_outputs = await asyncio.gather(
            *(critic(judge_input) for critic in self.critics)
        )

        kept_scores, kept_weights, lines = [], [], []
        for i, out in enumerate(critique_outputs):
            score = out.get("reward") if (out is not None and hasattr(out, "get")) else None
            crit = out.get("critique") if (out is not None and hasattr(out, "get")) else None
            if score is None:
                # Drop a rubric whose panelist failed (None / no reward) so a
                # transient glitch on one criterion doesn't sink the judgment.
                lines.append(f"- [dropped] {self.rubrics[i]}: no score returned")
                continue
            score = float(score)
            kept_scores.append(score)
            kept_weights.append(self.rubrics_weights[i])
            lines.append(
                f"- [{score:.2f} w={self.rubrics_weights[i]:g}] {self.rubrics[i]}: {crit}"
            )

        if not kept_scores:
            return RubricsJudgment(
                critique="all rubrics failed to return a score — scoring 0.0",
                reward=0.0,
            )

        reward = _reduce(kept_scores, kept_weights, self.rubrics_reduction)
        header = (
            f"rubrics_reduction={self.rubrics_reduction} over "
            f"{len(kept_scores)}/{len(self.rubrics)} rubric(s) -> {reward:.3f}"
        )
        return RubricsJudgment(
            critique=header + "\n" + "\n".join(lines),
            reward=float(reward),
        )

    def get_config(self):
        config = {
            "rubrics": self.rubrics,
            "rubrics_weights": self.rubrics_weights,
            "rubrics_reduction": self.rubrics_reduction,
            "prompt_template": self.prompt_template,
            "examples": self.examples,
            "instructions": self.instructions,
            "temperature": self.temperature,
            "reasoning_effort": self.reasoning_effort,
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


class RubricsAsJudge(ProgramAsJudge):
    """Rubrics-as-judge reward.

    Same surface as `synalinks.rewards.LMAsJudge`, but the prediction is scored
    against a *list of rubrics* — one criterion each, judged in parallel by a
    `SelfCritique` panel — and the per-rubric scores are mixed into one reward
    by `rubrics_reduction` (optionally weighted by `rubrics_weights`).

    Note `rubrics_reduction` (how per-rubric scores combine into one judgment)
    is distinct from the base `reduction` (how per-sample rewards combine over
    a batch); both are configurable.

    Example:

    ```python
    program.compile(
        reward=RubricsAsJudge(
            language_model="openai/gpt-4o",
            rubrics=[
                "Check that every factual claim in the answer is supported by "
                "the gold answer, with no fabricated or contradicted facts.",
                "Check that the answer addresses every part of the question, "
                "not just the easiest sub-question.",
                "Check that the answer is concise and well-structured, with no "
                "hedging or filler.",
            ],
            rubrics_weights=[0.6, 0.3, 0.1],
            rubrics_reduction="mean",
        ),
    )
    ```

    Args:
        language_model: The model scoring every rubric. Accepts a
            `LanguageModel`, a config dict, or a string identifier (e.g.
            `"openai/gpt-4o"`).
        rubrics (list): One instruction per rubric — each string becomes the
            `instructions` of its own SelfCritique, telling it the single
            criterion to score the prediction against. Required, non-empty.
        rubrics_weights (list): Optional. Per-rubric weights aligned with
            `rubrics`; same length, non-negative. Defaults to equal weights.
            Used only by the `mean` reduction (`min`/`max` ignore weights).
        rubrics_reduction (str): How to mix the per-rubric scores (all bounded
            in [0, 1], ordered `min` <= `mean` <= `max`) — one of `"mean"`
            (default, a weighted arithmetic mean), `"min"` (strict: every
            rubric must pass), `"max"` (lenient: any rubric suffices).
        prompt_template (str): The default jinja2 prompt template forwarded to
            every panelist.
        examples (list): The default examples forwarded to every panelist.
        instructions (str): The shared judging preamble spliced before each
            rubric (the rubric text and 0.0–1.0 scale are appended
            automatically). Defaults to a generic "score vs. gold, or on its
            own merits" preamble.
        temperature (float): Sampling temperature for the panelists
            (default 0.0).
        reasoning_effort (str): Forwarded to the panelists (for
            reasoning-capable LMs).
        name (str): Optional. string name of the reward instance
            (default `"rubrics_as_judge"`).
        in_mask (list): Optional. list of keys to keep to compute the reward.
        out_mask (list): Optional. list of keys to remove to compute the
            reward.
        reduction (str): Optional. How per-sample rewards are reduced over a
            batch (distinct from `rubrics_reduction`) — one of `"mean"`
            (default), `"sum"`, `"min"`, `"max"`, `"none"`.
        in_mask_pattern (str): Optional. Regex; fields whose names match are
            kept (OR-combined with `in_mask`).
        out_mask_pattern (str): Optional. Regex; fields whose names match are
            dropped (OR-combined with `out_mask`).
    """

    def __init__(
        self,
        language_model=None,
        rubrics=None,
        rubrics_weights=None,
        rubrics_reduction="mean",
        prompt_template=None,
        examples=None,
        instructions=None,
        temperature=0.0,
        reasoning_effort=None,
        name="rubrics_as_judge",
        in_mask=None,
        out_mask=None,
        reduction="mean",
        in_mask_pattern=None,
        out_mask_pattern=None,
        program=None,
    ):
        # `program` is normally built from the args above; it is accepted as a
        # kwarg only so `from_config` can pass the deserialized inner program
        # straight back in (round-tripping a saved reward).
        if program is None:
            program = RubricsAsJudgeProgram(
                language_model=language_model,
                rubrics=rubrics,
                rubrics_weights=rubrics_weights,
                rubrics_reduction=rubrics_reduction,
                prompt_template=prompt_template,
                examples=examples,
                instructions=instructions,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
            )
        super().__init__(
            program=program,
            name=name,
            reduction=reduction,
            in_mask=in_mask,
            out_mask=out_mask,
            in_mask_pattern=in_mask_pattern,
            out_mask_pattern=out_mask_pattern,
        )
