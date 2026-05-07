# License Apache 2.0: (c) 2026 Athena-Reply

"""RLM-as-judge reward.

Mirrors `synalinks.rewards.LMAsJudge`, but the inner judging program is a
`RecursiveLanguageModelAgent` instead of a `SelfCritique`. The agent treats
the (y_true, y_pred) pair as an environment to inspect with code, recursively
delegates semantic comparisons to a sub-LM, and submits a structured
judgment containing a `reward` field — exactly what `ProgramAsJudge` reads.

Use it for tasks where a single-shot LLM judge has too much context to chew
on (long passages, structured outputs with many fields, code) and where the
judge would benefit from carving the inputs up programmatically before
asking semantic questions.
"""

import synalinks
from synalinks.src import ops
from synalinks.src.modules.agents.recursive_language_model_agent import (
    RecursiveLanguageModelAgent,
)
from synalinks.src.modules.agents.recursive_language_model_agent import (
    get_default_instructions as _get_rlm_default_instructions,
)
from synalinks.src.modules.language_models import get as _get_lm
from synalinks.src.modules.ttc.self_critique import CritiqueWithReward
from synalinks.src.programs import Program
from synalinks.src.rewards.reward_wrappers import ProgramAsJudge
from synalinks.src.saving import serialization_lib

# `CritiqueWithReward` is the synalinks-canonical judge-output schema
# (`critique: str` + `reward: synalinks.Score`). Reusing it gives schema-
# level [0, 1] enforcement via the `Score` enum (the LM is forced via
# structured output to pick one of `Score`'s 11 buckets) instead of
# accepting any unbounded float.


_DEFAULT_JUDGE_TASK = """\
You are scoring a model's prediction.

If gold-prefixed fields (e.g. `gold_content`) appear in `inputs`, those are
the ground truth — compare the un-prefixed prediction fields against them
and score on closeness. If no gold-prefixed fields are present, judge the
prediction on its own merits (correctness, coherence, completeness given
whatever task the prediction is answering). Probe `inputs` first to see
which case you're in.

Inspect with code, use `llm_query` only when semantic comparison or
judgment on a specific field is actually needed (a Python equality check
is fine for short literal answers), and submit
`{"critique": <one-or-two sentences>, "reward": <one of 0.0, 0.1, 0.2,
0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0>}` where 1.0 is a perfect /
excellent answer and 0.0 is wrong / poor. Partial credit is fine when
the prediction is close but not exact.\
"""


def _build_judge_instructions(user_task, max_llm_calls):
    """Combine the RLM workflow guide with the judging task.

    The agent's per-turn code generator needs the workflow instructions
    (how to use code-mode, llm_query, submit, ...) to function at all;
    `instructions=` on `RecursiveLanguageModelAgent` *overrides* those
    rather than appending, so we have to splice the user's judging task
    onto the workflow text ourselves.
    """
    base = _get_rlm_default_instructions().replace(
        "{max_llm_calls}", str(max_llm_calls)
    )
    task = (user_task or _DEFAULT_JUDGE_TASK).strip()
    return f"{base}\n\nJUDGING TASK:\n{task}"


class RecursiveLMAsJudgeProgram(Program):
    """Inner judge program backing `RecursiveLMAsJudge`.

    Takes `[y_true, y_pred]`, prefixes the gold side with `gold_`, concats
    the two into a single structured input, and dispatches to a
    `RecursiveLanguageModelAgent` whose output schema is `CritiqueWithReward`
    (so it has the `reward` field that `ProgramAsJudge` reads). When
    `y_true` is missing, the prediction is judged on its own merits.

    The user's `instructions` are spliced onto the RLM workflow
    instructions via `_build_judge_instructions` rather than replacing
    them, since the agent needs the workflow guide (code-mode,
    `llm_query`, `submit`, ...) to function at all.

    Example:

    ```python
    program = RecursiveLMAsJudgeProgram(
        language_model="openai/gpt-4o",
        sub_language_model="openai/gpt-4o-mini",
        instructions="Score 0.0–1.0 on factual correctness.",
        max_iterations=8,
        max_llm_calls=10,
    )
    judgment = await program([y_true, y_pred])
    ```

    Args:
        language_model: The primary model driving the per-turn code
            generator and the final-answer step. Accepts a `LanguageModel`,
            a config dict, or a string identifier (e.g. `"openai/gpt-4o"`).
        sub_language_model: Optional. The model used for recursive
            `llm_query` / `llm_query_batched` calls. Same forms as
            `language_model`. Defaults to `language_model`.
        prompt_template (str): The default jinja2 prompt template
            forwarded to the inner generator (see `Generator`).
        examples (list): The default examples forwarded to the inner
            generator.
        instructions (str): The judging-task description spliced into the
            agent's workflow instructions. Defaults to a generic
            "score 0.0–1.0 vs. gold (or on its own merits if no gold)"
            task.
        max_iterations (int): Max code-execution turns per judgment
            (default 20).
        max_llm_calls (int): Hard cap on sub-LM calls per judgment,
            shared between `llm_query` and `llm_query_batched`
            (default 50).
        timeout (int): Per-turn execution budget in seconds (default 60).
        max_output_chars (int): REPL-output cap per turn
            (default 10 000).
        temperature (float): Sampling temperature for the inner
            generators (default 0.0).
        name (str): Optional. The name of the program.
        description (str): Optional. The description of the program.
        trainable (bool): Whether the program's variables should be
            trainable.
    """

    def __init__(
        self,
        language_model=None,
        sub_language_model=None,
        prompt_template=None,
        examples=None,
        instructions=None,
        max_iterations=20,
        max_llm_calls=50,
        timeout=60,
        max_output_chars=10_000,
        temperature=0.0,
        name=None,
        description=None,
        trainable=True,
    ):
        super().__init__(
            name=name,
            description=description,
            trainable=trainable,
        )
        # Resolve string / dict / instance identifiers up front, matching
        # the pattern used inside synalinks (e.g. `ChainOfThought`).
        language_model = _get_lm(language_model)
        sub_language_model = (
            _get_lm(sub_language_model) if sub_language_model is not None else language_model
        )
        self.judge = RecursiveLanguageModelAgent(
            data_model=CritiqueWithReward,
            language_model=language_model,
            sub_language_model=sub_language_model,
            prompt_template=prompt_template,
            examples=examples,
            instructions=_build_judge_instructions(instructions, max_llm_calls),
            temperature=temperature,
            max_iterations=max_iterations,
            max_llm_calls=max_llm_calls,
            timeout=timeout,
            max_output_chars=max_output_chars,
            return_inputs_with_trajectory=False,
            name="rlm_judge_" + self.name,
        )
        self.language_model = language_model
        self.sub_language_model = sub_language_model
        self.prompt_template = prompt_template
        self.examples = examples
        self.instructions = instructions
        self.max_iterations = max_iterations
        self.max_llm_calls = max_llm_calls
        self.timeout = timeout
        self.max_output_chars = max_output_chars
        self.temperature = temperature

    async def call(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            raise ValueError("The inputs should be a list or tuple.")
        if len(inputs) != 2:
            raise ValueError("The inputs of the program should have a length of 2.")
        y_true, y_pred = inputs
        if not y_pred:
            # ProgramAsJudge.call reads `result.get("reward", 0.0)` — return a
            # CritiqueWithReward-shaped object (matching the success path) so
            # the wrapper's `.get()` call doesn't crash on a bare float.
            return CritiqueWithReward(
                critique="empty prediction — nothing to judge",
                reward=synalinks.Score.VERY_BAD,
            )
        if y_true:
            y_true = await ops.prefix(y_true, prefix="gold", name="gold_y_true")
            return await self.judge(
                await ops.concat(y_true, y_pred, name="y_true_with_y_pred")
            )
        return await self.judge(y_pred)

    def get_config(self):
        config = {
            "prompt_template": self.prompt_template,
            "examples": self.examples,
            "instructions": self.instructions,
            "max_iterations": self.max_iterations,
            "max_llm_calls": self.max_llm_calls,
            "timeout": self.timeout,
            "max_output_chars": self.max_output_chars,
            "temperature": self.temperature,
            "name": self.name,
            "description": self.description,
            "trainable": self.trainable,
        }
        lm_config = {
            "language_model": serialization_lib.serialize_synalinks_object(
                self.language_model
            ),
            "sub_language_model": serialization_lib.serialize_synalinks_object(
                self.sub_language_model
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
        if "sub_language_model" in config:
            config["sub_language_model"] = serialization_lib.deserialize_synalinks_object(
                config["sub_language_model"]
            )
        return cls(**config)


class RecursiveLMAsJudge(ProgramAsJudge):
    """Recursive-LM-as-judge reward.

    Same surface as `synalinks.rewards.LMAsJudge`, but the inner judge is a
    `RecursiveLanguageModelAgent`. Use `language_model` for the primary
    (code-generating) LM and the optional `sub_language_model` for the
    cheaper LM that handles `llm_query` / `llm_query_batched` calls; the
    latter defaults to `language_model`.

    Example:

    ```python
    program.compile(
        reward=RecursiveLMAsJudge(
            language_model="openai/gpt-4o",
            sub_language_model="openai/gpt-4o-mini",
            instructions="Score the assistant's answer 0.0–1.0 based on factual correctness.",
        ),
    )
    ```

    Args:
        language_model: The primary model driving the per-turn code
            generator and the final-answer step. Accepts a `LanguageModel`,
            a config dict, or a string identifier (e.g. `"openai/gpt-4o"`).
        sub_language_model: Optional. The model used for recursive
            `llm_query` calls. Same forms as `language_model`. Defaults
            to `language_model`.
        prompt_template (str): The default jinja2 prompt template
            forwarded to the inner generator (see `Generator`).
        examples (list): The default examples forwarded to the inner
            generator.
        instructions (str): The judging-task description spliced into the
            agent's workflow instructions. Defaults to a generic
            "score 0.0–1.0 vs. gold" task.
        max_iterations (int): Max code-execution turns per judgment
            (default 20).
        max_llm_calls (int): Hard cap on sub-LM calls per judgment,
            shared between `llm_query` and `llm_query_batched`
            (default 50).
        timeout (int): Per-turn execution budget in seconds (default 60).
        max_output_chars (int): REPL-output cap per turn
            (default 10 000).
        temperature (float): Sampling temperature for the inner
            generators (default 0.0).
        name (str): Optional. string name of the reward instance
            (default `"recursive_lm_as_judge"`).
        in_mask (list): Optional. list of keys to keep to compute the
            reward.
        out_mask (list): Optional. list of keys to remove to compute the
            reward.
    """

    def __init__(
        self,
        language_model=None,
        sub_language_model=None,
        prompt_template=None,
        examples=None,
        instructions=None,
        max_iterations=20,
        max_llm_calls=50,
        timeout=60,
        max_output_chars=10_000,
        temperature=0.0,
        name="recursive_lm_as_judge",
        in_mask=None,
        out_mask=None,
    ):
        program = RecursiveLMAsJudgeProgram(
            language_model=language_model,
            sub_language_model=sub_language_model,
            prompt_template=prompt_template,
            examples=examples,
            instructions=instructions,
            max_iterations=max_iterations,
            max_llm_calls=max_llm_calls,
            timeout=timeout,
            max_output_chars=max_output_chars,
            temperature=temperature,
        )
        super().__init__(
            program=program,
            name=name,
            in_mask=in_mask,
            out_mask=out_mask,
        )
