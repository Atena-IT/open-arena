# License Apache 2.0: (c) 2026 Athena-Reply

"""Agent-as-judge reward.

Mirrors `synalinks.rewards.LMAsJudge`, but the inner judging program is a plain
`FunctionCallingAgent` instead of a `SelfCritique`. The agent receives the
(y_true, y_pred) pair as structured fields, calls the `tools` you give it to
verify the prediction, and returns a structured judgment containing a `reward`
field — exactly what `ProgramAsJudge` reads.

This is the lightest agentic judge: a bare function-calling loop with no
sandbox and no built-in tools. Contrast with the siblings:

  - `agent_as_judge` (this one): plain `FunctionCallingAgent` — you supply every
    tool. Use when judging means *calling your tools* (a unit-test runner, a
    retrieval/search API, a domain validator) and you don't need a filesystem
    or a code REPL.
  - `deep_as_judge`: `DeepAgent` with built-in filesystem + Python-execution
    tools over a copy-on-write sandbox. Use when judging means *running code or
    files*.
  - `rlm_as_judge`: `RecursiveLanguageModelAgent` that writes code and delegates
    semantic sub-questions to a cheaper LM. Use for long / structured inputs
    the judge should carve up programmatically.

Because `FunctionCallingAgent` has no built-in tools, `tools` is **required**
(an agent with nothing to call would just be `lm_as_judge`). Bare callables are
accepted and wrapped into `Tool`s; `Tool` instances pass through unchanged.
"""

import synalinks
from synalinks.src import ops
from synalinks.src.modules.agents.function_calling_agent import FunctionCallingAgent
from synalinks.src.modules.core.tool import Tool
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

If gold-prefixed fields (e.g. `gold_content`) appear in the input, those are
the ground truth — compare the un-prefixed prediction fields against them and
score on closeness. If no gold-prefixed fields are present, judge the
prediction on its own merits (correctness, coherence, completeness given
whatever task the prediction is answering). Read the input first to see which
case you're in.

Call the tools available to you whenever they help you verify the prediction
rather than guessing; a direct comparison is fine for short literal answers.
When you're done, stop calling tools and return your judgment: a one-or-two
sentence `critique` and a `reward` of one of 0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
0.6, 0.7, 0.8, 0.9, 1.0, where 1.0 is a perfect / excellent answer and 0.0 is
wrong / poor. Partial credit is fine when the prediction is close but not
exact.\
"""


class AgentAsJudgeProgram(Program):
    """Inner judge program backing `AgentAsJudge`.

    Takes `[y_true, y_pred]`, prefixes the gold side with `gold_`, concats the
    two into a single structured input, and dispatches to a
    `FunctionCallingAgent` whose output schema is `CritiqueWithReward` (so it
    has the `reward` field that `ProgramAsJudge` reads). When `y_true` is
    missing, the prediction is judged on its own merits.

    Unlike `deep_as_judge` / `rlm_as_judge`, there is no built-in tool plan to
    preserve, so the user's `instructions` are the agent's instructions
    directly (no splicing).

    Example:

    ```python
    program = AgentAsJudgeProgram(
        language_model="openai/gpt-4o",
        tools=[run_unit_tests],
        instructions="Score 0.0–1.0: does the prediction pass run_unit_tests?",
        max_iterations=8,
    )
    judgment = await program([y_true, y_pred])
    ```

    Args:
        language_model: The model driving the agent loop and the final-answer
            step. Accepts a `LanguageModel`, a config dict, or a string
            identifier (e.g. `"openai/gpt-4o"`).
        tools (list): Tools the judge may call — callables or `Tool` instances
            (bare callables are wrapped into `Tool`s). Required and non-empty:
            a `FunctionCallingAgent` has no built-in tools.
        prompt_template (str): The default jinja2 prompt template forwarded to
            the inner tool-call generator (see `Generator`).
        examples (list): The default examples forwarded to the inner generator.
        instructions (str): The judging-task description used as the agent's
            instructions. Defaults to a generic "score 0.0–1.0 vs. gold (or on
            its own merits if no gold), call tools to verify" task.
        temperature (float): Sampling temperature for the inner generators
            (default 0.0).
        reasoning_effort (str): Forwarded to the generators (for
            reasoning-capable LMs).
        use_chain_of_thought (bool): When True, the tool-call generator emits a
            `thinking` field per round (default False).
        max_iterations (int): Max tool-call rounds per judgment (default 5).
        name (str): Optional. The name of the program.
        description (str): Optional. The description of the program.
        trainable (bool): Whether the program's variables should be trainable.
    """

    def __init__(
        self,
        language_model=None,
        tools=None,
        prompt_template=None,
        examples=None,
        instructions=None,
        temperature=0.0,
        reasoning_effort=None,
        use_chain_of_thought=False,
        max_iterations=5,
        name=None,
        description=None,
        trainable=True,
    ):
        super().__init__(
            name=name,
            description=description,
            trainable=trainable,
        )
        if not tools:
            raise ValueError(
                "`tools` must be a non-empty list — a FunctionCallingAgent has "
                "no built-in tools (an agent with nothing to call is just "
                "`lm_as_judge`)."
            )
        # Resolve string / dict / instance identifiers up front, matching the
        # pattern used inside synalinks (e.g. `ChainOfThought`).
        language_model = _get_lm(language_model)

        # Normalize tools to `Tool` instances so a bare callable works as well
        # as a pre-built `Tool`, and `self.tools` serializes uniformly.
        tools = [t if isinstance(t, Tool) else Tool(t) for t in tools]

        self.judge = FunctionCallingAgent(
            data_model=CritiqueWithReward,
            language_model=language_model,
            tools=tools,
            prompt_template=prompt_template,
            examples=examples,
            instructions=(instructions or _DEFAULT_JUDGE_TASK).strip(),
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            use_chain_of_thought=use_chain_of_thought,
            max_iterations=max_iterations,
            return_inputs_with_trajectory=False,
            name="agent_judge_" + self.name,
        )
        self.language_model = language_model
        self.tools = tools
        self.prompt_template = prompt_template
        self.examples = examples
        self.instructions = instructions
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.use_chain_of_thought = use_chain_of_thought
        self.max_iterations = max_iterations

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
            "temperature": self.temperature,
            "reasoning_effort": self.reasoning_effort,
            "use_chain_of_thought": self.use_chain_of_thought,
            "max_iterations": self.max_iterations,
            "name": self.name,
            "description": self.description,
            "trainable": self.trainable,
        }
        lm_config = {
            "language_model": serialization_lib.serialize_synalinks_object(
                self.language_model
            ),
        }
        # `self.tools` is already a list of `Tool`s (normalized in __init__).
        tools_config = {
            "tools": [
                serialization_lib.serialize_synalinks_object(t) for t in self.tools
            ],
        }
        return {**lm_config, **config, **tools_config}

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        if "language_model" in config:
            config["language_model"] = serialization_lib.deserialize_synalinks_object(
                config["language_model"]
            )
        config["tools"] = [
            serialization_lib.deserialize_synalinks_object(t)
            for t in config.pop("tools", [])
        ] or None
        return cls(**config)


class AgentAsJudge(ProgramAsJudge):
    """Agent-as-judge reward.

    Same surface as `synalinks.rewards.LMAsJudge`, but the inner judge is a
    plain `FunctionCallingAgent` — the lightest agentic judge, with no sandbox
    and no built-in tools. Use it when scoring means *calling the tools you
    provide* (a unit-test runner, a search/retrieval API, a domain validator).
    For a filesystem + code sandbox use `deep_as_judge`; for a code/recursive
    judge use `rlm_as_judge`.

    Example:

    ```python
    program.compile(
        reward=AgentAsJudge(
            language_model="openai/gpt-4o",
            tools=[run_unit_tests],
            instructions="Score 0.0–1.0: does the prediction pass the tests?",
        ),
    )
    ```

    Args:
        language_model: The model driving the agent loop and the final-answer
            step. Accepts a `LanguageModel`, a config dict, or a string
            identifier (e.g. `"openai/gpt-4o"`).
        tools (list): Tools the judge may call — callables or `Tool` instances
            (bare callables are wrapped into `Tool`s). Required and non-empty.
        prompt_template (str): The default jinja2 prompt template forwarded to
            the inner tool-call generator.
        examples (list): The default examples forwarded to the inner generator.
        instructions (str): The judging-task description used as the agent's
            instructions. Defaults to a generic "score 0.0–1.0 vs. gold, call
            tools to verify" task.
        temperature (float): Sampling temperature for the inner generators
            (default 0.0).
        reasoning_effort (str): Forwarded to the generators (for
            reasoning-capable LMs).
        use_chain_of_thought (bool): When True, the tool-call generator emits a
            `thinking` field per round (default False).
        max_iterations (int): Max tool-call rounds per judgment (default 5).
        name (str): Optional. string name of the reward instance
            (default `"agent_as_judge"`).
        in_mask (list): Optional. list of keys to keep to compute the reward.
        out_mask (list): Optional. list of keys to remove to compute the
            reward.
        reduction (str): Optional. How per-sample rewards are reduced over a
            batch — one of `"mean"` (default), `"sum"`, `"min"`, `"max"`,
            `"none"`.
        in_mask_pattern (str): Optional. Regex; fields whose names match are
            kept (OR-combined with `in_mask`).
        out_mask_pattern (str): Optional. Regex; fields whose names match are
            dropped (OR-combined with `out_mask`).
    """

    def __init__(
        self,
        language_model=None,
        tools=None,
        prompt_template=None,
        examples=None,
        instructions=None,
        temperature=0.0,
        reasoning_effort=None,
        use_chain_of_thought=False,
        max_iterations=5,
        name="agent_as_judge",
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
            program = AgentAsJudgeProgram(
                language_model=language_model,
                tools=tools,
                prompt_template=prompt_template,
                examples=examples,
                instructions=instructions,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                use_chain_of_thought=use_chain_of_thought,
                max_iterations=max_iterations,
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
