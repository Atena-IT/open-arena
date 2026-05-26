# License Apache 2.0: (c) 2026 Athena-Reply

"""Deep-agent-as-judge reward.

Mirrors `synalinks.rewards.LMAsJudge`, but the inner judging program is a
`DeepAgent` instead of a `SelfCritique`. The agent receives the
(y_true, y_pred) pair as structured fields in its prompt and, on top of that,
gets filesystem + Python-execution tools (`read_file`, `list_files`,
`search_files`, `write_file`, `edit_file`, `run_python_code`,
`run_python_file`) backed by a sandboxed, copy-on-write workspace. It can
therefore *verify* a prediction empirically — write the predicted code to a
file and run it, execute a quick computation to check a numeric answer, or
diff the prediction against gold files living in the workdir — before
submitting a structured judgment containing a `reward` field, exactly what
`ProgramAsJudge` reads.

Use it for tasks where judging means *running something*: code generation
(does it compile / pass tests?), data transforms (does the output match when
re-derived?), or any case where ground truth is easier to check by execution
than by reading. For pure semantic comparison of long text, prefer
`rlm_as_judge` (recursive-LM judge) or a plain `lm_as_judge`.

## Sandboxing

As of synalinks 0.8.32 `DeepAgent` is **host-safe by construction**: its tools
run against a `MontySandbox` copy-on-write overlay over the workdir. Reads
fall through to the real workdir, but the agent's writes, edits and code
execution land only in an in-memory overlay and can never modify the workdir
or reach the host. There is therefore nothing to gate (the old
`allow_write` / `allow_bash` knobs are gone) and no scratch directory to clean
up — a `workdir=None` judge runs entirely in an in-memory filesystem.
"""

import synalinks
from synalinks.src import ops
from synalinks.src.modules.agents.deep_agent import DeepAgent
from synalinks.src.modules.agents.deep_agent import (
    get_default_instructions as _get_deep_default_instructions,
)
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

You have filesystem + Python-execution tools backed by a sandboxed,
copy-on-write workspace. Use them when verification is easier by *running
something* than by reading: write the predicted code to a file and run it,
execute a quick computation to check a numeric answer, or compare against gold
files if any were placed in the workdir. A direct comparison is fine for short
literal answers — don't reach for code execution when a string check settles
it.

When you're done, stop calling tools and return your judgment: a one-or-two
sentence `critique` and a `reward` of one of 0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
0.6, 0.7, 0.8, 0.9, 1.0, where 1.0 is a perfect / excellent answer and 0.0 is
wrong / poor. Partial credit is fine when the prediction is close but not
exact.\
"""


def _build_judge_instructions(user_task, workdir):
    """Combine the deep-agent tool plan with the judging task.

    `instructions=` on `DeepAgent` *overrides* the default tool-plan
    instructions (it flows straight to the inner `FunctionCallingAgent`)
    rather than appending, so to keep the workdir/tool guidance the agent
    relies on we rebuild that default ourselves and splice the user's
    judging task onto it.

    `get_default_instructions` takes only the `workdir` now (the
    `allow_write` / `allow_bash` flags it used to branch on are gone — the
    copy-on-write sandbox makes every tool always safe to expose).
    """
    base = _get_deep_default_instructions(workdir)
    task = (user_task or _DEFAULT_JUDGE_TASK).strip()
    return f"{base}\n\nJUDGING TASK:\n{task}"


class DeepAsJudgeProgram(Program):
    """Inner judge program backing `DeepAsJudge`.

    Takes `[y_true, y_pred]`, prefixes the gold side with `gold_`, concats
    the two into a single structured input, and dispatches to a `DeepAgent`
    whose output schema is `CritiqueWithReward` (so it has the `reward` field
    that `ProgramAsJudge` reads). When `y_true` is missing, the prediction is
    judged on its own merits.

    The user's `instructions` are spliced onto the deep-agent tool-plan
    instructions via `_build_judge_instructions` rather than replacing them,
    since `DeepAgent` would otherwise lose the workdir/tool guidance.

    The workdir is optional and passed straight through to `DeepAgent`. When
    omitted, the agent's sandbox is a fresh in-memory filesystem (no temp
    directory, nothing to clean up). When you pass an explicit `workdir`, it
    is mounted read-through in the sandbox: that's how you hand the judge gold
    files or a checked-out project to run tests against, and because the
    sandbox is copy-on-write the agent's writes never mutate it.

    Example:

    ```python
    program = DeepAsJudgeProgram(
        language_model="openai/gpt-4o",
        instructions="Score 0.0–1.0: does the predicted function pass the "
                     "tests in tests/?",
        workdir="/path/to/checked-out/task",
        max_iterations=15,
    )
    judgment = await program([y_true, y_pred])
    ```

    Args:
        language_model: The model driving the agent loop and the
            final-answer step. Accepts a `LanguageModel`, a config dict, or a
            string identifier (e.g. `"openai/gpt-4o"`).
        workdir (str): Optional. Directory mounted read-through in the
            agent's copy-on-write sandbox (must exist when provided). Defaults
            to `None` — an empty in-memory workspace.
        timeout (float): Per-snippet execution budget in seconds for
            `run_python_code` / `run_python_file` (default 30).
        prompt_template (str): The default jinja2 prompt template forwarded
            to the inner tool-call generator (see `Generator`).
        examples (list): The default examples forwarded to the inner
            generator.
        instructions (str): The judging-task description spliced into the
            agent's tool-plan instructions. Defaults to a generic "score
            0.0–1.0 vs. gold (or on its own merits if no gold), verify by
            running when useful" task.
        temperature (float): Sampling temperature for the inner generators
            (default 0.0).
        reasoning_effort (str): Forwarded to the generators (for
            reasoning-capable LMs).
        use_chain_of_thought (bool): When True, the tool-call generator emits
            a `thinking` field per round (default False).
        max_iterations (int): Max tool-call rounds per judgment (default 10).
        tools (list): Optional. Extra tools (callables or `Tool` instances)
            exposed to the judging agent on top of its built-in filesystem +
            Python-execution tools — e.g. a unit-test runner or a domain API
            to verify a prediction against. Defaults to `None` (built-ins
            only).
        name (str): Optional. The name of the program.
        description (str): Optional. The description of the program.
        trainable (bool): Whether the program's variables should be
            trainable.
    """

    def __init__(
        self,
        language_model=None,
        workdir=None,
        timeout=30.0,
        prompt_template=None,
        examples=None,
        instructions=None,
        temperature=0.0,
        reasoning_effort=None,
        use_chain_of_thought=False,
        max_iterations=10,
        tools=None,
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

        # Normalize tools to `Tool` instances up front so a bare callable works
        # for both agentic judges (DeepAgent auto-wraps, the RLM agent does
        # not), and so `self.tools` serializes uniformly.
        tools = (
            [t if isinstance(t, Tool) else Tool(t) for t in tools]
            if tools
            else None
        )

        # workdir is passed straight through: `DeepAgent` treats `None` as a
        # fresh in-memory sandbox and a real path as a read-through mount of a
        # copy-on-write overlay (the LM's writes never touch it), so there is
        # no scratch directory for us to create or clean up.
        self.judge = DeepAgent(
            data_model=CritiqueWithReward,
            language_model=language_model,
            workdir=workdir,
            timeout=timeout,
            tools=tools,
            prompt_template=prompt_template,
            examples=examples,
            instructions=_build_judge_instructions(instructions, workdir),
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            use_chain_of_thought=use_chain_of_thought,
            max_iterations=max_iterations,
            return_inputs_with_trajectory=False,
            name="deep_judge_" + self.name,
        )
        self.language_model = language_model
        self.workdir = workdir
        self.timeout = timeout
        self.prompt_template = prompt_template
        self.examples = examples
        self.instructions = instructions
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.use_chain_of_thought = use_chain_of_thought
        self.max_iterations = max_iterations
        self.tools = tools

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
            # `None` round-trips as "auto-create a fresh in-memory sandbox".
            "workdir": self.workdir,
            "timeout": self.timeout,
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
        # Mirror `DeepAgent`'s own tool serialization: wrap bare callables in
        # `Tool` so they round-trip the same way the agent's do.
        tools_config = {
            "tools": [
                serialization_lib.serialize_synalinks_object(
                    t if isinstance(t, Tool) else Tool(t)
                )
                for t in (self.tools or [])
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


class DeepAsJudge(ProgramAsJudge):
    """Deep-agent-as-judge reward.

    Same surface as `synalinks.rewards.LMAsJudge`, but the inner judge is a
    `DeepAgent` with filesystem + Python-execution tools backed by a
    sandboxed, copy-on-write workspace. Use it when scoring means running
    something — code generation, data transforms, anything where ground truth
    is easier to check by execution than by reading.

    Example:

    ```python
    program.compile(
        reward=DeepAsJudge(
            language_model="openai/gpt-4o",
            instructions="Score 0.0–1.0: does the predicted function pass "
                         "the tests in tests/?",
            workdir="/path/to/checked-out/task",
        ),
    )
    ```

    Args:
        language_model: The model driving the agent loop and the
            final-answer step. Accepts a `LanguageModel`, a config dict, or a
            string identifier (e.g. `"openai/gpt-4o"`).
        workdir (str): Optional. Directory mounted read-through in the
            agent's copy-on-write sandbox (must exist when provided). Defaults
            to `None` — an empty in-memory workspace.
        timeout (float): Per-snippet execution budget in seconds for
            `run_python_code` / `run_python_file` (default 30).
        prompt_template (str): The default jinja2 prompt template forwarded
            to the inner tool-call generator (see `Generator`).
        examples (list): The default examples forwarded to the inner
            generator.
        instructions (str): The judging-task description spliced into the
            agent's tool-plan instructions. Defaults to a generic "score
            0.0–1.0 vs. gold, verify by running when useful" task.
        temperature (float): Sampling temperature for the inner generators
            (default 0.0).
        reasoning_effort (str): Forwarded to the generators (for
            reasoning-capable LMs).
        use_chain_of_thought (bool): When True, the tool-call generator emits
            a `thinking` field per round (default False).
        max_iterations (int): Max tool-call rounds per judgment (default 10).
        tools (list): Optional. Extra tools (callables or `Tool` instances)
            exposed to the judging agent on top of its built-in filesystem +
            Python-execution tools. Defaults to `None` (built-ins only).
        name (str): Optional. string name of the reward instance
            (default `"deep_as_judge"`).
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
        workdir=None,
        timeout=30.0,
        prompt_template=None,
        examples=None,
        instructions=None,
        temperature=0.0,
        reasoning_effort=None,
        use_chain_of_thought=False,
        max_iterations=10,
        tools=None,
        name="deep_as_judge",
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
            program = DeepAsJudgeProgram(
                language_model=language_model,
                workdir=workdir,
                timeout=timeout,
                prompt_template=prompt_template,
                examples=examples,
                instructions=instructions,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                use_chain_of_thought=use_chain_of_thought,
                max_iterations=max_iterations,
                tools=tools,
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
