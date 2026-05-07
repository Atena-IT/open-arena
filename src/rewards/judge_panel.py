# License Apache 2.0: (c) 2026 Athena-Reply

"""Judge-panel reward.

Multiple small `LanguageModel`s judge each (y_true, y_pred) pair in parallel.
If the panelists agree (max-min reward spread within `agreement_threshold`),
the panelist whose score is closest to the mean wins and its verdict is
returned. If they disagree, a separate `smart_language_model` is invoked
and its verdict overrides the panel.

Cost per example: M small-LM calls always, +1 smart-LM call only on
disagreement. Cheaper than a smart-LM-on-everything judge while still
catching the cases where the small judges genuinely diverge.
"""

import asyncio

import synalinks
from synalinks.src import ops
from synalinks.src.modules import SelfCritique
from synalinks.src.modules.language_models import get as _get_lm
from synalinks.src.modules.ttc.self_critique import CritiqueWithReward
from synalinks.src.programs import Program
from synalinks.src.rewards.reward_wrappers import ProgramAsJudge
from synalinks.src.saving import serialization_lib

# `CritiqueWithReward` is the synalinks-canonical judge-output schema and
# what `SelfCritique` (each panelist) already produces on the success path.
# Reusing it for the empty-prediction sentinel keeps both paths
# schema-aligned and inherits Score-enum [0, 1] enforcement for free.


class JudgePanelProgram(Program):
    """Inner judge program backing `JudgePanel`.

    Builds one `SelfCritique` per panelist plus one for the smart
    escalator. In `call`, prefixes the gold side with `gold_`, concats it
    with the prediction, and runs the panel concurrently via
    `asyncio.gather`. Aggregates by score spread:

      - `spread <= agreement_threshold`: return the panelist whose
        `reward` is closest to the mean (its full output, so the
        `critique` survives in the trajectory).
      - otherwise: dispatch to the smart judge and return its output.

    Panelists that fail to return a well-formed `reward` field are
    silently dropped; if the entire panel fails, the smart judge is the
    fallback.

    Example:

    ```python
    program = JudgePanelProgram(
        panel_language_models=["ollama/llama3.2", "ollama/mistral"],
        smart_language_model="openai/gpt-4o",
        agreement_threshold=0.2,
        instructions="Score 0.0–1.0 on factual correctness.",
    )
    judgment = await program([y_true, y_pred])
    ```

    Args:
        panel_language_models (list): The panelists, each given as a
            `LanguageModel`, a config dict, or a string identifier (e.g.
            `"ollama/llama3.2"`). Required, non-empty.
        smart_language_model: Escalation model used on disagreement. Same
            forms as panelists (instance / dict / string identifier).
            Required.
        agreement_threshold (float): Maximum allowed (max - min) panel
            score spread for the panel to be deemed "in agreement"
            (default 0.2).
        prompt_template (str): The default jinja2 prompt template
            forwarded to every `SelfCritique` (panelists + smart).
        examples (list): The default examples forwarded to every judge.
        instructions (str): The default judging-task instructions
            forwarded to every judge (same prompt, different LMs).
        name (str): Optional. The name of the program.
        description (str): Optional. The description of the program.
        trainable (bool): Whether the program's variables should be
            trainable.
    """

    def __init__(
        self,
        panel_language_models=None,
        smart_language_model=None,
        agreement_threshold=0.2,
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
        if not panel_language_models:
            raise ValueError("`panel_language_models` must be a non-empty list.")
        if smart_language_model is None:
            raise ValueError("`smart_language_model` is required.")

        # Resolve string / dict / instance identifiers up front, matching
        # the pattern used inside synalinks (e.g. `ChainOfThought`).
        panel_lms = [_get_lm(lm) for lm in panel_language_models]
        smart_lm = _get_lm(smart_language_model)

        self.panel = [
            SelfCritique(
                language_model=lm,
                prompt_template=prompt_template,
                examples=examples,
                instructions=instructions,
                name=f"panelist_{i}_{self.name}",
            )
            for i, lm in enumerate(panel_lms)
        ]
        self.smart = SelfCritique(
            language_model=smart_lm,
            prompt_template=prompt_template,
            examples=examples,
            instructions=instructions,
            name=f"smart_{self.name}",
        )
        self.agreement_threshold = float(agreement_threshold)
        self.panel_language_models = panel_lms
        self.smart_language_model = smart_lm
        self.prompt_template = prompt_template
        self.examples = examples
        self.instructions = instructions

    async def call(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            raise ValueError("The inputs should be a list or tuple.")
        if len(inputs) != 2:
            raise ValueError("The inputs of the program should have a length of 2.")
        y_true, y_pred = inputs
        if not y_pred:
            # ProgramAsJudge.call reads `result.get("reward", 0.0)` — return a
            # CritiqueWithReward (matching what each SelfCritique panelist
            # produces on the success path) so the wrapper's `.get()` call
            # doesn't crash on a bare float.
            return CritiqueWithReward(
                critique="empty prediction — nothing to judge",
                reward=synalinks.Score.VERY_BAD,
            )

        if y_true:
            y_true = await ops.prefix(y_true, prefix="gold", name="gold_y_true")
            judge_input = await ops.concat(y_true, y_pred, name="y_true_with_y_pred")
        else:
            judge_input = y_pred

        # Run the panel concurrently.
        panel_outputs = await asyncio.gather(*(p(judge_input) for p in self.panel))

        # Collect (score, output) for panelists that returned a well-formed
        # reward — silently drop None / missing-reward results so a single
        # LLM glitch doesn't sink the whole panel.
        scored = []
        for out in panel_outputs:
            if out is None:
                continue
            r = out.get("reward") if hasattr(out, "get") else None
            if r is None:
                continue
            scored.append((float(r), out))

        if not scored:
            # Panel failed end-to-end — fall back to the smart judge.
            return await self.smart(judge_input)

        scores = [s for s, _ in scored]
        spread = max(scores) - min(scores)

        if spread <= self.agreement_threshold:
            # Agreement — return the panelist whose score is nearest the
            # mean. Its `reward` becomes the trial's score, its `critique`
            # text shows up in the trajectory.
            mean_score = sum(scores) / len(scores)
            _, out = min(scored, key=lambda so: abs(so[0] - mean_score))
            return out

        # Disagreement — escalate.
        return await self.smart(judge_input)

    def get_config(self):
        config = {
            "agreement_threshold": self.agreement_threshold,
            "prompt_template": self.prompt_template,
            "examples": self.examples,
            "instructions": self.instructions,
            "name": self.name,
            "description": self.description,
            "trainable": self.trainable,
        }
        lm_config = {
            "panel_language_models": [
                serialization_lib.serialize_synalinks_object(lm)
                for lm in self.panel_language_models
            ],
            "smart_language_model": serialization_lib.serialize_synalinks_object(
                self.smart_language_model
            ),
        }
        return {**lm_config, **config}

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        if "panel_language_models" in config:
            config["panel_language_models"] = [
                serialization_lib.deserialize_synalinks_object(lm)
                for lm in config["panel_language_models"]
            ]
        if "smart_language_model" in config:
            config["smart_language_model"] = (
                serialization_lib.deserialize_synalinks_object(
                    config["smart_language_model"]
                )
            )
        return cls(**config)


class JudgePanel(ProgramAsJudge):
    """Judge panel: multiple small LMs vote, smart LM breaks ties.

    Each (y_true, y_pred) is scored in parallel by every model in
    `panel_language_models`. If the spread between the lowest and highest
    panel scores is `<= agreement_threshold`, the panel is treated as in
    agreement and the panelist whose score is closest to the mean wins. If
    the spread exceeds the threshold, `smart_language_model` is invoked
    and its verdict overrides the panel.

    Example:

    ```python
    program.compile(
        reward=JudgePanel(
            panel_language_models=[
                "ollama/llama3.2",
                "ollama/mistral",
                "ollama/qwen",
            ],
            smart_language_model="openai/gpt-4o",
            agreement_threshold=0.2,
            instructions="Score 0.0–1.0 on factual correctness.",
        ),
    )
    ```

    Args:
        panel_language_models (list): The panelists, each given as a
            `LanguageModel`, a config dict, or a string identifier
            (e.g. `"ollama/llama3.2"`). Required.
        smart_language_model: Escalation model used on disagreement. Same
            forms as panelists. Required.
        agreement_threshold (float): Maximum allowed (max - min) panel
            score spread for the panel to be deemed "in agreement"
            (default 0.2).
        prompt_template (str): The default jinja2 prompt template
            forwarded to every panelist and the smart judge.
        examples (list): The default examples forwarded to every judge.
        instructions (str): The default judging-task instructions
            forwarded to every judge (same prompt, different LMs).
        name (str): Optional. string name of the reward instance
            (default `"judge_panel"`).
        in_mask (list): Optional. list of keys to keep to compute the
            reward.
        out_mask (list): Optional. list of keys to remove to compute the
            reward.
    """

    def __init__(
        self,
        panel_language_models=None,
        smart_language_model=None,
        agreement_threshold=0.2,
        prompt_template=None,
        examples=None,
        instructions=None,
        name="judge_panel",
        in_mask=None,
        out_mask=None,
    ):
        program = JudgePanelProgram(
            panel_language_models=panel_language_models,
            smart_language_model=smart_language_model,
            agreement_threshold=agreement_threshold,
            prompt_template=prompt_template,
            examples=examples,
            instructions=instructions,
        )
        super().__init__(
            program=program,
            name=name,
            in_mask=in_mask,
            out_mask=out_mask,
        )
