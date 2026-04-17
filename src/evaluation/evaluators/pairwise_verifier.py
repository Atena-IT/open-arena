"""LLM-as-a-Verifier, pairwise (group) evaluator.

Both outputs are placed in a single verifier prompt, the grader emits two score tags, and
we extract logprob distributions at `<score_A>` and `<score_B>`. Each
pair is evaluated in both orderings (position-bias mitigation) and K
times per ordering; per-model score = mean reward across all matches.
"""

import asyncio
import itertools
import json
from dataclasses import asdict
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from src import default_prompts
from src.evaluation.base import GroupEvaluator
from src.evaluation.evaluators._verifier_reward import (
    build_score_tokens,
    render_verifier_prompt,
    verifier_pairwise_reward,
)
from src.execution import ExecutionResult
from src.llms import AgentStep


class LLMPairwiseVerifierEvaluator(GroupEvaluator):
    """Round-robin pairwise verifier with continuous per-match rewards."""

    def __init__(
        self,
        groups: list[dict[str, ExecutionResult]],
        llm_config: dict[str, Any],
        system_prompt: str = default_prompts["evaluation"]["pairwise_verifier"],
        system_prompt_no_reference: str = default_prompts["evaluation"][
            "pairwise_verifier_no_reference"
        ],
        granularity: int = 8,
        repeats: int = 1,
        score_name: str = "verifier_score",
        max_concurrency: int = 10,
        callbacks: list[BaseCallbackHandler] | None = None,
    ):
        super().__init__(groups=groups, score_name=score_name, max_concurrency=max_concurrency)
        self.llm_config = llm_config
        self.granularity = granularity
        self.repeats = max(1, repeats)
        self._callbacks = list(callbacks or [])
        self._score_tokens = build_score_tokens(granularity)
        self.system_prompt = render_verifier_prompt(system_prompt, self._score_tokens)
        self.system_prompt_no_reference = render_verifier_prompt(
            system_prompt_no_reference, self._score_tokens
        )

    async def _score_group(
        self,
        input: str,
        outputs: dict[str, str],
        expected_output: str | None = None,
        trajectories: dict[str, list[AgentStep]] | None = None,
    ) -> tuple[dict[str, float] | None, str | None, str | None]:
        models = list(outputs)
        if len(models) < 2:
            return None, None, "Pairwise verifier requires at least 2 models"

        pairs = list(itertools.combinations(models, 2))
        match_specs = [(a, b) for a, b in pairs] + [(b, a) for a, b in pairs]

        results = await asyncio.gather(*[
            self._verify_pair(input, a, outputs[a], b, outputs[b], expected_output,
                              (trajectories or {}).get(a), (trajectories or {}).get(b))
            for a, b in match_specs
        ])

        per_model_rewards: dict[str, list[float]] = {m: [] for m in models}
        wins: dict[str, float] = {m: 0.0 for m in models}
        lines: list[str] = []
        failures = 0
        for (a, b), pair_result in zip(match_specs, results):
            if pair_result is None:
                failures += 1
                continue
            r_a, r_b = pair_result
            per_model_rewards[a].append(r_a)
            per_model_rewards[b].append(r_b)
            if r_a > r_b:
                wins[a] += 1.0
            elif r_b > r_a:
                wins[b] += 1.0
            else:
                wins[a] += 0.5
                wins[b] += 0.5
            lines.append(f"{a} vs {b}: R_A={r_a:.3f} R_B={r_b:.3f}")

        if not any(per_model_rewards.values()):
            return None, None, "all pair verifications failed"

        scores = {
            m: (sum(rs) / len(rs)) if rs else None
            for m, rs in per_model_rewards.items()
        }
        tournament = ", ".join(f"{m}:{wins[m]:.1f}W" for m in models)
        explanation = (
            f"G={self.granularity} K={self.repeats}  wins: {tournament}\n"
            + "\n".join(lines)
        )
        error = f"{failures}/{len(match_specs)} pair verifications failed" if failures else None
        return {m: s for m, s in scores.items() if s is not None}, explanation, error

    async def _verify_pair(
        self,
        input: str,
        name_a: str,
        output_a: str,
        name_b: str,
        output_b: str,
        expected_output: str | None,
        trajectory_a: list[AgentStep] | None,
        trajectory_b: list[AgentStep] | None,
    ) -> tuple[float, float] | None:
        payload_obj: dict[str, Any] = {
            "input": input,
            "output_A": output_a,
            "output_B": output_b,
        }
        if expected_output is None:
            system = self.system_prompt_no_reference
        else:
            system = self.system_prompt
            payload_obj["expected_output"] = expected_output
        if trajectory_a:
            payload_obj["trajectory_A"] = [asdict(step) for step in trajectory_a]
        if trajectory_b:
            payload_obj["trajectory_B"] = [asdict(step) for step in trajectory_b]
        payload = json.dumps(payload_obj, indent=2, default=str)

        rewards, error = await verifier_pairwise_reward(
            llm_config=self.llm_config,
            system_prompt=system,
            user_payload=payload,
            granularity=self.granularity,
            repeats=self.repeats,
        )
        if rewards is None:
            return None
        return rewards
