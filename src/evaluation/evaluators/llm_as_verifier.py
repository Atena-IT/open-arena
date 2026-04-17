"""LLM-as-a-Verifier evaluator (pointwise).

Token-level `top_logprobs` scoring scaled over:
  G — score-token granularity (number of discrete score letters)
  K — repeated verification samples (averaged)
  C — optional criteria decomposition: score each criterion independently
      and return their mean (Kwok et al. 2026).
"""

import json
from dataclasses import asdict
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from src import default_prompts
from src.evaluation.base import PointwiseEvaluator
from src.evaluation.evaluators._verifier_reward import (
    build_score_tokens,
    render_verifier_prompt,
    verifier_reward,
)
from src.execution import ExecutionResult
from src.llms import AgentStep


class LLMAsVerifierEvaluator(PointwiseEvaluator):
    """Verifier that produces a continuous score per item by taking the
    expectation of scalar scores under the model's `top_logprobs`
    distribution at a single `<score>` position. Averages over K repeats
    and (optionally) over C criteria.
    """

    def __init__(
        self,
        results: list[ExecutionResult],
        llm_config: dict[str, Any],
        system_prompt: str = default_prompts["evaluation"]["llm_as_verifier"],
        system_prompt_no_reference: str = default_prompts["evaluation"]["llm_as_verifier_no_reference"],
        granularity: int = 8,
        repeats: int = 1,
        criteria: list[str] | None = None,
        score_name: str = "verifier_score",
        max_concurrency: int = 10,
        callbacks: list[BaseCallbackHandler] | None = None,
    ):
        super().__init__(results=results, score_name=score_name, max_concurrency=max_concurrency)
        self.llm_config = llm_config
        self.granularity = granularity
        self.repeats = max(1, repeats)
        self.criteria = list(criteria) if criteria else None
        self._callbacks = list(callbacks or [])
        self._score_tokens = build_score_tokens(granularity)
        self._base_prompt = system_prompt
        self._base_prompt_no_reference = system_prompt_no_reference

    async def _score(
        self,
        input: str,
        output: str,
        expected_output: str | None = None,
        trajectory: list[AgentStep] | None = None,
    ) -> tuple[float | None, str | None, str | None]:
        # The default criterion is mode-specific so the no-`criteria` case
        # stays close to the pre-Phase-3 prompt semantics (reference mode
        # originally judged correctness; no-reference judged overall quality).
        if expected_output is not None:
            base, default_criterion = self._base_prompt, "correctness"
        else:
            base, default_criterion = self._base_prompt_no_reference, "overall quality"

        payload_obj: dict[str, Any] = {"input": input, "output": output}
        if expected_output is not None:
            payload_obj["expected_output"] = expected_output
        if trajectory:
            payload_obj["trajectory"] = [asdict(step) for step in trajectory]
        payload = json.dumps(payload_obj, indent=2, default=str)

        # Decomposition: run one verifier call per criterion (sequential to keep
        # rate-limit behaviour predictable; row-level concurrency is handled by
        # the base PointwiseEvaluator). When `criteria` is None, a single call
        #
        criteria = self.criteria or [default_criterion]
        per_criterion: list[tuple[str, float]] = []
        for criterion in criteria:
            system = render_verifier_prompt(base, self._score_tokens, criterion)
            reward, error = await verifier_reward(
                llm_config=self.llm_config,
                system_prompt=system,
                user_payload=payload,
                granularity=self.granularity,
                repeats=self.repeats,
            )
            if reward is None:
                return None, None, f"criterion '{criterion}': {error}"
            per_criterion.append((criterion, reward))

        mean = sum(v for _, v in per_criterion) / len(per_criterion)
        if self.criteria:
            breakdown = ", ".join(f"{c}={v:.2f}" for c, v in per_criterion)
            explanation = (
                f"R(t,τ)={mean:.4f} ({breakdown}; G={self.granularity}, K={self.repeats})"
            )
        else:
            explanation = f"R(t,τ)={mean:.4f} (G={self.granularity}, K={self.repeats})"
        return mean, explanation, None
