import json
import logging
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.exceptions import OutputParserException

_RETRY_EXCEPTIONS = (OutputParserException,)

from dataclasses import asdict

from src.evaluation.base import PointwiseEvaluator
from src.evaluation.types import JudgeResponse
from src.execution import ExecutionResult
from src.llms import AgentStep
from src.llms.base import build_chat_model, to_langchain_messages

_logger = logging.getLogger(__name__)


class LLMAsJudgeEvaluator(PointwiseEvaluator):
    """Score each output by asking a judge LLM. Two modes:

    - With `expected_output`: judges similarity to ground truth.
    - Without: judges output quality from input alone.

    Uses `ChatLiteLLM.with_structured_output(JudgeResponse, method="json_schema",
    strict=True)` so the judge returns a validated Pydantic instance via the
    provider's constrained-decoding / strict JSON schema path when available.
    """

    def __init__(
        self,
        results: list[ExecutionResult],
        llm_config: dict[str, Any],
        system_prompt: str,
        system_prompt_no_reference: str,
        score_name: str = "evaluation_score",
        max_concurrency: int = 10,
        max_retries: int = 3,
        callbacks: list[BaseCallbackHandler] | None = None,
        timeout_s: float | None = None,
    ):
        super().__init__(
            results=results,
            score_name=score_name,
            max_concurrency=max_concurrency,
            timeout_s=timeout_s,
        )
        self.system_prompt = system_prompt
        self.system_prompt_no_reference = system_prompt_no_reference
        self._callbacks = list(callbacks or [])
        self._judge = build_chat_model(llm_config).with_structured_output(
            JudgeResponse, method="json_schema", strict=True
        ).with_retry(retry_if_exception_type=_RETRY_EXCEPTIONS, stop_after_attempt=max_retries)

    async def _score(
        self,
        input: str,
        output: str,
        expected_output: str | None = None,
        trajectory: list[AgentStep] | None = None,
    ) -> tuple[float | None, str | None, str | None]:
        payload_obj: dict = {"input": input, "output": output}
        if expected_output is None:
            system = self.system_prompt_no_reference
        else:
            system = self.system_prompt
            payload_obj["expected_output"] = expected_output
        if trajectory:
            payload_obj["trajectory"] = [asdict(step) for step in trajectory]
        payload = json.dumps(payload_obj, indent=2, default=str)

        messages = to_langchain_messages([
            {"role": "system", "content": system},
            {"role": "user", "content": payload},
        ])

        try:
            parsed: JudgeResponse = await self._judge.ainvoke(
                messages,
                config={"callbacks": self._callbacks},
            )
        except Exception as e:
            return None, None, str(e)

        return (parsed.score - 1) / 4.0, parsed.thinking, None
