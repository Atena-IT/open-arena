import json
import logging
from typing import Any, TypeVar

from src import default_prompts
from src.llms import LLMClient
from src.datasets.item_models import DatasetItem
from src.execution.types import ExecutionResult
from src.evaluation.types import EvaluationResult, JudgeResponse
from src.evaluation.methods.base_method import EvaluationMethod

_logger = logging.getLogger(__name__)
T = TypeVar('T', bound=DatasetItem)


class LLMAsJudge(EvaluationMethod[T]):
    """
    LLM-as-a-Judge evaluation method.

    Uses another LLM to judge the quality of outputs by comparing
    them against expected outputs and providing scores + explanations.

    The judge LLM receives:
    - input: The original user input
    - output: The model's generated output
    - expected_output: The ground truth or expected answer

    And returns:
    - score: A numerical score (e.g., 0-5)
    - explanation: Text explaining the score
    """

    def __init__(
        self,
        llm_client: LLMClient, # TODO: simpler implementation that does not require a graph
        system_prompt: str = default_prompts["evaluation"]["llm_as_judge"],
    ):
        """
        :param llm_client: LLM client for judge completions
        :param system_prompt: Optional system prompt that defines how the judge should evaluate
        """
        super().__init__(name="LLM-as-Judge")
        self.client = llm_client
        self.system_prompt = system_prompt

    async def evaluate(self, result: ExecutionResult[T]) -> EvaluationResult[T]:
        """
        Evaluate a single execution result using LLM-as-Judge.

        :param result: Execution result to evaluate
        :return: Evaluation result with score and explanation
        """
        try:
            payload = self._build_judge_payload(result)

            messages = self.client.format_messages(
                system=self.system_prompt,
                user=payload
            )

            judge_output = await self.client.achat(
                messages=messages
            )

            score, explanation = self._parse_judge_response(judge_output)

            return EvaluationResult(
                item=result.item,
                output=result.output or "",
                model_name=result.model_name,
                score=score,
                explanation=explanation,
            )

        except Exception as e:
            return EvaluationResult(
                item=result.item,
                output=result.output or "",
                model_name=result.model_name,
                score=None,
                explanation=None,
                error=str(e),
            )

    def _build_judge_payload(self, result: ExecutionResult[T]) -> str:
        """
        Build the user prompt for the judge based on execution result.

        Creates a JSON payload with:
        - input: Original user input from the dataset item
        - output: Model's generated output
        - expected_output: Ground truth (if available)

        :param result: Execution result to evaluate
        :return: JSON string with evaluation payload
        """
        payload = {
            "input": result.item.input(),
            "output": result.output or "",
            "expected_output": result.item.expected_output(),
        }

        return json.dumps(payload, indent=2)

    def _parse_judge_response(self, raw_response: str) -> tuple[int | None, str | None]:
        """
        Parse the judge's response into score and explanation using Pydantic validation.

        Expects JSON format: {"score": 4, "explanation": "Good answer because..."}

        :param raw_response: Raw response from judge LLM
        :return: Tuple of (score, explanation)
        """
        try:
            if isinstance(raw_response, str):
                response_dict = json.loads(raw_response)
            else:
                response_dict = raw_response

            judge_response = JudgeResponse(**response_dict)

            return judge_response.score, judge_response.explanation

        except json.JSONDecodeError as e:
            _logger.error(f"Failed to parse judge response as JSON: {e}")
            _logger.debug(f"Raw response: {raw_response}")
            return None, None

        except Exception as e:
            _logger.error(f"Failed to validate judge response: {e}")
            _logger.debug(f"Raw response: {raw_response}")
            return None, None
