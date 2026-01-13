import json
import logging
from abc import ABC, abstractmethod
from typing import List, Generic, TypeVar, Dict, Any, Optional, Tuple

from src.llms import LLMClient
from src.datasets.item_models import DatasetItem
from src.execution.types import ExecutionResult
from src.evaluator.types import EvaluationResult, JudgeResponse

_logger = logging.getLogger(__name__)
T = TypeVar('T', bound=DatasetItem)


class Evaluator(ABC, Generic[T]):
    """
    Abstract base class for all evaluators.
    Defines the common interface that all evaluator implementations must follow.
    """
    
    def __init__(
        self,
        results: List[ExecutionResult[T]],
        llm_client: LLMClient,
        system_prompt: str,
        model_config: Dict[str, Any]
    ):
        """
        :param results: Results from executor to evaluate
        :param llm_client: LLM client for judge completions
        :param system_prompt: System prompt that defines how the judge should evaluate
        :param model_config: Model configuration for the judge to use
        """
        self.results = results
        self.client = llm_client
        self.system_prompt = system_prompt
        self.model_config = model_config
    
    @abstractmethod
    async def evaluate(self) -> List[EvaluationResult[T]]:
        """
        Evaluate the execution results.
        
        :return: List of evaluation results with scores
        """
        pass
    
    async def _evaluate_result(self, result: ExecutionResult[T]) -> EvaluationResult[T]:
        """
        Evaluate a single execution result.
        
        :param execution_result: Single execution result to evaluate
        :return: Evaluation result with score and explanation
        """
        try:
            user_prompt = self._build_judge_payload(result)
            
            messages = self.client.format_messages(
                system=self.system_prompt,
                user=user_prompt
            )
            
            judge_output = await self.client.achat(
                messages=messages,
                model_config=self.model_config
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
            _logger.error(f"Evaluation failed: {e}")
            return EvaluationResult(
                item=result.item,
                output=result.output or "",
                model_name=result.model_name,
                score=None,
                explanation=None,
                error=str(e),
            )
    
    def _build_judge_payload(self, execution_result: ExecutionResult[T]) -> str:
        """
        Build the user prompt for the judge based on execution result.
        
        Creates a JSON payload with:
        - input: Original user input from the dataset item
        - output: Model's generated output
        - expected_output: Ground truth (if available)
        
        :param execution_result: Execution result to evaluate
        :return: JSON string with evaluation payload
        """
        payload = {
            "input": execution_result.item.user_prompt(),
            "output": execution_result.output or "",
            "expected_output": execution_result.item.expected_output(),
        }
        
        return json.dumps(payload, indent=2)
    
    def _parse_judge_response(self, raw_response: str) -> Tuple[Optional[float], Optional[str]]:
        """
        Parse the judge's response into score and explanation using Pydantic validation.
        
        Expects JSON format: {"score": 4.5, "explanation": "Good answer because..."}
        
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
