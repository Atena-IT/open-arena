import asyncio
import logging
from typing import List, TypeVar

from langfuse import get_client, observe

from src.datasets.item_models import DatasetItem
from src.execution.types import ExecutionResult
from src.evaluation.base_evaluator import Evaluator
from src.evaluation.types import EvaluationResult
from src.evaluation.methods import EvaluationMethod

_logger = logging.getLogger(__name__)
T = TypeVar('T', bound=DatasetItem)


class LangfuseEvaluator(Evaluator[T]):
    """
    Evaluator that scores results in Langfuse using the Scores API.
    
    Takes ExecutionResult[T] from LangfuseExecutor (which contains trace_id),
    evaluates them using the provided method, and writes scores back to Langfuse.
    
    This allows scores to be visible in the Langfuse dashboard for each trace,
    enabling analysis and comparison of model performance.
    """
    
    def __init__(
        self,
        results: List[ExecutionResult[T]],
        method: EvaluationMethod[T],
        score_name: str = "evaluation_score",
        max_concurrency: int = 10
    ):
        """
        :param results: Execution results from LangfuseExecutor (with trace_id metadata)
        :param method: Evaluation method (e.g., LLMAsJudge)
        :param score_name: Name for the score in Langfuse (e.g., "correctness", "quality")
        :param max_concurrency: Max parallel evaluations
        """
        super().__init__(results=results, method=method)
        self.score_name = score_name
        self.langfuse = get_client()
        self.max_concurrency = max_concurrency
    
    async def _evaluate_and_score_item(
        self, 
        result: ExecutionResult[T]
    ) -> EvaluationResult[T]:
        """
        Evaluate a single result and write score to Langfuse.
        
        1. Call evaluation method to get score + explanation
        2. Extract trace_id from result metadata
        3. Write score to Langfuse asynchronously
        4. Return EvaluationResult
        
        :param result: Execution result to evaluate
        :return: Evaluation result with score
        """
        with self.langfuse.start_as_current_observation(
            as_type="evaluator",
            name="evaluation-item-run",
            input={
                "item": result.item,
                "output": result.output
            }
        ) as root_span:
            eval_result = await self.method.evaluate(result)
            
            trace_id = result.metadata.get("lf_trace_id") if result.metadata else None
            
            if not trace_id:
                _logger.warning(f"No lf_trace_id found in result metadata. Cannot score in Langfuse.")
                return eval_result
            
            if eval_result.score is not None:
                try:
                    self.langfuse.create_score(
                        trace_id=str(trace_id),
                        name=self.score_name,
                        value=float(eval_result.score),
                        comment=str(eval_result.explanation) if eval_result.explanation else None
                    )
                except Exception as e:
                    _logger.error(f"Failed to write score to Langfuse for trace {trace_id}: {e}")
            
            root_span.update(output={
                "score": eval_result.score, 
                "explanation": eval_result.explanation
            })
        
        return eval_result
    
    async def evaluate(self) -> List[EvaluationResult[T]]:
        """
        Evaluate all results with controlled concurrency and write scores to Langfuse.
        
        Uses asyncio.Semaphore to limit concurrent evaluations and prevent
        overwhelming the evaluation method (e.g., LLM rate limits).
        
        :return: List of evaluation results
        """
        semaphore = asyncio.Semaphore(self.max_concurrency)
        
        async def evaluate_with_semaphore(result: ExecutionResult[T]):
            async with semaphore:
                return await self._evaluate_and_score_item(result)
        
        eval_results = await asyncio.gather(
            *[evaluate_with_semaphore(r) for r in self.results]
        )
        
        return eval_results
