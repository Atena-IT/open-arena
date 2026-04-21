import asyncio
import logging
from abc import ABC, abstractmethod
from typing import ClassVar, Literal

from langfuse import get_client
from tqdm.asyncio import tqdm as async_tqdm

from src.evaluation.types import EvaluationResult
from src.execution import ExecutionResult
from src.llms import AgentStep

_logger = logging.getLogger(__name__)

Mode = Literal["pointwise", "group"]


class Evaluator(ABC):
    """Base evaluator. Subclasses come in two shapes — see `PointwiseEvaluator`
    and `GroupEvaluator`. The `mode` class attribute tells the runner which
    shape to feed it.

    Score contract: `_score` / `_score_group` must return floats in [0, 1]
    (or `None` on failure). Built-in evaluators normalise to this range so
    scores are comparable across methods in Langfuse — custom subclasses
    must follow the same contract.
    """

    mode: ClassVar[Mode]

    def __init__(
        self,
        score_name: str = "evaluation_score",
        max_concurrency: int = 10,
        timeout_s: float | None = None,
    ):
        self.score_name = score_name
        self.max_concurrency = max_concurrency
        self.timeout_s = timeout_s
        self.langfuse = get_client()

    @abstractmethod
    async def evaluate(self) -> list[EvaluationResult]:
        raise NotImplementedError

    def _write_score(self, trace_id: str | None, score: float | None, explanation: str | None) -> None:
        if score is None or not trace_id:
            return
        try:
            self.langfuse.create_score(
                trace_id=str(trace_id),
                name=self.score_name,
                value=float(score),
                comment=str(explanation) if explanation else None,
            )
        except Exception as e:
            _logger.error(f"Failed to write score to Langfuse for trace {trace_id}: {e}")


class PointwiseEvaluator(Evaluator):
    """Scores each ExecutionResult independently.

    Subclasses implement `_score(input, output, expected_output)` and get the
    Langfuse-wrapped per-item driver loop + score writeback for free.
    """

    mode: ClassVar[Mode] = "pointwise"

    def __init__(
        self,
        results: list[ExecutionResult],
        score_name: str = "evaluation_score",
        max_concurrency: int = 10,
        timeout_s: float | None = None,
    ):
        super().__init__(
            score_name=score_name, max_concurrency=max_concurrency, timeout_s=timeout_s
        )
        self.results = results

    @abstractmethod
    async def _score(
        self,
        input: str,
        output: str,
        expected_output: str | None = None,
        trajectory: list[AgentStep] | None = None,
    ) -> tuple[float | None, str | None, str | None]:
        raise NotImplementedError

    async def evaluate(self) -> list[EvaluationResult]:
        queue: asyncio.Queue[ExecutionResult] = asyncio.Queue()
        for r in self.results:
            queue.put_nowait(r)

        eval_results: list[EvaluationResult] = []
        pbar = async_tqdm(total=len(self.results), desc="Evaluating items")

        async def _worker() -> None:
            while not queue.empty():
                try:
                    result = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                eval_results.append(await self._evaluate_one(result))
                pbar.update(1)

        worker_count = min(self.max_concurrency, len(self.results))
        try:
            await asyncio.gather(*[_worker() for _ in range(worker_count)])
        finally:
            pbar.close()
            self.langfuse.flush()
        return eval_results

    async def _evaluate_one(self, result: ExecutionResult) -> EvaluationResult:
        if result.error:
            return EvaluationResult(
                input=result.input,
                expected_output=result.expected_output,
                output=result.output or "",
                model_name=result.model_name,
                experiment_name=result.experiment_name,
                error=result.error,
                metadata=dict(result.metadata),
            )

        trace_id = result.metadata.get("lf_trace_id")
        if not trace_id:
            _logger.warning("Result missing 'lf_trace_id' in metadata")

        with self.langfuse.start_as_current_observation(
            as_type="evaluator",
            name="evaluation-item-run",
            input={"input": result.input, "expected_output": result.expected_output, "output": result.output},
        ) as span:
            try:
                coro = self._score(
                    input=result.input,
                    output=result.output or "",
                    expected_output=result.expected_output or None,
                    trajectory=result.trajectory,
                )
                if self.timeout_s is not None:
                    coro = asyncio.wait_for(coro, timeout=self.timeout_s)
                score, explanation, error = await coro
            except asyncio.TimeoutError:
                score, explanation, error = None, None, f"evaluator timeout after {self.timeout_s}s"
            eval_result = EvaluationResult(
                input=result.input,
                expected_output=result.expected_output,
                output=result.output or "",
                model_name=result.model_name,
                experiment_name=result.experiment_name,
                score=score,
                explanation=explanation,
                error=error,
                metadata=dict(result.metadata),
            )
            if error:
                _logger.error(f"Evaluation error (trace {trace_id}): {error}")
            span.update(
                output={"score": score, "explanation": explanation, "error": error},
                level="ERROR" if error else "DEFAULT",
            )
            self._write_score(trace_id, score, explanation)

        return eval_result


class GroupEvaluator(Evaluator):
    """Scores multiple models' outputs for the same dataset item together.

    Input is a list of groups; each group maps `model_name -> ExecutionResult`
    for one dataset row (lf_item_id). Subclasses implement `_score_group`
    returning a per-model score dict. The base class writes one Langfuse score
    per model back to the corresponding experiment trace.
    """

    mode: ClassVar[Mode] = "group"

    def __init__(
        self,
        groups: list[dict[str, ExecutionResult]],
        score_name: str = "evaluation_score",
        max_concurrency: int = 10,
        timeout_s: float | None = None,
    ):
        super().__init__(
            score_name=score_name, max_concurrency=max_concurrency, timeout_s=timeout_s
        )
        self.groups = groups

    @abstractmethod
    async def _score_group(
        self,
        input: str,
        outputs: dict[str, str],
        expected_output: str | None = None,
        trajectories: dict[str, list[AgentStep]] | None = None,
    ) -> tuple[dict[str, float] | None, str | None, str | None]:
        """Return (scores_by_model, shared_explanation, error)."""
        raise NotImplementedError

    async def evaluate(self) -> list[EvaluationResult]:
        queue: asyncio.Queue[dict[str, ExecutionResult]] = asyncio.Queue()
        for g in self.groups:
            queue.put_nowait(g)

        all_results: list[EvaluationResult] = []
        pbar = async_tqdm(total=len(self.groups), desc="Evaluating groups")

        async def _worker() -> None:
            while not queue.empty():
                try:
                    group = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                all_results.extend(await self._evaluate_group(group))
                pbar.update(1)

        worker_count = min(self.max_concurrency, len(self.groups))
        try:
            await asyncio.gather(*[_worker() for _ in range(worker_count)])
        finally:
            pbar.close()
            self.langfuse.flush()
        return all_results

    async def _evaluate_group(self, group: dict[str, ExecutionResult]) -> list[EvaluationResult]:
        # All entries in a group share input/expected_output by construction.
        any_result = next(iter(group.values()))
        input_ = any_result.input
        expected = any_result.expected_output or None

        with self.langfuse.start_as_current_observation(
            as_type="evaluator",
            name="evaluation-group-run",
            input={
                "input": input_,
                "expected_output": expected,
                "outputs": {m: r.output for m, r in group.items()},
            },
        ) as span:
            trajectories = {m: r.trajectory for m, r in group.items() if r.trajectory}
            try:
                coro = self._score_group(
                    input=input_,
                    outputs={m: (r.output or "") for m, r in group.items()},
                    expected_output=expected,
                    trajectories=trajectories or None,
                )
                if self.timeout_s is not None:
                    coro = asyncio.wait_for(coro, timeout=self.timeout_s)
                scores, explanation, error = await coro
            except asyncio.TimeoutError:
                scores, explanation, error = None, None, f"evaluator timeout after {self.timeout_s}s"
            if error:
                _logger.error(f"Group evaluation error: {error}")
            span.update(
                output={"scores": scores, "explanation": explanation, "error": error},
                level="ERROR" if error else "DEFAULT",
            )

            eval_results: list[EvaluationResult] = []
            for exp_name, result in group.items():
                model_score = scores.get(exp_name) if scores else None
                eval_results.append(EvaluationResult(
                    input=result.input,
                    expected_output=result.expected_output,
                    output=result.output or "",
                    model_name=result.model_name,
                    experiment_name=result.experiment_name or exp_name,
                    score=model_score,
                    explanation=explanation,
                    error=error,
                    metadata=dict(result.metadata),
                ))
                self._write_score(result.metadata.get("lf_trace_id"), model_score, explanation)

        return eval_results
