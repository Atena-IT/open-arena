from __future__ import annotations

import asyncio
from unittest.mock import patch

from src.evaluation.base import PointwiseEvaluator
from src.execution import ExecutionResult


class _FakeObservation:
    def update(self, **kwargs) -> None:
        pass


class _ObservationContext:
    def __init__(self, owner: "_FakeLangfuse"):
        self._owner = owner

    def __enter__(self):
        self._owner.observation_count += 1
        return _FakeObservation()

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class _FakeLangfuse:
    def __init__(self):
        self.observation_count = 0
        self.scores: list[dict] = []

    def start_as_current_observation(self, **kwargs):
        return _ObservationContext(self)

    def create_score(self, **kwargs) -> None:
        self.scores.append(kwargs)

    def flush(self) -> None:
        pass


class _CountingEvaluator(PointwiseEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.score_calls = 0

    async def _score(self, input, output, expected_output=None, trajectory=None):
        self.score_calls += 1
        return 1.0, "ok", None


def test_pointwise_evaluator_skips_scoring_failed_execution_results():
    fake_langfuse = _FakeLangfuse()
    with patch("src.evaluation.base.get_client", return_value=fake_langfuse):
        evaluator = _CountingEvaluator(results=[])

    execution_result = ExecutionResult(
        input="question",
        expected_output="answer",
        output=None,
        model_name="test-model",
        experiment_name="exp",
        error="llm timeout",
        metadata={"lf_trace_id": "trace-1"},
    )

    eval_result = asyncio.run(evaluator._evaluate_one(execution_result))

    assert evaluator.score_calls == 0
    assert eval_result.score is None
    assert eval_result.error == "llm timeout"
    assert fake_langfuse.observation_count == 0
    assert fake_langfuse.scores == []
