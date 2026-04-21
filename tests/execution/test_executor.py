from __future__ import annotations

import asyncio
from unittest.mock import patch

from src.execution import ExecutionResult, Executor


_SLOW_LLM_DELAY_SECONDS = 0.05


class _FailingLLM:
    def __init__(self):
        self.llm_config = {"model": "test-model"}

    async def achat_with_trajectory(self, messages):
        raise RuntimeError("boom")


class _SlowLLM:
    def __init__(self, delay_s: float = _SLOW_LLM_DELAY_SECONDS):
        self.llm_config = {"model": "test-model"}
        self.delay_s = delay_s

    async def achat_with_trajectory(self, messages):
        await asyncio.sleep(self.delay_s)
        return "ok", []


class _DummyProgress:
    def update(self, _count: int) -> None:
        pass

    def close(self) -> None:
        pass


def test_call_llm_returns_none_output_on_failure():
    with patch("src.execution.executor.get_client", return_value=object()):
        executor = Executor(
            dataset=[],
            llm_client=_FailingLLM(),
            system_prompt="system",
            experiment_name="exp",
        )

    result = asyncio.run(executor._call_llm("question", "answer", {"row_id": "1"}))

    assert result.output is None
    assert result.error == "boom"


def test_call_llm_returns_timeout_error_message():
    with patch("src.execution.executor.get_client", return_value=object()):
        executor = Executor(
            dataset=[],
            llm_client=_SlowLLM(delay_s=_SLOW_LLM_DELAY_SECONDS),
            system_prompt="system",
            experiment_name="exp",
            timeout_s=0.01,
        )

    result = asyncio.run(executor._call_llm("question", "answer", {"row_id": "1"}))

    assert result.output is None
    assert result.error == "llm timeout after 0.01s"


def test_execute_caps_worker_count_to_dataset_size():
    dataset = [
        ("q1", "a1", {"lf_dataset_id": "ds-1", "lf_item_id": "item-1"}),
        ("q2", "a2", {"lf_dataset_id": "ds-1", "lf_item_id": "item-2"}),
    ]
    observed: dict[str, int] = {}
    original_gather = asyncio.gather

    async def fake_execute_row(row, run_name, dataset_id):
        return ExecutionResult(
            input=row[0],
            expected_output=row[1],
            output="ok",
            model_name="test-model",
            experiment_name="exp",
            metadata=dict(row[2]),
        )

    async def recording_gather(*aws):
        observed["worker_count"] = len(aws)
        return await original_gather(*aws)

    with patch("src.execution.executor.get_client", return_value=object()), \
            patch("src.execution.executor.async_tqdm", return_value=_DummyProgress()), \
            patch("src.execution.executor.asyncio.gather", new=recording_gather):
        executor = Executor(
            dataset=dataset,
            llm_client=_FailingLLM(),
            system_prompt="system",
            experiment_name="exp",
            max_concurrency=10,
        )
        executor._execute_row = fake_execute_row  # type: ignore[method-assign]

        results = asyncio.run(executor.execute())

    assert observed["worker_count"] == len(dataset)
    assert len(results) == len(dataset)
