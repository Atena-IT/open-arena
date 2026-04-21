import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from langfuse import get_client
from langfuse.api import CreateDatasetRunItemRequest
from tqdm.asyncio import tqdm as async_tqdm

from src.datasets import Row
from src.llms import AgentStep, LLMCaller

_logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of executing a single dataset row."""

    input: str
    expected_output: str
    output: str | None
    model_name: str
    experiment_name: str = ""
    error: str | None = None
    trajectory: list[AgentStep] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Executor:
    """Iterate a dataset, call the LLM for each row, wrap every call in a
    Langfuse span, link it to the uploaded dataset item, and return the
    collected results."""

    def __init__(
        self,
        dataset: list[Row],
        llm_client: LLMCaller,
        system_prompt: str,
        experiment_name: str,
        experiment_description: str | None = None,
        max_concurrency: int = 50,
        fail_fast: bool = False,
        timeout_s: float | None = None,
    ):
        self.dataset = dataset
        self.client = llm_client
        self.system_prompt = system_prompt
        self.experiment_name = experiment_name
        self.experiment_description = (
            experiment_description or f"Experiment with {llm_client.llm_config['model']}"
        )
        self.max_concurrency = max_concurrency
        self.fail_fast = fail_fast
        self.timeout_s = timeout_s
        self._langfuse = get_client()

    async def execute(self) -> list[ExecutionResult]:
        if not self.dataset:
            _logger.warning("Dataset is empty, no rows to execute")
            return []

        dataset_id = self.dataset[0][2].get("lf_dataset_id")
        if not dataset_id:
            raise ValueError("Rows must have 'lf_dataset_id' in metadata")

        run_name = f"{self.experiment_name} - {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S+00:00')}"

        queue: asyncio.Queue[Row] = asyncio.Queue()
        for row in self.dataset:
            queue.put_nowait(row)

        results: list[ExecutionResult] = []
        pbar = async_tqdm(total=len(self.dataset), desc="Executing items")

        async def _worker() -> None:
            while not queue.empty():
                try:
                    row = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                results.append(await self._execute_row(row, run_name, dataset_id))
                pbar.update(1)

        worker_count = min(self.max_concurrency, len(self.dataset))
        try:
            await asyncio.gather(*[_worker() for _ in range(worker_count)])
        finally:
            pbar.close()
        return results

    async def _execute_row(
        self,
        row: Row,
        run_name: str,
        dataset_id: str,
    ) -> ExecutionResult:
        input_, expected, metadata = row
        dataset_item_id = metadata.get("lf_item_id")
        if not dataset_item_id:
            raise ValueError("Row must have 'lf_item_id' in metadata")

        with self._langfuse.start_as_current_observation(
            as_type="span",
            name="experiment-item-run",
            input=input_,
            metadata={
                "experiment_name": self.experiment_name,
                "experiment_run_name": run_name,
                "dataset_id": dataset_id,
                "dataset_item_id": dataset_item_id,
            },
        ) as span:
            result = await self._call_llm(input_, expected, metadata)

            span.update(
                output=str(result.output) if result.output is not None else None,
                level="ERROR" if result.error else "DEFAULT",
            )

            try:
                self._langfuse.api.dataset_run_items.create(
                    request=CreateDatasetRunItemRequest(
                        runName=run_name,
                        datasetItemId=dataset_item_id,
                        traceId=span.trace_id,
                        runDescription=self.experiment_description,
                    )
                )
            except Exception as e:
                _logger.error(f"Failed to create dataset run item for {dataset_item_id}: {e}")

            result.metadata.update({
                "lf_trace_id": span.trace_id,
                "lf_observation_id": span.id,
                "lf_experiment_name": self.experiment_name,
                "lf_experiment_run_name": run_name,
            })
            return result

    async def _call_llm(self, input_: str, expected: str, metadata: dict[str, Any]) -> ExecutionResult:
        model = self.client.llm_config["model"]
        try:
            coro = self.client.achat_with_trajectory(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": input_},
                ]
            )
            if self.timeout_s is not None:
                coro = asyncio.wait_for(coro, timeout=self.timeout_s)
            output, trajectory = await coro
            return ExecutionResult(
                input=input_,
                expected_output=expected,
                output=output,
                model_name=model,
                experiment_name=self.experiment_name,
                trajectory=trajectory,
                metadata=dict(metadata),
            )
        except asyncio.TimeoutError:
            error = f"llm timeout after {self.timeout_s}s"
            _logger.error(error)
            if self.fail_fast:
                raise
            return ExecutionResult(
                input=input_,
                expected_output=expected,
                output=None,
                model_name=model,
                experiment_name=self.experiment_name,
                error=error,
                metadata=dict(metadata),
            )
        except Exception as e:
            _logger.error(f"Execution failed: {e}")
            if self.fail_fast:
                raise
            return ExecutionResult(
                input=input_,
                expected_output=expected,
                output=None,
                model_name=model,
                experiment_name=self.experiment_name,
                error=str(e),
                metadata=dict(metadata),
            )
