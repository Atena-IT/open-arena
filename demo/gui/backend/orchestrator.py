"""Background orchestration for notebook-aligned GUI runs."""

from __future__ import annotations

import asyncio
import logging
import threading
from datetime import datetime, timezone
from typing import Any

from langfuse import get_client

from demo.gui.backend.models import ExecutionMode, MetricSpec, RunPhase, RunState
from demo.gui.backend.run_manager import RunManager
from demo.gui.backend.runtime import (
    build_execution_caller,
    build_metric_evaluator_kwargs,
    build_runtime_config,
    load_runtime_rows,
)
from src.datasets.langfuse_upload import upload_rows
from src.evaluation import build_evaluator, evaluator_mode
from src.execution import ExecutionResult, Executor

_logger = logging.getLogger(__name__)


class DemoOrchestrator:
    """Launch GUI runs in the background and stream updates into RunManager."""

    def __init__(self, manager: RunManager, fake_delay_s: float = 0.08) -> None:
        self._manager = manager
        self._fake_delay_s = fake_delay_s

    def start_run(self, run_id: str) -> None:
        thread = threading.Thread(
            target=self._run_in_thread,
            args=(run_id,),
            daemon=True,
            name=f"open-arena-gui-run-{run_id[:8]}",
        )
        thread.start()

    def _run_in_thread(self, run_id: str) -> None:
        asyncio.run(self.run(run_id))

    async def run(self, run_id: str) -> None:
        run = self._require_run(run_id)
        try:
            self._emit_phase(run_id, RunPhase.UPLOADING)
            if run.executionMode == ExecutionMode.FAKE:
                await self._run_fake(run)
            else:
                await self._run_real(run)
            self._emit_phase(run_id, RunPhase.COMPLETED)
            completed = self._require_run(run_id)
            self._manager.add_event(
                run_id,
                "run_completed",
                {
                    "completedItems": completed.completedItems,
                    "errorCount": completed.errorCount,
                },
            )
        except Exception as exc:
            _logger.exception("GUI run failed for %s", run_id)
            self._manager.add_error(run_id, str(exc))
            self._emit_phase(run_id, RunPhase.FAILED)
            self._manager.add_event(run_id, "run_failed", {"message": str(exc)})

    async def _run_fake(self, run: RunState) -> None:
        self._manager.add_event(
            run.runId,
            "dataset_ready",
            {
                "datasetName": run.datasetName,
                "itemCount": run.sampleLimit,
                "mode": "fake",
            },
        )
        await self._sleep_if_needed()
        self._emit_phase(run.runId, RunPhase.RUNNING)

        progress = {
            model.experimentKey: {"completed": 0, "errors": 0}
            for model in run.selectedModels
        }
        for item_index in range(run.sampleLimit):
            for model_index, model in enumerate(run.selectedModels):
                await self._sleep_if_needed()
                item_reference = f"demo-item-{item_index + 1}"
                stats = progress[model.experimentKey]
                stats["completed"] += 1
                self._manager.record_execution_result(
                    run.runId,
                    model.experimentKey,
                    item_reference=item_reference,
                    input_text=f"Demo mission sample {item_index + 1}",
                    expected_output="Notebook-aligned executive memo with citations and takeaways.",
                    output_text=f"Simulated memo draft {item_index + 1} from {model.experimentName}.",
                    error=None,
                )
                self._manager.record_model_progress(
                    run.runId,
                    model.experimentKey,
                    completed=stats["completed"],
                    total=run.sampleLimit,
                    errors=stats["errors"],
                )
                self._manager.add_event(
                    run.runId,
                    "execution_completed",
                    {
                        "experimentKey": model.experimentKey,
                        "itemIndex": item_index + 1,
                        "itemReference": item_reference,
                    },
                )
                for metric_index, metric in enumerate(run.selectedMetrics):
                    score = self._fake_score(model_index, metric_index, item_index, len(run.selectedModels))
                    self._manager.record_metric_result(
                        run.runId,
                        model.experimentKey,
                        metric.key,
                        item_reference=item_reference,
                        score=score,
                        explanation=f"Simulated {metric.label} score in fake showcase mode.",
                    )

        self._emit_phase(run.runId, RunPhase.EVALUATING)
        await self._sleep_if_needed()

    async def _run_real(self, run: RunState) -> None:
        runtime_config = build_runtime_config(run)
        rows = load_runtime_rows(runtime_config)
        if runtime_config.dataset.source.get("provider") == "langfuse":
            runtime_rows = rows
            self._manager.add_event(
                run.runId,
                "dataset_ready",
                {
                    "datasetName": runtime_config.dataset.name,
                    "itemCount": len(runtime_rows),
                    "mode": "langfuse-source",
                },
            )
        else:
            runtime_rows = await upload_rows(
                rows,
                dataset_name=runtime_config.dataset.name,
                description=runtime_config.dataset.description or "",
            )
            self._manager.add_event(
                run.runId,
                "dataset_uploaded",
                {
                    "datasetName": runtime_config.dataset.name,
                    "itemCount": len(runtime_rows),
                },
            )

        actual_total = len(runtime_rows)
        self._manager.sync_run_totals(run.runId, actual_total)
        self._emit_phase(run.runId, RunPhase.RUNNING)
        pointwise_metrics = [metric for metric in run.selectedMetrics if evaluator_mode(metric.method) == "pointwise"]
        group_metrics = [metric for metric in run.selectedMetrics if evaluator_mode(metric.method) == "group"]
        pointwise_queues = {metric.key: asyncio.Queue() for metric in pointwise_metrics}
        group_queues = {metric.key: asyncio.Queue() for metric in group_metrics}
        pointwise_workers = await self._start_pointwise_workers(run, runtime_config, pointwise_metrics, pointwise_queues)
        group_workers = await self._start_group_workers(run, runtime_config, group_metrics, group_queues)

        progress = {
            model.experimentKey: {"completed": 0, "errors": 0}
            for model in run.selectedModels
        }
        group_buffer: dict[str, dict[str, ExecutionResult]] = {}

        async def on_execution_result(experiment_key: str, result: ExecutionResult) -> None:
            stats = progress[experiment_key]
            item_reference = self._item_reference(
                result.metadata.get("lf_item_id"),
                fallback=result.metadata.get("lf_trace_id") or result.metadata.get("lf_observation_id"),
            )
            self._manager.record_execution_result(
                run.runId,
                experiment_key,
                item_reference=item_reference,
                input_text=result.input,
                expected_output=result.expected_output,
                output_text=result.output,
                error=result.error,
                trace_id=self._string_or_none(result.metadata.get("lf_trace_id")),
                observation_id=self._string_or_none(result.metadata.get("lf_observation_id")),
            )
            if result.error:
                stats["errors"] += 1
                self._manager.add_event(
                    run.runId,
                    "execution_error",
                    {
                        "experimentKey": experiment_key,
                        "message": result.error,
                        "itemId": result.metadata.get("lf_item_id"),
                        "itemReference": item_reference,
                    },
                )
            else:
                stats["completed"] += 1
                self._manager.add_event(
                    run.runId,
                    "execution_completed",
                    {
                        "experimentKey": experiment_key,
                        "itemId": result.metadata.get("lf_item_id"),
                        "itemReference": item_reference,
                    },
                )
                for metric in pointwise_metrics:
                    await pointwise_queues[metric.key].put((experiment_key, result))
                item_id = result.metadata.get("lf_item_id")
                if group_metrics and item_id is not None:
                    item_key = str(item_id)
                    group = group_buffer.setdefault(item_key, {})
                    group[experiment_key] = result
                    if len(group) == len(run.selectedModels):
                        ready_group = dict(group_buffer.pop(item_key))
                        for metric in group_metrics:
                            await group_queues[metric.key].put(ready_group)

            self._manager.record_model_progress(
                run.runId,
                experiment_key,
                completed=stats["completed"],
                total=actual_total,
                errors=stats["errors"],
            )

        await asyncio.gather(*[
            self._run_experiment(
                run.runId,
                runtime_config,
                model.experimentKey,
                experiment,
                runtime_rows,
                on_execution_result,
            )
            for model, experiment in zip(run.selectedModels, runtime_config.experiments, strict=True)
        ])

        self._emit_phase(run.runId, RunPhase.EVALUATING)
        await self._drain_workers(pointwise_queues, pointwise_workers)
        await self._drain_workers(group_queues, group_workers)
        get_client().flush()

    async def _run_experiment(
        self,
        run_id: str,
        config,
        experiment_key: str,
        experiment,
        rows,
        on_result,
    ) -> list[ExecutionResult]:
        caller_cls, caller_kwargs = build_execution_caller(experiment, rows)
        async with caller_cls(**caller_kwargs) as client:
            executor = Executor(
                dataset=rows,
                llm_client=client,
                system_prompt=config.system_prompt,
                experiment_name=experiment.name,
                experiment_description=f"Experiment: {experiment.name} with model {experiment.litellm.model}",
                timeout_s=experiment.timeout_s,
            )
            if not rows:
                return []
            dataset_id = rows[0][2].get("lf_dataset_id")
            if not dataset_id:
                raise ValueError("Rows must include 'lf_dataset_id' after dataset upload")
            run_name = f"{experiment.name} - {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S+00:00')}"
            queue: asyncio.Queue[Any] = asyncio.Queue()
            for row in rows:
                queue.put_nowait(row)

            results: list[ExecutionResult] = []
            self._manager.add_event(
                run_id,
                "model_started",
                {
                    "experimentKey": experiment_key,
                    "backendModel": experiment.litellm.model,
                },
            )

            async def worker() -> None:
                while not queue.empty():
                    try:
                        row = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    result = await executor._execute_row(row, run_name, dataset_id)
                    results.append(result)
                    await on_result(experiment_key, result)

            worker_count = min(executor.max_concurrency, len(rows))
            await asyncio.gather(*[worker() for _ in range(worker_count)])

        self._manager.add_event(
            run_id,
            "model_finished",
            {
                "experimentKey": experiment_key,
                "items": len(results),
                "errors": sum(1 for result in results if result.error),
            },
        )
        return results

    async def _start_pointwise_workers(self, run: RunState, config, metrics, queues):
        workers: dict[str, list[asyncio.Task[None]]] = {}
        for metric in metrics:
            evaluator = build_evaluator(results=[], **build_metric_evaluator_kwargs(metric, config))
            queue = queues[metric.key]
            worker_count = max(1, min(config.evaluation.max_concurrency or 1, run.sampleLimit))
            workers[metric.key] = [
                asyncio.create_task(self._pointwise_worker(run.runId, metric, evaluator, queue))
                for _ in range(worker_count)
            ]
        return workers

    async def _start_group_workers(self, run: RunState, config, metrics, queues):
        workers: dict[str, list[asyncio.Task[None]]] = {}
        for metric in metrics:
            evaluator = build_evaluator(groups=[], **build_metric_evaluator_kwargs(metric, config))
            queue = queues[metric.key]
            worker_count = max(1, min(config.evaluation.max_concurrency or 1, run.sampleLimit))
            workers[metric.key] = [
                asyncio.create_task(self._group_worker(run.runId, metric, evaluator, queue))
                for _ in range(worker_count)
            ]
        return workers

    async def _pointwise_worker(self, run_id: str, metric: MetricSpec, evaluator, queue: asyncio.Queue[Any]) -> None:
        while True:
            item = await queue.get()
            try:
                if item is None:
                    return
                experiment_key, result = item
                eval_result = await evaluator._evaluate_one(result)
                item_reference = self._item_reference(
                    result.metadata.get("lf_item_id"),
                    fallback=result.metadata.get("lf_trace_id") or result.metadata.get("lf_observation_id"),
                )
                if eval_result.score is not None:
                    self._manager.record_metric_result(
                        run_id,
                        experiment_key,
                        metric.key,
                        item_reference=item_reference,
                        score=eval_result.score,
                        explanation=eval_result.explanation,
                    )
                elif eval_result.error:
                    self._manager.record_metric_result(
                        run_id,
                        experiment_key,
                        metric.key,
                        item_reference=item_reference,
                        score=None,
                        explanation=eval_result.explanation,
                        error=eval_result.error,
                    )
            finally:
                queue.task_done()

    async def _group_worker(self, run_id: str, metric: MetricSpec, evaluator, queue: asyncio.Queue[Any]) -> None:
        while True:
            group = await queue.get()
            try:
                if group is None:
                    return
                eval_results = await evaluator._evaluate_group(group)
                for experiment_key, eval_result in zip(group, eval_results):
                    source_result = group[experiment_key]
                    item_reference = self._item_reference(
                        eval_result.metadata.get("lf_item_id") or source_result.metadata.get("lf_item_id"),
                        fallback=source_result.metadata.get("lf_trace_id") or source_result.metadata.get("lf_observation_id"),
                    )
                    if eval_result.score is not None:
                        self._manager.record_metric_result(
                            run_id,
                            experiment_key,
                            metric.key,
                            item_reference=item_reference,
                            score=eval_result.score,
                            explanation=eval_result.explanation,
                        )
                    elif eval_result.error:
                        self._manager.record_metric_result(
                            run_id,
                            experiment_key,
                            metric.key,
                            item_reference=item_reference,
                            score=None,
                            explanation=eval_result.explanation,
                            error=eval_result.error,
                        )
            finally:
                queue.task_done()

    async def _drain_workers(self, queues, workers) -> None:
        for queue in queues.values():
            await queue.join()
        for key, tasks in workers.items():
            queue = queues[key]
            for _ in tasks:
                await queue.put(None)
        for tasks in workers.values():
            if tasks:
                await asyncio.gather(*tasks)

    def _emit_phase(self, run_id: str, phase: RunPhase) -> None:
        self._manager.set_phase(run_id, phase)
        self._manager.add_event(run_id, "phase_changed", {"phase": phase.value})

    def _require_run(self, run_id: str) -> RunState:
        run = self._manager.get_run(run_id)
        if run is None:
            raise KeyError(f"Run not found: {run_id!r}")
        return run

    async def _sleep_if_needed(self) -> None:
        if self._fake_delay_s > 0:
            await asyncio.sleep(self._fake_delay_s)

    @staticmethod
    def _string_or_none(value: Any) -> str | None:
        return str(value) if value is not None else None

    @classmethod
    def _item_reference(cls, value: Any, fallback: Any = None) -> str:
        candidate = cls._string_or_none(value)
        if candidate:
            return candidate
        fallback_value = cls._string_or_none(fallback)
        if fallback_value:
            return fallback_value
        return "unknown-item"

    @staticmethod
    def _fake_score(model_index: int, metric_index: int, item_index: int, model_count: int) -> float:
        leader_index = (item_index + metric_index) % max(model_count, 1)
        leader_bonus = 0.14 if model_index == leader_index else -0.03
        score = 0.62 + (model_index * 0.025) + leader_bonus + ((item_index % 2) * 0.015) - (metric_index * 0.01)
        return max(0.0, min(1.0, score))
