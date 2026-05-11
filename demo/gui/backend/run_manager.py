"""Thread-safe in-memory RunManager for GUI demo run state."""

from __future__ import annotations

import copy
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from demo.gui.backend.config_loader import load_demo_config
from demo.gui.backend.models import (
    CreateRunRequest,
    EventRow,
    ExecutionMode,
    LeaderboardEntry,
    MetricHistoryPoint,
    MetricResults,
    MetricSpec,
    ModelProgress,
    ResolvedModel,
    ResultExample,
    RunPhase,
    RunResultsResponse,
    RunState,
)
from src.evaluation.evaluators import evaluator_mode

_RECENT_EVENTS_LIMIT = 20


@dataclass
class _ExecutionRecord:
    item_reference: str
    input: str
    expected_output: str
    output: str | None
    error: str | None
    trace_id: str | None
    observation_id: str | None


@dataclass
class _MetricRecord:
    item_reference: str
    score: float | None
    explanation: str | None
    error: str | None


@dataclass
class _RunData:
    run_id: str
    phase: RunPhase
    execution_mode: ExecutionMode
    dataset_name: str
    sample_limit: int
    items_total: int
    selected_models: list[ResolvedModel]
    selected_metrics: list[MetricSpec]
    active_metric_key: str | None
    scores: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    execution_results: dict[str, dict[str, _ExecutionRecord]] = field(default_factory=dict)
    metric_results: dict[str, dict[str, dict[str, _MetricRecord]]] = field(default_factory=dict)
    model_progress: dict[str, ModelProgress] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    events: list[EventRow] = field(default_factory=list)
    next_event_sequence: int = 1


class RunManager:
    """Thread-safe in-memory store for demo run state."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runs: dict[str, _RunData] = {}

    def create_run(self, req: CreateRunRequest) -> RunState:
        config = load_demo_config()
        key_to_mapping = {mapping.experimentKey: mapping for mapping in config.modelMapping}

        resolved: list[ResolvedModel] = []
        for experiment_key in req.selectedModels:
            mapping = key_to_mapping.get(experiment_key)
            if mapping is None:
                raise ValueError(f"Unknown experimentKey: {experiment_key!r}")
            resolved.append(
                ResolvedModel(
                    experimentKey=mapping.experimentKey,
                    experimentName=mapping.experimentName,
                    showcaseModel=mapping.showcaseModel,
                    backendModel=mapping.backendModel,
                )
            )

        for metric in req.metrics:
            mode = evaluator_mode(metric.method)
            if mode == "group" and len(resolved) < 2:
                raise ValueError(f"Metric requires at least 2 selected models: {metric.method!r}")

        items_total = req.sampleLimit * len(resolved)
        dataset_name = req.runtimeDatasetName or config.runtimeDatasetName
        execution_mode = ExecutionMode.FAKE if req.fakeMode else ExecutionMode.REAL
        active_metric_key = req.metrics[0].key
        model_progress = {
            model.experimentKey: ModelProgress(
                experimentKey=model.experimentKey,
                experimentName=model.experimentName,
                showcaseModel=model.showcaseModel,
                backendModel=model.backendModel,
                completed=0,
                total=req.sampleLimit,
                errors=0,
            )
            for model in resolved
        }

        run_data = _RunData(
            run_id=str(uuid.uuid4()),
            phase=RunPhase.CONFIGURING,
            execution_mode=execution_mode,
            dataset_name=dataset_name,
            sample_limit=req.sampleLimit,
            items_total=items_total,
            selected_models=resolved,
            selected_metrics=[metric.model_copy(deep=True) for metric in req.metrics],
            active_metric_key=active_metric_key,
            model_progress=model_progress,
        )

        with self._lock:
            self._runs[run_data.run_id] = run_data
            return self._snapshot(run_data)

    def get_run(self, run_id: str) -> RunState | None:
        with self._lock:
            run_data = self._runs.get(run_id)
            if run_data is None:
                return None
            return self._snapshot(run_data)

    def start_run(self, run_id: str) -> RunState:
        with self._lock:
            run_data = self._require_run(run_id)
            if run_data.phase != RunPhase.CONFIGURING:
                raise ValueError(f"Run cannot start from phase: {run_data.phase.value!r}")
            run_data.phase = RunPhase.UPLOADING
            return self._snapshot(run_data)

    def sync_run_totals(self, run_id: str, total_per_model: int) -> None:
        if total_per_model < 0:
            raise ValueError("total_per_model must be non-negative")
        with self._lock:
            run_data = self._require_run(run_id)
            for model_progress in run_data.model_progress.values():
                if model_progress.completed > total_per_model:
                    raise ValueError("completed cannot exceed synced total")
                if model_progress.errors > total_per_model:
                    raise ValueError("errors cannot exceed synced total")
                if model_progress.completed + model_progress.errors > total_per_model:
                    raise ValueError("completed plus errors cannot exceed synced total")
            run_data.items_total = total_per_model * len(run_data.selected_models)
            for model_progress in run_data.model_progress.values():
                model_progress.total = total_per_model

    def set_phase(self, run_id: str, phase: RunPhase) -> None:
        with self._lock:
            run_data = self._require_run(run_id)
            run_data.phase = phase

    def add_error(self, run_id: str, message: str) -> None:
        with self._lock:
            run_data = self._require_run(run_id)
            run_data.errors.append(message)

    def record_execution_result(
        self,
        run_id: str,
        experiment_key: str,
        *,
        item_reference: str,
        input_text: str,
        expected_output: str,
        output_text: str | None,
        error: str | None,
        trace_id: str | None = None,
        observation_id: str | None = None,
    ) -> None:
        with self._lock:
            run_data = self._require_run(run_id)
            self._require_model(run_data, experiment_key)
            run_data.execution_results.setdefault(experiment_key, {})[item_reference] = _ExecutionRecord(
                item_reference=item_reference,
                input=input_text,
                expected_output=expected_output,
                output=output_text,
                error=error,
                trace_id=trace_id,
                observation_id=observation_id,
            )

    def record_metric_result(
        self,
        run_id: str,
        experiment_key: str,
        metric_key: str,
        *,
        item_reference: str,
        score: float | None,
        explanation: str | None = None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            run_data = self._require_run(run_id)
            self._require_model(run_data, experiment_key)
            self._require_metric(run_data, metric_key)
            run_data.metric_results.setdefault(metric_key, {}).setdefault(experiment_key, {})[item_reference] = _MetricRecord(
                item_reference=item_reference,
                score=score,
                explanation=explanation,
                error=error,
            )
            if score is not None:
                scores = run_data.scores.setdefault(experiment_key, {}).setdefault(metric_key, [])
                scores.append(score)
                self._append_event_locked(
                    run_data,
                    "metric_scored",
                    {
                        "experimentKey": experiment_key,
                        "metricKey": metric_key,
                        "score": score,
                        "avgScore": sum(scores) / len(scores),
                        "scoredCount": len(scores),
                        "itemReference": item_reference,
                    },
                )
            elif error:
                self._append_event_locked(
                    run_data,
                    "metric_error",
                    {
                        "experimentKey": experiment_key,
                        "metricKey": metric_key,
                        "message": error,
                        "itemReference": item_reference,
                    },
                )

    def record_score(self, run_id: str, experiment_key: str, metric_key: str, score: float) -> None:
        with self._lock:
            run_data = self._require_run(run_id)
            self._require_model(run_data, experiment_key)
            self._require_metric(run_data, metric_key)
            run_data.scores.setdefault(experiment_key, {}).setdefault(metric_key, []).append(score)

    def record_model_progress(
        self,
        run_id: str,
        experiment_key: str,
        *,
        completed: int,
        total: int,
        errors: int,
    ) -> None:
        if completed < 0 or total < 0 or errors < 0:
            raise ValueError("progress counts must be non-negative")
        if completed > total:
            raise ValueError("completed cannot exceed total")
        if errors > total:
            raise ValueError("errors cannot exceed total")
        if completed + errors > total:
            raise ValueError("completed plus errors cannot exceed total")
        with self._lock:
            run_data = self._require_run(run_id)
            model_progress = self._require_model(run_data, experiment_key)
            model_progress.completed = completed
            model_progress.total = total
            model_progress.errors = errors

    def add_event(self, run_id: str, kind: str, payload: dict) -> EventRow:
        with self._lock:
            run_data = self._require_run(run_id)
            return self._append_event_locked(run_data, kind, payload)

    def get_leaderboard(self, run_id: str, metric_key: str) -> list[LeaderboardEntry]:
        with self._lock:
            run_data = self._require_run(run_id)
            self._require_metric(run_data, metric_key)
            return [entry.model_copy(deep=True) for entry in self._build_leaderboard(run_data, metric_key)]

    def get_results(self, run_id: str) -> RunResultsResponse:
        with self._lock:
            run_data = self._require_run(run_id)
            snapshot = self._snapshot(run_data)
            metrics = [
                MetricResults(
                    key=metric.key,
                    label=metric.label,
                    method=metric.method,
                    leaderboard=[entry.model_copy(deep=True) for entry in self._build_leaderboard(run_data, metric.key)],
                    history=self._build_metric_history(run_data, metric.key),
                    lowestScored=self._build_metric_examples(run_data, metric.key),
                )
                for metric in run_data.selected_metrics
            ]
            return RunResultsResponse(
                runId=snapshot.runId,
                phase=snapshot.phase,
                executionMode=snapshot.executionMode,
                datasetName=snapshot.datasetName,
                sampleLimit=snapshot.sampleLimit,
                itemsTotal=snapshot.itemsTotal,
                completedItems=snapshot.completedItems,
                errorCount=snapshot.errorCount,
                selectedModels=[model.model_copy(deep=True) for model in snapshot.selectedModels],
                selectedMetrics=[metric.model_copy(deep=True) for metric in snapshot.selectedMetrics],
                activeMetricKey=snapshot.activeMetricKey,
                modelProgress={
                    key: progress.model_copy(deep=True)
                    for key, progress in snapshot.modelProgress.items()
                },
                errors=list(snapshot.errors),
                recentEvents=[event.model_copy(deep=True) for event in snapshot.recentEvents],
                metrics=metrics,
            )

    def get_events(self, run_id: str, after: int | None = None) -> list[EventRow]:
        with self._lock:
            run_data = self._require_run(run_id)
            filtered = [
                event.model_copy(deep=True)
                for event in run_data.events
                if after is None or event.sequence > after
            ]
            return filtered

    def _require_run(self, run_id: str) -> _RunData:
        run_data = self._runs.get(run_id)
        if run_data is None:
            raise KeyError(f"Run not found: {run_id!r}")
        return run_data

    @staticmethod
    def _require_model(run_data: _RunData, experiment_key: str) -> ModelProgress:
        model_progress = run_data.model_progress.get(experiment_key)
        if model_progress is None:
            raise KeyError(f"Model not registered in run: {experiment_key!r}")
        return model_progress

    @staticmethod
    def _require_metric(run_data: _RunData, metric_key: str) -> MetricSpec:
        for metric in run_data.selected_metrics:
            if metric.key == metric_key:
                return metric
        raise ValueError(f"Metric not selected for run: {metric_key!r}")

    def _append_event_locked(self, run_data: _RunData, kind: str, payload: dict) -> EventRow:
        event = EventRow(
            sequence=run_data.next_event_sequence,
            eventId=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            kind=kind,
            payload=copy.deepcopy(payload),
        )
        run_data.next_event_sequence += 1
        run_data.events.append(event)
        return event.model_copy(deep=True)

    @staticmethod
    def _build_leaderboard(run_data: _RunData, metric_key: str) -> list[LeaderboardEntry]:
        entries: list[LeaderboardEntry] = []
        for model in run_data.selected_models:
            scores = run_data.scores.get(model.experimentKey, {}).get(metric_key, [])
            model_progress = run_data.model_progress[model.experimentKey]
            entries.append(
                LeaderboardEntry(
                    experimentKey=model.experimentKey,
                    experimentName=model.experimentName,
                    showcaseModel=model.showcaseModel,
                    backendModel=model.backendModel,
                    avgScore=(sum(scores) / len(scores)) if scores else None,
                    scoredCount=len(scores),
                    totalCount=model_progress.total,
                )
            )

        entries.sort(
            key=lambda entry: (
                entry.avgScore is None,
                -(entry.avgScore or 0.0),
                entry.experimentName,
            )
        )
        return entries

    @staticmethod
    def _build_metric_history(run_data: _RunData, metric_key: str) -> list[MetricHistoryPoint]:
        history: list[MetricHistoryPoint] = []
        latest_scores: dict[str, float] = {}
        for event in run_data.events:
            if event.kind != "metric_scored":
                continue
            event_metric_key = event.payload.get("metricKey")
            experiment_key = event.payload.get("experimentKey")
            avg_score = event.payload.get("avgScore")
            if event_metric_key != metric_key:
                continue
            if not isinstance(experiment_key, str) or not isinstance(avg_score, (int, float)):
                continue
            latest_scores[experiment_key] = float(avg_score)
            history.append(MetricHistoryPoint(sequence=event.sequence, averages=dict(latest_scores)))
        return history

    @staticmethod
    def _build_metric_examples(run_data: _RunData, metric_key: str) -> list[ResultExample]:
        examples: list[ResultExample] = []
        metric_results = run_data.metric_results.get(metric_key, {})
        for model in run_data.selected_models:
            execution_records = run_data.execution_results.get(model.experimentKey, {})
            model_metric_results = metric_results.get(model.experimentKey, {})
            for item_reference, metric_record in model_metric_results.items():
                execution_record = execution_records.get(item_reference)
                if execution_record is None:
                    continue
                examples.append(
                    ResultExample(
                        experimentKey=model.experimentKey,
                        experimentName=model.experimentName,
                        showcaseModel=model.showcaseModel,
                        backendModel=model.backendModel,
                        itemReference=item_reference,
                        input=execution_record.input,
                        expectedOutput=execution_record.expected_output,
                        output=execution_record.output,
                        score=metric_record.score,
                        explanation=metric_record.explanation,
                        executionError=execution_record.error,
                        metricError=metric_record.error,
                        traceId=execution_record.trace_id,
                        observationId=execution_record.observation_id,
                    )
                )

        examples.sort(
            key=lambda example: (
                example.score is None,
                example.score if example.score is not None else 2.0,
                example.experimentName,
                example.itemReference,
            )
        )
        return examples[:6]

    @staticmethod
    def _snapshot(run_data: _RunData) -> RunState:
        completed_items = sum(model.completed for model in run_data.model_progress.values())
        metric_error_count = sum(
            1
            for metric_results in run_data.metric_results.values()
            for model_results in metric_results.values()
            for record in model_results.values()
            if record.error is not None
        )
        error_count = len(run_data.errors) + sum(model.errors for model in run_data.model_progress.values()) + metric_error_count
        return RunState(
            runId=run_data.run_id,
            phase=run_data.phase,
            executionMode=run_data.execution_mode,
            datasetName=run_data.dataset_name,
            sampleLimit=run_data.sample_limit,
            itemsTotal=run_data.items_total,
            completedItems=completed_items,
            errorCount=error_count,
            selectedModels=[model.model_copy(deep=True) for model in run_data.selected_models],
            selectedMetrics=[metric.model_copy(deep=True) for metric in run_data.selected_metrics],
            activeMetricKey=run_data.active_metric_key,
            modelProgress={
                key: progress.model_copy(deep=True)
                for key, progress in run_data.model_progress.items()
            },
            errors=list(run_data.errors),
            recentEvents=[event.model_copy(deep=True) for event in run_data.events[-_RECENT_EVENTS_LIMIT:]],
        )
