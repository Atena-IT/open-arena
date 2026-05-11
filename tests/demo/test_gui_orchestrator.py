"""Tests for GUI background orchestration."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import demo.gui.backend.orchestrator as backend_orchestrator

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.gui.backend.config_loader import load_demo_config
from demo.gui.backend.models import CreateRunRequest, MetricSpec, RunPhase
from demo.gui.backend.orchestrator import DemoOrchestrator
from demo.gui.backend.run_manager import RunManager
from src.evaluation.types import EvaluationResult
from src.execution import ExecutionResult



def _all_model_keys() -> list[str]:
    return [mapping.experimentKey for mapping in load_demo_config().modelMapping]



def test_fake_orchestrator_completes_run_and_scores_models() -> None:
    manager = RunManager()
    keys = _all_model_keys()[:2]
    run = manager.create_run(
        CreateRunRequest(
            sampleLimit=2,
            selectedModels=keys,
            metrics=[MetricSpec(key="accuracy", label="Accuracy", method="llm_as_judge")],
            fakeMode=True,
        )
    )
    manager.start_run(run.runId)

    orchestrator = DemoOrchestrator(manager=manager, fake_delay_s=0)
    asyncio.run(orchestrator.run(run.runId))

    updated = manager.get_run(run.runId)
    assert updated is not None
    assert updated.phase == RunPhase.COMPLETED
    assert updated.completedItems == 4
    assert updated.errorCount == 0
    assert all(updated.modelProgress[key].completed == 2 for key in keys)

    leaderboard = manager.get_leaderboard(run.runId, "accuracy")
    assert len(leaderboard) == 2
    assert all(entry.scoredCount == 2 for entry in leaderboard)
    assert all(entry.avgScore is not None for entry in leaderboard)

    results = manager.get_results(run.runId)
    assert results.metrics[0].history
    assert results.metrics[0].lowestScored
    assert results.metrics[0].lowestScored[0].itemReference.startswith("demo-item-")

    events = manager.get_events(run.runId)
    assert any(event.kind == "phase_changed" and event.payload["phase"] == "running" for event in events)
    assert any(event.kind == "phase_changed" and event.payload["phase"] == "completed" for event in events)
    assert sum(event.kind == "metric_scored" for event in events) == 4



def test_fake_orchestrator_records_multiple_metrics() -> None:
    manager = RunManager()
    key = _all_model_keys()[0]
    run = manager.create_run(
        CreateRunRequest(
            sampleLimit=3,
            selectedModels=[key],
            metrics=[
                MetricSpec(key="accuracy", label="Accuracy", method="llm_as_judge"),
                MetricSpec(key="clarity", label="Clarity", method="llm_as_judge"),
            ],
            fakeMode=True,
        )
    )
    manager.start_run(run.runId)

    orchestrator = DemoOrchestrator(manager=manager, fake_delay_s=0)
    asyncio.run(orchestrator.run(run.runId))

    accuracy = manager.get_leaderboard(run.runId, "accuracy")
    clarity = manager.get_leaderboard(run.runId, "clarity")
    assert accuracy[0].scoredCount == 3
    assert clarity[0].scoredCount == 3
    assert accuracy[0].avgScore is not None
    assert clarity[0].avgScore is not None


def test_real_mode_syncs_totals_to_loaded_rows() -> None:
    manager = RunManager()
    keys = _all_model_keys()[:2]
    run = manager.create_run(
        CreateRunRequest(
            sampleLimit=5,
            selectedModels=keys,
            metrics=[MetricSpec(key="accuracy", label="Accuracy", method="llm_as_judge")],
            fakeMode=False,
        )
    )
    manager.start_run(run.runId)

    orchestrator = DemoOrchestrator(manager=manager, fake_delay_s=0)
    rows = [
        ("input 1", "expected 1", {"lf_dataset_id": "ds-1", "lf_item_id": "item-1"}),
        ("input 2", "expected 2", {"lf_dataset_id": "ds-1", "lf_item_id": "item-2"}),
    ]

    class FakePointwiseEvaluator:
        async def _evaluate_one(self, result: ExecutionResult) -> EvaluationResult:
            return EvaluationResult(
                input=result.input,
                expected_output=result.expected_output,
                output=result.output or "",
                model_name=result.model_name,
                experiment_name=result.experiment_name,
                score=0.75,
                metadata=dict(result.metadata),
            )

    async def fake_upload_rows(input_rows, dataset_name: str, description: str):
        return input_rows

    async def fake_run_experiment(self, run_id: str, config, experiment_key: str, experiment, rows, on_result):
        results: list[ExecutionResult] = []
        for input_text, expected_output, metadata in rows:
            result = ExecutionResult(
                input=input_text,
                expected_output=expected_output,
                output=f"output-{experiment_key}-{metadata['lf_item_id']}",
                model_name=experiment.litellm.model,
                experiment_name=experiment.name,
                metadata=dict(metadata),
            )
            results.append(result)
            await on_result(experiment_key, result)
        return results

    original_load_runtime_rows = backend_orchestrator.load_runtime_rows
    original_upload_rows = backend_orchestrator.upload_rows
    original_build_evaluator = backend_orchestrator.build_evaluator
    original_run_experiment = DemoOrchestrator._run_experiment
    backend_orchestrator.load_runtime_rows = lambda config: rows
    backend_orchestrator.upload_rows = fake_upload_rows
    backend_orchestrator.build_evaluator = lambda **kwargs: FakePointwiseEvaluator()
    DemoOrchestrator._run_experiment = fake_run_experiment
    try:
        asyncio.run(orchestrator.run(run.runId))
    finally:
        backend_orchestrator.load_runtime_rows = original_load_runtime_rows
        backend_orchestrator.upload_rows = original_upload_rows
        backend_orchestrator.build_evaluator = original_build_evaluator
        DemoOrchestrator._run_experiment = original_run_experiment

    updated = manager.get_run(run.runId)
    assert updated is not None
    assert updated.phase == RunPhase.COMPLETED
    assert updated.itemsTotal == 4
    assert updated.completedItems == 4
    assert all(updated.modelProgress[key].total == 2 for key in keys)

    leaderboard = manager.get_leaderboard(run.runId, "accuracy")
    assert len(leaderboard) == 2
    assert all(entry.scoredCount == 2 for entry in leaderboard)
    assert all(entry.avgScore == 0.75 for entry in leaderboard)
