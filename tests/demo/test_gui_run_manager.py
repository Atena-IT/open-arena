"""Tests for RunManager and run-state API endpoints."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import demo.gui.backend.main as backend_main
from demo.gui.backend.config_loader import load_demo_config
from demo.gui.backend.main import _run_manager, app
from demo.gui.backend.models import CreateRunRequest, ExecutionMode, MetricSpec, RunPhase
from demo.gui.backend.run_manager import RunManager

client = TestClient(app)


def _first_model_key() -> str:
    return load_demo_config().modelMapping[0].experimentKey


def _all_model_keys() -> list[str]:
    return [mapping.experimentKey for mapping in load_demo_config().modelMapping]


def _make_request(**overrides) -> dict:
    payload: dict = {
        "sampleLimit": 5,
        "selectedModels": [_first_model_key()],
        "metrics": [{"key": "accuracy", "label": "Accuracy", "method": "llm_as_judge"}],
        "fakeMode": True,
    }
    payload.update(overrides)
    return payload


# ── unit tests: RunManager ────────────────────────────────────────────────────


def test_create_run_initial_phase() -> None:
    manager = RunManager()
    state = manager.create_run(
        CreateRunRequest(
            sampleLimit=3,
            selectedModels=[_first_model_key()],
            metrics=[MetricSpec(key="acc", label="Acc", method="llm_as_judge")],
        )
    )
    assert state.phase == RunPhase.CONFIGURING
    assert state.executionMode == ExecutionMode.REAL


def test_create_run_items_total_from_sample_limit() -> None:
    manager = RunManager()
    keys = _all_model_keys()[:2]
    state = manager.create_run(
        CreateRunRequest(
            sampleLimit=7,
            selectedModels=keys,
            metrics=[MetricSpec(key="acc", label="Acc", method="llm_as_judge")],
            fakeMode=True,
        )
    )
    assert state.sampleLimit == 7
    assert state.itemsTotal == 14
    assert state.executionMode == ExecutionMode.FAKE


def test_create_run_model_progress_initialized() -> None:
    manager = RunManager()
    key = _first_model_key()
    state = manager.create_run(
        CreateRunRequest(
            sampleLimit=4,
            selectedModels=[key],
            metrics=[MetricSpec(key="acc", label="Acc", method="llm_as_judge")],
        )
    )
    progress = state.modelProgress[key]
    assert progress.completed == 0
    assert progress.total == 4
    assert progress.errors == 0
    assert state.completedItems == 0
    assert state.errorCount == 0


def test_record_score_updates_leaderboard_avg() -> None:
    manager = RunManager()
    key = _first_model_key()
    run = manager.create_run(
        CreateRunRequest(
            sampleLimit=5,
            selectedModels=[key],
            metrics=[MetricSpec(key="acc", label="Acc", method="llm_as_judge")],
        )
    )

    entries = manager.get_leaderboard(run.runId, "acc")
    assert entries[0].avgScore is None
    assert entries[0].scoredCount == 0

    manager.record_score(run.runId, key, "acc", 0.6)
    manager.record_score(run.runId, key, "acc", 0.8)

    entries = manager.get_leaderboard(run.runId, "acc")
    assert entries[0].avgScore == pytest.approx(0.7)
    assert entries[0].scoredCount == 2


def test_leaderboard_sorted_descending() -> None:
    keys = _all_model_keys()[:2]
    manager = RunManager()
    run = manager.create_run(
        CreateRunRequest(
            sampleLimit=5,
            selectedModels=keys,
            metrics=[MetricSpec(key="acc", label="Acc", method="llm_as_judge")],
        )
    )

    manager.record_score(run.runId, keys[0], "acc", 0.3)
    manager.record_score(run.runId, keys[1], "acc", 0.9)

    entries = manager.get_leaderboard(run.runId, "acc")
    assert entries[0].experimentKey == keys[1]
    assert entries[1].experimentKey == keys[0]


def test_leaderboard_none_scores_last() -> None:
    keys = _all_model_keys()[:2]
    manager = RunManager()
    run = manager.create_run(
        CreateRunRequest(
            sampleLimit=5,
            selectedModels=keys,
            metrics=[MetricSpec(key="acc", label="Acc", method="llm_as_judge")],
        )
    )

    manager.record_score(run.runId, keys[0], "acc", 0.5)

    entries = manager.get_leaderboard(run.runId, "acc")
    assert entries[0].experimentKey == keys[0]
    assert entries[1].avgScore is None


def test_record_model_progress_updates_state() -> None:
    manager = RunManager()
    key = _first_model_key()
    run = manager.create_run(
        CreateRunRequest(
            sampleLimit=10,
            selectedModels=[key],
            metrics=[MetricSpec(key="acc", label="Acc", method="llm_as_judge")],
        )
    )

    manager.record_model_progress(run.runId, key, completed=4, total=10, errors=1)
    updated = manager.get_run(run.runId)
    assert updated is not None
    assert updated.modelProgress[key].completed == 4
    assert updated.completedItems == 4
    assert updated.errorCount == 1


def test_record_model_progress_rejects_invalid_counts() -> None:
    manager = RunManager()
    key = _first_model_key()
    run = manager.create_run(
        CreateRunRequest(
            sampleLimit=3,
            selectedModels=[key],
            metrics=[MetricSpec(key="acc", label="Acc", method="llm_as_judge")],
        )
    )

    with pytest.raises(ValueError, match="completed cannot exceed total"):
        manager.record_model_progress(run.runId, key, completed=4, total=3, errors=0)

    with pytest.raises(ValueError, match="errors cannot exceed total"):
        manager.record_model_progress(run.runId, key, completed=1, total=3, errors=4)

    with pytest.raises(ValueError, match="completed plus errors cannot exceed total"):
        manager.record_model_progress(run.runId, key, completed=2, total=3, errors=2)


def test_record_score_rejects_unknown_metric() -> None:
    manager = RunManager()
    key = _first_model_key()
    run = manager.create_run(
        CreateRunRequest(
            sampleLimit=3,
            selectedModels=[key],
            metrics=[MetricSpec(key="acc", label="Acc", method="llm_as_judge")],
        )
    )

    with pytest.raises(ValueError, match="Metric not selected"):
        manager.record_score(run.runId, key, "other", 0.5)


def test_get_run_returns_snapshot_not_live_reference() -> None:
    manager = RunManager()
    key = _first_model_key()
    run = manager.create_run(
        CreateRunRequest(
            sampleLimit=3,
            selectedModels=[key],
            metrics=[MetricSpec(key="acc", label="Acc", method="llm_as_judge")],
        )
    )

    snapshot = manager.get_run(run.runId)
    manager.record_model_progress(run.runId, key, completed=2, total=3, errors=0)
    refreshed = manager.get_run(run.runId)

    assert snapshot is not None and refreshed is not None
    assert snapshot.modelProgress[key].completed == 0
    assert refreshed.modelProgress[key].completed == 2


def test_add_event_appended_with_sequence() -> None:
    manager = RunManager()
    key = _first_model_key()
    run = manager.create_run(
        CreateRunRequest(
            sampleLimit=3,
            selectedModels=[key],
            metrics=[MetricSpec(key="acc", label="Acc", method="llm_as_judge")],
        )
    )

    event_one = manager.add_event(run.runId, "score_recorded", {"model": key, "score": 0.8})
    event_two = manager.add_event(run.runId, "score_recorded", {"model": key, "score": 0.9})

    assert event_one.sequence == 1
    assert event_two.sequence == 2
    assert manager.get_events(run.runId, after=1)[0].sequence == 2


def test_add_event_copies_nested_payload() -> None:
    manager = RunManager()
    key = _first_model_key()
    run = manager.create_run(
        CreateRunRequest(
            sampleLimit=3,
            selectedModels=[key],
            metrics=[MetricSpec(key="acc", label="Acc", method="llm_as_judge")],
        )
    )

    payload = {"model": key, "scores": [{"metric": "acc", "value": 0.8}]}
    manager.add_event(run.runId, "score_recorded", payload)
    payload["scores"][0]["value"] = 0.1

    stored = manager.get_events(run.runId)
    assert stored[0].payload["scores"][0]["value"] == pytest.approx(0.8)


def test_recent_events_are_truncated_in_run_state() -> None:
    manager = RunManager()
    key = _first_model_key()
    run = manager.create_run(
        CreateRunRequest(
            sampleLimit=3,
            selectedModels=[key],
            metrics=[MetricSpec(key="acc", label="Acc", method="llm_as_judge")],
        )
    )

    for index in range(25):
        manager.add_event(run.runId, "tick", {"index": index})

    updated = manager.get_run(run.runId)
    assert updated is not None
    assert len(updated.recentEvents) == 20
    assert updated.recentEvents[0].sequence == 6
    assert updated.recentEvents[-1].sequence == 25


def test_get_run_unknown_returns_none() -> None:
    manager = RunManager()
    assert manager.get_run("does-not-exist") is None


def test_start_run_moves_phase_to_uploading() -> None:
    manager = RunManager()
    key = _first_model_key()
    run = manager.create_run(
        CreateRunRequest(
            sampleLimit=3,
            selectedModels=[key],
            metrics=[MetricSpec(key="acc", label="Acc", method="llm_as_judge")],
        )
    )

    started = manager.start_run(run.runId)
    assert started.phase == RunPhase.UPLOADING


def test_sync_run_totals_updates_items_total_and_model_totals() -> None:
    manager = RunManager()
    keys = _all_model_keys()[:2]
    run = manager.create_run(
        CreateRunRequest(
            sampleLimit=5,
            selectedModels=keys,
            metrics=[MetricSpec(key="acc", label="Acc", method="llm_as_judge")],
        )
    )

    manager.sync_run_totals(run.runId, 2)
    updated = manager.get_run(run.runId)
    assert updated is not None
    assert updated.itemsTotal == 4
    assert all(updated.modelProgress[key].total == 2 for key in keys)


def test_sync_run_totals_rejects_totals_below_existing_progress() -> None:
    manager = RunManager()
    key = _first_model_key()
    run = manager.create_run(
        CreateRunRequest(
            sampleLimit=5,
            selectedModels=[key],
            metrics=[MetricSpec(key="acc", label="Acc", method="llm_as_judge")],
        )
    )
    manager.record_model_progress(run.runId, key, completed=2, total=5, errors=1)

    with pytest.raises(ValueError, match="completed plus errors cannot exceed synced total"):
        manager.sync_run_totals(run.runId, 2)


def test_get_results_returns_metric_history_and_examples() -> None:
    manager = RunManager()
    key = _first_model_key()
    run = manager.create_run(
        CreateRunRequest(
            sampleLimit=3,
            selectedModels=[key],
            metrics=[MetricSpec(key="acc", label="Acc", method="llm_as_judge")],
            fakeMode=True,
        )
    )

    manager.record_execution_result(
        run.runId,
        key,
        item_reference="item-1",
        input_text="mission brief",
        expected_output="expected memo",
        output_text="generated memo",
        error=None,
        trace_id="trace-1",
        observation_id="obs-1",
    )
    manager.record_metric_result(
        run.runId,
        key,
        "acc",
        item_reference="item-1",
        score=0.25,
        explanation="Thin evidence discipline.",
    )
    manager.set_phase(run.runId, RunPhase.COMPLETED)

    results = manager.get_results(run.runId)
    assert results.phase == RunPhase.COMPLETED
    assert results.metrics[0].leaderboard[0].avgScore == pytest.approx(0.25)
    assert results.metrics[0].history[0].averages[key] == pytest.approx(0.25)
    example = results.metrics[0].lowestScored[0]
    assert example.itemReference == "item-1"
    assert example.traceId == "trace-1"
    assert example.explanation == "Thin evidence discipline."


def test_metric_errors_increment_run_error_count() -> None:
    manager = RunManager()
    key = _first_model_key()
    run = manager.create_run(
        CreateRunRequest(
            sampleLimit=3,
            selectedModels=[key],
            metrics=[MetricSpec(key="acc", label="Acc", method="llm_as_judge")],
            fakeMode=True,
        )
    )

    manager.record_execution_result(
        run.runId,
        key,
        item_reference="item-2",
        input_text="mission brief",
        expected_output="expected memo",
        output_text="generated memo",
        error=None,
    )
    manager.record_metric_result(
        run.runId,
        key,
        "acc",
        item_reference="item-2",
        score=None,
        error="judge timeout",
    )

    updated = manager.get_run(run.runId)
    assert updated is not None
    assert updated.errorCount == 1
    results = manager.get_results(run.runId)
    assert results.metrics[0].lowestScored[0].metricError == "judge timeout"


# ── API tests ─────────────────────────────────────────────────────────────────


def test_post_runs_returns_valid_run_state() -> None:
    key = _first_model_key()
    response = client.post("/api/runs", json=_make_request(selectedModels=[key]))
    assert response.status_code == 201, response.text

    data = response.json()
    assert data["runId"]
    assert data["phase"] == "configuring"
    assert data["executionMode"] == "fake"
    assert data["sampleLimit"] == 5
    assert data["itemsTotal"] == 5
    assert data["completedItems"] == 0
    assert data["errorCount"] == 0
    assert data["selectedModels"][0]["experimentKey"] == key
    assert data["selectedMetrics"][0]["key"] == "accuracy"
    assert data["modelProgress"][key]["completed"] == 0


def test_post_runs_duplicate_model_keys_returns_422() -> None:
    key = _first_model_key()
    response = client.post("/api/runs", json=_make_request(selectedModels=[key, key]))
    assert response.status_code == 422


def test_post_runs_duplicate_metric_keys_returns_422() -> None:
    response = client.post(
        "/api/runs",
        json=_make_request(
            metrics=[
                {"key": "accuracy", "label": "Accuracy", "method": "llm_as_judge"},
                {"key": "accuracy", "label": "Accuracy v2", "method": "llm_as_judge"},
            ]
        ),
    )
    assert response.status_code == 422


def test_post_runs_unknown_metric_method_returns_422() -> None:
    response = client.post(
        "/api/runs",
        json=_make_request(metrics=[{"key": "accuracy", "label": "Accuracy", "method": "not-real"}]),
    )
    assert response.status_code == 422


def test_post_runs_group_metric_requires_multiple_models() -> None:
    response = client.post(
        "/api/runs",
        json=_make_request(metrics=[{"key": "verifier", "label": "Verifier", "method": "llm_as_verifier"}]),
    )
    assert response.status_code == 422


def test_post_runs_unknown_model_key_returns_422() -> None:
    response = client.post("/api/runs", json=_make_request(selectedModels=["not-a-real-model-key"]))
    assert response.status_code == 422


def test_get_run_returns_same_state() -> None:
    key = _first_model_key()
    created = client.post("/api/runs", json=_make_request(selectedModels=[key]))
    run_id = created.json()["runId"]

    response = client.get(f"/api/runs/{run_id}")
    assert response.status_code == 200
    assert response.json()["runId"] == run_id


def test_get_run_not_found() -> None:
    response = client.get("/api/runs/nonexistent-run-id-xyz")
    assert response.status_code == 404


def test_post_start_run_returns_uploading_state() -> None:
    key = _first_model_key()
    created = client.post("/api/runs", json=_make_request(selectedModels=[key]))
    run_id = created.json()["runId"]

    class StubOrchestrator:
        def __init__(self) -> None:
            self.started_run_id: str | None = None

        def start_run(self, candidate_run_id: str) -> None:
            self.started_run_id = candidate_run_id

    stub = StubOrchestrator()
    original = backend_main._orchestrator
    backend_main._orchestrator = stub
    try:
        response = client.post(f"/api/runs/{run_id}/start")
    finally:
        backend_main._orchestrator = original

    assert response.status_code == 202
    assert response.json()["phase"] == "uploading"
    assert stub.started_run_id == run_id


def test_post_start_run_not_found() -> None:
    response = client.post("/api/runs/bad-id/start")
    assert response.status_code == 404


def test_post_start_run_twice_returns_409() -> None:
    key = _first_model_key()
    created = client.post("/api/runs", json=_make_request(selectedModels=[key]))
    run_id = created.json()["runId"]

    class StubOrchestrator:
        def start_run(self, candidate_run_id: str) -> None:
            pass

    original = backend_main._orchestrator
    backend_main._orchestrator = StubOrchestrator()
    try:
        first = client.post(f"/api/runs/{run_id}/start")
        second = client.post(f"/api/runs/{run_id}/start")
    finally:
        backend_main._orchestrator = original

    assert first.status_code == 202
    assert second.status_code == 409


def test_get_leaderboard_returns_entries() -> None:
    key = _first_model_key()
    created = client.post("/api/runs", json=_make_request(selectedModels=[key]))
    run_id = created.json()["runId"]

    response = client.get(f"/api/runs/{run_id}/leaderboard", params={"metric": "accuracy"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["runId"] == run_id
    assert payload["metric"] == "accuracy"
    assert payload["entries"][0]["experimentKey"] == key
    assert payload["entries"][0]["avgScore"] is None
    assert payload["entries"][0]["scoredCount"] == 0
    assert payload["entries"][0]["totalCount"] == 5


def test_get_leaderboard_invalid_metric_returns_422() -> None:
    key = _first_model_key()
    created = client.post("/api/runs", json=_make_request(selectedModels=[key]))
    run_id = created.json()["runId"]

    response = client.get(f"/api/runs/{run_id}/leaderboard", params={"metric": "unknown"})
    assert response.status_code == 422


def test_get_leaderboard_missing_metric_returns_422() -> None:
    key = _first_model_key()
    created = client.post("/api/runs", json=_make_request(selectedModels=[key]))
    run_id = created.json()["runId"]

    response = client.get(f"/api/runs/{run_id}/leaderboard")
    assert response.status_code == 422


def test_get_events_empty_initially() -> None:
    key = _first_model_key()
    created = client.post("/api/runs", json=_make_request(selectedModels=[key]))
    run_id = created.json()["runId"]

    response = client.get(f"/api/runs/{run_id}/events")
    assert response.status_code == 200
    payload = response.json()
    assert payload["runId"] == run_id
    assert payload["events"] == []
    assert payload["nextCursor"] is None


def test_get_events_supports_cursor() -> None:
    key = _first_model_key()
    created = client.post("/api/runs", json=_make_request(selectedModels=[key]))
    run_id = created.json()["runId"]

    _run_manager.add_event(run_id, "step", {"n": 1})
    _run_manager.add_event(run_id, "step", {"n": 2})

    response = client.get(f"/api/runs/{run_id}/events", params={"after": 1})
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["events"]) == 1
    assert payload["events"][0]["sequence"] == 2
    assert payload["nextCursor"] == 2


def test_get_events_not_found() -> None:
    response = client.get("/api/runs/bad-id/events")
    assert response.status_code == 404


def test_get_results_returns_final_payload() -> None:
    key = _first_model_key()
    created = client.post("/api/runs", json=_make_request(selectedModels=[key]))
    run_id = created.json()["runId"]

    _run_manager.record_execution_result(
        run_id,
        key,
        item_reference="item-1",
        input_text="mission brief",
        expected_output="expected memo",
        output_text="generated memo",
        error=None,
        trace_id="trace-1",
        observation_id="obs-1",
    )
    _run_manager.record_metric_result(
        run_id,
        key,
        "accuracy",
        item_reference="item-1",
        score=0.8,
        explanation="Strong final memo.",
    )
    _run_manager.set_phase(run_id, RunPhase.COMPLETED)

    response = client.get(f"/api/runs/{run_id}/results")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["runId"] == run_id
    assert payload["phase"] == "completed"
    assert payload["metrics"][0]["leaderboard"][0]["avgScore"] == pytest.approx(0.8)
    assert payload["metrics"][0]["history"][0]["averages"][key] == pytest.approx(0.8)
    assert payload["metrics"][0]["lowestScored"][0]["traceId"] == "trace-1"


def test_get_results_while_run_active_returns_409() -> None:
    key = _first_model_key()
    created = client.post("/api/runs", json=_make_request(selectedModels=[key]))
    run_id = created.json()["runId"]

    response = client.get(f"/api/runs/{run_id}/results")
    assert response.status_code == 409
