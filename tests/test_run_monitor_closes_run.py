# License Apache 2.0: (c) 2026 Atena-Reply

"""The sweep's MLflow monitor closes each cell's run with its metrics logged.

`_RunMonitor` used to override the framework's `_log_metrics` with a two-argument
signature; the framework now calls it with `prefix` and `model_id`, so every
cell's closing log raised, the metrics were never written and the run stayed
open for the next cell to clean up.
"""

import os

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import warnings

import mlflow
import pytest

from openarena.evaluate import _RunMonitor


@pytest.fixture
def tracking_uri(tmp_path):
    before = mlflow.get_tracking_uri()
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    yield uri
    mlflow.set_tracking_uri(before)


def test_a_cell_ends_with_a_finished_run_and_its_metrics(tracking_uri):
    monitor = _RunMonitor(
        "sweep-experiment",
        tracking_uri,
        "gsm8k_test_20",
        "ollama/qwen3:8b",
        tags={"run": "abc"},
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        monitor.on_test_begin({})
        monitor.on_test_batch_end(0, {"reward": 0.5})
        monitor.on_test_end({"reward": 0.8, "avg_latency": 9.16, "cached": True})
    ours = [str(w.message) for w in caught if "MLflow:" in str(w.message)]
    assert ours == [], ours
    assert mlflow.active_run() is None

    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name("sweep-experiment")
    runs = mlflow.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    run = runs.iloc[0]
    assert run["status"] == "FINISHED"
    assert run["tags.dataset"] == "gsm8k_test_20"
    assert run["tags.language_model"] == "ollama/qwen3:8b"
    assert run["metrics.reward"] == pytest.approx(0.8)
    assert run["metrics.avg_latency"] == pytest.approx(9.16)
