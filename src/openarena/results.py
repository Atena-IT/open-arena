# License Apache 2.0: (c) 2026 Atena-Reply

"""The result of a sweep, read off its tuners: one row per model, dataset and metric.

Each dataset's tuner holds one trial per model, with the metrics the trial
scored and its status. `SweepResults` gathers them as the sweep finishes a
dataset, keeps the order the metrics were met in, works out each dataset's
Pareto frontier over its objectives, and lays everything out as the long
rows the CLI prints and the leaderboard reads, beside the run's `meta`.
"""

from __future__ import annotations

from typing import Any

from openarena.leaderboard import pareto_frontier

# A cell's status, as keras-tuner records a trial's.
COMPLETED = "COMPLETED"


class SweepResults:
    """What a sweep scored, cell by cell."""

    def __init__(self, models: list[str], datasets: list[str]):
        self.models = list(models)
        self.datasets = list(datasets)
        self.scores: dict[tuple[str, str], dict[str, float | None]] = {}  # (model, dataset) -> metric -> value
        self.statuses: dict[tuple[str, str], str] = {}
        self.reward_directions: dict[str, str] = {}  # per dataset: the primary metric's
        self.metric_directions: dict[str, str] = {}  # per alias, in the order the aliases were met
        self.objectives: dict[str, list[tuple[str, str]]] = {}  # per dataset: `(alias, direction)` of its tuner's objectives
        self.errors: dict[str, dict[str, str]] = {}  # per dataset: why each cell that did not complete failed

    @property
    def metrics(self) -> list[str]:
        """The result's metric columns: the reward, then every alias in the order it was met."""
        return ["reward", *self.metric_directions]

    def read(self, dataset: str, oracle, metrics: list[tuple[str, str]], reward_direction: str, objectives: list[tuple[str, str]]) -> None:
        """Take a dataset's cells off its tuner: `metrics` is `(alias, direction)` per metric beside the reward."""
        self.reward_directions[dataset] = reward_direction
        self.objectives[dataset] = list(objectives)
        for alias, direction in metrics:
            self.metric_directions.setdefault(alias, direction)
        self.errors[dataset] = {}
        for trial in oracle.trials.values():
            model = trial.hyperparameters.get("language_model")
            if model is None:
                continue  # a trial of another search space, left in the cache
            self.statuses[model, dataset] = trial.status
            self.scores[model, dataset] = {key: _best(trial, key) for key in ("reward", *(alias for alias, _ in metrics))}
            if trial.status != COMPLETED and getattr(trial, "message", None):
                self.errors[dataset][model] = _last_line(trial.message)

    def status(self, model: str, dataset: str) -> str | None:
        return self.statuses.get((model, dataset))

    def value(self, model: str, dataset: str, metric: str = "reward") -> float | None:
        return self.scores.get((model, dataset), {}).get(metric)

    def frontier(self, dataset: str) -> list[str]:
        """The models on the dataset's Pareto frontier, in axis order; none under a single objective, or for a model with a value missing."""
        axes = self.objectives.get(dataset, [])
        if len(axes) < 2:
            return []
        points = {}
        for model in self.models:
            values = {alias: self.value(model, dataset, alias) for alias, _ in axes}
            if None not in values.values():
                points[model] = values
        kept = pareto_frontier(points, axes)
        return [model for model in self.models if model in kept]

    def rows(self) -> list[dict[str, Any]]:
        """One row per dataset, model and metric, each carrying what it takes to read it alone."""
        rows = []
        for dataset in self.datasets:
            frontier = set(self.frontier(dataset))
            for model in self.models:
                for metric in self.metrics:
                    rows.append(
                        {
                            "model": model,
                            "dataset": dataset,
                            "metric": metric,
                            "value": self.value(model, dataset, metric),
                            "status": self.status(model, dataset),
                            "direction": self.reward_directions.get(dataset, "max") if metric == "reward" else self.metric_directions.get(metric, "max"),
                            "on_frontier": model in frontier,
                        }
                    )
        return rows

    def meta(self, config_path: str) -> dict[str, Any]:
        return {
            "config": str(config_path),
            "models": list(self.models),
            "datasets": list(self.datasets),
            "metrics": self.metrics,
            "directions": dict(self.reward_directions),
            "metric_directions": dict(self.metric_directions),
            "objectives_by_ds": {dataset: [list(axis) for axis in axes] for dataset, axes in self.objectives.items()},
            "frontier_by_ds": {dataset: self.frontier(dataset) for dataset in self.objectives},
        }


def _best(trial, metric: str) -> float | None:
    try:
        return trial.metrics.get_best_value(metric)
    except (KeyError, ValueError):
        return None


def _last_line(message: str) -> str:
    """The exception itself, off a trial's traceback."""
    lines = [line.strip() for line in str(message).splitlines() if line.strip()]
    return lines[-1] if lines else str(message)
