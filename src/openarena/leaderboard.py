# License Apache 2.0: (c) 2026 Atena-Reply

"""Leaderboards: the scores of finished runs, gathered and ranked.

Two steps, the same two the web app takes:

1. `board(runs)` gathers the latest cell per `(model, dataset)` from sweep
   results, newest run first, the way `/leaderboard` does: a cell is taken
   whole from one run, its `reward` plus every metric that run recorded,
   and `eol_days` is added to every cell of a model.
2. `rank(board, ...)` scores the models of a leaderboard's scope, the port
   of the frontend's `lib/criteria.ts`: each metric normalized per dataset to
   `[FLOOR, 1]` against the models in scope (linear for the reward and
   `eol_days`, log for magnitudes), combined as a weighted geometric mean,
   and averaged over the datasets. A model must be scored on every dataset in
   scope, and a metric only carries weight when every cell in scope records
   it. The Pareto frontier is marked over the weighted metrics.

Plain dicts in and out, so the CLI prints them and a caller can dump them.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timezone


FLOOR = 0.05

# The importance ladder: each step weighs three times the one below.
LEVELS = {"Off": 0, "Low": 1, "Medium": 3, "High": 9}

# Facts about the model rather than measurements of a run; presets start them off.
MODEL_FACTS = {"eol_days"}

# Normalized linearly: the reward, and calendar time, where a day is a day.
# `preference` is the app's: the votes' Bradley-Terry rating, an Elo-like scale where differences, not ratios, mean something.
LINEAR = {"reward", "eol_days", "preference"}


def board(runs: list[dict], *, today: date | None = None, eol: bool = True) -> dict:
    """The leaderboard over `runs`, each `{"id", "result": <run_sweep result>, "model_ids": {name: id}}`, newest first.

    The one aggregation: the backend's `/leaderboard` and `openarena
    leaderboards run` both read the board from here. A row's `model` is what
    the board reports; `model_ids` maps it to the id it called, for
    `eol_days` and the entry's `model_id` (the name itself when absent).

    Returns `{"datasets", "metrics", "entries"}`, entries being
    `{"model", "model_id", "scores": {dataset: {"value", "direction", "run_id",
    "metrics": {name: {"value", "direction"}}}}, "mean"}`, best mean first.
    `eol` off leaves `eol_days` out (it asks the outside world).
    """
    scores: dict[str, dict[str, dict]] = {}
    ids: dict[str, str] = {}
    datasets: list[str] = []
    # Datasets only imports scored (`openarena.imports`): in the board, but out of an empty scope.
    imported: set[str] = set()
    measured: set[str] = set()
    # A dataset's group (suite, criterion), as an import names it; a dataset in none is its own.
    groups: dict[str, str] = {}
    metrics: list[str] = ["reward"]
    for run in runs:
        groups.update(run.get("groups") or {})
        model_ids = run.get("model_ids") or {}
        by_cell: dict[tuple[str, str], dict[str, dict]] = {}
        for row in run["result"].get("rows", []):
            if row.get("value") is None:
                continue
            model = row["model"]
            ids.setdefault(model, model_ids.get(model, model))
            by_cell.setdefault((model, row["dataset"]), {})[row["metric"]] = {
                "value": float(row["value"]),
                "direction": row.get("direction", "max"),
                # An imported score's uncertainty and sample count, carried as given.
                **{k: row[k] for k in ("ci_low", "ci_high", "n") if row.get(k) is not None},
            }
        for (model, dataset), cell in by_cell.items():
            reward = cell.get("reward")
            if reward is None:
                continue
            per_model = scores.setdefault(model, {})
            if dataset in per_model:
                continue  # a newer run already scored this cell
            if dataset not in datasets:
                datasets.append(dataset)
            (imported if str(run.get("id", "")).startswith("import:") else measured).add(dataset)
            for name in cell:
                if name not in metrics:
                    metrics.append(name)
            per_model[dataset] = {
                "value": reward["value"],
                "direction": reward["direction"],
                "run_id": run.get("id"),
                "metrics": cell,
            }
    if eol and scores:
        # Imported here: the lookup reads litellm's tables and Azure's schedule.
        from openarena.lookups.eol import eol_days

        today = today or datetime.now(timezone.utc).date()
        for model, per_dataset in scores.items():
            days = eol_days(ids.get(model, model), today)
            for cell in per_dataset.values():
                cell["metrics"]["eol_days"] = {"value": days, "direction": "max"}
        if "eol_days" not in metrics:
            metrics.append("eol_days")
    entries = []
    for model, per_dataset in scores.items():
        values = [s["value"] for s in per_dataset.values()]
        entries.append(
            {
                "model": model,
                "model_id": ids.get(model),
                "scores": per_dataset,
                "mean": sum(values) / len(values) if values else None,
            }
        )
    entries.sort(key=lambda e: (e["mean"] is None, -(e["mean"] or 0.0), e["model"]))
    return {
        "datasets": sorted(datasets),
        "imported": sorted(imported - measured),
        "groups": {d: g for d, g in groups.items() if d in datasets},
        "metrics": metrics,
        "entries": entries,
    }


def dataset_weights(
    datasets: list[str],
    groups: dict[str, str] | None = None,
    group_shares: dict[str, float] | None = None,
    dataset_shares: dict[str, float] | None = None,
) -> dict[str, float]:
    """Each dataset's weight in a model's score: its group's share, split among the group's datasets by theirs.

    A group weighs what its share says whatever the number of datasets in it;
    a share left out is 1, so with no shares every dataset weighs the same and
    the score is the plain mean:

        weight(d) = group_shares[group(d)] * dataset_shares[d] / sum of dataset_shares in group(d)

    A dataset in no group (every dataset of a CLI run, which names none) is a
    group of one keyed by its name, with nothing to split: dividing by its own
    share would cancel it, so it weighs `group_shares[d] * dataset_shares[d]`.
    """
    groups, group_shares, dataset_shares = groups or {}, group_shares or {}, dataset_shares or {}
    group_of = {d: groups.get(d, d) for d in datasets}
    totals: dict[str, float] = {}
    for d in datasets:
        totals[group_of[d]] = totals.get(group_of[d], 0.0) + dataset_shares.get(d, 1.0)
    weights = {}
    for d in datasets:
        total = totals[group_of[d]] if d in groups else 1.0
        weights[d] = group_shares.get(group_of[d], 1.0) * dataset_shares.get(d, 1.0) / total if total > 0 else 0.0
    return weights


def default_datasets(data: dict) -> list[str]:
    """What an empty dataset scope reads: the arena's own datasets, imports only when nothing else is scored.

    An import's datasets enter a leaderboard by being named in it: read as
    "every dataset", they would put models scored on disjoint benches in one
    scope, and a model is ranked only when scored on all of it.
    """
    own = [d for d in data["datasets"] if d not in set(data.get("imported", []))]
    return own or list(data["datasets"])


def _lin(x: float, xs: list[float]) -> float:
    mn, mx = min(xs), max(xs)
    return 1.0 if mx == mn else FLOOR + (1 - FLOOR) * ((x - mn) / (mx - mn))


def normalize(name: str, direction: str, values: list[float]) -> list[float]:
    """One metric column to `[FLOOR, 1]`, higher better: linear for `LINEAR`, else log when every value is positive."""
    if name not in LINEAR and all(v > 0 for v in values):
        oriented = values if direction == "max" else [1 / v for v in values]
        logs = [math.log(v) for v in oriented]
        return [_lin(v, logs) for v in logs]
    oriented = values if direction == "max" else [-v for v in values]
    return [_lin(v, oriented) for v in oriented]


def _in_scope(data: dict, models: list[str] | None) -> list[dict]:
    return [e for e in data["entries"] if models is None or e["model"] in models]


def coverage(data: dict, datasets: list[str], models: list[str] | None = None) -> dict[str, tuple[int, int]]:
    """Per metric, how many cells in scope record it, out of how many cells there are."""
    out = {m: [0, 0] for m in data["metrics"]}
    for e in _in_scope(data, models):
        for d in datasets:
            cell = e["scores"].get(d)
            if cell is None:
                continue
            for m in data["metrics"]:
                out[m][1] += 1
                out[m][0] += m in cell["metrics"]
    return {m: (have, total) for m, (have, total) in out.items()}


def weighable(cov: dict[str, tuple[int, int]]) -> list[str]:
    """The metrics every cell in scope records, the only ones that can carry weight."""
    return [m for m, (have, total) in cov.items() if total > 0 and have == total]


def default_shares(metrics: list[str]) -> dict[str, float]:
    """The page's default preset, "Reward first": the reward high, the other measurements low, model facts off."""
    levels = {m: "Off" if m in MODEL_FACTS else ("High" if m == "reward" else "Low") for m in metrics}
    return {m: float(LEVELS[level]) for m, level in levels.items() if LEVELS[level] > 0}


def _active(weights: dict[str, float]) -> dict[str, float]:
    positive = {m: w for m, w in weights.items() if w > 0}
    total = sum(positive.values()) or 1.0
    return {m: w / total for m, w in positive.items()}


def pareto_frontier(points: dict[str, dict[str, float]], axes: list[tuple[str, str]]) -> set[str]:
    """The ids no other point dominates: at least as good on every axis and strictly better on one."""

    def better(a: float, b: float, direction: str) -> bool:
        return a > b if direction == "max" else a < b

    dominated = set()
    for a, va in points.items():
        for b, vb in points.items():
            if a == b:
                continue
            if any(better(va[m], vb[m], d) for m, d in axes):
                continue
            if any(better(vb[m], va[m], d) for m, d in axes):
                dominated.add(a)
                break
    return set(points) - dominated


def violates(budget: dict, value: float | None) -> bool:
    """Whether `value` breaks `budget` (`{"metric", "op": "lte" | "gte", "value"}`); a metric the model lacks cannot."""
    if value is None:
        return False
    return value > budget["value"] if budget["op"] == "lte" else value < budget["value"]


def rank(
    data: dict,
    *,
    datasets: list[str] | None = None,
    models: list[str] | None = None,
    shares: dict[str, float] | None = None,
    budgets: list[dict] | None = None,
    group_shares: dict[str, float] | None = None,
    dataset_shares: dict[str, float] | None = None,
) -> dict:
    """Rank the models of the scope, as the leaderboard page does.

    `datasets` / `models` empty or `None` mean everything the board has, as
    in a saved leaderboard. `shares` are percents per metric; a share on a
    metric the scope only partly records is dropped, and none at all means
    the default preset. `group_shares` / `dataset_shares` weigh the datasets
    (`dataset_weights`); one at 0 is out of the score and not required.
    Returns `{"datasets", "models", "shares", "axes", "rows", "skipped",
    "gaps", "excluded"}`: `rows` best first, each `{"rank", "model", "score", "values",
    "contributions", "per_dataset", "dataset_contributions", "on_frontier",
    "runs"}`; `skipped`
    the models in scope left unranked and `gaps` why, per model a list of
    `{"dataset", "missing"}` (`missing` None: never scored there).

    `budgets` are hard limits, `{"metric", "op": "lte" | "gte", "value"}`: a
    model breaking one on any dataset is left out before anything is
    normalized, so it cannot stretch the scale of the ones kept, and comes
    back in `excluded`, `{"model", "failed": [budget, ...]}`.
    """
    budgets = list(budgets or [])
    scope_datasets = [d for d in datasets if d in data["datasets"]] if datasets else default_datasets(data)
    weight = dataset_weights(scope_datasets, data.get("groups"), group_shares, dataset_shares)
    # A dataset weighed 0 is out of the score, so a model need not be scored on it.
    scope_datasets = [d for d in scope_datasets if weight[d] > 0]
    scope_models = list(models) if models else None
    can_weigh = weighable(coverage(data, scope_datasets, scope_models))
    kept = {m: w for m, w in (shares or {}).items() if m in can_weigh and w > 0}
    w = _active(kept or default_shares(can_weigh))
    metrics = sorted(w, key=lambda m: (-w[m], m))  # most important first

    entries = _in_scope(data, scope_models)
    directions: dict[str, str] = {}
    for e in entries:
        for cell in e["scores"].values():
            for m, v in cell["metrics"].items():
                directions.setdefault(m, v["direction"])

    gaps: dict[str, list[dict]] = {}
    acc: dict[str, dict] = {}
    excluded: dict[str, list[dict]] = {}
    for d in scope_datasets:
        present = []
        for e in entries:
            cell = e["scores"].get(d)
            if cell is None:
                gaps.setdefault(e["model"], []).append({"dataset": d, "missing": None})
                continue
            missing = [m for m in metrics if m not in cell["metrics"]]
            if missing:
                gaps.setdefault(e["model"], []).append({"dataset": d, "missing": missing})
                continue
            present.append(e)
        # Budgets partition the field before anything is normalized.
        kept_here = []
        for e in present:
            cell = e["scores"][d]["metrics"]
            failed = [b for b in budgets if violates(b, (cell.get(b["metric"]) or {}).get("value"))]
            if failed:
                broke = excluded.setdefault(e["model"], [])
                broke += [b for b in failed if b not in broke]
            else:
                kept_here.append(e)
        present = kept_here
        if not present:
            continue
        columns = []
        for m in metrics:
            values = [e["scores"][d]["metrics"][m]["value"] for e in present]
            direction = present[0]["scores"][d]["metrics"][m]["direction"]
            directions[m] = direction
            columns.append((m, values, normalize(m, direction, values)))
        for i, e in enumerate(present):
            a = acc.setdefault(e["model"], {"score": 0.0, "weight": 0.0, "n": 0, "sums": {}, "norms": {}, "runs": [], "per_dataset": {}})
            s = 1.0
            for m, values, norm in columns:
                s *= norm[i] ** w[m]
                a["sums"][m] = a["sums"].get(m, 0.0) + values[i]
                a["norms"][m] = a["norms"].get(m, 0.0) + norm[i]
            a["score"] += s * weight[d]
            a["weight"] += weight[d]
            a["per_dataset"][d] = s
            a["n"] += 1
            a["runs"].append(e["scores"][d].get("run_id"))

    rows = []
    for model, a in acc.items():
        if a["n"] != len(scope_datasets) or model in excluded:
            continue  # only models scored on every dataset in scope, and over budget on none, are comparable
        n = a["n"]
        rows.append(
            {
                "model": model,
                "score": a["score"] / a["weight"],
                # Each dataset's own score, and its share of the total: the score, decomposed.
                "per_dataset": a["per_dataset"],
                "dataset_contributions": {d: weight[d] * s / a["weight"] for d, s in a["per_dataset"].items()},
                "values": {m: a["sums"][m] / n for m in metrics},
                "contributions": [
                    {
                        "metric": m,
                        "value": a["sums"][m] / n,
                        "normalized": a["norms"][m] / n,
                        "weight": w[m],
                        "factor": (a["norms"][m] / n) ** w[m],
                    }
                    for m in metrics
                ],
                "on_frontier": False,
                "runs": a["runs"],
            }
        )
    axes = [(m, directions.get(m, "max")) for m in metrics]
    frontier = pareto_frontier({r["model"]: r["values"] for r in rows}, axes)
    rows.sort(key=lambda r: (-r["score"], r["model"]))
    for i, r in enumerate(rows, 1):
        r["rank"] = i
        r["on_frontier"] = r["model"] in frontier
    ranked = {r["model"] for r in rows}
    return {
        "datasets": scope_datasets,
        "models": [e["model"] for e in entries],
        "shares": {m: w[m] * 100 for m in metrics},
        "axes": [{"name": m, "direction": d} for m, d in axes],
        "rows": rows,
        "skipped": [e["model"] for e in entries if e["model"] not in ranked and e["model"] not in excluded],
        "gaps": {m: g for m, g in gaps.items() if m not in ranked},
        "excluded": [{"model": m, "failed": f} for m, f in sorted(excluded.items())],
    }
