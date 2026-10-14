# License Apache 2.0: (c) 2026 Atena-Reply

"""Files other tools write, read into the canonical import (`openarena.imports`).

`read` knows three shapes: the canonical JSON itself, a tidy CSV (one score
per line: model, dataset or task, metric, value, optional direction,
primary, ci_low, ci_high, n) and lm-evaluation-harness's `results_*.json`.
It says which directions it had to guess from a metric's name and what
looks off. Anything else is `NotRecognised`: converting it is an agent's
job (the arena's assistant, or the `arena-import` skill). Every reader
parses data and nothing else; nothing in a file is ever run.
"""

from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path

from openarena.imports import MAX_BYTES, MAX_ROWS, ImportFile, _not_json, parse


class NotRecognised(ValueError):
    """No reader knows the file: an agent has to convert it."""


# ponytail: a name heuristic for a missing direction, shown to the user to confirm.
_LOWER_IS_BETTER = (
    "ttft",
    "latency",
    "_ms",
    "memory",
    "cost",
    "loss",
    "perplexity",
    "error",
    "wer",
    "seconds",
)
_CSV_COLUMNS = {
    "model",
    "dataset",
    "task",
    "metric",
    "value",
    "direction",
    "primary",
    "ci_low",
    "ci_high",
    "n",
}


def guess_direction(metric: str) -> str:
    stem = metric.split("@")[0].lower()
    return "min" if any(word in stem for word in _LOWER_IS_BETTER) else "max"


def _slug(filename: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "-", Path(filename or "import").stem).strip(
        "-._"
    )
    return (stem or "import")[:64]


def _rewarded(rows: list[dict]) -> tuple[list[dict], list[str]]:
    """`rows` with a `reward` row added to each cell lacking one, from its `primary` row or its only metric."""
    cells: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        cells.setdefault((r["model"], r["dataset"]), []).append(r)
    out, ambiguous = [], set()
    for (_, dataset), cell in cells.items():
        plain = [{k: v for k, v in r.items() if k != "primary"} for r in cell]
        out.extend(plain)
        if any(r["metric"] == "reward" for r in cell):
            continue
        primary = [p for p, r in zip(plain, cell) if r.get("primary")] or (
            plain if len(plain) == 1 else []
        )
        if len(primary) == 1:
            out.append({**primary[0], "metric": "reward"})
        else:
            ambiguous.add(dataset)
    return out, sorted(ambiguous)


def _mixed_metrics(rows: list[dict]) -> list[str]:
    """Datasets whose cells were measured under different `@` parameters: unlike numbers ranked together."""
    by_dataset: dict[str, set[str]] = {}
    for r in rows:
        if r["metric"] != "reward":
            by_dataset.setdefault(r["dataset"], set()).add(r["metric"])
    return [
        f"dataset {d!r} mixes {', '.join(sorted(m))}: check every model was measured the same way"
        for d, m in sorted(by_dataset.items())
        if len(m) > 1 and len({x.split("@")[0] for x in m}) == 1
    ]


def _tidy_csv(raw: bytes, filename: str) -> dict:
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    columns = {c.strip().lower(): c for c in reader.fieldnames or []}
    if not (
        {"model", "metric", "value"} <= columns.keys()
        and ({"dataset", "task"} & columns.keys())
    ):
        raise NotRecognised(
            "not a tidy CSV: it needs the columns model, dataset (or task), metric and value"
        )
    dataset_column = "dataset" if "dataset" in columns else "task"

    rows, guessed = [], set()
    for number, line in enumerate(reader, start=2):
        if number > MAX_ROWS + 1:
            raise ValueError(f"over {MAX_ROWS} rows")
        cell = {
            c: (line.get(columns[c]) or "").strip()
            for c in _CSV_COLUMNS & columns.keys()
        }
        if not cell["value"]:
            continue
        direction = cell.get("direction", "").lower()
        if direction not in ("max", "min"):
            direction = guess_direction(cell["metric"])
            guessed.add(cell["metric"])
        try:
            row = {
                "model": cell["model"],
                "dataset": cell[dataset_column],
                "metric": cell["metric"],
                "value": float(cell["value"]),
                "direction": direction,
                "primary": cell.get("primary", "").lower() in ("1", "true", "yes"),
            }
            for key in ("ci_low", "ci_high"):
                if cell.get(key):
                    row[key] = float(cell[key])
            if cell.get("n"):
                row["n"] = int(float(cell["n"]))
        except ValueError as e:
            raise ValueError(f"line {number}: {e}") from e
        rows.append(row)

    ignored = sorted(columns.keys() - _CSV_COLUMNS)
    return {
        "meta": {
            "name": _slug(filename),
            "title": Path(filename or "import").stem[:200] or "Import",
        },
        "rows": rows,
        "guessed": guessed,
        "warnings": [f"columns not read: {', '.join(ignored)}"] if ignored else [],
    }


def _lm_eval(data: dict, filename: str) -> dict:
    model = str(
        data.get("model_name")
        or (data.get("config") or {}).get("model_args")
        or "model"
    )[:200]
    better = data.get("higher_is_better") or {}
    samples = data.get("n-samples") or {}
    rows, guessed = [], set()
    for task, scores in data["results"].items():
        first = True
        for key, value in (scores or {}).items():
            # `acc,none` is `acc`; `exact_match,strict-match` and `exact_match,flexible-extract`
            # are two measurements, so the filter stays in the name, the arena's `@` way.
            base, _, flt = key.partition(",")
            metric = base if flt in ("", "none") else f"{base}@{flt}"
            if (
                base == "alias"
                or base.endswith("_stderr")
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
            ):
                continue
            known = better.get(task) or {}
            if base in known:
                direction = "max" if known[base] else "min"
            else:
                direction = guess_direction(metric)
                guessed.add(metric)
            row = {
                "model": model,
                "dataset": task,
                "metric": metric,
                "value": float(value),
                "direction": direction,
                "primary": first,
            }
            n = (samples.get(task) or {}).get("effective")
            if isinstance(n, int):
                row["n"] = n
            rows.append(row)
            first = False
    return {
        "meta": {
            "name": _slug(filename),
            "title": f"{model} (lm-evaluation-harness)"[:200],
            "source": "lm-evaluation-harness",
        },
        "rows": rows,
        "guessed": guessed,
        "warnings": [
            "each task's first metric is its reward: check it is the one to rank on"
        ],
    }


def _malformed(reader, what: str, *args):
    """`reader(*args)`, its stumbles on a malformed file (a list where a table was, an infinite count) as `ValueError`."""
    try:
        return reader(*args)
    except (TypeError, AttributeError, KeyError, OverflowError) as e:
        raise ValueError(f"malformed {what}: {type(e).__name__}: {e}") from e


def read(raw: bytes, filename: str = "", *, directions: dict[str, str] | None = None, meta: dict[str, str] | None = None) -> dict:
    """A file as a canonical import: `{"format", "document", "guessed", "warnings"}`.

    `document` validates as `ImportFile`; `guessed` lists the metrics whose
    direction came from their name. `directions` (metric to `max` / `min`)
    and `meta` (`name`, `title`) are the user's corrections, applied before
    each cell's `reward` is taken. `NotRecognised` when no reader knows the
    file, `ValueError` when one does and the file is wrong.
    """
    if not isinstance(directions or {}, dict) or not isinstance(meta or {}, dict):
        raise ValueError('corrections are {"directions": {metric: "max" | "min"}, "meta": {"name", "title"}}')
    directions = {m: d for m, d in (directions or {}).items() if d in ("max", "min")}
    if len(raw) > MAX_BYTES:
        raise ValueError(f"the file is {len(raw)} bytes, over the {MAX_BYTES} limit")
    try:
        data = json.loads(raw, parse_constant=_not_json)
    except (UnicodeDecodeError, json.JSONDecodeError):
        data = None

    if isinstance(data, dict) and "meta" in data and "rows" in data:
        fmt, found = "arena", parse(raw).model_dump(exclude_none=True)
        found["guessed"], found["warnings"] = set(), []
    elif isinstance(data, dict) and isinstance(data.get("results"), dict):
        fmt, found = "lm-evaluation-harness", _malformed(_lm_eval, "lm-evaluation-harness results", data, filename)
    elif data is None and raw.lstrip()[:1] not in (b"<", b"{", b"["):
        try:
            fmt, found = "csv", _malformed(_tidy_csv, "CSV", raw, filename)
        except UnicodeDecodeError as e:
            raise NotRecognised("not a text file") from e
    else:
        raise NotRecognised("not a shape the arena reads natively (arena JSON, tidy CSV, lm-evaluation-harness results)")

    for row in found["rows"]:
        if row["metric"] in directions:
            row["direction"] = directions[row["metric"]]
    # What the user may correct: every metric the file states, a `reward` it writes out included; a
    # `reward` taken from a primary metric follows that metric, so it is not offered on its own.
    stated = {r["metric"]: r["direction"] for r in found["rows"]}
    rows, ambiguous = _rewarded(found["rows"])
    if ambiguous:
        raise ValueError(
            f"cells with several metrics and none marked primary, e.g. dataset {ambiguous[0]!r}: "
            "add a `primary` column, true on the metric to rank on, or a `reward` row per cell"
        )
    document = {
        "meta": {**found["meta"], **{k: v for k, v in (meta or {}).items() if k in ("name", "title") and v}},
        "rows": rows,
        # What the arena's own JSON says about its datasets and models, kept as given.
        **{k: found[k] for k in ("datasets", "models", "leaderboard", "charts") if found.get(k)},
    }
    renamed = (meta or {}).get("name")
    if renamed and isinstance(document.get("leaderboard"), dict):
        # Imported under another name, its board goes by that name too, beside the one it was exported from.
        document["leaderboard"] = {**document["leaderboard"], "name": renamed}
    imported = ImportFile.model_validate(document)
    guessed = found["guessed"] - directions.keys()
    return {
        "format": fmt,
        "document": document,
        "imported": imported,
        "metrics": [{"name": m, "direction": d, "guessed": m in guessed} for m, d in sorted(stated.items())],
        "guessed": sorted(found["guessed"] - directions.keys()),
        "warnings": found["warnings"] + _mixed_metrics(rows),
    }
