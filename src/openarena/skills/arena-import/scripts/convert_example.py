"""An example: one HTML dashboard's embedded JSON, converted into an arena import.

Usage: cat dashboard.html | python3 convert_example.py > out.json

The page keeps its data as JSON in a <script> tag: `criteria[].tasks[].cells`
per model (value, ci_low, ci_high, n), `operational.dimensions` measured once
per model, `run` for provenance. Each quality cell's own metric becomes its
`reward`, kept beside it with ci and n; the operational measurements are
repeated on every cell of their model, as the arena does with its own latency
and cost. What the page says about tasks (their criterion, the license
status of their data) and the run goes into `datasets` and `meta`. Nothing
is rounded, averaged or filled in.
"""

import json
import re
import sys

page = sys.stdin.read()
block = re.search(r"<script[^>]*>(.*?)</script>", page, re.S)
if block is None:
    sys.exit(
        "no JSON <script> block: this is not the dashboard this script reads, so refuse it"
    )
data = json.loads(block.group(1))

operational = data["operational"]
ops = [
    (d["name"], "min" if d["direction"] == "lower" else "max", d["values"])
    for d in operational["dimensions"]
]
rows, datasets, ranked, scores = [], {}, [], 0
for criterion in data["criteria"]:
    for task in criterion["tasks"]:
        datasets[task["name"]] = {
            "group": criterion["name"],
            # Read: the criterion's weight class and the data's license status. Inferred: the language, from `_it`.
            "tags": [
                criterion["weight"],
                task["license_status"].replace("_", "-"),
                "italian" if task["name"].endswith(("_it", "_it_ll", "_it_node")) or criterion["name"].startswith("italian") else "english",
            ],
        }
        if criterion["weight"] != "diagnostic" and not criterion.get("not_measured"):
            ranked.append(task["name"])
        for model, cell in task["cells"].items():
            scores += 1
            base = {"model": model, "dataset": task["name"]}
            extra = {
                k: cell[k]
                for k in ("ci_low", "ci_high", "n")
                if cell.get(k) is not None
            }
            rows.append(
                {
                    **base,
                    "metric": "reward",
                    "value": cell["value"],
                    "direction": "max",
                    **extra,
                }
            )
            rows.append(
                {
                    **base,
                    "metric": task["metric"],
                    "value": cell["value"],
                    "direction": "max",
                    **extra,
                }
            )
            for name, way, values in ops:
                if model in values:
                    rows.append(
                        {
                            **base,
                            "metric": name,
                            "value": values[model],
                            "direction": way,
                        }
                    )

run, point = data["run"], operational["point"]
doc = {
    "meta": {
        "name": "bench",
        "title": "Clinical QA Suite",
        "description": (
            f"Run {run['id']}: {len(data['models'])} models on {len(datasets)} medical tasks in "
            f"{len(data['criteria'])} groups (primary, secondary, diagnostic). Operational metrics "
            f"({', '.join(n for n, _, _ in ops)}) measured once per model at len={point['length']}, "
            f"concurrency={point['concurrency']}, repeated on each of its cells."
            + (
                f" Source warnings: {'; '.join(run['warnings'])}."
                if run.get("warnings")
                else ""
            )
        )[:2000],
        "source": run["id"],
        "protocol": run["protocol"],
        "hash": run["protocol_hash"],
        "provisional": run["provisional"],
        "tags": ["health", "medical", "italian", "english"] + (["provisional"] if run["provisional"] else []),
    },
    "datasets": datasets,
    # Inferred from the names: the reasoning setting a suffix spells.
    "models": {
        m["name"]: {"tags": [f"reasoning-{m['name'].rsplit('-', 1)[1]}"]}
        for m in data["models"]
        if m["name"].rsplit("-", 1)[-1] in ("xhigh", "high", "medium", "low")
    },
    # The source ranks on primary and secondary criteria; diagnostic ones are imported, not ranked.
    # The source's weight per criterion (primary 2, secondary 1) is the board's weight per group.
    "leaderboard": {
        "datasets": ranked,
        "group_shares": {c["name"]: c["default_weight"] for c in data["criteria"] if c["default_weight"] > 0},
    },
    # Charts the arena has no default for, as specs it draws in its own style.
    "charts": [
        {"type": "heatmap", "title": "Score by group", "metric": "reward", "aggregate": "mean", "stack": "group"},
        {"type": "scatter", "title": "Score against p95 time to first token", "metric": "reward", "x": "p95_ttft_ms", "aggregate": "mean", "unit": "ms"},
    ],
    "rows": rows,
}
json.dump(doc, sys.stdout, separators=(",", ":"))
print(f"{scores} source scores -> {len(rows)} rows, 0 skipped", file=sys.stderr)
