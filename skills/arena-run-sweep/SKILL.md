---
name: arena-run-sweep
description: Run the arena evaluation sweep and interpret last_run.tsv and the leaderboard matrix.
---

Read `README.md` (Run section) and `AUTORESEARCH.md` (Output format section) for full context. This skill covers the common invocations and how to read the output.

## Prerequisites

```bash
uv sync           # installs deps + the `arena` console script
cp .env.example .env   # fill in provider API keys you use
```

## CLI flags

```bash
arena                            # reads ./config.yaml, stores state in .open-arena/
arena -c configs/eval.yaml       # -c / --config: different config file
arena --no-cache                 # discard per-dataset trial caches before running
arena --state-dir runs/exp1      # store trial state + TSVs under a custom dir
arena -v                         # show synalinks progress bar (verbose=1)
arena -vv                        # per-batch lines (verbose=2, good for log files)
arena --json results.json        # also write the full result matrix as JSON
arena --json -                   # emit JSON on stdout only, skip TSV + tables
```

Always invoke via `uv run arena` (or inside the activated venv). Plain `python -m src.evaluate` fails outside the venv because `keras_tuner` is not on the system path.

## Keep multiple runs side by side

```bash
arena --state-dir runs/baseline
arena --state-dir runs/exp1 --no-cache
```

Each `--state-dir` gets its own `last_run.tsv`, `frontier.tsv`, and per-dataset keras-tuner trial cache. Runs never clobber each other.

## When to pass `--no-cache`

Pass `--no-cache` (or `rm -rf .open-arena/*/`) whenever:
- you added, renamed, or removed a model in `experiments.language_models`
- you added, renamed, or removed a candidate metric alias under the top-level `metrics:` block
- you changed the dataset list in `experiments.datasets`

Completed trials for unchanged axes are still reused — `--no-cache` only drops trials whose HP space changed.

## Reading `.open-arena/last_run.tsv`

Long-format TSV, one row per `(model, dataset, metric)` cell:

```
model           dataset     metric      value       direction
ollama/mistral  mmlu_test   reward      0.440000    max
ollama/mistral  mmlu_test   lm_judge    0.612000    max
ollama/llama3.2 mmlu_test   reward      0.520000    max
```

Columns:
- `model` — litellm model identifier (e.g. `ollama/mistral`, `openai/gpt-4o`)
- `dataset` — dataset name from `experiments.datasets`
- `metric` — `reward` (primary) or a candidate alias from `metrics:`
- `value` — score in `[0, 1]` (higher = better for `max`, lower for `min`)
- `direction` — `max` or `min`

Failed trials are omitted from the TSV (they appear in the JSON output with `value: null`).

## Multi-objective datasets: `frontier.tsv`

When a dataset has `objective: true` on more than one metric entry, a Pareto frontier is computed. Models on the frontier are printed as a markdown table to stdout and written to `.open-arena/frontier.tsv`:

```
dataset     model           axis        direction   value
my_dataset  ollama/mistral  reward      max         0.520000
my_dataset  ollama/mistral  lm_judge    max         0.710000
```

## Quick agreement check (Python one-liner)

```bash
uv run python -c "
import csv, collections
rows = list(csv.DictReader(open('.open-arena/last_run.tsv'), delimiter='\t'))
# argmax per dataset for 'reward' and a candidate alias
best = collections.defaultdict(dict)
for r in rows:
    ds, model, metric, val = r['dataset'], r['model'], r['metric'], float(r['value'])
    if metric in ('reward', 'lm_judge'):
        if model not in best[ds] or best[ds].get(metric, -1) < val:
            best[ds][metric] = (model, val)
for ds, d in best.items():
    agree = d.get('reward', ('?',))[0] == d.get('lm_judge', ('?',))[0]
    print(ds, 'agree' if agree else 'DISAGREE', d)
"
```

## Smoke-test before a full sweep

Test that every dataset in `experiments.datasets` actually loads:

```bash
uv run python -c "
import yaml
from src.datasets import load_dataset_from_yaml
cfg = yaml.safe_load(open('config.yaml'))
for n in cfg['experiments']['datasets']:
    it = iter(load_dataset_from_yaml('config.yaml', name=n))
    next(it); print('ok', n)
"
```

## Kill a runaway run

```bash
pkill -f 'src.evaluate'
```

Each trial should take under 5 minutes. If a sweep exceeds 10 minutes, kill it, lower `limit:` in `config.yaml`, or drop a heavy candidate from the `metrics:` block.
