---
name: openarena-run-evaluation
description: Use when models (or A2A agents) have to be evaluated on datasets with openarena — writing a saved evaluation in evaluations.yaml (which models, which datasets, batch size, extra metrics), launching it with `openarena evaluations run`, resuming or restarting it, and reading the score matrix it prints and writes to result.json.
---

# openarena-run-evaluation

An evaluation is a saved sweep: every model of `language_models` on every
dataset of `datasets`, one **cell** per pair. `openarena evaluations run NAME`
runs it on this machine and prints the matrix.

## The entry

```yaml
evaluations:
  math-compare:
    description: Local Qwen against the hosted baseline on GSM8K.
    language_models: [qwen38, gpt-4o-mini]   # required: models.yaml keys (a2a_agents keys for an a2a dataset)
    datasets: [gsm8k-20]                     # required: datasets.yaml keys
    batch_size: 4                            # optional: rows at once for every dataset, over each one's own
    metrics: [answer_accuracy]               # optional: columns added to every dataset (see openarena-add-dataset)
    seed: 0                                  # optional
    tags: [math]
```

Every dataset named must have a `reward:`; every model named must exist.
`batch_size` is the concurrency knob: rows sent at once, hence requests in
flight against the provider. Keep it low for a local model or a rate-limited
key, at 1 for Harbor datasets on a small machine.

## Running it

From the workspace's parent folder (or with `uv run` in the openarena project):

```bash
uv run openarena evaluations list --configs my-arena/configs
uv run openarena check my-arena/configs
uv run openarena evaluations run math-compare --configs my-arena/configs
```

| option | does |
|---|---|
| `--configs DIR` | the configs folder (default `./configs`) |
| `--state-dir DIR` | where runs live (default `./.open-arena`), one folder per evaluation |
| `--no-cache` | start over instead of resuming |
| `--batch-size N` | override the evaluation's batch size for this run |
| `--json FILE` | also write the result there |
| `-v` / `-vv` | progress bar / one line per batch |

The run writes `<state-dir>/<name>/`:

```
config.yaml       the evaluation and every entry it names, gathered: what ran
result.json       the matrix (openarena-read-results)
harbor/           Harbor trials, for harbor datasets
environments/ tasks/   copies of the Harbor folders the run used
```

**Resuming.** Finished cells are cached in that folder. Running the same
name again picks up where it stopped (an interrupted run, a failed cell);
after changing a dataset, a model's params or a reward, use `--no-cache` so
old cells are not reused. Two evaluations do not share a cache.

## Cost and time

Each row is at least one model call per model, plus one per judge reward;
a Harbor row is a whole agent session in a sandbox. Before a full run:

1. Set `limit: 5` on the datasets, or run a copy of the evaluation with
   one model, and check the matrix makes sense.
2. Then the real size. Say to the user how many rows × models you are about
   to run when it is large or uses paid models.

Hosted models need their key in the environment (`OPENAI_API_KEY`, ...);
a missing key fails the cell, it does not fail silently.

## The output

While it runs, a line per cell (`<model> running…`, then `done reward ...`),
and after each dataset `<model> failed: <why>` for a cell that failed. At the
end, one table per dataset: a row per model, a column per metric (`reward`
first, then cost, tokens, latency and any extra metric), `—` where a cell has
no value. The same is in `result.json`; read it as openarena-read-results
says, and rank several runs with openarena-rank-leaderboard.
