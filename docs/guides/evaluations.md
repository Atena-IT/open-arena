# Running evaluations

An evaluation is a saved sweep: every model of `language_models` on every
dataset of `datasets`, one **cell** per pair.

```yaml
evaluations:
  math-compare:
    description: Local Qwen against the hosted baseline on GSM8K.
    language_models: [qwen38, gpt-4o-mini]   # models (A2A agents for an a2a dataset)
    datasets: [gsm8k-20]
    batch_size: 4                            # rows at once, over each dataset's own
    metrics: [answer_accuracy]               # columns added to every dataset
    seed: 0
    tags: [math]
```

| field | |
|---|---|
| `language_models` | required: the models to compare, by name |
| `datasets` | required: the datasets, by name; each needs a `reward` |
| `batch_size` | rows sent at once for every dataset, over each one's own |
| `metrics` | extra columns for every dataset ([Datasets](datasets.md#extra-metrics)) |
| `seed` | for reproducibility |
| `default_language_model`, `default_embedding_model` | the model a reward uses when its config names none |

## Run

```shell
openarena evaluations list
openarena evaluations run math-compare
```

| option | |
|---|---|
| `--configs DIR` | the configs folder (default `./configs`) |
| `--state-dir DIR` | where runs live (default `./.open-arena`) |
| `--no-cache` | start over instead of resuming |
| `--batch-size N` | override the batch size for this run |
| `--json FILE` | also write the result there |
| `-v`, `-vv` | a progress bar, or a line per batch |

While it runs it prints a line per cell, and after each dataset
`<model> failed: <why>` for any cell that failed. At the end, a table per
dataset: a row per model, a column per metric, `—` where a cell has no value.

## What a run writes

```
.open-arena/math-compare/
  config.yaml       the evaluation and every entry it names, gathered: what ran
  result.json       the scores
  harbor/           Harbor trials, for Harbor datasets
  tasks/ environments/   copies of the Harbor folders the run used
```

## Resuming

Finished cells are kept in that folder. Running the same evaluation again
continues where it stopped: an interrupted run, a cell that failed on a missing
key. After changing what a finished cell depends on (a dataset, a model's
params, a reward), add `--no-cache`, or old cells are reused.

## Cost and time

Each row is one model call per model, plus one per judge reward; a Harbor row
is a whole agent session in a sandbox. Run a few rows first (`limit: 5` on the
datasets), read the answers, then run the full size. Every run reports its cost
per cell, from litellm's prices.
