---
name: openarena-rank-leaderboard
description: Use when finished openarena runs have to be turned into a ranking — writing a leaderboard in leaderboards.yaml (which datasets and models are in scope, how much each metric weighs, hard budgets such as a cost cap, per-dataset weights) and printing it with `openarena leaderboards run`, including why a model was left unranked.
---

# openarena-rank-leaderboard

A leaderboard ranks the models of finished runs; it never calls a model. It
reads every `<state-dir>/*/result.json`, keeps one cell per (model, dataset)
taken whole from the newest run that scored it, keeps the datasets and models
in scope, drops models over budget, normalizes each weighted metric per
dataset to `[0.05, 1]` against the models left, combines them as a weighted
geometric mean, and averages over the datasets (weighted). It also marks the
Pareto frontier. Full reference:
https://atena-it.github.io/open-arena/guides/leaderboards/

## The entry

```yaml
leaderboards:
  math-value:
    title: Math, quality for the money      # required, 1 to 200 characters
    description: Reward first, cost second, nothing above 5 cents a cell.
    tags: [math, value]
    datasets: [gsm8k-20]                    # datasets keys; [] = every dataset scored
    models: [qwen38, gpt-4o-mini]           # models keys (agent names for A2A); [] = every model scored
    shares:                                 # weight per metric; {} = the default preset
      reward: 70
      cost: 30
    budgets:                                # hard limits: a model breaking one is left out
      - {metric: cost, op: lte, value: 0.05}     # lte: at most; gte: at least
      - {metric: reward, op: gte, value: 0.5}
```

Any other key is an error. `group_shares` / `dataset_shares` weigh datasets
(below).

## What to know before writing it

- **Metric names** are the run's own: `reward`, plus only the columns listed
  under the evaluation's or the dataset's `metrics:` (`cost`, `reward_cost`,
  `embedding_cost`, `input_tokens`, `output_tokens`, `total_tokens`,
  `avg_latency`, a metric's `alias`), plus `eol_days`. Cost, tokens and
  latency exist only when listed there. The exact list of a run is
  `meta.metrics` in its `result.json`.
- **Direction comes from the run**; a leaderboard never sets it. A metric
  under `metrics:` without `direction:` is `min` for a cost, tokens or latency
  measurement and `max` otherwise, so a bare `- cost` ranks the cheapest model
  best; an explicit `direction:` wins. Before ranking, read the directions a
  run recorded (`meta.metric_directions` in its `result.json`) and rerun if
  one is wrong:

  ```yaml
  evaluations:
    math-compare:
      # ...
      metrics:
        - cost                            # min, unset
        - avg_latency                     # min, unset
  ```

- **`shares`** are proportions (`70/30` = `7/3`), each 0 or more, at least
  one above 0. Empty is the default preset: `reward` 9, every other metric
  the scope records 1, `eol_days` 0.
- **A metric carries weight only if every cell in scope records it.** A share
  on any other metric (a typo, a metric only some datasets have, `eol_days`
  under `--no-eol`) is dropped silently; if all are dropped, the default
  preset is used. Always compare the printed `Shares:` line with what you
  wrote.
- **Budgets** are checked per dataset on that cell's value, before
  normalizing; breaking one on any dataset leaves the model out. A cell
  without the budget's metric cannot break it, so a misspelled metric
  excludes no one. `cost` is what the whole cell spent (all its rows), not
  one request.
- **Scope is part of the score.** A model ranks only if it has a cell on
  every dataset in scope. Unranked models (a gap, or a budget broken on
  another dataset) still count in normalization where they have a cell, so
  leave non-candidates out of `models`. Min-max normalization stretches any
  gap to the full `0.05`–`1` range, so a near-saturated dataset weighs as
  much as a discriminating one: keep only datasets that measure the same
  thing for the same decision.
- **Names, silently.** A `datasets` / `models` name with no scored cell is
  dropped (all unknown datasets = nothing ranked); `openarena check` only
  warns about names missing from the registries. An empty `datasets` means
  every dataset scored, and changes meaning with every new run: list it for
  a board you rely on.
- **The newest run per cell wins**, whatever it was (a `limit: 5` smoke test
  included); a cell with no reward does not replace an older one. Move stale
  run folders out of the state dir.

## Weighing datasets

Runs from `openarena evaluations run` carry no groups: every dataset is a
group of its own, keyed by its name. Either `group_shares` or
`dataset_shares`, keyed by dataset name, then weighs it directly (left out =
1); given both, a dataset weighs their product. `0` takes a dataset out of
the score, and it is no longer required:

```yaml
leaderboards:
  support-mix:
    title: Support, weighted by traffic
    datasets: [faq-answers, refund-requests, escalations]
    dataset_shares: {faq-answers: 6, refund-requests: 3, escalations: 0}   # 0: out of the score, not required
    shares: {reward: 80, cost: 20}
```

Datasets that do carry a group (an imported benchmark's suites) weigh their
group's `group_shares`, split among the group's datasets by their
`dataset_shares`.

## `eol_days`

Days to the model's end of life, looked up for the model id the newest run
recorded: litellm's deprecation date, Azure's OpenAI retirement schedule (for
`openai/`, `azure/`, `azure_ai/` ids; fetched, cached a day, shipped snapshot
offline), OpenRouter's expiration date (for `openrouter/` ids). No date, or
one over five years out, counts as 1825; a past date is negative. It is off in
the default preset; give it a share to penalize models near retirement.
`--no-eol` skips the online lookups, and a share or budget on it then does
nothing.

## Ranking

```bash
uv run openarena leaderboards list --configs my-arena/configs
uv run openarena leaderboards run math-value --configs my-arena/configs
uv run openarena leaderboards run math-value --configs my-arena/configs --state-dir my-arena/.open-arena --no-eol --json ranking.json
```

| option | does |
|---|---|
| `--configs DIR` | the folder holding `leaderboards.yaml` (default `configs`) |
| `--state-dir DIR` | where the runs are (default `.open-arena`, as for `evaluations run`); point it where the evaluations wrote |
| `--no-eol` | leave out `eol_days`, which looks dates up online |
| `--json FILE` | also write the ranking (with `contributions`, `per_dataset`, the `runs` behind each cell) |

`leaderboards run` validates the saved entry as `openarena check` does and
refuses an invalid one (exit 1): ``the leaderboard '<name>' does not
validate:``, each problem with its fix.

It prints the title, `Datasets:` and `Shares:` actually used, then a table:
rank, model, score, each weighted metric's raw mean with ↑/↓, ★ on the Pareto
frontier. Then one line per model it could not rank: `Not ranked <model>: not
scored on <dataset>` (run that cell, drop the model, narrow `datasets`, or
weigh the dataset 0) or `<dataset> lacks <metric>` (that run did not record
the metric), and `Over budget <model>: cost at most 0.05` for a budget it
broke. No finished run at all fails with `no finished runs under <dir>`.

## Steps

1. Run the evaluations first (openarena-run-evaluation), with cost, tokens or
   latency listed under `metrics:` if they will be ranked.
2. Write the leaderboard; scope it to what those runs covered, and write
   `shares` explicitly for a board that drives a decision.
3. `uv run openarena check my-arena/configs` (openarena-check-configs).
4. `leaderboards run`; check the `Datasets:` and `Shares:` lines against the
   entry, then explain the ranking from the printed table and the unranked
   lines, not from memory.
