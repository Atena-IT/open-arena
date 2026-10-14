---
name: openarena-rank-leaderboard
description: Use when finished openarena runs have to be turned into a ranking — writing a leaderboard in leaderboards.yaml (which datasets and models are in scope, how much each metric weighs, hard budgets such as a cost cap, per-dataset weights) and printing it with `openarena leaderboards run`, including why a model was left unranked.
---

# openarena-rank-leaderboard

A leaderboard ranks the models of finished runs. It reads every
`result.json` under the state dir, newest run first per cell, keeps the
datasets and models in its scope, normalizes each metric per dataset against
the models in scope, and combines them with the weights you give into one
score per model. It also marks the Pareto frontier.

## The entry

```yaml
leaderboards:
  math-value:
    title: Math, quality for the money      # required
    description: Reward first, cost second, no model above 5 cents a cell.
    tags: [math]
    datasets: [gsm8k-20]                    # datasets keys in scope; [] = every dataset scored
    models: []                              # models keys in scope; [] = every model scored
    shares:                                 # percent of the score per metric; {} = the default
      reward: 70
      cost: 30
    budgets:                                # hard limits: a model breaking one is left out
      - {metric: cost, op: lte, value: 0.05}     # lte: at most; gte: at least
      - {metric: reward, op: gte, value: 0.5}
    # group_shares: {}                      # weight per group of datasets (an import's suites); left out = 1
    # dataset_shares: {gsm8k-20: 2}         # weight per dataset within its group; 0 = out of the score
```

- **Metric names** are the result's own: `reward`, `cost`, `input_tokens`,
  `output_tokens`, latency, any extra metric's alias, and `eol_days` (days to
  the model's end of life, looked up online). The exact list of a run is
  `meta.metrics` in its `result.json`.
- **`shares`** need not add to 100; they are read as proportions. Empty is
  the default: `reward` high, every other measured metric low,
  `eol_days` off.
- **Direction** comes from the run (a cost is better low); you never write it.
- **Scope is part of the score.** A model must be scored on every dataset in
  scope to rank; a metric carries weight only when every cell in scope
  records it. Narrow `datasets` / `models` to compare like with like.

## Ranking

From the workspace's parent folder (or with `uv run` in the openarena project):

```bash
uv run openarena leaderboards list --configs my-arena/configs
uv run openarena leaderboards run math-value --configs my-arena/configs
uv run openarena leaderboards run math-value --configs my-arena/configs --no-eol --json ranking.json
```

| option | does |
|---|---|
| `--state-dir DIR` | where the runs are (default `./.open-arena`, the run command's default) |
| `--no-eol` | leave out `eol_days`, which asks the outside world |
| `--json FILE` | write the ranking for another program |

It prints the datasets and shares used, then a table: rank, model, score
(0–1), each weighted metric, ★ on the Pareto frontier. Then one line per
model it could not rank and why: `Not ranked <model>: not scored on
<dataset>` (run it there) or `<dataset> lacks <metric>` (a metric that run
did not record), and `Over budget <model>: cost at most 0.05` for a budget
it broke. No finished run at all exits 1 and says to run an evaluation first.

## Steps

1. Run the evaluations first (openarena-run-evaluation).
2. Write the leaderboard; scope it to what those runs covered.
3. `uv run openarena check my-arena/configs` (a dataset or model in scope
   that no longer exists is a warning).
4. `leaderboards run`; explain the ranking to the user from the printed
   table and the unranked lines, not from memory.
