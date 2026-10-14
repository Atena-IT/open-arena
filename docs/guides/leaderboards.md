# Leaderboards

A leaderboard ranks the models of finished runs. It reads every `result.json`
under the state folder, keeps the newest run of each cell, keeps the datasets
and models in its scope, normalizes each metric per dataset against the models
in scope, and combines them with your weights into one score per model. It
also marks the Pareto frontier.

```yaml
leaderboards:
  math-value:
    title: Math, quality for the money      # required
    description: Reward first, cost second, nothing above 5 cents a cell.
    datasets: [gsm8k-20]                    # [] = every dataset with scores
    models: []                              # [] = every model with scores
    shares:                                 # weight per metric; {} = the default
      reward: 70
      cost: 30
    budgets:                                # hard limits: a model breaking one is left out
      - {metric: cost, op: lte, value: 0.05}     # lte: at most; gte: at least
      - {metric: reward, op: gte, value: 0.5}
```

| field | |
|---|---|
| `title` | required |
| `datasets`, `models` | the scope; empty takes everything scored |
| `shares` | each metric's weight, read as proportions; empty is the default |
| `budgets` | `{metric, op, value}`, `op` being `lte` or `gte` |
| `group_shares` | a weight per group of datasets (an import's suites); a group left out weighs 1 |
| `dataset_shares` | a weight per dataset within its group; 0 leaves it out of the score |

## Metrics

The names are those of the runs: `reward`, `cost`, `input_tokens`,
`output_tokens`, latency, any extra metric's alias, and `eol_days`, the days
left before the model's announced end of life, looked up online. A run's exact
list is `meta.metrics` in its `result.json`. Each metric's direction comes from
the run (a cost is better low); you never write it.

The default weights: `reward` high, every other measured metric low, `eol_days`
off.

## Scope is part of the score

A model is ranked only if it is scored on every dataset in scope, and a metric
carries weight only when every cell in scope records it. Narrow `datasets` and
`models` to compare like with like.

## Rank

```shell
openarena leaderboards list
openarena leaderboards run math-value
openarena leaderboards run math-value --no-eol --json ranking.json
```

| option | |
|---|---|
| `--configs DIR` | the configs folder (default `./configs`) |
| `--state-dir DIR` | where the runs are (default `./.open-arena`) |
| `--no-eol` | leave out `eol_days`, which asks the outside world |
| `--json FILE` | write the ranking for another program |

It prints the datasets and weights used, then a table: rank, model, score from
0 to 1, each weighted metric, and ★ on the Pareto frontier. Then a line for
each model it could not rank:

```
Not ranked qwen38: not scored on mmlu-math
Not ranked gpt-4o-mini: gsm8k-20 lacks judged
Over budget big-model: cost at most 0.05
```

## How the score is computed

Per dataset, each weighted metric is normalized against the models in scope
to `[0.05, 1]`: linearly for the reward and calendar days, on a log scale for
magnitudes such as cost and tokens, inverted where lower is better. A model's
score on the dataset is the weighted geometric mean of its normalized metrics;
its score is the mean over the datasets, weighted by `group_shares` and
`dataset_shares`. Budgets remove models before anything is normalized.
