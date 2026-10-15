---
name: openarena-run-evaluation
description: Use when models (or A2A agents) have to be evaluated on datasets with openarena — writing a saved evaluation in evaluations.yaml (which models, which datasets, batch size, extra metrics), launching it with `openarena evaluations run`, resuming or restarting it, and reading the score matrix it prints and writes to result.json.
---

# openarena-run-evaluation

An evaluation is a saved sweep: every model of `language_models` on every
dataset of `datasets`, one **cell** per pair. `openarena evaluations run NAME`
runs it on this machine, prints a table per dataset and writes `result.json`.
Depth: [Running evaluations](https://atena-it.github.io/open-arena/guides/evaluations/),
[CLI](https://atena-it.github.io/open-arena/cli/).

## The entry

```yaml
evaluations:
  math-compare:
    description: Local Qwen against the hosted baseline on GSM8K.
    tags: [math, weekly]
    language_models: [qwen38, gpt-4o-mini]   # required: models keys (a2a_agents keys for an a2a sweep)
    datasets: [gsm8k-20]                     # required: datasets keys, each with a reward:
    batch_size: 4                            # optional: rows in flight at once, 1 to 1024
    metrics:                                 # optional: columns beside reward, on every dataset
      - cost                                 # better lower without saying so
      - avg_latency
      - answer_accuracy
    seed: 0                                  # optional: the engine's own randomness only
```

These are the only fields: `language_models`, `datasets`, `batch_size`,
`metrics`, `seed`, `default_language_model`, `default_embedding_model`,
`description`, `tags`. Any other key is an error. In particular there is no
`concurrency`, `repeats`, `limit`, `budget` or `retry`:

| you want | do |
|---|---|
| fewer rows | `limit: N` in the dataset's `config` (openarena-add-dataset) |
| each row answered N times | `repeat: N` in the dataset's `config` |
| temperature, max tokens | the model's `params` (openarena-add-model) |
| a spending cap | a leaderboard's `budgets:` (openarena-rank-leaderboard); nothing stops a run while it spends |
| more retries | not configurable: each model call gets up to five attempts |
| cells in parallel | not possible: cells run one at a time; raise `batch_size` instead |

Rules the run (and `openarena check`) enforces:

- Every dataset named needs a `reward:`; every model named must exist and be
  of the type the datasets need (`embedding` models for embedding datasets,
  `llm` otherwise, `a2a_agents` keys for A2A datasets).
- A2A, Harbor and embedding datasets cannot share a sweep with other kinds:
  write one evaluation per kind.

**`batch_size`** is the concurrency knob: rows sent at once, hence requests in
flight (more with a judge reward). A cell uses the first one set of:
`--batch-size` on the command line, the evaluation's `batch_size`, the
dataset's own `batch_size`, then 1. Keep it at 1 to 4 for a local model, a
low-tier key or an `agent` dataset, and at 1 for Harbor on a laptop. Too high
does not error: throttled calls exhaust their retries, the rows score 0, and
you see `All retries failed for ...` warnings and a too-low reward.

**`metrics`**: a bare name or `{class, alias, direction, objective, ...}`.
`direction` is optional: unset, it is `min` for a cost, tokens or latency
measurement (`cost`, `reward_cost`, `embedding_cost`, `input_tokens`,
`output_tokens`, `total_tokens`, `avg_latency`) and `max` for anything else;
an explicit `direction:` wins. Only listed columns are kept
in `result.json`: list `cost` if a leaderboard should rank on it. A dataset's
own metric of the same name wins over the evaluation's.

**`seed`** seeds the engine's random generators. It does not make provider
sampling repeatable: use `temperature: 0` on the models for that.

**`default_language_model` / `default_embedding_model`**: the model a reward or
metric uses when its config names none (a `models` key of type `llm` /
`embedding`). The run also writes it to the engine's settings file in your
home folder, where it outlives the run, so prefer naming the judge on the
reward itself.

## Running it

From the workspace folder (the one holding `configs/`):

```bash
openarena check configs
openarena evaluations list
openarena evaluations run math-compare -v
```

| option | does |
|---|---|
| `--configs DIR` | the configs folder (default `configs`); `.env`, `tasks/` and `environments/` are read from its parent |
| `--state-dir DIR` | where runs live (default `.open-arena`, relative to where you run), one folder per evaluation |
| `--no-cache` | start over instead of resuming |
| `--batch-size N` | the batch size for this run, over the evaluation's and the datasets' |
| `--json FILE` | also write the result there |
| `-v` / `-vv` | a progress bar per cell / a line per batch |

The run reads the workspace's `.env` first (never over a variable already set
in the shell), replaces each `${NAME}` in the config with that variable
(`$${NAME}` stays a literal `${NAME}`), validates the whole config and refuses
to start on a problem, a `${NAME}` set nowhere included
(`not set in the environment: NAME (at <where>)`). Keep secrets in `.env` and
reference them as `${NAME}`; `config.yaml` below keeps the references, not the
values.
It does not trace to MLflow (only the arena app does). It writes
`<state-dir>/<name>/`:

```
config.yaml            the evaluation and every entry it names, gathered: what ran
result.json            the matrix, written when the run ends (openarena-read-results)
<dataset>/             each dataset's saved cells, what a resumed run reads back
harbor/                Harbor trials, for harbor datasets
environments/ tasks/   copies of the Harbor folders the run used
```

**Exit code.** `0` when every cell scored; `1` on a config error, a crash, or
when any cell failed (`N cell(s) failed: <model> × <dataset>, ...`, after the
tables). `result.json` is written either way: read each cell's `status`
there.

A cell fails alone and the run goes on to the next one. It fails when its
program cannot be built (an agent whose MCP servers give no tools:
`could not load the MCP tools of [...]`), when an error stops it, or when
**no row got an answer** (`no row got an answer: all N row(s) failed; the
first error: <the provider's error>`): a missing key, a model server that is down,
an unknown model id, an A2A agent that cannot be reached. A row that alone
gets no answer (a flaky call, an A2A row the agent does not answer) scores 0,
with a warning, and the cell goes on.

## Resuming and starting over

Resume is the default: running the same name again reuses every finished
cell and runs only the failed, interrupted or never-reached ones. So after
fixing a missing key or starting a server that was down, a plain rerun is
enough: those cells failed, and they run again. A finished cell is kept even
when its inputs changed, and so is one where only some rows scored 0
(throttling, a flaky server). So run with `--no-cache` after:

- fixing what made some rows score 0 in a cell that finished;
- changing a dataset (rows, templates, `limit`, `repeat`), a reward, the
  `metrics`, a model's `model`/`params`, or the `language_models` list.

`--no-cache` wipes the saved cells of **every** dataset of the evaluation, so
everything runs, and is paid for, again. To redo part of a sweep, write a
smaller evaluation. Changing only `batch_size`, `description` or `tags` needs
no `--no-cache`. Two evaluations never share saved state.

## Cost and time

Each row is at least one model call per model, plus one per judge reward or
judge metric; an `agent` row is a loop of calls; a Harbor row is a whole agent
session in a sandbox. Before a full run:

1. Smoke it: set a small `limit` (e.g. `limit: 5`) in the datasets' `config`,
   run one cheap model with `cost` as a metric, and read the answers, not only
   the scores (a reward of 0 everywhere usually means a template or a
   reward that does not match the answers). Then raise `limit` and run with
   `--no-cache`.
2. Before the real size, tell the user how many rows × models you are about
   to run when it is large or uses paid models.

Hosted models need their key in `.env` or the shell (`OPENAI_API_KEY`, ...).
`openarena check` cannot see a missing key: it shows at run time as a failed
cell (`no row got an answer: ...`, the provider's error naming the key). A
`cost` of 0 on a hosted model means its price is unknown, not that it was
free.

## The output

While it runs: `<model> running…` when a cell starts, `<model> done reward
...` with its numbers when it ends, and after each dataset `<model> failed:
<why>` for every failed cell. At the end, one table per dataset: a row per
model, a column per metric (`reward` first, then the `metrics` columns you
listed, e.g. cost, tokens, latency), `—` where a cell has no value. The same
is in `result.json`; read it as openarena-read-results says, and rank several
runs with openarena-rank-leaderboard.
