---
name: openarena-read-results
description: Use when an openarena run has finished or failed and its outcome has to be understood — reading result.json (cells, metrics, status, the Pareto frontier), finding why a cell failed or scored 0 (missing API key, unreachable model, bad template, sandbox trouble) from trial.json, and reading a Harbor run's trials.jsonl and per-trial logs.
---

# openarena-read-results

Everything a run produced is under `<state-dir>/<evaluation>/` (default
`.open-arena/<evaluation>/`, see `openarena evaluations run --help`):

```
config.yaml                    what ran: the evaluation and every entry, gathered; rewritten each launch
result.json                    the scores; written only when the sweep finishes, replaced on each run
<dataset>/trial_N/trial.json   one per model: status, metrics, and `message` (the full traceback)
harbor/trials.jsonl            Harbor datasets only: one line per row
harbor/trials/<trial>/         that row's logs
```

Read `config.yaml` in the run folder, not `configs/`, to know what produced a
number. The CLI saves no per-row answers and does not trace into MLflow.
Full reference: `docs/guides/results.md`.

## Did it work?

`openarena evaluations run` exits 0 even when cells failed. Always check the
statuses. Do not rely on the exit code. A failed cell prints
`<model> failed: <last line of the error>` after its dataset.

## result.json

```json
{
  "meta": {
    "config": ".open-arena/smoke/config.yaml",
    "models": ["qwen38", "gpt-4o-mini"],
    "datasets": ["gsm8k-20"],
    "metrics": ["reward"],
    "directions": {"gsm8k-20": "max"},
    "metric_directions": {},
    "objectives_by_ds": {"gsm8k-20": [["reward", "max"]]},
    "frontier_by_ds": {"gsm8k-20": []}
  },
  "rows": [
    {"model": "gpt-4o-mini", "dataset": "gsm8k-20", "metric": "reward",
     "value": 0.85, "status": "COMPLETED", "direction": "max", "on_frontier": false}
  ]
}
```

- One row per model × dataset × metric. `reward` is the mean of the dataset's
  reward over its rows, usually 0 to 1. Its `direction` is the reward's own.
- `metrics` is `reward` plus only the aliases listed under a dataset's or
  the evaluation's `metrics:`. Cost, tokens and latency appear only when they
  are listed there, and they need `direction: min` because the default is `max`:

  ```yaml
  evaluations:
    smoke:
      # ...
      metrics:
        - {class: cost, direction: min}
        - {class: output_tokens, direction: min}
  ```

- `frontier_by_ds` is empty and every `on_frontier` is `false` unless a metric is
  marked `objective: true`, which gives the frontier a second axis.
- `status` is the cell's outcome, and it is the same on all of the cell's rows:

| status | meaning |
|---|---|
| `COMPLETED` | every row was asked and scored. The value can still be 0 |
| `FAILED` | a deliberate failure, e.g. a Harbor cell where no row got an answer |
| `INVALID` | any other error: key, connection, template, reward |
| `null` | no trial: the run never reached the cell |

A failed cell has `value: null`. Read the file with a script, not by eye:

```bash
python3 -c "
import json; r = json.load(open('.open-arena/smoke/result.json'))
for row in r['rows']:
    if row['metric'] == 'reward': print(row['dataset'], row['model'], row['value'], row['status'])"
```

## When a cell failed or scored 0

The full error is in the cell's `trial.json`:

```bash
grep -lE '"status": "(FAILED|INVALID)"' .open-arena/smoke/*/trial_*/trial.json
python3 -c "import json,sys; t=json.load(open(sys.argv[1])); print(t['hyperparameters']['values'], t['status']); print(t['message'])" \
  .open-arena/smoke/gsm8k-20/trial_0/trial.json
```

A provider error on a row does not fail the cell. The row gets no answer and
scores 0. So a missing key or an unreachable server usually shows as a
`COMPLETED` cell with reward 0 (and `output_tokens` near 0, if you listed it).
That 0 is cached, so after fixing the cause, rerun with `--no-cache`. A plain
rerun only retries the failed cells.

| symptom | likely cause | fix |
|---|---|---|
| one model at 0, others fine | its `*_API_KEY` unset, or `ollama/...` with no Ollama, or `vllm/...` without `HOSTED_VLLM_API_BASE` | set the key in `.env`; ask the user to start the server (do not start it yourself); rerun with `--no-cache` |
| model not found | a wrong id in `models.yaml` | fix `model:` |
| `INVALID` at the dataset | wrong `path` / `name` / `split`, or a template naming a missing column | look at a real row; fix `config` (openarena-add-dataset) |
| every model at 0, answers look right | the answer's form differs from the reference (`"42"` vs `"42.0"`, `"B"` vs `"B) ..."`) | tighten `output_schema` or the template, or use `answer_accuracy` / a judge |
| a judge reward fails | the judge's `language_model` has no key | set that provider's key |
| `none of the N Harbor task(s) got an answer` (`FAILED`) | the sandbox: no Docker / E2B key, image build failed, `tmux` missing | `harbor/trials.jsonl`, below |

On one dataset, the run stops after `max(3, number of models)` failed cells in
a row. When that happens, no `result.json` is written and the previous one stays
in place. Trial state is kept, so the next run resumes.

To see why a 0 happened row by row outside Harbor, run the sweep from Python with
`run_sweep(..., on_sample=..., samples_per_cell=N)` and a fresh `state_dir`.
See "From a number to the rows" in the docs page.

## Harbor runs

Each row's trial appends one line to `harbor/trials.jsonl`:
`trial`, `dir`, `dataset`, `model`, `agent`, `task` (task datasets),
`question`, `status` (`answered` or `failed`), `answer`, `steps`, `rewards`,
`error`, `seconds`, `input_tokens`, `output_tokens`, `cost_usd`.

```bash
jq -c '{model, task, status, rewards, error}' .open-arena/harbor/harbor/trials.jsonl
```

`dir` is that trial's folder: `agent/` (the session), `verifier/` (the tests'
output) and `trial.log`. It may sit under `ARENA_HARBOR_WORKDIR` or
`~/arena-harbor/` instead of the run folder, and `dir` always gives the real
location. A `failed` line means the sandbox never ran that row, and it scores 0. If no
row of a cell got an answer, the cell is `FAILED`. Look at the sandbox, the image
or a key, not the model. If a task folder scores 0 for every model, run its
solution first, because its `tests/test.sh` may be wrong (openarena-create-harbor-task).

## Reporting

Report to the user from the files:
- the reward per model and dataset, with its status
- what failed and why, quoting `trial.json` → `message`
- what you changed

To compare models, use a leaderboard (openarena-rank-leaderboard). Do not
average the rows yourself.
