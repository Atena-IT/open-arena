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

`openarena evaluations run` exits 1 when any cell failed, after printing
`N cell(s) failed: <model> × <dataset>, ...`; `result.json` is written
either way. A failed cell also prints `<model> failed: <last line of the
error>` after its dataset. Exit 0 means every cell scored, not that the
scores are good: still read the statuses and values.

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
  are listed there; written bare they are better lower (`direction: min`),
  any other metric higher, unless the entry sets `direction:`:

  ```yaml
  evaluations:
    smoke:
      # ...
      metrics:
        - cost
        - output_tokens
  ```

- `frontier_by_ds` is empty and every `on_frontier` is `false` unless a metric is
  marked `objective: true`, which gives the frontier a second axis.
- `status` is the cell's outcome, and it is the same on all of the cell's rows:

| status | meaning |
|---|---|
| `COMPLETED` | every row was asked and scored, at least one got an answer. The value can still be 0 |
| `FAILED` | no row got an answer (a key, a model server or an A2A agent unreachable), or an agent's MCP servers gave no tools |
| `INVALID` | any other error: template, reward, ... |
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

A provider error on one row does not fail the cell: that row gets no answer
and scores 0, with a warning. When **no** row got an answer, the cell is
`FAILED` with `no row got an answer: all N row(s) failed; the first error: <the
error>`: a missing key or an unreachable server shows there, not as a 0. A
plain rerun runs the failed cells again, so after fixing the cause no
`--no-cache` is needed. A `COMPLETED` cell is reused as it is, its 0s too:
rerun with `--no-cache` to redo one where only some rows failed.

| symptom | likely cause | fix |
|---|---|---|
| one model `FAILED`, `no row got an answer`, others fine | its `*_API_KEY` unset, or `ollama/...` with no Ollama, or `vllm/...` without `HOSTED_VLLM_API_BASE`, or `sglang/...` with no SGLang server | set the key in `.env`; ask the user to start the server (do not start it yourself); rerun |
| `FAILED`, `could not load the MCP tools of [...]` | an MCP server down, its command not on the PATH, or it exposes no tools | openarena-add-mcp-tools; rerun |
| model not found | a wrong id in `models.yaml` | fix `model:` |
| `INVALID` at the dataset | wrong `path` / `name` / `split`, or a template naming a missing column | look at a real row; fix `config` (openarena-add-dataset) |
| every model at 0, answers look right | the answer's form differs from the reference (`"42"` vs `"42.0"`, `"B"` vs `"B) ..."`) | tighten `output_schema` or the template, or use `answer_accuracy` / a judge |
| a judge reward fails | the judge's `language_model` has no key | set that provider's key |
| the run refuses to start: `not set in the environment: NAME (at ...)` | a `${NAME}` in the config is set neither in the shell nor in `.env` | add `NAME=...` to the workspace's `.env` |
| `none of the N Harbor task(s) got an answer` (`FAILED`) | the sandbox: no Docker / E2B key, image build failed, `tmux` missing | `harbor/trials.jsonl`, below |

However many cells fail, the run goes on to the end and writes
`result.json`. Only an interruption (Ctrl-C, a crash) leaves no new
`result.json`; the previous one stays in place, and the trial state is kept,
so the next run resumes.

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
