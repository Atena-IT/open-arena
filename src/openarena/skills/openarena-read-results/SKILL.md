---
name: openarena-read-results
description: Use when an openarena run has finished or failed and its outcome has to be understood — reading result.json (cells, metrics, status, the Pareto frontier), finding why a cell failed or scored 0 (missing API key, unreachable model, bad template, sandbox trouble), and reading a Harbor run's trials.jsonl and per-trial logs.
---

# openarena-read-results

Everything a run produced is under `<state-dir>/<evaluation>/` (default
`.open-arena/<evaluation>/`): `config.yaml` (what ran), `result.json` (the
scores) and, for Harbor datasets, `harbor/`.

## result.json

```json
{
  "meta": {
    "models": ["qwen38", "gpt-4o-mini"],
    "datasets": ["gsm8k-20"],
    "metrics": ["reward", "cost", "input_tokens", "output_tokens", "..."],
    "directions": {"gsm8k-20": "max"},
    "metric_directions": {"cost": "min"},
    "frontier_by_ds": {"gsm8k-20": ["gpt-4o-mini"]}
  },
  "rows": [
    {"model": "gpt-4o-mini", "dataset": "gsm8k-20", "metric": "reward",
     "value": 0.85, "status": "COMPLETED", "direction": "max", "on_frontier": true}
  ]
}
```

One row per model × dataset × metric. `reward` is the mean of the dataset's
reward over its rows, 0 to 1. `value: null` with a status other than
`COMPLETED` is a failed cell. Read it with a short script, never by eye on a
large file:

```bash
python3 -c "
import json; r = json.load(open('.open-arena/smoke/result.json'))
for row in r['rows']:
    if row['metric'] == 'reward': print(row['dataset'], row['model'], row['value'], row['status'])"
```

`config.yaml` beside it is the exact evaluation and entries that ran, ids
resolved: check it when a result looks wrong for the config you expected.

## When a cell failed or scored 0

The run prints `<model> failed: <why>` after each dataset for a failed cell;
start from that line.

| symptom | likely cause | where to look / fix |
|---|---|---|
| failed, auth / key error | the provider's `*_API_KEY` unset | the printed error; set the key, rerun (resumes) |
| failed, connection refused | `ollama/...` with no Ollama, `vllm/...` without `HOSTED_VLLM_API_BASE` | ask the user to start it; do not start it yourself |
| failed, model not found | a wrong id in `models.yaml` | fix `model:` |
| failed at the dataset | wrong `path` / `name` / `split`, a template using a missing column | look at a real row; fix `config` (openarena-add-dataset) |
| reward 0 everywhere, answers look right | the reference does not match the answer's form (`"42"` vs `"42.0"`, `"B"` vs `"B) ..."`) | tighten `output_schema` (`enum`, description) or the template; or use `answer_accuracy` / a judge |
| a judge reward fails | the judge's `language_model` has no key | the judge model's provider key |
| every Harbor row of a cell failed | the sandbox: no Docker / E2B key, image build failed, `tmux` missing | `harbor/trials.jsonl`, below |

After a fix, rerun the same command: finished cells are kept, failed ones run
again. After changing what a finished cell depends on, add `--no-cache`.

## Harbor runs

```
harbor/
  trials.jsonl        one line per trial: dataset, model, question, outcome and why
  trials/<trial>/     that trial's logs: the agent's session, the verifier's output
```

```bash
python3 -c "
import json
for line in open('.open-arena/harbor/harbor/trials.jsonl'):
    t = json.loads(line); print({k: t[k] for k in list(t)[:8]})"
```

A row whose trial left no trajectory (the sandbox would not start, the agent
crashed at once) scores 0 and its line says why. A cell where no row got an
answer fails as a whole: that is the sandbox, the image or a key, not the
model. For a task folder that scores 0 for every model, run its solution
first: the task's `tests/test.sh` may be wrong (openarena-create-harbor-task).

## Reporting

Report to the user from the files: the reward per model and dataset, what
failed and why, what you changed. Compare models with a leaderboard
(openarena-rank-leaderboard) rather than by averaging rows yourself.
