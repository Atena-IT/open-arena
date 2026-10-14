# Reading results

Everything a run produced is under `.open-arena/<evaluation>/`: `config.yaml`,
what ran; `result.json`, the scores; and for Harbor datasets, `harbor/`.

## result.json

```json
{
  "meta": {
    "models": ["qwen38", "gpt-4o-mini"],
    "datasets": ["gsm8k-20"],
    "metrics": ["reward", "cost", "input_tokens", "output_tokens"],
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

One row per model, dataset and metric. `reward` is the mean of the dataset's
reward over its rows, from 0 to 1. A row with `value: null` and a status other
than `COMPLETED` is a failed cell.

```shell
python -c "
import json; r = json.load(open('.open-arena/smoke/result.json'))
for row in r['rows']:
    if row['metric'] == 'reward': print(row['dataset'], row['model'], row['value'], row['status'])"
```

## When a cell failed or scored 0

The run prints `<model> failed: <why>` after each dataset; start from that line.

| symptom | likely cause | fix |
|---|---|---|
| failed, authentication error | the provider's key is not set | set it in `.env`, run again (it resumes) |
| failed, connection refused | no Ollama running, or `HOSTED_VLLM_API_BASE` unset | start the model server |
| failed, model not found | a wrong id in `models.yaml` | fix `model:` |
| failed reading the dataset | a wrong `path`, `name` or `split`, or a template naming a missing column | look at a real row, fix `config` |
| reward 0 everywhere, answers look right | the answer's form differs from the expected one (`"42"` and `"42.0"`, `"B"` and `"B) ..."`) | tighten the `output_schema` or the template, or use `answer_accuracy` or a judge |
| a judge reward fails | the judge model's key is not set | set its provider's key |
| every Harbor row of a cell failed | the sandbox: no Docker or E2B key, a failed image build, no `tmux` | read `harbor/trials.jsonl` |

After a fix, run the same command again: finished cells are kept and failed
ones run again. After changing what finished cells depend on, add `--no-cache`.

## Harbor runs

```
harbor/
  trials.jsonl        one line per trial: dataset, model, question, outcome and why
  trials/<trial>/     that trial's logs: the agent's session, the verifier's output
```

A row whose trial left no trajectory (the sandbox would not start, the agent
crashed at once) scores 0, and its line says why. A cell where no row got an
answer fails as a whole: that is the sandbox, the image or a key, not the
model. When a task scores 0 for every model, check the task itself: its
solution should make its tests pass.
