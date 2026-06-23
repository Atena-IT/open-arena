---
name: arena-prepare-data
description: Prepare evaluation datasets: write prepare_data.py, wire loaders in config.yaml, smoke-test that rows load.
---

Read `PREPARE_DATA.md` end-to-end before writing any prep script — it covers the synalinks `Generator` pattern for synthetic data, schema vs ChatMessages choices, and every pitfall. Read `README.md` (Dataset providers table) and `config.example.yaml` (dataset entries) for reference YAML.

## Two-stage pipeline

```
prepare_data.py  ──►  raw_data/<name>.jsonl  ──►  config.yaml (local/folder/…)  ──►  src/evaluate.py
```

`prepare_data.py` is a free-form script — fill it for the dataset's needs. The loaders in `src/datasets/` read the files it produces. The platform has no synthetic-data engine; put generation, filtering, deduplication, and formatting here.

## Typical prepare_data.py pattern (pull-filter-dump)

```python
import json
from datasets import load_dataset

src = load_dataset("gsm8k", "main", split="test")
with open("raw_data/gsm8k_test.jsonl", "w") as f:
    for row in src:
        if not row["question"]:
            continue
        f.write(json.dumps({
            "question": row["question"],
            "answer": row["answer"].split("####")[-1].strip(),
        }) + "\n")
```

Run it: `uv run python prepare_data.py`

Output convention: write files under `raw_data/`. Path is not enforced — any path works as long as `config.yaml` points to it.

## Synthetic generation via synalinks

See `PREPARE_DATA.md` (Using synalinks section) for the full pattern. Key points:
- Declare input/output as `DataModel` subclasses with `Field(description=...)`.
- Wire `Input` → `Generator` → `Program` and `await program(input_instance)` per seed.
- Always `asyncio.run(main())` at the top — programs are async.
- Use `temperature=0` for determinism.
- A second `Generator` can act as a quality filter (generate → judge → keep).

## Wiring into config.yaml

### Local file

```yaml
datasets:
  gsm8k_test:
    type: local
    path: raw_data/gsm8k_test.jsonl
    input_schema:
      type: object
      properties:
        question: { type: string }
      required: [question]
    input_template: |
      {"question": {{ question | tojson }}}
    output_schema:
      type: object
      properties:
        answer: { type: string }
      required: [answer]
    output_template: |
      {"answer": {{ answer | tojson }}}
    batch_size: 8
    limit: 100
    reward:
      name: exact_match
      out_mask: [question]    # mask input field re-attached by return_inputs=True
```

### Folder of per-case files

```yaml
datasets:
  cases:
    type: folder
    path: raw_data/cases
    pattern: "*.json"
    recursive: false
    batch_size: 4
    input_template: |
      {"messages":[{"role":"user","content": {{ question | tojson }} }]}
    output_template: |
      {"role":"assistant","content": {{ answer | tojson }} }
```

`folder` rows expose every file's parsed dict plus `_filename`, `_stem`, `_path` metadata.

### HuggingFace direct (no prepare_data.py needed)

```yaml
datasets:
  mmlu_test:
    type: huggingface
    path: cais/mmlu
    name: all
    split: test
    streaming: true
    limit: 50
    batch_size: 1
    input_template: |
      {"messages":[{"role":"user","content": {{ question | tojson }} }]}
    output_template: |
      {"role":"assistant","content": {{ ["A","B","C","D"][answer] | tojson }} }
    reward:
      name: exact_match
      in_mask: [content]
```

## Schema vs ChatMessages

- Use `input_schema` / `output_schema` for structured tasks (multiple-choice, JSON, numeric). Rewards see clean fields; the LM is constrained to valid JSON.
- Default (`ChatMessages` / `ChatMessage`) for free-form chat. `y_pred.content` carries the answer; use `in_mask: [content]` on comparison rewards.

Do not pass both `input_schema` and `input_data_model` — the constructor raises.

## Smoke-test

After wiring, verify every dataset loads at least one batch:

```bash
uv run python -c "
import yaml
from src.datasets import load_dataset_from_yaml
cfg = yaml.safe_load(open('config.yaml'))
for n in cfg['experiments']['datasets']:
    it = iter(load_dataset_from_yaml('config.yaml', name=n))
    next(it); print('ok', n)
"
```

## Common pitfalls

- **Templates rendering invalid JSON** — always use `{{ field | tojson }}`, never bare `{{ field }}`.
- **`StrictUndefined`** — templates fail loudly on missing keys. `prepare_data.py` must emit every key the template references.
- **Schema / data-model collision** — don't pass both `input_schema` and `input_data_model`.
- **Streaming HF datasets have no `len`** — pin `limit:` if a caller needs a bounded epoch.
- **`return_inputs=True` leaks input fields into `y_pred`** — comparison rewards need `out_mask` (schema datasets) or `in_mask: [content]` (chat-message defaults) to avoid scoring the prompt.
- **`batch_size` and `limit` interactions** — `limit` caps raw pre-`repeat` rows. Final batch count is `ceil(limit * repeat / batch_size)`.
- **Cache invalidation** — changing the dataset list invalidates keras-tuner caches. Run `--no-cache` or `rm -rf .open-arena/*/` after edits.

## Autoresearch caveat

Inside the autoresearch loop, `prepare_data.py` is **not** edited autonomously. Pause, propose the change to the human, and wait for explicit approval before touching it.
