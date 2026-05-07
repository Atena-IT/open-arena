# Preparing datasets

`prepare_data.py` and `src/datasets/` form a two-stage pipeline:

1. **`prepare_data.py`** — free-form, project-local script that turns
   *raw* sources (HF dumps, scraped pages, prod logs, synthetic
   generations, …) into *evaluation-ready* files on disk. The platform
   does **not** ship a synthetic-data engine; anything you need —
   generation, selection, deduplication, filtering, formatting — lives
   here. Output convention: write files under `raw_data/` (or another
   path of your choice) in a shape the loaders can read directly.
2. **`src/datasets/`** — Jinja2-templated streaming loaders, declared
   in `config.yaml` and instantiated by `load_dataset_from_yaml`. The
   loaders are the bridge between the on-disk files (or remote
   service) and `evaluate.py`.

End-to-end flow:

```
prepare_data.py  ──►  raw_data/<name>.jsonl  ──►  config.yaml (local/folder/...)  ──►  evaluate.py
```

## `prepare_data.py`

Intentionally free-form. The file ships as a stub; fill it with
whatever the dataset needs. Typical patterns:

- **Pull → filter → dump JSONL.** Read a HF or local source, drop
  unusable rows, write one JSON object per line:
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
- **Synthetic generation.** Loop a generator program over seed prompts
  and dump the outputs. Keep determinism (set seeds, log model
  versions) so reruns reproduce.
- **Folder corpora.** When each example is its own file (long docs,
  per-case YAML), write to `raw_data/<name>/<id>.json` and use the
  `folder` loader.

Run it directly:

```bash
uv run python prepare_data.py
```

Output should land in a path your `config.yaml` will point to.
`raw_data/` is the convention but not enforced.

**Autoresearch caveat**: `prepare_data.py` is editable but **not
autonomously**. Inside the autoresearch loop, pause and propose the
change to the human before touching it.

## Using synalinks inside `prepare_data.py`

Synalinks isn't only the eval runtime — the same `Generator` /
`Program` primitives that `evaluate.py:build_program` uses for the
sweep work just as well for *producing* rows offline. The typical use
cases are synthetic question generation, paraphrase / augmentation,
LM-assisted filtering, and labeling unlabeled corpora into the
`(input, target)` shape the loaders expect.

The idiomatic pattern: declare the input and output as `DataModel`
classes, chain `Input` → `Generator` → `Program`, then `await
program(input_instance)` per row. Same primitives `evaluate.py` uses,
called offline.

```python
# prepare_data.py
import asyncio
import json

import synalinks


class Seed(synalinks.DataModel):
    topic: str = synalinks.Field(description="A short topic to write a problem about")


class QA(synalinks.DataModel):
    question: str = synalinks.Field(description="A grade-school math problem")
    answer:   str = synalinks.Field(description="The numeric answer as a string")


async def main():
    inputs = synalinks.Input(data_model=Seed)
    outputs = await synalinks.Generator(
        data_model=QA,
        language_model="ollama/llama3.2",   # plain string is fine
        instructions="Write one GSM8K-style problem about the given topic.",
        temperature=0.0,
    )(inputs)
    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="gsm8k_synth",
        description="Generate one (question, answer) row per seed topic.",
    )

    seeds = ["coins", "apples", "trains", "ages", "discounts"]
    with open("raw_data/gsm8k_synth.jsonl", "w") as f:
        for topic in seeds:
            row = await program(Seed(topic=topic))
            data = row.get_json()
            f.write(json.dumps({
                "question": data["question"],
                "answer":   data["answer"],
            }) + "\n")


asyncio.run(main())
```

Notes:

- **`DataModel` + `Field(description=...)` is the canonical shape.**
  Synalinks infers the JSON schema from the data model and uses it
  as a structured-output constraint. The `Field` descriptions are
  part of that schema (they help the LM honor each field's intent),
  **not** the prompt — the prompt comes from the `instructions=`
  param on the `Generator`. `await program(...)` returns an instance
  you can `.get_json()` on. (Raw `schema={...}` dicts also work —
  that's the pattern `program.py` uses when the dataset declares its
  schema in YAML — but for a one-off prep script, the `DataModel` is
  shorter and self-documenting.)
- **Always `asyncio.run` at the top.** Synalinks programs are async;
  `await` them under a top-level `main()`.
- **Invoke with `await program(input_instance)`**, not `program(...)`.
  Forgetting `await` returns a coroutine instead of the result.
- **Feed seeds, not the full dataset.** `prepare_data.py` runs once
  and writes files; it doesn't need to stream. A list of seeds + a
  loop is simpler than building a `Dataset`.
- **Cheap models for prep, fixed models for eval.** Generation /
  filtering / augmentation can use a small local model
  (`ollama/llama3.2`, `ollama/mistral`); the eval grid in
  `experiments.language_models` stays whatever you set for the sweep.
- **Determinism.** Set `temperature=0` (or a fixed seed if the
  provider supports it) on the `Generator` so reruns reproduce. Log
  the model id and any prompt template alongside the output file so
  the corpus is reproducible.
- **Filtering loop.** A common pattern is generate → judge → keep.
  Wire a second `Generator` (or any of the project-local rewards from
  `src/rewards/`) as a quality filter, drop rows below a threshold,
  write the survivors. Same `Generator` API, just used for scoring
  instead of producing rows.
- **Output schema = loader schema.** Whatever JSON shape you write
  here must match the `input_template` / `output_template` (or
  `input_schema` / `output_schema`) of the corresponding entry in
  `config.yaml`. Easiest: pick the schema first, then use it both
  here and in the loader entry.

## Loader providers

`src/datasets/__init__.py:_DATASET_TYPES` registers the available
providers. Each is selected via `type:` in `config.yaml`:

| `type:`       | Class               | Source                                                    |
| ------------- | ------------------- | --------------------------------------------------------- |
| `local`       | `LocalDataset`      | Single file: `.jsonl` / `.ndjson` / `.csv` / `.tsv` / `.parquet` |
| `folder`      | `FolderDataset`     | One file per record under a directory (json/yaml/text)    |
| `huggingface` | `HuggingFaceDataset`| `datasets.load_dataset(path, name, split, …)`             |
| `langfuse`    | `LangfuseDataset`   | Langfuse-managed dataset                                  |
| `langsmith`   | `LangSmithDataset`  | LangSmith-managed dataset                                 |
| `opik`        | `OpikDataset`       | Comet Opik dataset                                        |
| `phoenix`     | `PhoenixDataset`    | Arize Phoenix dataset                                     |
| `braintrust`  | `BraintrustDataset` | Braintrust dataset                                        |

Pick `local` / `folder` for offline files produced by
`prepare_data.py`; pick the others for remote-managed sources where
the upstream system already curates rows.

## The base `Dataset` contract

All loaders subclass `src.datasets.dataset.Dataset` and yield
`(x, y)` tuples — numpy object arrays of `DataModel` (or
`JsonDataModel`) instances — sized by `batch_size`. The shape per row
is controlled by Jinja2 templates plus either a `DataModel` class or a
raw JSON Schema.

Constructor knobs (forwarded from YAML verbatim):

- `input_template` / `output_template` — **required**, Jinja2 strings
  rendering one raw row to JSON. Use the `tojson` filter for safe
  string escaping.
- `input_data_model` / `output_data_model` — Python class describing
  the rendered row. Defaults to `synalinks.ChatMessages` (input) and
  `synalinks.ChatMessage` (output) when neither this nor a schema is
  set.
- `input_schema` / `output_schema` — raw JSON Schema (dict or
  JSON-encoded string), mutually exclusive with the data-model knobs.
  Schema-based rows wrap as `JsonDataModel`, the schema flows into the
  LM as a structured-output constraint, and `y_true` / `y_pred` lose
  the ChatMessage auxiliary noise (`thinking`, `tool_calls`, …).
- `batch_size` — examples per yielded batch.
- `limit` — cap raw rows (pre-`repeat`). Useful for smoke tests and
  for keeping the autoresearch sweep at ~5 minutes wall clock.
- `repeat` — emit each raw row N times in a row. `repeat == batch_size`
  gives GRPO-style batches of N rollouts of one prompt.

`input_data_model` / `output_data_model` are **not** exposed in YAML —
they default to `ChatMessages` / `ChatMessage` inside the loader. Use
`input_schema` / `output_schema` from YAML when you need
structured-output constraints.

## Wiring into `config.yaml`

Each entry under `datasets:` becomes a kwargs bag for the chosen
loader. `type:` is consumed by the dispatcher; `generator:` and
`reward:` are consumed by `evaluate.py`; everything else is forwarded
verbatim to the loader's constructor.

### Local file (typical `prepare_data.py` output)

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
      out_mask: [question]
```

### HuggingFace direct

```yaml
datasets:
  mmlu_test:
    type: huggingface
    path: cais/mmlu
    name: all
    split: test
    input_template: |
      {"messages":[{"role":"user","content": {{ question | tojson }} }]}
    output_template: |
      {"role":"assistant","content": {{ ["A","B","C","D"][answer] | tojson }} }
    batch_size: 8
    limit: 100
    reward:
      name: exact_match
      in_mask: [content]
```

### Folder of per-case files

```yaml
datasets:
  cases:
    type: folder
    path: raw_data/cases
    pattern: "*.json"
    recursive: false
    input_template: |
      {"messages":[{"role":"user","content": {{ question | tojson }} }]}
    output_template: |
      {"role":"assistant","content": {{ answer | tojson }} }
    batch_size: 4
```

`folder` rows expose every file's parsed dict at the top level **plus**
`_filename`, `_stem`, `_path` metadata for the templates.

### Smoke-testing it loads

After `prepare_data.py` writes the files and you've added the entry:

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

## Choosing schema vs ChatMessages

- **Use `input_schema` / `output_schema`** when the task has a
  well-defined structured shape (multiple-choice answer, JSON object,
  numeric value). Rewards see clean fields; the LM is constrained to
  produce valid JSON.
- **Default to `ChatMessages` / `ChatMessage`** for free-form text /
  chat tasks. `y_pred.content` carries the model's response;
  `y_true.content` carries the gold.

The two interact with reward masking — see `REWARDS_BUILDING.md` for
`in_mask` / `out_mask` semantics, especially that `return_inputs=True`
re-attaches input fields onto `y_pred`.

## Adding a new provider

Only when an existing provider can't express the source. Most
real-world cases are better served by writing a `prepare_data.py`
script that dumps to JSONL and using `local`.

If you do need a new provider:

1. Create `src/datasets/<name>_dataset.py` with a class subclassing
   `Dataset`. Implement `_iter_rows()` as a generator yielding raw row
   dicts (one per example). Each dict's keys must match the Jinja2
   variables your YAML templates reference.
2. Forward all base-class kwargs through `super().__init__(...)`.
   Provider-specific kwargs (API key, host, dataset id) become YAML
   keys.
3. Implement `__len__` if a bounded count is cheaply available (e.g.
   parquet metadata) — needed only when callers ask for bounded
   epochs.
4. Register it in `src/datasets/__init__.py:_DATASET_TYPES` under the
   snake_case name you want users to type as `type:` in YAML.

**Autoresearch caveat**: `src/datasets/` is read-only inside the
autoresearch loop. Add new providers outside the loop.

## Pitfalls

- **Templates rendering invalid JSON**. Always use the Jinja `tojson`
  filter for string interpolation. Plain `{{ x }}` will break on
  quotes, newlines, unicode.
- **`StrictUndefined`**. Templates fail loudly on missing fields —
  good for catching prep bugs, but means `prepare_data.py` must emit
  every key the template references.
- **Schema vs data-model collision**. Don't pass both `input_schema`
  and `input_data_model` (or both on the output side); the
  constructor raises.
- **Streaming HF datasets have no `len`**. Without `limit:` they
  iterate until the source is exhausted — fine for evaluation, but
  `__len__` raises. Pin `limit:` if a caller needs a bounded epoch.
- **`return_inputs=True` leaks input fields into `y_pred`**. The
  evaluation harness re-attaches inputs to `y_pred` so judge rewards
  can see the prompt. Comparison rewards (`exact_match`,
  `cosine_similarity`, …) need `out_mask: [<input field names>]`
  (schema datasets) or `in_mask: [content]` (chat-message defaults)
  so the comparison only spans fields that exist in `y_true`. See
  `REWARDS_BUILDING.md`.
- **`batch_size` and `limit` interactions**. `limit` caps *raw*
  pre-`repeat` rows. Final batch count is
  `ceil(limit * repeat / batch_size)`.
- **Cache invalidation**. Changing the dataset list in `config.yaml`
  invalidates the keras-tuner cache at `.kt/open_arena/`. Run with
  `--no-cache` or `rm -rf .kt/open_arena` after edits.
- **CSV / JSONL row counts are unknown without a scan**. `LocalDataset`
  raises `NotImplementedError` from `__len__` for those formats unless
  you set `limit:`. Parquet metadata gives the count for free.
