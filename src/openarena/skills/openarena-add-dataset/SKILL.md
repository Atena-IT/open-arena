---
name: openarena-add-dataset
description: Use when a benchmark has to become an openarena dataset (datasets.yaml) — from Hugging Face, a local JSONL/CSV/Parquet file, a folder of files, or an LLM-observability platform — writing its provider config, the Jinja2 input and output templates over the rows, the structured output schemas, the block that says what answers each row (a model, MCP tools, A2A agents, embeddings, tool calls, a Harbor agent), and its reward.
---

# openarena-add-dataset

A dataset entry says where rows come from (`type` + `config`), how each row
becomes a question and an expected answer (two Jinja2 templates in `config`),
what answers it (at most one kind block), and which reward scores it.
Full reference: `docs/guides/datasets.md`.

## The entry, field by field

```yaml
datasets:
  gsm8k-20:
    type: huggingface                  # the provider, below
    description: GSM8K test split, 20 rows.
    tags: [math]
    config:                            # the provider's settings + templates + schemas
      path: openai/gsm8k
      name: main
      split: test
      limit: 20                        # the FIRST 20 rows; leave out for all
      batch_size: 4                    # rows in flight at once (default 1)
      input_template: '{"question": {{ question | tojson }}}'
      output_template: '{"answer": {{ answer.split(''####'')[-1] | replace('','', '''') | trim | tojson }}}'
      input_schema:
        type: object
        properties:
          question: {type: string, description: The question to answer}
        required: [question]
      output_schema:
        type: object
        properties:
          answer: {type: string, description: "The final number only, no units or commas"}
        required: [answer]
    generator:                         # the kind block: a model answers, in one call
      chain_of_thought: true
      instructions: Solve the problem. Reply with the final number only.
    reward: exact-match                # a rewards.yaml key; required to run
```

`type`, `config`, the kind block, `reward`, `metrics`, `description` and
`tags` are the entry's fields. `reward`, `metrics` and the kind block never go
inside `config` (check refuses it).

Settings every provider takes, in `config`:

- `input_template` (required, except `harbor_tasks`), `output_template`,
  `input_schema`, `output_schema`: below.
- `limit`: read at most this many rows, the **first** ones in the source's
  order. Nothing samples or shuffles; for a representative subset, make one
  (a `local` file of picked rows, a Phoenix split, an Opik `filter_string`).
- `batch_size` (default `1`): rows in flight at once. An evaluation's
  `batch_size` and `openarena evaluations run --batch-size` override it.
- `repeat` (default `1`): each row is sent this many times, each copy scored
  as a row. For measuring how much a sampled model varies; keep `1` for a
  benchmark score.

## The templates: rows → JSON

Both templates are Jinja2 over one row's columns, and each must render to
**JSON** (an object). Always pipe values through `tojson`, which quotes and
escapes them; never write `"{{ question }}"`.

- `input_template`: what the program receives. `{"question": ...}` for a
  question-answering dataset.
- `output_template`: the reference the reward compares with. Optional when a
  judge scores without one (the dataset is then inputs-only); required for
  `exact_match` and the other deterministic rewards.
- Columns: `{{ question }}`, a field `{{ input.question }}`, an item
  `{{ answers.text[0] }}`. Filters: `| tojson`, `| trim`, `| lower`,
  `| string`, `| replace('a', 'b')`, `| join(', ')`, Python string methods
  (`answer.split('####')[-1]`).
- **Strict.** A variable the row lacks is an error, not an empty string. For
  an optional column write `{{ hint | default("") | tojson }}`. A column name
  must be a valid identifier (`gold-label` cannot be named): rename it at the
  source.
- **No metadata template.** What the model should see goes in the input;
  rows you only want to select, select at the source.
- **Quoting.** Write the template as a single-quoted YAML string; a single
  quote inside it is doubled: `split(''####'')` in YAML is `split('####')` in
  Jinja. Double quotes need no escaping inside single quotes.

Lettered multiple choice, the answer an index:

```yaml
datasets:
  mmlu-math:
    type: huggingface
    config:
      path: cais/mmlu
      name: high_school_mathematics
      split: test
      limit: 50
      input_template: '{"question": {{ (question ~ "\n" ~ (choices | map("string") | list | join("\n"))) | tojson }}}'
      output_template: '{"answer": {{ "ABCD"[answer] | tojson }}}'
      input_schema: {type: object, properties: {question: {type: string, description: The question to answer}}, required: [question]}
      output_schema:
        type: object
        properties:
          answer: {type: string, enum: [A, B, C, D], description: The letter of the correct choice}
        required: [answer]
    generator:
      instructions: Answer with the letter of the correct choice only.
    reward: exact-match
```

Look at real rows before writing a template; never guess column names:

```bash
uv run python -c "import datasets; print(next(iter(datasets.load_dataset('openai/gsm8k', 'main', split='test', streaming=True))))"
head -n 2 data/rows.jsonl
```

For a platform, the row variables are the provider's fields (table below),
your data one level down (`{{ input.question | tojson }}`).

### Schemas

`input_schema` / `output_schema` are JSON Schemas of what the templates
render; with `output_schema` the model answers in that structure (an `enum`
stops "Option B" where "B" is expected, a field `description` is the
instruction for that field). List `required`; every object is closed for you
(`additionalProperties: false` unless you set it).

**Give both whenever a template renders anything but chat messages**, which is
nearly always: without `input_schema` a row is read as a list of chat messages
(`{"messages": [...]}`), and without `output_schema` the expected answer as one
chat message (`{"role": ..., "content": ...}`), so `{"question": ...}` and
`{"answer": ...}` fail when the run reads the rows. A free-text answer still
gets a schema: one `string` field, no `enum`. An `output_schema` without an
`output_template` is refused. Retrieval (`embedding`) and `harbor_tasks`
datasets are the exception: their shapes are set for them.

## Providers (`type`) and their `config`

| type | config | row variables |
|---|---|---|
| `huggingface` | `path` (repo), `name` (subset), `split`, `revision` (pin a commit), `streaming` (default true); other keys go to `datasets.load_dataset` | the columns |
| `local` | `path` (.jsonl/.ndjson/.json/.csv/.tsv/.parquet), `format` (only for another extension) | the columns (CSV values are strings) |
| `folder` | `path` (one record per file), `pattern` (glob, default `*`), `recursive`, `format` (json/yaml/text), `encoding` | the file's keys, or `text` / `record`; `_filename`, `_stem`, `_path` |
| `langfuse` | `dataset_name`, `version` (an ISO time), `streaming`, `page_size` | `input`, `expected_output`, `metadata`, `id` |
| `langsmith` | `dataset_name`, `as_of` (a version **tag**; a date does not work) | `inputs`, `outputs`, `metadata`, `id` |
| `opik` | `dataset_name`, `version`, `filter_string` | `input`, `expected_output`, `metadata`, `id` |
| `phoenix` | `dataset_name`, `version_id`, `splits` | `input`, `output`, `metadata`, `id` |
| `braintrust` | `project` or `project_id`, `dataset_name`, `version` (a transaction id), `org_name`, `fetch_batch_size` | `input`, `expected`, `metadata`, `id` |
| `mlflow` | `name` or `dataset_id`, `tracking_uri` | `inputs`, `expectations`, `outputs`, `tags`, `record_id` |
| `unity_catalog` | `table` (`catalog.schema.table`), `version` (an int) or `as_of`, `storage_path`, `read_batch_size` | the columns |
| `harbor_tasks` | `path` (folder of Harbor tasks), `tasks` (names); no templates — see openarena-create-harbor-task | fixed |

Pitfalls:

- A relative `local` / `folder` path is read from the directory you run
  `openarena` in: run from the workspace root, or give an absolute path.
- `huggingface` without `split` reads every split in turn, so `limit` takes
  the first rows of the first split (often `train`). Unknown settings are
  passed on, so check cannot catch a misspelled one there.
- `mlflow` reads from `tracking_uri`, else `MLFLOW_SOURCE_TRACKING_URI`, else
  the tracing server (`MLFLOW_TRACKING_URI`). It has no version to pin.
- Braintrust needs `project` (or `project_id`) and `dataset_name`, MLflow
  `name` or `dataset_id`: `check` refuses the entry without them.
- Keys and endpoints go in the workspace's `.env`, never written out in
  `datasets.yaml` (the env vars per provider are in `docs/guides/datasets.md`).
  A value the entry itself must hold can reference one as `${NAME}`, replaced
  when the run reads its config.

The platform providers (`langfuse`, `langsmith`, `opik`, `phoenix`,
`braintrust`, `unity_catalog`) need their SDK: an extra of openarena named
after the provider (`unity-catalog` for Unity Catalog, `all` for every one),
e.g. `uv tool install "openarena[langfuse] @ git+https://github.com/Atena-IT/open-arena"`.
`mlflow` needs none. Without it the run says which extra to install; `check`
does not need it. Tell the user rather than installing packages yourself.

## What answers each row: one kind block at most

| block | answered by | the evaluation's `language_models` are | skill |
|---|---|---|---|
| none or `generator: {...}` | a model, one call (`instructions`, `chain_of_thought`) | `models` keys | — |
| `agent: {mcp_servers: [...]}` | a model calling MCP tools | `models` keys | openarena-add-mcp-tools |
| `a2a: {}` | remote A2A agents | `a2a_agents` keys | openarena-compare-a2a-agents |
| `embedding: {k: 3}` | retrieval with an embedding model | embedding `models` keys | below |
| `tools: {tool_schemas: [...]}` | one call with tool declarations, nothing runs | `models` keys | below |
| `harbor: {...}` | an agent the model drives in a sandbox | `models` keys | openarena-create-harbor-task, openarena-add-environment |

Retrieval: the input renders `query` and the `chunks` to search, the output
the gold `chunks`; `retrieval_recall` scores it. No schemas to write.

```yaml
datasets:
  passages:
    type: local
    config:
      path: retrieval.jsonl            # {"query": "...", "passages": [...], "gold": [...]}
      input_template: '{"query": {{ query | tojson }}, "chunks": {{ passages | tojson }}}'
      output_template: '{"chunks": {{ gold | tojson }}}'
    embedding:
      k: 3
    reward: recall                     # a rewards.yaml entry of type retrieval_recall (openarena-add-reward)
```

Tool calls: the output renders the gold `tool_calls` (`{name, arguments}`
each), with an `output_schema` for that target (it shapes only the target: the model's
answer is its chat message, tool calls included); `tool_calls_accuracy` scores it.

```yaml
datasets:
  weather-calls:
    type: local
    config:
      path: calls.jsonl                # {"question": "...", "calls": [{"name": "get_weather", "arguments": {"city": "Paris"}}]}
      input_template: '{"question": {{ question | tojson }}}'
      output_template: '{"tool_calls": {{ calls | tojson }}}'
      input_schema: {type: object, properties: {question: {type: string, description: The question to answer}}, required: [question]}
      output_schema: {type: object, properties: {tool_calls: {type: array, description: The gold calls}}, required: [tool_calls]}
    tools:
      instructions: Call the function that answers the question.
      tool_schemas:
        - type: function
          function:
            name: get_weather
            description: The weather in a city.
            parameters: {type: object, properties: {city: {type: string}}, required: [city]}
    reward: tool-calls                 # type: tool_calls_accuracy
```

## Extra metrics

`metrics:` adds columns to every cell of the dataset, beside `reward`. Each is
a catalog reward type, averaged over the rows (no extra call for a
deterministic one), or a plain metric (`accuracy`, `f1_score`, ...); a bare name or a mapping with
`class`, an `alias` (the column's name, never `reward`), `direction`
(optional, see below) and the type's settings:

```yaml
datasets:
  gsm8k-20:
    # ...
    reward: exact-match
    metrics:
      - answer_accuracy                          # partial credit beside the strict reward
      - {class: lm_as_judge, alias: judged, language_model: gpt-4o-mini}
      - cost                                     # USD spent on the model under test
      - avg_latency                              # seconds per model call
```

Cost, tokens and latency are **not** kept unless listed here (or under the
evaluation's `metrics`): `cost`, `reward_cost`, `embedding_cost`,
`input_tokens`, `output_tokens`, `total_tokens`, `avg_latency`. Without
`direction:`, a cost, tokens or latency measurement is better lower (`min`)
and anything else higher (`max`); write `direction:` only to override that.
See `docs/guides/evaluations.md`.

## Steps

1. Look at a few real rows (above).
2. Write the entry; start from `gsm8k-20` in the workspace's own
   `configs/datasets.yaml` and change what differs. Pin a version
   (`revision`, `version`, `as_of` tag, ...) for anything you will compare
   over time.
3. Make sure `reward:` names an entry of `rewards.yaml` (openarena-add-reward).
4. Run `openarena check configs` from the workspace root (it reads no rows),
   then a small first run with `limit: 5` (openarena-run-evaluation) before
   the full size. Keep the smoke and full entries under different keys: a
   resumed run reuses cells by dataset key.
