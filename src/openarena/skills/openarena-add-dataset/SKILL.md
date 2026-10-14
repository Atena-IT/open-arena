---
name: openarena-add-dataset
description: Use when a benchmark has to become an openarena dataset (datasets.yaml) — from Hugging Face, a local JSONL/CSV/Parquet file, a folder of files, or an LLM-observability platform — writing its provider config, the Jinja2 input and output templates over the rows, the structured output schemas, the block that says what answers each row (a model, MCP tools, A2A agents, embeddings, tool calls, a Harbor agent), and its reward.
---

# openarena-add-dataset

A dataset entry says where rows come from (`type` + `config`), how each row
becomes a question and an expected answer (two Jinja2 templates in `config`),
what answers it (at most one kind block), and which reward scores it.

## The entry, field by field

```yaml
datasets:
  gsm8k-20:
    type: huggingface                  # the provider, below
    description: GSM8K test split, 20 rows.
    tags: [math]
    config:                            # the provider's kwargs + templates + schemas
      path: openai/gsm8k
      name: main
      split: test
      limit: 20                        # rows used; leave out for all
      batch_size: 4                    # rows sent to the model at once
      input_template: '{"question": {{ question | tojson }}}'
      output_template: '{"answer": {{ answer.split(''####'')[-1] | replace('','', '''') | trim | tojson }}}'
      input_schema:
        type: object
        properties:
          question: {type: string, description: The question to answer}
        required: [question]
        additionalProperties: false
      output_schema:
        type: object
        properties:
          answer: {type: string, description: "The final number only, no units or commas"}
        required: [answer]
        additionalProperties: false
    generator:                         # the kind block: a model answers, in one call
      chain_of_thought: true
      instructions: Solve the problem. Reply with the final number only.
    reward: exact-match                # a rewards.yaml key; required to run
```

`type`, `config`, the kind block, `reward`, `metrics`, `description` and
`tags` are the entry's fields. `reward` and the kind block never go inside
`config` (check refuses it).

## The templates: rows → JSON

Both templates are Jinja2 over one row's columns, and each must render to
**JSON** (an object). Always pipe values through `tojson`, which quotes and
escapes them; never write `"{{ question }}"`.

- `input_template` (required, except `harbor_tasks`): what the program
  receives. `{"question": ...}` for a question-answering dataset.
- `output_template`: the reference the reward compares with. Optional when a
  judge scores without one; required for `exact_match` and the other
  deterministic rewards.
- Columns: `{{ question }}`, nested `{{ choices.text }}`, an item
  `{{ answers.text[0] }}`. Filters: `| tojson`, `| trim`, `| lower`,
  `| replace('a', 'b')`, `| join(', ')`, Python string methods
  (`answer.split('####')[-1]`).
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
        additionalProperties: false
    generator:
      instructions: Answer with the letter of the correct choice only.
    reward: exact-match
```

Look at real rows before writing a template; never guess column names:

```bash
uv run python -c "import datasets; print(next(iter(datasets.load_dataset('openai/gsm8k', 'main', split='test', streaming=True))))"
head -2 data/rows.jsonl
```

### Schemas

`input_schema` / `output_schema` are JSON Schemas of what the templates
render; with `output_schema` the model answers in that structure (an `enum`
stops "Option B" where "B" is expected, a field `description` is the
instruction for that field). Keep `additionalProperties: false` and list
`required`.

**Give both whenever a template renders anything but chat messages**, which is
nearly always: without `input_schema` a row is read as a list of chat messages
(`{"messages": [...]}`), and without `output_schema` the expected answer as one
chat message (`{"role": ..., "content": ...}`), so `{"question": ...}` and
`{"answer": ...}` fail when the run reads the rows. A free-text answer still
gets a schema: one `string` field, no `enum`. Retrieval datasets are the
exception: their shapes are set for them.

## Providers (`type`) and their `config`

| type | config |
|---|---|
| `huggingface` | `path` (repo), `name` (subset), `split`, `revision`, `streaming` (default true) |
| `local` | `path` (a .jsonl/.json/.csv/.parquet file, relative to where you run), `format` |
| `folder` | `path` (one record per file), `pattern` (glob), `recursive`, `format` (json/yaml/text), `encoding` |
| `mlflow` | `name` or `dataset_id` (MLFLOW_TRACKING_URI) |
| `langfuse` | `dataset_name`, `version`, `streaming`, `page_size` |
| `langsmith` | `dataset_name`, `as_of` |
| `opik` | `dataset_name`, `version`, `filter_string` |
| `phoenix` | `dataset_name`, `version_id`, `splits` |
| `braintrust` | `project` or `project_id`, `dataset_name`, `version`, `org_name` |
| `unity_catalog` | `table` (`catalog.schema.table`), `version` or `as_of`, `storage_path` |
| `harbor_tasks` | `path` (folder of Harbor tasks), `tasks` (names) — see openarena-create-harbor-task |

Every provider also takes `limit`, `batch_size`, `input_template`,
`output_template`, `input_schema`, `output_schema`.

The platform providers (`langfuse`, `langsmith`, `opik`, `phoenix`,
`braintrust`, `unity_catalog`) need their SDK: an extra of openarena named
after the provider (`unity-catalog` for Unity Catalog, `all` for every one),
e.g. `uv tool install "openarena[langfuse] @ git+https://github.com/Atena-IT/open-arena"`.
Without it the run says which extra to install; tell the user rather than
installing packages yourself.

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
the gold `chunks`; `retrieval_recall` scores it.

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
    reward: recall                     # type: retrieval_recall
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
`class`, an `alias` (the column's name, never `reward`), `direction` and the
type's kwargs:

```yaml
datasets:
  gsm8k-20:
    # ...
    reward: exact-match
    metrics:
      - answer_accuracy                          # partial credit beside the strict reward
      - {class: lm_as_judge, alias: judged, language_model: gpt-4o-mini}
```

Cost, tokens (`input_tokens`, `output_tokens`) and latency are measured
without asking.

## Steps

1. Look at a few real rows (above).
2. Write the entry; start from `gsm8k-20` in the workspace's own
   `datasets.yaml` and change what differs.
3. Make sure `reward:` names an entry of `rewards.yaml` (openarena-add-reward).
4. `uv run openarena check my-arena/configs`; then a small first run with
   `limit: 5` (openarena-run-evaluation) before the full size.
