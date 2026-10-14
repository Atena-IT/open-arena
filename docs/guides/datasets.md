# Datasets

A dataset says where its rows come from (`type` and `config`), how each row
becomes a question and an expected answer (two Jinja2 templates), what answers
it, and which reward scores it.

```yaml
datasets:
  gsm8k-20:
    type: huggingface                  # the provider
    description: GSM8K test split, 20 rows.
    tags: [math]
    config:                            # the provider's settings, the templates, the schemas
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
    generator:                         # what answers: a model, in one call
      chain_of_thought: true
      instructions: Solve the problem. Reply with the final number only.
    reward: exact-match                # a rewards.yaml name; required to run
```

The entry's own fields are `type`, `config`, at most one block saying what
answers ([below](#what-answers-each-row)), `reward`, `metrics`, `description`
and `tags`. `reward` and the blocks never go inside `config`.

## Templates: from a row to JSON

Both templates are Jinja2 over one row's columns, and each must render to a
**JSON object**.

- `input_template` (required, except for [Harbor task folders](harbor-tasks.md))
  is what the model receives: `{"question": ...}` for question answering.
- `output_template` is the expected answer the reward compares with. A judge
  can score without one; `exact_match` and the other deterministic rewards
  cannot.

Pipe every value through `| tojson`: it quotes and escapes. Never write
`"{{ question }}"`, which breaks on the first quote in a question.

| you write | for |
|---|---|
| `{{ question \| tojson }}` | a column |
| `{{ choices.text \| tojson }}` | a field of a column |
| `{{ answers.text[0] \| tojson }}` | an item of a list |
| `{{ answer.split('####')[-1] \| trim \| tojson }}` | Python string methods, Jinja filters |
| `{{ "ABCD"[answer] \| tojson }}` | an index turned into a letter |

!!! warning "Quoting in YAML"
    Write a template as a single-quoted YAML string, and double every single
    quote inside it: `split(''####'')` in YAML is `split('####')` in Jinja.
    Double quotes need no escaping.

Look at real rows before writing a template. For a Hugging Face dataset:

```shell
python -c "import datasets; print(next(iter(datasets.load_dataset('openai/gsm8k', 'main', split='test', streaming=True))))"
```

### Multiple choice

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

### Schemas

`input_schema` and `output_schema` are the JSON Schemas of what the templates
render. With an `output_schema` the model answers in that structure: an `enum`
keeps "Option B" out where "B" is expected, and a field's `description` is the
instruction for that field. Keep `additionalProperties: false` and list
`required`.

**Give both whenever a template renders anything but chat messages**, which is
nearly always: without `input_schema` a row is read as a list of chat messages
(`{"messages": [...]}`), and without `output_schema` the expected answer as one
chat message (`{"role": ..., "content": ...}`), so `{"question": ...}` and
`{"answer": ...}` fail when the run reads the rows. A free-text answer still
gets a schema: one `string` field, no `enum`. Retrieval datasets are the
exception: their shapes are set for them.

## Providers

| `type` | `config` |
|---|---|
| `huggingface` | `path` (the repo), `name` (the subset), `split`, `revision`, `streaming` (default true) |
| `local` | `path` (a `.jsonl`, `.json`, `.csv` or `.parquet` file), `format` |
| `folder` | `path` (one record per file), `pattern` (a glob), `recursive`, `format` (`json`, `yaml`, `text`), `encoding` |
| `mlflow` | `name` or `dataset_id` (reads `MLFLOW_TRACKING_URI`) |
| `langfuse` | `dataset_name`, `version`, `streaming`, `page_size` |
| `langsmith` | `dataset_name`, `as_of` |
| `opik` | `dataset_name`, `version`, `filter_string` |
| `phoenix` | `dataset_name`, `version_id`, `splits` |
| `braintrust` | `project` or `project_id`, `dataset_name`, `version`, `org_name` |
| `unity_catalog` | `table` (`catalog.schema.table`), `version` or `as_of`, `storage_path` |
| `harbor_tasks` | `path` (a folder of Harbor tasks), `tasks` ([Harbor tasks](harbor-tasks.md)) |

Every provider also takes `limit`, `batch_size`, the templates and the schemas.
The [API reference](../api/datasets.md) documents each provider's settings.

The platform providers need their SDK, an extra of openarena named after the
provider: `langfuse`, `langsmith`, `opik`, `phoenix`, `braintrust`,
`unity-catalog`, or `all` for every one:

```shell
uv tool install "openarena[langfuse] @ git+https://github.com/Atena-IT/open-arena"
```

Without it, the provider says which extra to install when it reads its rows.

## What answers each row

At most one block. Without one, a model answers in one call.

| block | answered by | the evaluation's `language_models` are |
|---|---|---|
| none, or `generator:` | a model in one call; `instructions`, `chain_of_thought` | models |
| `agent:` | a model calling MCP tools ([MCP tools](mcp-tools.md)) | models |
| `a2a: {}` | remote agents ([A2A agents](a2a-agents.md)) | A2A agents |
| `embedding:` | retrieval with an embedding model ([below](#retrieval)) | embedding models |
| `tools:` | one call with tool declarations, nothing executed ([below](#tool-calls)) | models |
| `harbor:` | an agent the model drives in a sandbox ([Environments](environments.md), [Harbor tasks](harbor-tasks.md)) | models |

### Retrieval

The input renders a `query` and the `chunks` to search, the expected answer the
gold `chunks`. The embedding model under test ranks the chunks, the top `k`
are returned, and `retrieval_recall` scores them. No schemas to write: an embedding
dataset gets these shapes by default.

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

### Tool calls

The model gets the declared functions and answers with the calls it would
make; nothing is executed. The expected answer renders the gold `tool_calls`,
each `{name, arguments}`, with an `output_schema` saying so. That schema only shapes the
expected answer: the model's answer is its chat message, tool calls included.

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

`metrics:` adds columns beside `reward`. Each is a catalog reward, averaged
over the rows, or a plain metric (`accuracy`, `f1_score`, ...): a bare name, or
a mapping with `class`, an `alias` (the column's name, never `reward`), a
`direction` and the type's settings.

```yaml
datasets:
  gsm8k-20:
    # ...
    reward: exact-match
    metrics:
      - answer_accuracy
      - {class: lm_as_judge, alias: judged, language_model: gpt-4o-mini}
```

Cost, tokens and latency are measured without asking.

## Tips

- Start with `limit: 5`, run, read the answers, then raise it.
- `batch_size` is how many requests are in flight at once: keep it low for a
  local model or a rate-limited key.
- `openarena check configs` after every edit ([Checking configs](check.md)).
