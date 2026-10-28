---
name: openarena-add-model
description: Use when a language model, an embedding model or a decision model has to be added to, changed in, or chosen from an openarena workspace (models.yaml) — writing the entry with its litellm id, type and params, picking the provider prefix and the environment variables it needs, and referencing it from evaluations, judges and leaderboards by its name.
---

# openarena-add-model

`configs/models.yaml` maps a **name** to the litellm id it calls. Everything
else refers to models by that name, never by id: an evaluation's
`language_models`, a reward's `language_model` / `embedding_model`, a
leaderboard's `models`. Results are stored under the name. Two entries may
call the same id with different params: they are two models, each with its
own row. Depth: `docs/guides/models.md`.

## The entry

```yaml
models:
  gpt-4o-mini:                       # the name everything refers to
    model: openai/gpt-4o-mini        # required: <provider>/<id>, as litellm routes it
    type: llm                        # llm (default), embedding or decision
    params:                          # optional; only what the type takes, below
      temperature: 0
      max_tokens: 2048
    modalities: [text]               # optional, default [text]; a label only, the run never reads it
    description: OpenAI's small model, the hosted baseline.
    tags: [openai, hosted]           # lower-case labels; never sent to the model
  qwen3-local:
    model: ollama/qwen3:8b
    params:
      temperature: 0
      reasoning_effort: disable
    tags: [local]
  minilm:
    model: ollama/all-minilm
    type: embedding
    tags: [local]
  jev:
    model: typesafe/jev-latest       # needs TYPESAFE_API_KEY
    type: decision                   # grades typed questions; a rubric judge's `decision_model`
    tags: [judge]
```

No other field is accepted (`provider:`, or a param beside `model:`, is an
error). The name: a letter or digit, then letters, digits, `_`, `.`, `-`.

### `params` by type

| type | param | value |
|---|---|---|
| `llm` | `temperature` | number, 0 to 2 |
| `llm` | `max_tokens` | integer, at least 1 |
| `llm` | `reasoning_effort` | `disable`, `low`, `medium` or `high` |
| `embedding` | `dimensions` | integer, at least 1 |
| `decision` | none | leave `params` out |

Nothing else: `api_base`, `top_p`, `seed`, `stop` are refused. Leave a param
out to keep the provider's default. Where params actually apply:

- Model under test on a generator, agent or tools dataset: sent with every call.
- Model under test on a Harbor dataset: passed to the agent where it takes them.
- A judge, a reward's or metric's model, a `default_*_model`: built with its
  entry's params too (an embedding model's `dimensions` included). Give a
  judge's entry `temperature: 0` so its scores repeat as far as the provider
  allows.
- An embedding model under test (retrieval dataset): built with its
  `dimensions`.

For a setting ablation, add two entries for the same id (`o4-mini-low`,
`o4-mini-high`) rather than editing one.

### The `model` id and its environment

The prefix picks the provider. Keys and server addresses come from the
environment, never from `models.yaml`:

| prefix | example | needs |
|---|---|---|
| `openai/` | `openai/gpt-4o-mini` | `OPENAI_API_KEY` |
| `anthropic/` | `anthropic/claude-sonnet-4-5` | `ANTHROPIC_API_KEY` |
| `gemini/` | `gemini/gemini-2.5-flash` | `GEMINI_API_KEY` |
| `mistral/` | `mistral/mistral-small-latest` | `MISTRAL_API_KEY` |
| `groq/` | `groq/llama-3.3-70b-versatile` | `GROQ_API_KEY` |
| `openrouter/` | `openrouter/mistralai/ministral-14b-2512` | `OPENROUTER_API_KEY` |
| `ollama/` | `ollama/qwen3:8b` | a running Ollama; `OLLAMA_API_BASE` if not `http://localhost:11434` |
| `vllm/` | `vllm/Qwen/Qwen3-4B` | `HOSTED_VLLM_API_BASE`, the server's endpoint with `/v1` |
| `sglang/` | `sglang/Qwen/Qwen3-4B` | `SGLANG_API_BASE`, the server's endpoint with `/v1` (default `http://localhost:30000/v1`) |

Any provider litellm supports works the same way. Do not guess an id: use one
the user gave, one the provider documents, or one already in the file. A base
URL is per provider, so every `vllm/` entry hits the one server
`HOSTED_VLLM_API_BASE` names, and every `sglang/` entry the one
`SGLANG_API_BASE` names.

Put variables in the workspace's `.env` (beside `configs/`; copy
`.env.template`). Only `openarena evaluations run` loads it, and it never
overrides a variable already set in the shell. A model entry has no field for
a key or a base URL (`api_key:`, `api_base:` are refused): the provider reads
its own variable.

### Type, cost, end of life

- `llm` models go in a sweep's `language_models` and in judge settings
  (`language_model`, `smart_language_model`, `sub_language_model`,
  `panel_language_models`, `default_language_model`).
- `embedding` models go in `embedding_model`, `default_embedding_model`, and in
  the `language_models` of a sweep over retrieval (`embedding:`) datasets only;
  such a sweep cannot mix in other dataset kinds.
- `decision` models go in a rubric judge's `decision_model` only
  (`rubrics_as_judge` and the ready-made rubric judges): they grade typed
  questions with calibrated probabilities, write nothing, and cannot be under
  test. `typesafe/...` ids need `TYPESAFE_API_KEY`; `ollama/laya` and
  `sglang/kev` (a decision checkpoint SGLang serves) run locally.
- Local ids (`ollama/`, `vllm/`, `sglang/`, ...) cost 0. A hosted id no price source knows
  still runs but also shows a cost of 0: check its cost after the first run.
- Price and end-of-life date are per id; entries sharing an id share them.

## Steps

1. `openarena models list --configs <workspace>/configs` — what exists.
2. Add the entry under `models:`; pick a short, stable name that says what is
   compared (`o4-mini-high`), not just the vendor id.
3. Reference it by name where needed, e.g. in `evaluations.yaml`:
   ```yaml
   evaluations:
     # ...
     my-sweep:
       language_models: [gpt-4o-mini, qwen3-local]
   ```
   or `language_model: <name>` in a judge reward's `config` (see
   `openarena-add-reward`).
4. `openarena check <workspace>/configs`. It reads files only: a wrong id after
   the prefix, a missing key, an unpulled Ollama model or a down server fail
   the cell at run time.
5. Make sure the needed variable is in the shell or `.env`.

Renaming a model splits its leaderboard history: past results keep the old
name, new runs get the new one. `check` lists every config reference the
rename broke (an error, a warning on a leaderboard); it cannot rename results.
Prefer adding a new entry over renaming one that already has results.
