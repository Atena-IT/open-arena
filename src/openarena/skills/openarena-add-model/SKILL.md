---
name: openarena-add-model
description: Use when a language model or an embedding model has to be added to, changed in, or chosen from an openarena workspace (models.yaml) — writing the entry with its litellm id, type and params, picking the provider prefix and the API key it needs, and referencing it from evaluations, judges and leaderboards by its name.
---

# openarena-add-model

`configs/models.yaml` maps a **name** to the litellm id it calls. Everything
else refers to models by that name: an evaluation's `language_models`, a
judge reward's `language_model`, a leaderboard's `models`. Two entries may
call the same id with different params: they are two models.

## The entry

```yaml
models:
  gpt-4o-mini:                       # the name everything refers to
    model: openai/gpt-4o-mini        # required: <provider>/<id>, as litellm routes it
    type: llm                        # llm (default) or embedding
    params:                          # optional; what the type takes, below
      temperature: 0
    modalities: [text]               # optional: text, image, audio — what it reads
    description: OpenAI's small model, the hosted baseline.
    tags: [openai, hosted]
  minilm:
    model: ollama/all-minilm
    type: embedding
    params:
      dimensions: 384
    tags: [local]
```

### `params` by type

| type | param | value |
|---|---|---|
| `llm` | `temperature` | number, 0 to 2 |
| `llm` | `max_tokens` | integer, at least 1 |
| `llm` | `reasoning_effort` | `disable`, `low`, `medium` or `high` |
| `embedding` | `dimensions` | integer, at least 1, where the model lets you pick |

Leave a param out to keep the provider's default.

### The `model` id and its key

The prefix picks the provider; a hosted one needs its key in the environment
of the process that runs `openarena`:

| prefix | example | needs |
|---|---|---|
| `openai/` | `openai/gpt-4o-mini` | `OPENAI_API_KEY` |
| `anthropic/` | `anthropic/claude-sonnet-4-5` | `ANTHROPIC_API_KEY` |
| `gemini/` | `gemini/gemini-2.5-flash` | `GEMINI_API_KEY` |
| `mistral/` | `mistral/mistral-small-latest` | `MISTRAL_API_KEY` |
| `ollama/` | `ollama/qwen3:8b` | a local Ollama on :11434, no key |
| `vllm/` | `vllm/meta-llama/Llama-3.1-8B-Instruct` | `HOSTED_VLLM_API_BASE` |

Any provider litellm supports works the same way. Do not guess an id: use
one the user gave, one the provider documents, or one already in the file.
Only `llm` models can be evaluated; `embedding` models serve retrieval
datasets (`embedding:` block) and embedding-based rewards
(`cosine_similarity`'s `embedding_model`).

## Steps

1. `uv run openarena models list --configs my-arena/configs` — what exists.
2. Add the entry under `models:`; pick a short, stable name.
3. Reference it where needed: `language_models: [gpt-4o-mini]` in
   `evaluations.yaml`, `language_model: gpt-4o-mini` in a judge reward.
4. `uv run openarena check my-arena/configs`.

Renaming a model means renaming every reference to it; `check` lists each
one it can no longer resolve. Results of earlier runs keep the old name.
