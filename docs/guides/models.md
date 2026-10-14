# Models

`configs/models.yaml` gives each model a **name** and says which litellm id it
calls. Everything else refers to a model by that name: an evaluation's
`language_models`, a judge reward's `language_model`, a leaderboard's `models`.
Two entries may call the same id with different params; they are two models.

```yaml
models:
  gpt-4o-mini:                       # the name everything refers to
    model: openai/gpt-4o-mini        # required: <provider>/<id>, as litellm routes it
    type: llm                        # llm (default) or embedding
    params:
      temperature: 0
    modalities: [text]               # what it reads: text, image, audio
    description: OpenAI's small model, the hosted baseline.
    tags: [openai, hosted]
  minilm:
    model: ollama/all-minilm
    type: embedding
    params:
      dimensions: 384
    tags: [local]
```

## Fields

| field | |
|---|---|
| `model` | required. The litellm id: a provider prefix and the provider's model id. |
| `type` | `llm` (the default) or `embedding`. Only `llm` models are evaluated; embedding models serve retrieval datasets and embedding-based rewards. |
| `params` | what the type takes, below. Leave a param out to keep the provider's default. |
| `modalities` | what the model reads: `text`, `image`, `audio`. |
| `description`, `tags` | for listings. |

### Params by type

| type | param | value |
|---|---|---|
| `llm` | `temperature` | a number, 0 to 2 |
| `llm` | `max_tokens` | an integer, at least 1 |
| `llm` | `reasoning_effort` | `disable`, `low`, `medium` or `high` |
| `embedding` | `dimensions` | an integer, at least 1, where the model lets you pick |

## Providers and keys

The prefix picks the provider. A hosted provider needs its key in the
environment, from your shell or the workspace's `.env`:

| prefix | example | needs |
|---|---|---|
| `openai/` | `openai/gpt-4o-mini` | `OPENAI_API_KEY` |
| `anthropic/` | `anthropic/claude-sonnet-4-5` | `ANTHROPIC_API_KEY` |
| `gemini/` | `gemini/gemini-2.5-flash` | `GEMINI_API_KEY` |
| `mistral/` | `mistral/mistral-small-latest` | `MISTRAL_API_KEY` |
| `ollama/` | `ollama/qwen3:8b` | a local Ollama, no key |
| `vllm/` | `vllm/meta-llama/Llama-3.1-8B-Instruct` | `HOSTED_VLLM_API_BASE` |

Any provider [litellm supports](https://docs.litellm.ai/docs/providers) works
the same way.

## Commands

```shell
openarena models list            # the models of ./configs
openarena check configs          # after an edit
```

Renaming a model renames it everywhere it is used: `check` lists every
reference it can no longer resolve. Results of earlier runs keep the old name.
