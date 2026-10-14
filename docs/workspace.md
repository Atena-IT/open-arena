# The workspace

A workspace is a folder: the registries under `configs/`, the Harbor folders
beside them, and the runs under `.open-arena/`. `openarena init` writes one
that runs as it is.

```shell
openarena init my-arena        # DIR defaults to the current folder
```

It refuses to overwrite a file that is already there, and writes nothing in
that case; `--force` overwrites.

## Layout

```
my-arena/
  configs/                        the registries, one YAML file each
    models.yaml                   name -> litellm id + params
    rewards.yaml                  name -> catalog reward + kwargs
    datasets.yaml                 name -> rows, templates, what answers, reward
    mcp_servers.yaml              name -> MCP tool server
    a2a_agents.yaml               name -> remote A2A agent
    environments.yaml             name -> Harbor sandbox
    evaluations.yaml              name -> models x datasets
    leaderboards.yaml             name -> how to rank runs
  tasks/hello-world/              a Harbor task folder
  environments/python/environment/   the `python` environment's build context
  .agents/skills/                 Agent Skills, one per task
  AGENTS.md                       the guide a coding agent reads first (CLAUDE.md links to it)
  README.md
  .env.template                   every key and endpoint a run reads
  .gitignore                      keeps .env and .open-arena/ out of git
  .open-arena/                    runs, created by the first one
```

## What the example runs

| evaluation | runs | needs |
|---|---|---|
| `smoke` | `qwen38` on `gsm8k-20`, 20 grade-school math questions, scored by exact match | a local Ollama |
| `harbor` | `gpt-4o-mini` driving the `terminus-2` agent on the `hello-world` task and on 5 GSM8K questions worked in a Python sandbox | `OPENAI_API_KEY`, and Docker or `E2B_API_KEY` |

And two leaderboards: `overall` (every model on every dataset, reward first)
and `math` (GSM8K, the reward alone).

## How every registry file is written

```yaml
# configs/<registry>.yaml
<registry>:              # the one top-level key, the file's own name
  <name>:                # an entry; others refer to it by this name
    <field>: <value>
```

- **Names** match `^[A-Za-z0-9][A-Za-z0-9_.-]*$`, at most 128 characters:
  `gpt-4o-mini`, `gsm8k-20`.
- **References are by name, never by id.** A dataset's `reward` is a
  `rewards` key; an evaluation's `language_models` are `models` keys; a judge's
  `language_model` is a `models` key. A litellm id such as
  `openai/gpt-4o-mini` appears once, in `models.yaml`.
- **Unknown fields are errors.** A misspelled field is reported with the
  rename that fixes it ([Checking configs](guides/check.md)).
- **`tags`** are lowercase labels matching `^[a-z0-9][a-z0-9:+./-]*$`:
  `[math, reasoning]`. **`description`** is free text.
- **An empty registry** is `<registry>: {}`.

Each file of a new workspace starts with a comment listing every field its
entries take, and gives the optional ones as commented examples.

## Keys and endpoints

`openarena evaluations run` reads the workspace's `.env` (beside `configs/`)
before anything else, and never overrides a variable your shell already set.

```shell
cp .env.template .env
```

| variable | for |
|---|---|
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `MISTRAL_API_KEY`, ... | hosted models, the one matching each `models.yaml` id |
| `OLLAMA_API_BASE` | `ollama/...` ids, when Ollama is not on `localhost:11434` |
| `HOSTED_VLLM_API_BASE` | `vllm/...` ids: your server's OpenAI endpoint, ending in `/v1` |
| `E2B_API_KEY`, `ARENA_HARBOR_SANDBOX` | Harbor trials: E2B with a key, else the local Docker; or another provider |
| `MLFLOW_TRACKING_URI` | tracing every run into MLflow |

## In the arena app

The layout is the arena's own: a workspace made by `init` is one the arena web
app opens as it is, forms and all.
