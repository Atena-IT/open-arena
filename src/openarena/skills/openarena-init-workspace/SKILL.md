---
name: openarena-init-workspace
description: Use when an evaluation project has to be started from nothing with the openarena engine, or when you need to know how an openarena workspace is laid out and how its YAML files are written in general. Scaffolds a working workspace (configs/, tasks/, environments/) with `openarena init`, explains the rules every registry file follows, and the edit → check → run loop the other openarena skills build on.
---

# openarena-init-workspace

`openarena` is the arena's evaluation engine as a command: it reads YAML
registries, sweeps every model of an evaluation over every dataset of it, and
ranks the results on leaderboards. No server, no web app: files in, files out.

## Running the command

Installed as a tool, `openarena` is on the PATH:

```bash
uv tool install "git+https://github.com/Atena-IT/open-arena"
openarena --help
```

From a clone of open-arena instead, `uv sync` once, then prefix every command
with `uv run` (`uv run openarena --help`). The commands below are written
`uv run openarena ...`; drop the `uv run` for the tool. Paths you pass are
relative to where you run it.

## Make a workspace

```bash
uv run openarena init my-arena          # DIR defaults to the current folder
uv run openarena check my-arena/configs
```

`init` refuses to overwrite a file that is already there and writes nothing in
that case; `--force` overwrites. What it writes:

```
my-arena/
  configs/                      one registry per file
    models.yaml                 name -> litellm id + params           (openarena-add-model)
    rewards.yaml                name -> catalog reward + kwargs       (openarena-add-reward)
    datasets.yaml               name -> rows, templates, kind, reward (openarena-add-dataset)
    mcp_servers.yaml            name -> MCP tool server connection    (openarena-add-mcp-tools)
    a2a_agents.yaml             name -> remote A2A agent              (openarena-compare-a2a-agents)
    environments.yaml           name -> Harbor sandbox                (openarena-add-environment)
    evaluations.yaml            name -> models x datasets sweep       (openarena-run-evaluation)
    leaderboards.yaml           name -> how to rank results           (openarena-rank-leaderboard)
  tasks/hello-world/            a Harbor task folder                  (openarena-create-harbor-task)
  environments/python/environment/   the `python` environment's build context
  .agents/skills/               these skills, the workspace's own copy
  AGENTS.md                     the guide a coding agent reads first; CLAUDE.md links to it
  README.md                     the quickstart, for people
  .env.template                 every key and endpoint a run reads; copy to .env
  .gitignore                    keeps .env and .open-arena/ out of git
```

Each file starts with a comment listing every field its entries take; what is
optional is there as commented examples. Read the header before writing an
entry. As written the workspace runs: `smoke` (a local Ollama model on 20
GSM8K rows) and `harbor` (Harbor tasks and terminal rows, `gpt-4o-mini`).

## How every registry file is written

```yaml
# configs/<registry>.yaml
<registry>:              # the one top-level key: the file's own name (models, datasets, ...)
  <name>:                # an entry; other files refer to it by this name
    <field>: <value>
```

- **Names** match `^[A-Za-z0-9][A-Za-z0-9_.-]*$`, at most 128 characters
  (`gpt-4o-mini`, `gsm8k-20`). They are what everything else references.
- **References are by name, never by id.** A dataset's `reward:` is a
  `rewards` key, an evaluation's `language_models` are `models` keys, a judge's
  `language_model` is a `models` key, a `harbor.environment` is an
  `environments` key. The litellm id (`openai/gpt-4o-mini`) appears once, in
  `models.yaml`.
- **Unknown fields are errors.** Every entry refuses a field it does not know;
  a misspelling is reported with the rename that fixes it.
- **`tags`** are lowercase labels matching `^[a-z0-9][a-z0-9:+./-]*$`, unique
  per entry: `[math, reasoning]`.
- **`description`** is free text, shown in listings.
- An empty registry is `<registry>: {}`, never a bare `<registry>:`.
- Jinja templates are single-quoted YAML strings; a single quote inside one
  is doubled (`''####''`). See openarena-add-dataset.
- A dataset's templates come with their `input_schema` / `output_schema`:
  without them a row is read as chat messages, and `{"question": ...}` fails
  when the run reads it. See openarena-add-dataset.

## The loop

1. Read the file's header comment and the entries already there:
   `uv run openarena <registry> list --configs my-arena/configs`
   (`models`, `rewards`, `datasets`, `mcp-servers`, `a2a-agents`,
   `environments`, `evaluations`, `leaderboards`).
2. Edit the YAML.
3. `uv run openarena check my-arena/configs` until it says no problems
   (openarena-check-configs).
4. Run (openarena-run-evaluation), then rank (openarena-rank-leaderboard).

Never invent a field, a reward type or a provider: the lists in these skills
come from the engine, and `check` names the valid choices when you are wrong.

## Keys and services

`evaluations run` reads the workspace's `.env` (beside `configs/`) before
anything else, never over a variable the shell already set:
`cp .env.template .env`, then uncomment what the models need. Hosted models need their provider's key in the environment
(`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, ...); `ollama/...`
ids need a local Ollama; `vllm/...` ids need `HOSTED_VLLM_API_BASE`. Harbor
datasets run trials on E2B with `E2B_API_KEY`, else on the local Docker
(`ARENA_HARBOR_SANDBOX` picks another provider).
`MLFLOW_TRACKING_URI`, when set, traces every run into MLflow. Do not start
or stop servers yourself: if a model or service does not answer, say so.

## The same workspace in the arena app

The layout is the app's own: a folder made by `init` is a workspace the arena
web app opens as it is.
