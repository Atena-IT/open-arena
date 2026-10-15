---
name: openarena-init-workspace
description: Use when an evaluation project has to be started from nothing with the openarena engine, or when you need to know how an openarena workspace is laid out and how its YAML files are written in general. Scaffolds a working workspace (configs/, tasks/, environments/) with `openarena init`, explains the rules every registry file follows, where the command looks for each file, and the edit → check → run loop the other openarena skills build on.
---

# openarena-init-workspace

`openarena` is the arena's evaluation engine as a command: it reads YAML
registries, sweeps every model of an evaluation over every dataset of it, and
ranks the results on leaderboards. No server, no web app: files in, files out.
Depth: `docs/workspace.md` (the workspace), `docs/index.md` (quickstart),
`docs/cli.md` (every option).

## Running the command

Installed as a tool, `openarena` is on the PATH:

```bash
uv tool install "git+https://github.com/Atena-IT/open-arena"
openarena --help
```

From a clone of open-arena instead, `uv sync` once, then activate its `.venv`
(`source .venv/bin/activate`), or prefix each command with
`uv run --project <path-to-open-arena> openarena ...`, which keeps the folder
you are in. Run every command **from the workspace root** (see below).

## Make a workspace

```bash
openarena init my-arena          # DIR defaults to the current folder; created if missing
cd my-arena
openarena check configs
```

`init` checks every target file first: if any one already exists it writes
nothing and names every file in the way. `--force` overwrites them all,
`configs/*.yaml` and `AGENTS.md` included, so never use it on an edited
workspace just to refresh the skills: run `init` into an empty scratch folder
and copy `.agents/skills/` across instead. `--template default` is the only
template. What it writes:

```
my-arena/
  configs/                      one registry per file
    models.yaml                 name -> litellm id, type, params      (openarena-add-model)
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
  AGENTS.md                     the guide a coding agent reads first; CLAUDE.md is a link to it
  README.md                     the quickstart, for people
  .env.template                 the keys and endpoints a run may read; copy to .env
  .gitignore                    keeps .env and .open-arena/ out of git
```

`init` makes every `.sh` it writes executable; a script you add needs
`chmod +x`. Each config file starts with a comment listing every field its
entries take; what is optional is there as commented examples. Read the
header before writing an entry. As written the workspace runs: `smoke`
(`qwen38`, a local Ollama model, on 20 GSM8K rows) and `harbor`
(`gpt-4o-mini` on the `hello-world` task and 5 GSM8K rows in a Python
sandbox), ranked by the leaderboards `overall` and `math`.

## How every registry file is written

```yaml
# configs/<registry>.yaml
<registry>:              # the one top-level key: the file's own name (models, datasets, ...)
  <name>:                # an entry; other files refer to it by this name
    <field>: <value>
    # ...
```

- **Only an exact `<registry>.yaml` is read**, under its one top-level key.
  `model.yaml`, `models.yml` or another registry's key in a file are not read
  by a run; `check` reports them with the name to use. A missing file is an
  empty registry; write an empty one as `<registry>: {}`.
- **Names** match `^[A-Za-z0-9][A-Za-z0-9_.-]*$`, at most 128 characters
  (`gpt-4o-mini`, `gsm8k-20`). They are what everything else references.
- **References are by name, never by id.** A dataset's `reward:` is a
  `rewards` key, an evaluation's `language_models` are `models` keys, a judge's
  `language_model` is a `models` key, a `harbor.environment` is an
  `environments` key. The litellm id (`openai/gpt-4o-mini`) appears once, in
  `models.yaml`.
- **Unknown fields are errors**, reported with the rename that fixes it. **A
  key written twice** in one mapping is an error too, with its line.
- **`tags`** are lowercase labels matching `^[a-z0-9][a-z0-9:+./-]*$`, unique
  per entry: `[math, reasoning]`. **`description`** is free text, shown in
  listings.
- **No keys in YAML**: provider keys come from the environment (below). A
  secret an entry must hold (an A2A or MCP `headers` token, an MCP server's
  or an environment's `env`) goes in `.env` and is referenced as `${NAME}`:
  any string value may hold one, replaced when a run reads its config; a run
  refuses to start when it is set nowhere, and `$${NAME}` writes a literal
  `${NAME}`.
- Jinja templates are single-quoted YAML strings; a single quote inside one
  is doubled (`''####''`). A dataset's templates come with their
  `input_schema` / `output_schema`: without them a row is read as chat
  messages, and `{"question": ...}` fails when the run reads it. See
  openarena-add-dataset.

## Where the command looks

No search up the folder tree, no global config: every path comes from the
folder you run in and two options.

| what | looked for in | change with |
|---|---|---|
| registries | `./configs` | `--configs DIR` |
| `.env` | the parent of the configs folder | export the variables instead |
| `tasks/`, `environments/` | the parent of the configs folder | an absolute `harbor_tasks` `path` |
| `local` / `folder` data, relative `path` | the folder you run in | an absolute `path` |
| runs | `./.open-arena` | `--state-dir DIR` |

Run from elsewhere and a run stops with `no configs folder at configs`. Keep
data files in `data/` at the workspace root so relative paths work for
everyone.

## The loop

1. Read the file's header comment and the entries already there:
   `openarena <registry> list` (`models`, `rewards`, `datasets`,
   `mcp-servers`, `a2a-agents`, `environments`, `evaluations`,
   `leaderboards`).
2. Edit the YAML.
3. `openarena check configs` until every file says no problems (it exits 1
   on an error; `openarena check configs/models.yaml` checks one file against
   its neighbours). See openarena-check-configs.
4. `openarena evaluations run <name>` (openarena-run-evaluation), then
   `openarena leaderboards run <name>` (openarena-rank-leaderboard); read the
   scores with openarena-read-results.

Never invent a field, a reward type or a provider: the lists in these skills
come from the engine, and `check` names the valid choices when you are wrong.

## Runs: .open-arena/

`openarena evaluations run NAME` writes `.open-arena/NAME/`: `config.yaml`
(what ran), `result.json` (the scores, the run's record), per-dataset trial
state and, for Harbor, `harbor/` plus copies of `tasks/` and
`environments/`. Running NAME again resumes it; `--no-cache` starts over. The
new `result.json` replaces the old one. `leaderboards run` reads every
`*/result.json` under the state dir, newest first per cell. An evaluation's
`default_language_model` / `default_embedding_model` is also saved to a
settings file in the home folder, and a later run that sets none can inherit
it: give rewards and metrics their `language_model` explicitly.

## Keys and services

Only `evaluations run` loads `.env`, from beside `configs/`, and a variable
already set in the shell always wins over it: `cp .env.template .env`, then
uncomment what the models need (each variable there is commented with what
uses it). `check` reads `.env` only to warn about a `${NAME}` set neither
there nor in the shell; `check`, `list` and `leaderboards run` need no keys.
Hosted models need their provider's key (`OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, ...); `ollama/...` ids need a local
Ollama (`OLLAMA_API_BASE` if it is not on `localhost:11434`); `vllm/...` ids
need `HOSTED_VLLM_API_BASE` (ending in `/v1`). Harbor datasets run trials on
E2B with `E2B_API_KEY`, else on the local Docker (`ARENA_HARBOR_SANDBOX`
picks another provider). `openarena evaluations run` does not trace into
MLflow, even with `MLFLOW_TRACKING_URI` set: only the arena app traces. Do
not start or stop servers yourself: if a model or service does not answer,
say so.

## The same workspace in the arena app

The layout is the app's own: a folder made by `init` is a workspace the arena
web app opens as it is.
