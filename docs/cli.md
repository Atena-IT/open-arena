# CLI reference

Everything `openarena` does from a terminal: start a workspace, check it, list
what it holds, run an evaluation, rank the results. This page lists every
command, every option and every exit code. The guides explain what goes in the
YAML; this page explains how to drive it.

```shell
openarena --help                      # the command groups
openarena evaluations run --help      # one command's options
```

The command is installed with the package (`openarena` is its entry point).
From a checkout of the library, prefix it with `uv run`:
`uv run openarena check configs`.

## At a glance

| command | what it does | reads | writes |
|---|---|---|---|
| [`openarena init`](#openarena-init) | writes a new workspace | the packaged template | `DEST/` |
| [`openarena check`](#openarena-check) | validates registry files | `configs/` | nothing |
| [`openarena <registry> list`](#registry-listings) | prints one registry as a table | `configs/` | nothing |
| [`openarena evaluations run`](#openarena-evaluations-run) | runs a saved evaluation | `configs/`, `.env`, `tasks/`, `environments/` | `.open-arena/<name>/` |
| [`openarena leaderboards run`](#openarena-leaderboards-run) | ranks models from finished runs | `configs/`, `.open-arena/*/result.json` | nothing, unless `--json` |

A typical session:

```shell
openarena init my-arena && cd my-arena
cp .env.template .env                 # the keys your models need
openarena check configs               # fix what it reports
openarena evaluations run smoke -v    # a first, small run
openarena leaderboards run overall    # rank what ran
```

## Paths, global options and exit codes

### Run it from the workspace folder

Every command resolves its paths against the folder you run it from. Two
defaults matter:

| option | default | used by | what it points at |
|---|---|---|---|
| `--configs DIR` | `configs` | every `list`, `evaluations run`, `leaderboards run` | the folder holding `models.yaml`, `datasets.yaml`, ... ([The workspace](workspace.md)) |
| `--state-dir DIR` | `.open-arena` | `evaluations run`, `leaderboards run` | where runs are kept, one folder per evaluation |

From the workspace folder neither is needed. From anywhere else, pass both,
and pass the same `--state-dir` to every command that should see the same runs.

!!! warning "The two paths resolve differently"
    `--state-dir` is relative to the folder you run from. The files around the
    configs are not: `evaluations run` reads `.env`, `tasks/` and
    `environments/` from the **parent of `--configs`**. Run
    `openarena evaluations run smoke --configs my-arena/configs` from outside
    `my-arena`, and the keys and Harbor tasks come from `my-arena/`, but the run
    lands in `./.open-arena/smoke/`, not in `my-arena/.open-arena/`. A later
    `openarena leaderboards run` from inside `my-arena` will not find it.

### Global options

The top-level command takes only `--help`. There is no global `--configs`, no
`--version` and no config file of its own: each option belongs to the command
that uses it, and goes after the command (`openarena evaluations run smoke
--configs DIR`, not `openarena --configs DIR evaluations run smoke`).

### Exit codes

| code | when |
|---|---|
| `0` | the command did what it was asked. For `check`: no error (warnings are allowed). For `evaluations run`: the sweep finished, **even if some cells failed** |
| `1` | an error the command reports as `Error: ...`: no configs folder, an unknown evaluation or leaderboard name, files in the way of `init`, no finished runs to rank, an invalid evaluation config. Also `check` finding at least one error, a crash inside a run (a traceback), and an interrupt (Ctrl-C prints `Aborted!`) |
| `2` | a usage error, found before anything runs: an unknown command or option, a missing `NAME`, `--batch-size 0`, an unknown `--template` |

!!! warning "A finished run is not a successful run"
    `openarena evaluations run` exits `0` when the sweep completes, whatever
    its cells scored. A cell that hit a hard error is printed as
    `<model> failed: <why>` and recorded in `result.json` with `value: null`
    and a status other than `COMPLETED`. A missing key or an unreachable model
    is quieter: the cell completes with a reward of 0. In CI, check the
    statuses and the rewards yourself ([Reading results](guides/results.md)).

### Environment variables

The CLI has no environment variables of its own. `openarena evaluations run`
loads the workspace's `.env` (beside `configs/`) before anything else, and
never overrides a variable your shell already set; when it finds one it prints
`Read <path>/.env`. No other command reads `.env`.

What a run reads from the environment, through that file or your shell:

| variable | for |
|---|---|
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, ... | the hosted models your `models.yaml` ids call ([Models](guides/models.md)) |
| `OLLAMA_API_BASE`, `HOSTED_VLLM_API_BASE` | local model servers not on their default address |
| `E2B_API_KEY`, `ARENA_HARBOR_SANDBOX`, `ARENA_HARBOR_WORKDIR` | Harbor trials: the sandbox provider, and where task folders and trials go ([Environments](guides/environments.md)) |
| `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT_NAME`, `ARENA_MLFLOW_TRACING` | MLflow tracing, for the arena app and `run_sweep` called from Python; `openarena evaluations run` does not switch tracing on |

The full list, with what each one does, is in
[The workspace](workspace.md#env-keys-and-endpoints).

!!! tip "In CI, prefer the shell"
    Set the keys as CI secrets in the environment. They win over `.env`, so a
    `.env` left in the folder cannot override them.

## Workspace commands

### `openarena init`

Writes a ready-to-use workspace: every registry file with each field explained
in comments, an example Harbor task, an environment's build context, and what a
coding agent needs to work in the folder.

```shell
openarena init [DEST] [--template NAME] [--force]
```

| argument / option | type | default | effect |
|---|---|---|---|
| `DEST` | folder | `.` (here) | where to write; created if missing |
| `--template NAME` | choice | `default` | the template to copy. `default` is the only one shipped today |
| `--force` | flag | off | overwrite files already there |

It writes:

```
DEST/
  configs/          one registry per file: models, rewards, datasets, mcp_servers,
                    a2a_agents, environments, evaluations, leaderboards
  tasks/            Harbor task folders (task.toml, instruction.md, environment/,
                    tests/test.sh, solution/solve.sh)
  environments/     an environment's build context
  .agents/skills/   the openarena Agent Skills, one per action
  AGENTS.md         the guide a coding agent reads first; CLAUDE.md links to it
  README.md  .env.template  .gitignore
```

Then it prints a table of the files and the entries each registry holds, and
the next steps. `.sh` files are made executable, since Harbor runs them as
they are. `CLAUDE.md` is a symbolic link to `AGENTS.md`; where links cannot be
made, it is a copy.

**Without `--force`, `init` is all or nothing.** If any file it would write
already exists, it writes nothing, names every file in the way, and exits 1.

**When to use it.** Once per workspace, into a fresh folder. Use `--force` on
an existing workspace only to reset it to the template: it overwrites your
`configs/*.yaml`, `AGENTS.md` and the example tasks with the template's, and
leaves files the template does not have alone. To pick up newer Agent Skills
after an upgrade without touching your configs, run `init` into an empty
folder and copy its `.agents/skills/` over.

```shell
openarena init my-arena               # a new folder
openarena init                        # the current folder
openarena init my-arena --force       # back to the template; your configs are overwritten
```

See [The workspace](workspace.md) for what each file holds.

### `openarena check`

Validates registry files against the same schema a run loads, and every
reference between them (a dataset's `reward`, an evaluation's models). Each
problem is printed with its line, the dotted path to the field, what is wrong
and the edit that fixes it.

```shell
openarena check [PATH] [--configs DIR]
```

| argument / option | type | default | effect |
|---|---|---|---|
| `PATH` | folder or file | `configs` | a configs folder (every `<registry>.yaml` in it), or one registry file |
| `--configs DIR` | folder | the file's own folder | only for a single file: the folder whose registries its references are checked against. Ignored when `PATH` is a folder |

A file's registry is read off its name only (`datasets.yaml` holds `datasets`);
a file with any other name is an error, whatever its top-level key. In a
folder, any other `.yaml` or `.yml` file is reported too, since nothing reads it.

| exit | when |
|---|---|
| `0` | no error; warnings are printed but allowed |
| `1` | at least one error; or `PATH` does not exist; or the folder has no registry files |

**When to use it.** After every edit, and as the first step of any CI job: a
run refuses a config `check` rejects, and `check` is fast, offline and free.
It reads files only. It does not call a model, download a dataset or start a
sandbox, so a wrong dataset path or a missing key still shows up at run time.

```shell
openarena check                                  # ./configs
openarena check configs/datasets.yaml            # one file, references against configs/
openarena check draft/datasets.yaml --configs configs   # a draft, against the real registries
```

Use the last form to validate a file before you move it into `configs/`.
See [Checking configs](guides/check.md) for how to read the output.

## Registry listings

```shell
openarena models list         [--configs DIR]
openarena rewards list        [--configs DIR]
openarena datasets list       [--configs DIR]
openarena mcp-servers list    [--configs DIR]
openarena a2a-agents list     [--configs DIR]
openarena environments list   [--configs DIR]
openarena evaluations list    [--configs DIR]
openarena leaderboards list   [--configs DIR]
```

Each prints one registry of the configs folder as a table, a row per entry.
They read the YAML as it is, without validating it: run `openarena check` for
that. A registry with no file prints an empty table.

| option | type | default | effect |
|---|---|---|---|
| `--configs DIR` | folder | `configs` | the configs folder to read; exits 1 if it does not exist |

| command | columns | guide |
|---|---|---|
| `openarena models list` | `model` (the litellm id), `type`, `description` | [Models](guides/models.md) |
| `openarena rewards list` | `type`, `direction`, `description` | [Rewards](guides/rewards.md) |
| `openarena datasets list` | `type`, `reward`, `description` | [Datasets](guides/datasets.md) |
| `openarena mcp-servers list` | `transport`, `command`, `url` | [MCP tools](guides/mcp-tools.md) |
| `openarena a2a-agents list` | `url`, `description` | [A2A agents](guides/a2a-agents.md) |
| `openarena environments list` | `docker_image`, `cpus`, `memory_mb`, `network_mode` | [Environments](guides/environments.md) |
| `openarena evaluations list` | `language_models`, `datasets`, `description` | [Running evaluations](guides/evaluations.md) |
| `openarena leaderboards list` | `title`, `datasets`, `models`, `description`; an empty scope shows as `all` | [Leaderboards](guides/leaderboards.md) |

**When to use them.** To find the exact name to pass to
`openarena evaluations run` or `openarena leaderboards run`, or to see which
model id a name calls before you spend money on it.

```shell
openarena evaluations list
openarena models list --configs ../other-arena/configs
```

## Running and ranking

### `openarena evaluations run`

Runs the evaluation saved as `NAME`: every model of its `language_models` on
every dataset of its `datasets`, one cell per pair.

```shell
openarena evaluations run NAME [--configs DIR] [--state-dir DIR] [--no-cache]
                               [--batch-size N] [--json FILE] [-v | -vv]
```

| argument / option | type | default | effect |
|---|---|---|---|
| `NAME` | text | required | a key of `evaluations.yaml`; an unknown name exits 1 and lists the ones there |
| `--configs DIR` | folder | `configs` | the configs folder. Its parent is where `.env`, `tasks/` and `environments/` are read |
| `--state-dir DIR` | folder | `.open-arena` | where the run is kept, in `<state-dir>/NAME/`; relative to the current folder |
| `--no-cache` | flag | off | drop the run's saved state and start every cell over |
| `--batch-size N` | integer, at least 1 | the evaluation's `batch_size`, else each dataset's own, else 1 | rows sent at once, for every dataset of this run |
| `--json FILE` | file | none | also write the result to `FILE`; parent folders are created |
| `-v`, `--verbose` | count | silent | `-v` a progress bar per cell, `-vv` a line per batch |

What it does, in order:

1. Loads `.env` from the parent of `--configs`, without overriding your shell.
2. Gathers the evaluation and every entry it names (its datasets, their
   rewards and MCP servers, Harbor environments, A2A agents; all models) into
   one config, and validates it. An invalid config exits 1 before any call.
3. Writes that config to `<state-dir>/NAME/config.yaml`: the exact record of
   what ran.
4. Copies the Harbor folders the run uses beside it, fresh each time:
   `environments/<name>/` for each Harbor environment, and the folder of each
   `harbor_tasks` dataset (`tasks/`). A file deleted from the source is gone
   from the copy too.
5. Runs the sweep. It prints each dataset, then each model as it starts and
   finishes with its metrics, then `<model> failed: <why>` for any cell that
   failed.
6. Writes `<state-dir>/NAME/result.json`, prints a table per dataset (a row
   per model, a column per metric, `—` where a cell has no value), and writes
   `--json FILE` if asked.

```
.open-arena/NAME/
  config.yaml            what ran
  result.json            the scores
  tasks/ environments/   copies of the Harbor folders, for Harbor datasets
  harbor/                Harbor trials, for Harbor datasets
```

!!! tip "Check first"
    The run validates only the entries the evaluation names. Run
    `openarena check` before it to catch every problem at once, with the fix.

#### Resuming, and when to use `--no-cache`

The run keeps each cell's state in `<state-dir>/NAME/`. Running the same
`NAME` again with the same `--state-dir` skips every cell that completed,
reading its scores back, and re-runs the cells that failed, were interrupted
or were never reached. So:

- **Interrupted** (Ctrl-C, a closed laptop, a crash): run the same command
  again.
- **A cell failed** (`<model> failed: <why>`): fix the cause, run the same
  command again. Only the failed cells run.
- **A cell scored 0** because of a missing key or a model server that was
  down: fix that, and run with `--no-cache`. A plain rerun keeps the 0.
- **You changed what a finished cell depends on** (a dataset's rows or
  templates, a reward, a model's params, the model list): add `--no-cache`.
  Without it, the old cells are reused and the table mixes old and new.

!!! warning "`--no-cache` throws away finished work"
    It deletes the saved state before starting, so every cell is paid for
    again. Never add it to resume a run, and do not leave it in a script you
    rerun after a failure.

#### Choosing `--batch-size`

A batch is the rows sent at once, so it is the concurrency knob. Higher is
faster, until you hit a provider's rate limit, a local server's memory, or
the number of sandboxes you can start for Harbor rows. The order of precedence
is `--batch-size`, then the evaluation's `batch_size`, then each dataset's own,
then 1.

| situation | batch size |
|---|---|
| a local model on one GPU (Ollama, vLLM) | low, 1 to 4; raise it while the server keeps up |
| a hosted API with a generous rate limit | 8 to 32 |
| rate-limit errors in the failed cells | lower it, then rerun: only the failed cells run |
| Harbor tasks, where each row is a sandbox | as many sandboxes as your provider or Docker host can hold |
| debugging one row's answer | 1, so errors are easy to follow |

Use the flag for a one-off change; put a value you want every time in
`evaluations.yaml`.

#### `--verbose` and `--json`

Without `-v`, a run prints one line when each cell starts and ends: enough for
CI logs. `-v` adds a progress bar per cell, useful in a terminal on a long
dataset. `-vv` prints a line per batch, useful to see where a run stalls.

`result.json` is always written in the run folder. `--json FILE` writes the
same content to a path you choose: an artifact path for CI, or a dated file to
keep a history. Its format is in
[Reading results](guides/results.md#resultjson).

#### When to use which setup

| goal | command |
|---|---|
| a first smoke run, on a few rows (`limit: 5` on the datasets) | `openarena evaluations run smoke -v` |
| resume after an interruption or a fixed failure | `openarena evaluations run smoke` (the same command, no `--no-cache`) |
| rerun after editing a dataset, reward or model | `openarena evaluations run smoke --no-cache` |
| a fast run on a hosted API | `openarena evaluations run compare --batch-size 16` |
| CI: a clean run with a result to archive | `openarena evaluations run nightly --no-cache --json out/nightly.json` |
| keep two variants' runs apart | `openarena evaluations run smoke --state-dir .open-arena-b` |

??? details "A CI job, step by step"

    ```shell
    set -e
    cd my-arena
    openarena check configs                       # exit 1 on any config error
    openarena evaluations run nightly --no-cache --json out/nightly.json
    # The run exits 0 even when cells failed: fail the job on them.
    python -c "
    import json, sys
    r = json.load(open('out/nightly.json'))
    bad = [(x['dataset'], x['model'], x['status']) for x in r['rows']
           if x['metric'] == 'reward' and x['status'] != 'COMPLETED']
    print(*bad, sep='\n'); sys.exit(1 if bad else 0)"
    openarena leaderboards run overall --no-eol --json out/ranking.json
    ```

    `--no-cache` makes each nightly run independent of the last one; drop it
    if you would rather a retried job resume. `--no-eol` keeps the ranking
    from depending on the network. Keys come from CI secrets, which win over
    any `.env`.

See [Running evaluations](guides/evaluations.md) for the `evaluations.yaml`
fields, and [Harbor tasks](guides/harbor-tasks.md) for sandboxed runs.

### `openarena leaderboards run`

Ranks the models on the leaderboard saved as `NAME`, from the runs already
finished. It runs no model.

```shell
openarena leaderboards run NAME [--configs DIR] [--state-dir DIR] [--no-eol] [--json FILE]
```

| argument / option | type | default | effect |
|---|---|---|---|
| `NAME` | text | required | a key of `leaderboards.yaml`; an unknown name exits 1 and lists the ones there |
| `--configs DIR` | folder | `configs` | the configs folder holding `leaderboards.yaml` |
| `--state-dir DIR` | folder | `.open-arena` | where the runs are; every `*/result.json` under it is read |
| `--no-eol` | flag | off | leave out `eol_days`, the days before each model's announced end of life, which is looked up online |
| `--json FILE` | file | none | also write the ranking to `FILE`; parent folders are created |

It reads every `result.json` under `--state-dir`, newest first, keeps the
newest score of each cell, applies the leaderboard's scope, weights and
budgets, and prints the datasets in scope, the share of each metric, and a
table: rank, model, score, each metric with ↑ or ↓ for which way is better,
and ★ on the Pareto frontier. Then a line for each model it left out:
`Not ranked <model>: <why>` (not scored on a dataset in scope, or a metric
missing) or `Over budget <model>: <limit>`.

| exit | when |
|---|---|
| `0` | a ranking was printed, even if some models were not ranked |
| `1` | no configs folder; no leaderboard `NAME`; no `result.json` under `--state-dir` |

**About `--no-eol`.** End-of-life dates come from litellm's pricing table,
Azure's published retirement schedule, and OpenRouter for `openrouter/...` ids.
The online sources are cached for a day, and a shipped snapshot answers when
the network does not. Add `--no-eol` when you want the ranking to depend only
on your runs: offline, in CI, or when a model's lifetime is not something you
rank on. It removes the `eol_days` metric entirely, so a share you gave to
`eol_days` in `leaderboards.yaml` has nothing to weigh.

**The JSON.** `--json FILE` writes `leaderboard` (the saved entry, with its
`name`) and the ranking: `datasets`, `models`, `shares`, `axes`, `rows` (each
with `rank`, `model`, `score`, `values`, `on_frontier`), `skipped`, `gaps` and
`excluded`. Use it to feed a dashboard or to diff two rankings.

```shell
openarena leaderboards list
openarena leaderboards run overall
openarena leaderboards run math --no-eol --json out/math.json
openarena leaderboards run overall --state-dir ../shared-runs   # rank runs kept elsewhere
```

!!! tip "One state folder, many evaluations"
    A leaderboard ranks across every evaluation under the state folder. Run
    your evaluations into the same `--state-dir`, and one leaderboard can
    compare models that never ran in the same evaluation, as long as each is
    scored on every dataset in scope.

See [Leaderboards](guides/leaderboards.md) for scope, shares, budgets and how
the score is computed.
