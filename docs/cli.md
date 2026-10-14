# CLI reference

```shell
openarena [COMMAND] --help
```

Commands that read a workspace take `--configs DIR` (default `./configs`);
those that read or write runs take `--state-dir DIR` (default `./.open-arena`).
Run them from the workspace folder and neither is needed.

## `openarena init`

```shell
openarena init [DEST] [--template NAME] [--force]
```

Writes a ready-to-use workspace into `DEST` (default: the current folder):
`configs/`, `tasks/`, `environments/`, `.agents/skills/`, `AGENTS.md` and
`CLAUDE.md`, `README.md`, `.env.template`, `.gitignore`. See
[The workspace](workspace.md).

| option | |
|---|---|
| `--template NAME` | the template to start from (default `default`) |
| `--force` | overwrite files already there; without it, `init` writes nothing when one is in the way |

## `openarena check`

```shell
openarena check [PATH] [--configs DIR]
```

Checks `PATH`: a configs folder, every registry file in it (default
`./configs`), or one registry file. Each problem is printed with its line, what
is wrong and how to fix it. Exits 1 on an error. See
[Checking configs](guides/check.md).

| option | |
|---|---|
| `--configs DIR` | for a single file: the folder whose registries its references are checked against (default: the file's own) |

## `openarena evaluations`

### `evaluations list`

```shell
openarena evaluations list [--configs DIR]
```

Lists the saved evaluations.

### `evaluations run`

```shell
openarena evaluations run NAME [--configs DIR] [--state-dir DIR] [--no-cache]
                               [--batch-size N] [--json FILE] [-v]
```

Runs the evaluation saved as `NAME` into `<state-dir>/NAME/`, reading the
workspace's `.env` first. Running it again resumes. See
[Running evaluations](guides/evaluations.md).

| option | |
|---|---|
| `--configs DIR` | the configs folder (default `configs`) |
| `--state-dir DIR` | where runs keep their config, trial state and result, a folder per evaluation (default `.open-arena`) |
| `--no-cache` | start over instead of resuming a previous run |
| `--batch-size N` | override the evaluation's batch size, at least 1 |
| `--json FILE` | also write the result as JSON |
| `-v`, `--verbose` | show progress: `-v` a bar, `-vv` a line per batch |

## `openarena leaderboards`

### `leaderboards list`

```shell
openarena leaderboards list [--configs DIR]
```

Lists the saved leaderboards, with their scope.

### `leaderboards run`

```shell
openarena leaderboards run NAME [--configs DIR] [--state-dir DIR] [--no-eol] [--json FILE]
```

Ranks the models on the leaderboard saved as `NAME`, from the finished runs.
See [Leaderboards](guides/leaderboards.md).

| option | |
|---|---|
| `--configs DIR` | the configs folder (default `configs`) |
| `--state-dir DIR` | where the runs are (default `.open-arena`) |
| `--no-eol` | leave out time to end of life, which looks dates up online |
| `--json FILE` | also write the ranking as JSON |

## Registry listings

```shell
openarena models list         [--configs DIR]
openarena rewards list        [--configs DIR]
openarena datasets list       [--configs DIR]
openarena mcp-servers list    [--configs DIR]
openarena a2a-agents list     [--configs DIR]
openarena environments list   [--configs DIR]
```

Each prints its registry as a table.
