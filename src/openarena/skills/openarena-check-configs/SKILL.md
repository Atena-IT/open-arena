---
name: openarena-check-configs
description: Use after writing or editing any openarena YAML (models, rewards, datasets, MCP servers, A2A agents, environments, evaluations, leaderboards), or when an openarena command fails on a config. Validates a configs folder or one registry file with `openarena check`, explains how to read each problem and apply its fix, and loops until the files are clean.
---

# openarena-check-configs

`openarena check` validates registry YAML with the engine's own strict schema
(a field it does not know is an error) and checks every reference between
files. It reads every entry and reports every problem at once, each with its
line and a fix. Run it after every edit and before every run.

`evaluations run` validates only the evaluation it runs and the entries it
selects, and stops at the first problem: a broken entry no evaluation uses
yet waits there. So check the whole folder, not just the file you edited.

Depth: [Checking configs](https://atena-it.github.io/open-arena/guides/check/).

## Commands

From the workspace (the folder holding `configs/`):

```bash
openarena check                                          # ./configs: every registry file in it
openarena check configs/datasets.yaml                    # one file, references checked against its folder
openarena check drafts/datasets.yaml --configs configs   # a file elsewhere, references checked against configs/
```

- `--configs DIR` is for a single file only; it is ignored when PATH is a folder.
- Exit 0: no error (warnings allowed). Exit 1: at least one error, nothing at
  PATH (`nothing at <PATH>; pass a configs folder or a registry file`), or a
  folder with no registry file
  (``no registry files in <PATH>; write some with `openarena init` ``).
- There is no JSON output. For a machine-readable result, call
  `openarena.config.check.check_folder(Path("configs"))` from Python (each
  `Report` has `.problems`, `.errors`, `.warnings`).

A registry is read **only** from the file named after it: `models.yaml`,
`rewards.yaml`, `datasets.yaml`, `mcp_servers.yaml`, `a2a_agents.yaml`,
`environments.yaml`, `evaluations.yaml`, `leaderboards.yaml`, each holding one
top-level key, its own. Any other `.yaml`/`.yml` in the folder is an error,
even with the right top-level key, and a run never reads it:

```
✗ configs/my-models.yaml: 1 error

  error    my-models.yaml
           not a registry file: a registry is read only from `<registry>.yaml`
     fix   rename it to `models.yaml`
```

## Reading the output

A clean file is one line; `✗` means at least one error, `!` warnings only:

```
✓ configs/models.yaml: 2 models, no problems
```

Each problem gives level, line, where, what is wrong, and the fix:

```
✗ configs/rewards.yaml: 3 rewards, 2 errors

  error    line 3  rewards.exact-match.type
           unknown reward 'exact_matsh'
     fix   change 'exact_matsh' to 'exact_match'

  error    line 8  rewards.judge.config.language_modl
           lm_as_judge does not take `language_modl`
     fix   rename `language_modl:` to `language_model:` under rewards.judge.config
```

- **where** is the dotted path: `<registry>.<entry>.<field>`, list items as
  `[i]` (`datasets.agent.agent.mcp_servers[0]`). A file-level problem shows
  the registry or the file name.
- Two levels only. **error**: the engine would refuse it; it fails `check`
  and blocks a run that loads that entry. **warning**: it loads, or could not
  be checked; it never fails `check` and never blocks a run.
- **fix** is literal: apply it as written. A multi-line fix is a snippet to
  paste. Shapes you will see:
  - a rename: ``rename `pth:` to `path:` under datasets.local.config``;
  - a value: `change 'hugingface' to 'huggingface'`;
  - a removal, with what the entry reads: ``remove `foo:` from datasets.local; an entry reads: ...``;
  - a reference: `change 'exact-mach' to 'exact-match'`, or
    `name the entry that calls it: 'gpt-4o-mini'` for a raw model id where a
    `models` key belongs;
  - a snippet, e.g. the templates to paste for
    ``no `config.input_template`: nothing says what each row asks``.

Warnings to know:

- ``no `models` registry to check references against; they are not checked``:
  a single file with no registry beside it. It means **not checked**, not
  correct: pass `--configs DIR` or check the folder.
- ``'retired' is not in `datasets`; only scores from runs made before it was removed can be ranked``:
  a leaderboard scoped to a missing entry. Fine for a retired dataset,
  otherwise fix the name.
- ``not validated whole: `datasets` it names does not read``: an evaluation
  whose referenced registry has errors. The fix names that registry's file;
  check and fix it, and the warning goes.
- ``` `${NAME}` is not set in your shell or in .env; a run that reads it refuses to start ```:
  a value references a variable set nowhere (check reads the `.env` beside
  the configs folder for it). Add `NAME=...` to that `.env`, or export it.

## The loop

1. `openarena check configs`.
2. For each error, open the file at its line and apply the fix. With no fix,
   the message names the rule.
3. Run again until it exits 0. After a rename, the errors land in the files
   that still use the old name: that is why the folder is checked.
4. Read the warnings left and decide on each (see above).

## What check does not catch

It reads files only. These pass and fail later, in the run (see
openarena-read-results):

- **Environment**: `.env` is read only for the `${NAME}` references a file
  holds; a provider key a model needs (`OPENAI_API_KEY`, ...) is not
  noticed.
- **Reachability**: no model id is tried, no MCP server started, no A2A URL
  called, no Docker image pulled, no dataset downloaded or opened (a wrong
  Hugging Face path or a missing local file passes).
- **Packages**: whether a provider's optional packages are installed.
- **Templates**: only that `input_template` is set; the Jinja is not parsed
  and its columns are not matched against the data.

Check proves the config is well formed; a run on a few rows (`limit:`) with
one cheap model proves it works.
