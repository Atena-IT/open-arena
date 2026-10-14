---
name: openarena-check-configs
description: Use after writing or editing any openarena YAML (models, rewards, datasets, MCP servers, A2A agents, environments, evaluations, leaderboards), or when an openarena command fails on a config. Validates a configs folder or one registry file with `openarena check`, explains how to read each problem and apply its fix, and loops until the files are clean.
---

# openarena-check-configs

`openarena check` validates YAML with the engine's own strict schema, the same
one a run loads, and checks every reference between files. Run it after every
edit; a run refuses a config `check` would reject.

## Commands

From the workspace's parent folder (or with `uv run` in the openarena project):

```bash
uv run openarena check my-arena/configs                    # the folder: every registry file in it
uv run openarena check my-arena/configs/datasets.yaml      # one file, references checked against its folder
uv run openarena check other/datasets.yaml --configs my-arena/configs   # one file, references checked elsewhere
```

A file's registry is read off its name (`datasets.yaml` holds `datasets`),
else off its top-level key. Exit code 0: no error (warnings allowed);
1: at least one error.

## Reading the output

A clean file:

```
✓ my-arena/configs/rewards.yaml: 2 rewards, no problems
```

A problem is printed with where it is, its line, what is wrong and the fix:

```
✗ my-arena/configs/rewards.yaml: 2 rewards, 2 errors

  error    line 3  rewards.exact-match.type
           unknown reward 'exact_matsh'
     fix   change 'exact_matsh' to 'exact_match'

  error    line 6  rewards.j.config.language_modl
           lm_as_judge does not take `language_modl`
     fix   rename `language_modl:` to `language_model:` under rewards.j.config
```

- **where** is the dotted path to the field: `<registry>.<entry>.<field>`,
  list items as `[i]` (`datasets.agent.agent.mcp_servers[0]`).
- **error** stops a run; **warning** does not (a leaderboard scoped to a
  dataset that no longer exists, a registry missing beside a single file).
- **fix** is literal: apply it as written. Typical fixes:
  - a misspelled field: ``rename `pth:` to `path:` under datasets.x.config``;
  - a near-miss value: ``change 'hugingface' to 'huggingface'``;
  - a field the entry does not take: ``remove `foo:` ``;
  - a missing `input_template`: a template to paste.

## The loop

1. `uv run openarena check <configs>`.
2. For each error, open the file at its line and apply the fix. If no fix is
   given, the message names the rule (e.g. two kind blocks in one dataset:
   keep one).
3. Run again until it exits 0. Then deal with warnings that matter (a
   leaderboard scoped to a removed dataset ranks nothing).

## What check does not do

It reads files only: it does not call a model, download a dataset or start a
sandbox. A provider's kwargs are checked by name, not by fetching the data;
a model id is not tried. A run can still fail on a wrong Hugging Face path or
a missing API key: see openarena-read-results for those failures.
