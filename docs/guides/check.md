# Checking configs

`openarena check` validates the YAML with the schema a run loads, and every
reference between files. Run it after every edit: a run refuses a config that
`check` rejects.

```shell
openarena check configs                         # every registry file of the folder
openarena check configs/datasets.yaml           # one file, references checked against its folder
openarena check other/datasets.yaml --configs configs   # references checked elsewhere
```

A file's registry is read off its name (`datasets.yaml` holds `datasets`),
else off its top-level key. It exits 0 when there is no error, warnings
allowed, and 1 otherwise.

## Reading the output

```
✓ configs/models.yaml: 2 models, no problems
✗ configs/rewards.yaml: 2 rewards, 2 errors

  error    line 3  rewards.exact-match.type
           unknown reward 'exact_matsh'
     fix   change 'exact_matsh' to 'exact_match'

  error    line 6  rewards.j.config.language_modl
           lm_as_judge does not take `language_modl`
     fix   rename `language_modl:` to `language_model:` under rewards.j.config
```

Each problem gives:

- its **level**: an `error` stops a run, a `warning` does not (a leaderboard
  scoped to a dataset that no longer exists);
- its **line** in the file;
- **where**: the dotted path to the field, list items as `[i]`;
- what is wrong, and a **fix** to apply as written: a rename, a replacement
  value, a field to remove, a template to paste.

## What it does not check

It reads files only. It does not call a model, download a dataset or start a
sandbox: a wrong Hugging Face path or a missing key shows up when the run
reaches it ([Reading results](results.md#when-a-cell-failed-or-scored-0)).
