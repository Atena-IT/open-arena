# Agent guide: open-arena

The repository of **openarena**, an evaluation engine and its `openarena`
command: YAML registries (models, rewards, datasets, MCP servers, A2A agents,
environments, evaluations, leaderboards) in, a sweep of every model on every
dataset out, ranked on leaderboards. A uv project, Python 3.12+.

## Layout

- `src/openarena/config/` is the one strict schema. `entries.py` holds a model
  per registry entry, each refusing a field it does not know; `__init__.py`
  (`Config`) reads a run's `config.yaml` and resolves references by name;
  `check.py` lists a registry's problems, each with its line and fix.
- `src/openarena/evaluate.py` (`run_sweep`) runs the cells, model by model, one
  `GridSearch` tuner per dataset; a finished cell is cached in the run's state.
- `src/openarena/programs/` builds the program a dataset kind is answered by:
  `generator`, `agent` (MCP tools), `a2a`, `embedding`, `tools`, `harbor`.
- `src/openarena/datasets/` are the dataset providers, `rewards/` the reward
  catalog and the engine's own rewards, `leaderboard.py` gathers and ranks.
- `src/openarena/cli/` is the `openarena` command; `templates/default/` is what
  `openarena init` writes, and `src/openarena/skills/` the Agent Skills it
  copies into a workspace.
- `docs/` + `mkdocs.yml` is the documentation site, built by Zensical,
  published to GitHub Pages from `main` (`.github/workflows/docs.yml`).

## Rules

- **Docs, skills and templates follow the code.** A change to a field, a
  reward, a provider or a command changes the guide and the skill that teach
  it. `tests/test_docs.py` and `tests/test_skills.py` check every YAML example
  they show against the engine, and that the CLI reference names every command
  and option.
- **The init template runs as written**: `openarena check` is clean on it, and
  `tests/test_cli.py` checks what it writes.
- **No invented field or name**: lists in the docs (reward types, providers,
  model params, Harbor agents) are read from the engine.
- **Tests run without network or keys**; fake the sweep or the provider.

## Commands

```bash
uv sync
uv run pytest
uvx ruff check .
uv run --group docs zensical build --clean --strict
uv run --group docs zensical serve -a localhost:8100
```
