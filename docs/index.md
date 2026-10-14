# Quickstart

**openarena** compares language models, and agents, on datasets: it runs every
model on every dataset you name, scores each answer, and ranks the models on
leaderboards. Everything is declared in YAML files and run from one command,
`openarena`, on your own machine. No server and no web app: files in, files out.

!!! info "Working with a coding agent?"
    `openarena init` writes an `AGENTS.md` guide and one Agent Skill per task
    into the workspace. Start Claude Code, Codex, Cursor or another coding agent
    in the folder and ask in plain words. See [With a coding agent](coding-agents.md).

## Install

openarena is a Python package (Python 3.12+), developed in
[open-arena](https://github.com/Atena-IT/open-arena). With
[uv](https://docs.astral.sh/uv/):

=== "As a tool"

    ```shell
    uv tool install "git+https://github.com/Atena-IT/open-arena"
    openarena --help
    ```

=== "From a clone"

    ```shell
    git clone https://github.com/Atena-IT/open-arena
    cd open-arena
    uv sync
    uv run openarena --help
    ```

From a clone, prefix every command on this site with `uv run`.

## Start a workspace

```shell
openarena init my-arena
cd my-arena
```

`init` writes a workspace that runs as it is: models, rewards, datasets and
evaluations ready to go, each YAML file explaining its own fields in comments.
See [The workspace](workspace.md) for what each file is.

## Set up a model

The example evaluation, `smoke`, runs a model served by a local
[Ollama](https://ollama.com), which needs no API key:

```shell
ollama pull qwen3.8:27b-mlx
```

Prefer a hosted model? Copy the environment template and fill in the key:

```shell
cp .env.template .env      # then set OPENAI_API_KEY=...
```

and point `smoke` at `gpt-4o-mini` in `configs/evaluations.yaml`:

```yaml
evaluations:
  smoke:
    language_models: [gpt-4o-mini]
    datasets: [gsm8k-20]
```

## Check, run, rank

```shell
openarena check configs              # every registry: problems, lines, fixes
openarena evaluations run smoke      # 20 GSM8K questions; results in .open-arena/smoke/
openarena leaderboards run overall   # rank every finished run
```

`evaluations run` prints a table per dataset: a row per model, a column per
metric (`reward` first, then cost, tokens and latency). `leaderboards run`
turns every finished run into one ranking.

## Next

- [Introduction](introduction.md): how a sweep works, from rows to a ranking.
- [Datasets](guides/datasets.md): bring your own benchmark.
- [Rewards](guides/rewards.md): choose what scores an answer.
- [Harbor tasks](guides/harbor-tasks.md): evaluate agents on tasks in a sandbox.
