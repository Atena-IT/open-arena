# openarena workspace

Compare language models (or agents) on datasets, and rank them, with the
**openarena** engine. Made by `openarena init`.

## Quickstart

```bash
# 1. Keys and endpoints: copy the template and fill in what your models need.
cp .env.template .env

# 2. Check the configs.
openarena check configs

# 3. Run the example evaluation: 20 GSM8K rows on a local Ollama model
#    (`ollama pull qwen3.8:27b-mlx`, or change `smoke` to `gpt-4o-mini` with OPENAI_API_KEY).
openarena evaluations run smoke

# 4. Rank the results.
openarena leaderboards run overall
```

`openarena` comes with the `openarena` Python package (`uv run openarena ...`
inside a uv project that depends on it).

## Layout

```
configs/          the registries: models, rewards, datasets, MCP servers, A2A agents,
                  environments, evaluations, leaderboards; each file explains its fields
tasks/            Harbor task folders, run by a `harbor_tasks` dataset
environments/     each environment's build context
.agents/skills/   Agent Skills for a coding agent working here
AGENTS.md         the guide a coding agent reads first (CLAUDE.md points to it)
.env.template     the keys and endpoints a run reads; copy to .env
```

Runs land in `.open-arena/<evaluation>/` (`result.json`, Harbor trials).

## With a coding agent

Start Claude Code, Codex, Cursor or another coding agent in this folder and
ask in plain words ("add Mistral Small and compare it on GSM8K"): `AGENTS.md`
and the skills in `.agents/skills/` tell it how the files are written and
which commands to run.
