# With a coding agent

An openarena workspace is written to be worked on by a coding agent (Claude
Code, Codex, Cursor, Copilot, ...) as much as by hand. Start one in the
workspace folder and ask in plain words:

> Add Mistral Small and compare it with gpt-4o-mini on GSM8K, then tell me which is better for the money.

## What guides it

`openarena init` writes two things for it:

- **`AGENTS.md`**, the guide an agent reads first: the layout, the rules (refer
  to entries by name, invent no field, templates render JSON, check after
  every edit, run small before running big), the commands, and which skill to
  read for what. `CLAUDE.md` is a link to it, for agents that look for that
  name.
- **`.agents/skills/`**, one [Agent Skill](https://agentskills.io) per task.
  Each teaches its YAML field by field and the commands that go with it:

| skill | when |
|---|---|
| `openarena-init-workspace` | the layout and the rules every YAML follows |
| `openarena-check-configs` | after any edit; reading and fixing `check`'s problems |
| `openarena-add-model` | adding or changing a model |
| `openarena-add-reward` | choosing and configuring what scores answers |
| `openarena-add-dataset` | turning a benchmark into rows, templates and a reward |
| `openarena-add-mcp-tools` | models that answer by calling MCP tools |
| `openarena-compare-a2a-agents` | comparing remote A2A agents instead of models |
| `openarena-add-environment` | rows worked by an agent in a Harbor sandbox |
| `openarena-create-harbor-task` | writing a Harbor task folder |
| `openarena-run-evaluation` | running a sweep |
| `openarena-rank-leaderboard` | ranking finished runs |
| `openarena-read-results` | reading results, and why a cell failed or scored 0 |

The skills ship with the package, under `openarena/skills/`. To use them
outside a workspace, copy a folder into `~/.claude/skills/` or
`~/.agents/skills/`.

## Keep it honest

The agent works from the files: it checks with `openarena check`, runs with
`openarena evaluations run`, and reports from `result.json` and the ranking.
Two habits are worth asking for:

- **run small first**: a few rows per dataset before the full size, since every
  row is a paid call per model;
- **report what failed**: a failed cell says why, and a missing key or a model
  server that is not running is yours to fix, not the agent's to work around.
