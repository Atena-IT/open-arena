# Agent guide: openarena workspace

An **openarena** workspace: YAML registries that say which models to compare,
on which datasets, scored how, and ranked how. The `openarena` command runs
the comparisons on this machine and ranks the results. No server and no web
app: files in, files out. The same folder is a workspace the arena web app
opens as it is.

## Layout

```
configs/                 the registries, one YAML each; read a file's header comment before editing it
  models.yaml            name -> litellm id + params
  rewards.yaml           name -> catalog reward + kwargs (what scores an answer)
  datasets.yaml          name -> rows, Jinja templates, what answers them, the reward
  mcp_servers.yaml       name -> MCP tool server
  a2a_agents.yaml        name -> remote A2A agent
  environments.yaml      name -> Harbor sandbox
  evaluations.yaml       name -> models x datasets, the sweep to run
  leaderboards.yaml      name -> how to rank finished runs
tasks/<name>/            Harbor task folders (task.toml, instruction.md, environment/, tests/, solution/)
environments/<name>/environment/   an environment's build context
.agents/skills/          one Agent Skill per thing to do here: read the one that fits before acting
.open-arena/             runs: config.yaml, result.json, harbor/ trials (created by the first run)
.env                     API keys and endpoints (copy .env.template); never commit it
```

## What to know

- **Everything is referenced by name.** An evaluation's `language_models`
  are `models.yaml` keys, a dataset's `reward` a `rewards.yaml` key, a judge's
  `language_model` a `models.yaml` key, never a litellm id. The id appears
  once, in `models.yaml`.
- **Entries refuse unknown fields.** Do not invent a field, a reward type or
  a provider: the skills list the real ones, and `openarena check` names the
  valid choices when one is wrong.
- **Templates render JSON.** A dataset's `input_template` / `output_template`
  are Jinja2 over a row's columns; pipe every value through `| tojson`.
  Look at real rows before writing them.
- **Check after every edit**, run small before running big: `limit: 5` on a
  dataset first, then the full size. A run costs a model call per row per
  model, plus one per judge.
- **Keys** come from `.env` (or the shell): `OPENAI_API_KEY`,
  `ANTHROPIC_API_KEY`, ... for hosted models; `ollama/...` ids need a local
  Ollama; Harbor datasets need `E2B_API_KEY` or a local Docker. Never start or
  stop model servers or Docker yourself: if one does not answer, say so.

## Commands

```bash
openarena check configs                         # validate every registry: problems, lines, fixes
openarena evaluations list                      # what can run
openarena evaluations run <name>                # sweep into .open-arena/<name>/; resumes if run again
openarena evaluations run <name> --no-cache     # start over after changing what a cell depends on
openarena leaderboards run <name>               # rank every result.json by a saved leaderboard
openarena <registry> list                       # models, rewards, datasets, mcp-servers, a2a-agents,
                                                # environments, evaluations, leaderboards
```

Run them from this folder: `--configs` defaults to `./configs` and
`--state-dir` to `./.open-arena`. Prefix with `uv run` when `openarena` is
installed in a uv project rather than on the PATH.

## Skills

Each skill in `.agents/skills/` teaches one action, its YAML field by field
and the commands that go with it:

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

## Reporting

Say what you ran and report from the files: the reward per model and
dataset from `result.json`, the ranking `leaderboards run` printed, what
failed and the error the run gave. Never fill a result in from memory.
