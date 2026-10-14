# Open Arena

<img src="open-arena.png" width="28%" align="right" alt="Open Arena logo">

**openarena** compares language models, and agents, on datasets: it runs every
model on every dataset you name, scores each answer, and ranks the models on
leaderboards. Everything is declared in YAML and run from one command,
`openarena`, on your own machine. No server and no web app: files in, files out.

- **Any model litellm routes**: OpenAI, Anthropic, Gemini, Mistral, Ollama, vLLM, ...
- **Any dataset source**: Hugging Face, local files, MLflow, Langfuse, LangSmith,
  Opik, Phoenix, Braintrust, Unity Catalog, each row shaped by Jinja2 templates.
- **Scored your way**: exact match, judge models, rubrics, multi-judge panels,
  agent judges, tool-call accuracy, retrieval recall.
- **Beyond one call**: models calling MCP tools, remote A2A agents, and agents
  working [Harbor](https://harborframework.com) tasks in a sandbox.
- **Ranked**: leaderboards weighing reward, cost, tokens and latency, with
  budgets and the Pareto frontier.
- **Ready for a coding agent**: every workspace comes with an `AGENTS.md` and
  one Agent Skill per task.

**Documentation: [atena-it.github.io/open-arena](https://atena-it.github.io/open-arena/)**

## Quickstart

```shell
uv tool install "git+https://github.com/Atena-IT/open-arena"

openarena init my-arena && cd my-arena
cp .env.template .env                # the keys your models need
openarena check configs              # every registry: problems, lines, fixes
openarena evaluations run smoke      # 20 GSM8K questions, results in .open-arena/smoke/
openarena leaderboards run overall   # rank every finished run
```

`smoke` runs a model on a local [Ollama](https://ollama.com)
(`ollama pull qwen3.8:27b-mlx`); point it at `gpt-4o-mini` in
`configs/evaluations.yaml` to use `OPENAI_API_KEY` instead.

## A workspace

`openarena init` writes a workspace that runs as it is:

```
configs/                          one registry per YAML, each field explained in comments
  models.yaml  rewards.yaml  datasets.yaml  mcp_servers.yaml  a2a_agents.yaml
  environments.yaml  evaluations.yaml  leaderboards.yaml
tasks/hello-world/                a Harbor task: task.toml, instruction.md, environment/,
                                  tests/test.sh, solution/solve.sh
environments/python/environment/  an environment's build context
.agents/skills/                   one Agent Skill per task
AGENTS.md                         the guide a coding agent reads first (CLAUDE.md links to it)
.env.template  .gitignore  README.md
```

```yaml
# configs/datasets.yaml
datasets:
  gsm8k-20:
    type: huggingface
    config:
      path: openai/gsm8k
      name: main
      split: test
      limit: 20
      input_template: '{"question": {{ question | tojson }}}'
      output_template: '{"answer": {{ answer.split(''####'')[-1] | replace('','', '''') | trim | tojson }}}'
      input_schema: {type: object, properties: {question: {type: string, description: The question to answer}}, required: [question]}
      output_schema: {type: object, properties: {answer: {type: string, description: The answer only}}, required: [answer]}
    generator:
      instructions: Solve the problem. Reply with the final number only.
    reward: exact-match
```

See the [guides](https://atena-it.github.io/open-arena/) for every registry.

## Development

```shell
git clone https://github.com/Atena-IT/open-arena && cd open-arena
uv sync
uv run openarena --help
uv run pytest                                          # no network needed
uvx ruff check .
uv run --group docs zensical build --clean --strict    # the docs site, into site/
uv run --group docs zensical serve -a localhost:8100   # preview while editing
```

| path | |
| --- | --- |
| `src/openarena/config/` | the schema: one model per registry entry, a run's config, `check` |
| `src/openarena/evaluate.py` | `run_sweep`, every model × dataset cell |
| `src/openarena/programs/` | the program each dataset kind is answered by |
| `src/openarena/datasets/` | dataset providers |
| `src/openarena/rewards/` | the reward catalog and openarena's own rewards |
| `src/openarena/leaderboard.py` | `board` gathers results, `rank` scores them |
| `src/openarena/harbor.py` | Harbor tasks, agents and sandboxes |
| `src/openarena/cli/` | the `openarena` command and its `init` template |
| `src/openarena/skills/` | the Agent Skills `init` copies into a workspace |
| `docs/`, `mkdocs.yml` | the documentation site, built by [Zensical](https://zensical.org) |

The tests keep the docs and skills honest: every YAML example in them is
checked against the engine (`tests/test_docs.py`, `tests/test_skills.py`), and
the CLI reference must name every command and option.

## License

[Apache 2.0](LICENSE).
