<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Atena-IT/open-arena/main/img/openarena-logo-dark.svg">
  <img height="160" alt="Open Arena by Atena" src="https://raw.githubusercontent.com/Atena-IT/open-arena/main/img/openarena-logo-light.svg">
</picture>
</div>

<div align="center">

<b>Which model, or agent, does this job best, at what cost?</b>

<em>openarena runs every model on every dataset you name, scores each answer, and ranks the models on leaderboards. Declared in YAML, run from one command, on your own machine.</em>

<p align="center">
  <a href="https://atena-it.github.io/open-arena/"><strong>Documentation</strong></a> ·
  <a href="https://atena-it.github.io/open-arena/introduction/"><strong>Introduction</strong></a> ·
  <a href="https://atena-it.github.io/open-arena/cli/"><strong>CLI reference</strong></a> ·
  <a href="https://atena-it.github.io/open-arena/coding-agents/"><strong>With a coding agent</strong></a>
</p>

[![Tests](https://github.com/Atena-IT/open-arena/actions/workflows/tests.yml/badge.svg)](https://github.com/Atena-IT/open-arena/actions/workflows/tests.yml)
[![Docs](https://github.com/Atena-IT/open-arena/actions/workflows/docs.yml/badge.svg)](https://atena-it.github.io/open-arena/)
![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://github.com/Atena-IT/open-arena/blob/main/LICENSE)

</div>

## What is Open Arena?

**openarena** compares language models, and agents, on datasets. You describe
the candidates and the job in YAML; openarena runs every candidate on every
dataset, scores each answer, and ranks them. No server and no web app: files
in, files out.

- **Any model litellm routes**: OpenAI, Anthropic, Gemini, Mistral, Ollama, vLLM, ...
- **Any dataset source**: Hugging Face, local files, MLflow, Langfuse, LangSmith,
  Opik, Phoenix, Braintrust, Datadog, Unity Catalog, each row shaped by Jinja2 templates.
- **Scored your way**: exact match, judge models, rubrics, multi-judge panels,
  agent judges, tool-call accuracy, retrieval recall.
- **Beyond one call**: models calling MCP tools, remote A2A agents, and agents
  working [Harbor](https://harborframework.com) tasks in a sandbox.
- **Ranked**: leaderboards weighing reward, cost, tokens and latency, with
  budgets and the Pareto frontier.
- **Ready for a coding agent**: every workspace comes with an `AGENTS.md` and
  an Agent Skill for each thing to do in it: add a model, run an evaluation, ...

## When to use it

Use it when you have to pick a model, or keep picking one. It measures on your
data, on your machine, and keeps every score so the next comparison builds on
the last one.

<div align="center">

| You want | openarena gives you |
| --- | --- |
| A cheaper model for a task that already works | The same dataset and reward on both, ranked by reward against cost |
| A local model to replace a hosted one | Ollama or vLLM next to the hosted model, on the same leaderboard |
| The model that uses your MCP tools best | Datasets with an `agent:` block: the tools really run and the final answer is scored. Or `tools:`, to grade the calls themselves |
| The deployed agent that answers your users best | Remote agents reached over A2A, listed in an evaluation in place of models and scored the same way |
| The coding agent that solves your tasks | Harbor tasks run in a sandbox, checked by each task's own tests |

</div>

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

`openarena init my-arena` writes a workspace that runs as it is:

```
my-arena/
├── configs/                  one registry per file, every field explained in comments
│   ├── models.yaml           the models to compare, and the judge models
│   ├── rewards.yaml          how an answer is scored, and which way is better
│   ├── datasets.yaml         where the rows come from, their templates, their reward
│   ├── mcp_servers.yaml      tool servers a model can call
│   ├── a2a_agents.yaml       remote agents reached over A2A
│   ├── environments.yaml     sandboxes: their Dockerfile or image, resources, network
│   ├── evaluations.yaml      sweeps: which models run on which datasets
│   └── leaderboards.yaml     how finished runs are ranked
├── tasks/
│   └── hello-world/          a Harbor task
│       ├── task.toml
│       ├── instruction.md
│       ├── environment/
│       │   └── Dockerfile
│       ├── tests/
│       │   └── test.sh
│       └── solution/
│           └── solve.sh
├── environments/             the files each sandbox's Dockerfile copies in
│   └── python/               the `python` entry of environments.yaml
│       └── environment/      its build context, as Harbor expects it
│           └── requirements.txt
├── .agents/
│   └── skills/               an Agent Skill per action: openarena-add-model, ...
├── AGENTS.md                 the guide a coding agent reads first
├── CLAUDE.md -> AGENTS.md    the same guide, under the name Claude Code reads
├── README.md
├── .env.template             the keys your models need
└── .gitignore
```

Runs write their results under `.open-arena/`, one folder per evaluation.
Each registry has its guide:
[models](https://atena-it.github.io/open-arena/guides/models/) ·
[rewards](https://atena-it.github.io/open-arena/guides/rewards/) ·
[datasets](https://atena-it.github.io/open-arena/guides/datasets/) ·
[MCP tools](https://atena-it.github.io/open-arena/guides/mcp-tools/) ·
[A2A agents](https://atena-it.github.io/open-arena/guides/a2a-agents/) ·
[environments](https://atena-it.github.io/open-arena/guides/environments/) ·
[evaluations](https://atena-it.github.io/open-arena/guides/evaluations/) ·
[leaderboards](https://atena-it.github.io/open-arena/guides/leaderboards/).

The dataset the `smoke` evaluation runs, as `init` writes it:

```yaml
# configs/datasets.yaml
datasets:
  gsm8k-20:
    type: huggingface
    description: Grade-school math word problems, GSM8K test split, 20 rows.
    tags: [math, reasoning]
    config:
      path: openai/gsm8k
      name: main
      split: test
      limit: 20
      batch_size: 4
      # Jinja2 over the row's columns, each rendering to the JSON its schema describes.
      input_template: '{"question": {{ question | tojson }}}'
      # The reference is what follows `####`, commas removed.
      output_template: '{"answer": {{ answer.split(''####'')[-1] | replace('','', '''') | trim | tojson }}}'
      input_schema:
        type: object
        properties:
          question: {type: string, description: The question to answer}
        required: [question]
        additionalProperties: false
      output_schema:
        type: object
        properties:
          answer: {type: string, description: "The final numeric answer only, no units or commas"}
        required: [answer]
        additionalProperties: false
    generator:
      chain_of_thought: true
      instructions: Solve the problem. Reply with the final number only, no units, no commas.
    reward: exact-match                # a `rewards` key
```

See the [guides](https://atena-it.github.io/open-arena/) for every registry.

## Telemetry

openarena sends one anonymous event each time an evaluation run starts, so
we can see which providers, datasets and rewards people use, and what to work
on next. Downloads alone do not say that.

**Turn it off** with either of these, set in the environment the run starts in:

```shell
export ARENA_TELEMETRY=0     # unset or empty is on; any value but 1, true, yes, on or print is off
export DO_NOT_TRACK=1        # the cross-tool convention, honoured too
```

Nothing is sent from a CI runner (`CI`, `TF_BUILD`, `JENKINS_URL`,
`CODEBUILD_BUILD_ID` or `TEAMCITY_VERSION` set) unless `ARENA_TELEMETRY=1`. A run that sends prints one line on stderr saying
so, once per process.

**See exactly what would be sent** with `ARENA_TELEMETRY=print`: the request
body is printed on stderr and nothing leaves your machine.

The event, `evaluation_run`, holds these properties and nothing else:

| property | example | why |
| --- | --- | --- |
| `source` | `sdk` | tells these events from the Open Arena platform's, which share the project |
| `$lib`, `$lib_version` | `openarena`, `0.2.0` | which releases are in use |
| `$process_person_profile`, `$geoip_disable` | `false`, `true` | tell PostHog to keep no profile and to look up no location from the request's address |
| `python_version` | `3.12` | which Pythons to keep supporting |
| `os`, `arch` | `linux`, `x86_64` | which platforms to test on |
| `models_count`, `datasets_count` | `3`, `2` | how big a sweep is (an A2A run counts its agents, and names none) |
| `model_providers` | `["anthropic", "ollama"]` | which providers litellm routes to; `other` for one it does not list |
| `models` | `["openai/gpt-4o-mini", "other"]` | only models in litellm's public price table; a fine-tune, a deployment or a local model is `other` |
| `dataset_kinds` | `["agent", "model"]` | what answers the rows: a model, an MCP agent, an A2A agent, Harbor, ... |
| `dataset_providers` | `["huggingface", "local"]` | where the rows come from |
| `reward_types` | `["exact_match", "lm_as_judge"]` | how answers are scored |

It never holds a prompt, a row, an answer, a score, a key, an endpoint, a
path, a host or user name, or a name you chose (a model's, a dataset's, an
evaluation's, an agent's). The event's `distinct_id` is a random id made for
each process and forgotten when it ends: nothing is written on your machine,
and two runs cannot be tied together. It goes to PostHog's EU Cloud
(`eu.i.posthog.com`), which sees your IP address as any server you reach
does; the event asks it to look up no location from it. It is sent in the background, with a 2-second timeout: it never
fails, slows or stops a run, and nothing is sent when openarena is imported.

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
| `src/openarena/results.py` | the cells read off the tuners, as the rows a run reports |
| `src/openarena/programs/` | the program each dataset kind is answered by |
| `src/openarena/datasets/` | dataset providers; `platform.py` is what the ones reading a platform share |
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

[Apache 2.0](https://github.com/Atena-IT/open-arena/blob/main/LICENSE).
