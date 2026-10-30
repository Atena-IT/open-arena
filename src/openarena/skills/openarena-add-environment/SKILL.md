---
name: openarena-add-environment
description: Use when the rows of an ordinary openarena dataset (questions from Hugging Face, a file, ...) should be worked by an agent inside a sandbox rather than answered in one model call — defining a Harbor environment in environments.yaml (image or Dockerfile, its build context folder, resources, network, env, MCP tools) and a `harbor:` dataset over it whose answer file the dataset's reward scores.
---

# openarena-add-environment

An environment is a sandbox definition. A dataset with a `harbor:` block
naming one turns each of its rows into a Harbor task in that sandbox: the
row's question (after `instructions`) is the task's instruction, the model
under test drives the agent, and the agent's answer is scored by the
dataset's own reward, on the arena's side. Every row gets its own sandbox.
Use task folders instead (openarena-create-harbor-task) when each task needs
its own files and tests. Full reference: `docs/guides/environments.md`.

## The environment

`openarena init` already ships this `python` entry and the dataset below;
copy them under new names rather than defining the same keys twice.

```yaml
environments:
  python:
    description: Python 3.12 with tmux.
    tags: [python]
    dockerfile: |                      # or `docker_image: python:3.12-slim`; exactly one of the two
      FROM python:3.12-slim
      RUN apt-get update && apt-get install -y --no-install-recommends tmux && rm -rf /var/lib/apt/lists/*
      WORKDIR /app
      COPY requirements.txt /tmp/requirements.txt
      RUN pip install --no-cache-dir -r /tmp/requirements.txt
    workdir: /app                      # where the agent starts; the image must have it
    cpus: 1                            # blank: the provider's default (local Docker: 2 CPUs, 2048 MB)
    memory_mb: 2048
    network_mode: public               # public | no-network | allowlist
```

Optional fields (any other key is an error):

```yaml
environments:
  python:
    # ...
    os: linux                          # or windows
    storage_mb: 4096
    gpus: 1                            # not on E2B
    gpu_types: [H100]                  # empty: any
    network_mode: allowlist
    allowed_hosts: [pypi.org]          # required by allowlist, refused with any other mode
    env: {PYTHONUNBUFFERED: "1"}       # strings: quote numbers
    build_timeout_sec: 600
    healthcheck: {command: "test -f /app/ready", interval_sec: 5, retries: 3}
    mcp_servers: [fetch]               # mcp_servers.yaml keys handed to the agent
    clock: {start: "2026-01-05T09:00:00Z", rate: 0}   # one virtual time for every server in the sandbox; 0 is frozen
```

- **Build context**: `environments/<name>/environment/` beside `configs/`
  holds what the Dockerfile `COPY`s
  (`environments/python/environment/requirements.txt` above). Each run copies
  `environments/<name>/` fresh beside its config. A `docker_image` needs no
  folder.
- **tmux**: `terminus-2`, the default agent, drives its terminal through
  `tmux`. When the image lacks it, terminus-2 installs it at the start of
  every task, which needs a package manager and the network; without them,
  every trial fails. Install `tmux` in the Dockerfile, and start from an
  image with `python3`.
- **workdir**: with a `docker_image`, a `workdir` the image lacks makes every
  command fail. Create it (`WORKDIR /app`) or leave `workdir` out.
- **Network**: `no-network` is the honest setting for a leaderboard.
  Agents installed in the sandbox (`claude-code`, `codex`, `openhands`, ...)
  call their model from there: checking the evaluation refuses them with
  `no-network`, and with `allowlist` the model provider's host must be
  listed. `terminus-2` runs on your machine and works with any mode.
- **`env` is not secret**: keep a key out of the file by putting it in the
  workspace's `.env` and writing `env: {API_KEY: "${API_KEY}"}`; the run
  replaces the reference when it reads its config, and the run's copy keeps
  the reference. But the agent can still read every value in the sandbox
  (and, with a network, send it anywhere): put in no key you would not hand
  to the model.
- **MCP servers**: Harbor carries `stdio`, `sse` and `streamable_http` only,
  and no `headers`, stdio `env` or `cwd`; a stdio command runs inside the
  sandbox, so the image must have it.

## The dataset over it

```yaml
datasets:
  gsm8k-terminal:
    type: huggingface
    config:
      path: openai/gsm8k
      name: main
      split: test
      limit: 5
      input_template: '{"question": {{ question | tojson }}}'
      output_template: '{"answer": {{ answer.split(''####'')[-1] | replace('','', '''') | trim | tojson }}}'
      input_schema: {type: object, properties: {question: {type: string, description: The question to answer}}, required: [question]}
      output_schema: {type: object, properties: {answer: {type: string, description: The answer only}}, required: [answer]}
    harbor:
      environment: python              # required here: an environments.yaml key
      agent: terminus-2                # the default
      instructions: Solve the problem; you may use Python. Write the bare number only.
      answer_file: /app/answer.txt     # an absolute path; the agent is told to write its answer there
      timeout_sec: 300                 # per row; blank keeps Harbor's
    reward: exact-match                # scores `answer` against the output_template's reference
```

The `input_template` must render `{"question": ...}`. The prediction is the
agent's trajectory plus `answer`: the content of `answer_file`, else its last
message. The reward reads `answer` alone, so any reward of
openarena-add-reward works; set `answer_file` for string rewards
(`exact_match`), and have `instructions` ask for the bare answer. A Harbor
dataset cannot share an evaluation with other kinds of dataset: give it an
evaluation of its own (openarena-run-evaluation).

## Where it runs

Set on the machine (shell or the workspace's `.env`), never in the YAML:

- `ARENA_HARBOR_SANDBOX`: `e2b` by default, or `docker`, `daytona`, `modal`,
  ... A name Harbor does not know is refused, with the known ones.
- `ARENA_HARBOR_SANDBOX_OPTIONS`: that provider's own settings, a JSON object
  Harbor hands its environment (`{"namespace": "sandboxes"}` for `ack`, the
  Kubernetes one). Not JSON, or not an object: refused.
- `E2B_API_KEY`: E2B's key. When the configured provider fails its
  pre-flight (E2B with no key), trials **silently** run on the local Docker
  instead, with the local defaults below.
- Local Docker, for fields the entry leaves blank: `ARENA_HARBOR_LOCAL_CPUS`
  (2, at most the machine's), `ARENA_HARBOR_LOCAL_MEMORY_MB` (2048);
  `ARENA_HARBOR_LOCAL_TRIALS` trials at once (1).
- `ARENA_HARBOR_WORKDIR`: where task folders and trials go, when Docker
  cannot read the default (a snap Docker uses `~/arena-harbor/`).

The library CLI does not refuse a provider that cannot give the GPUs or
network mode asked for (only the arena app checks that before a run): E2B
has no GPUs, and on the local Docker both depend on the host. Such trials
fail, and their rows score 0.

## Steps

1. Write the environment; put what the Dockerfile copies under
   `environments/<name>/environment/`.
2. Write the dataset with `harbor.environment: <name>`, templates and reward
   (openarena-add-dataset for the templates), and an evaluation that runs
   it alone.
3. `openarena check configs` from the workspace (openarena-check-configs).
   It starts nothing, so it cannot see a missing image, a failing build, a
   missing `workdir` or `tmux`, or a provider's limits.
4. Build on one row first: `limit: 1`, `ARENA_HARBOR_SANDBOX=docker`, then
   `openarena evaluations run <evaluation> -v`, and read the trial before the
   full grid (openarena-read-results). Rows scoring 0, or a cell failing
   because no row got an answer, point at the sandbox, the image or a key,
   not the model.
