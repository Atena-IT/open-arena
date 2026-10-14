# Environments

An environment is a sandbox definition. A dataset whose `harbor:` block names
one turns each of its rows into a [Harbor](https://harborframework.com) task in
that sandbox: the row's question is the task's instruction, the model under
test drives an agent that works there, and the agent's answer is scored by the
dataset's own reward.

Use this when ordinary questions should be **worked**, with a terminal, files
and tools, rather than answered in one call. When each task needs its own
files and tests, write [Harbor tasks](harbor-tasks.md) instead.

```yaml
environments:
  python:
    description: Python 3.12 with tmux.
    dockerfile: |                      # or docker_image: python:3.12-slim
      FROM python:3.12-slim
      RUN apt-get update && apt-get install -y --no-install-recommends tmux && rm -rf /var/lib/apt/lists/*
      WORKDIR /app
      COPY requirements.txt /tmp/requirements.txt
      RUN pip install --no-cache-dir -r /tmp/requirements.txt
    workdir: /app
    cpus: 1
    memory_mb: 2048
    network_mode: public
```

## Fields

| field | |
|---|---|
| `docker_image` or `dockerfile` | a prebuilt image, or the Dockerfile built from the build context |
| `workdir` | where the agent starts; the image must have it |
| `cpus`, `memory_mb`, `storage_mb`, `gpus`, `gpu_types` | resources; leave them out for the provider's default |
| `network_mode` | `public` (the default), `no-network`, or `allowlist` with `allowed_hosts` |
| `env` | environment variables in the container |
| `build_timeout_sec` | how long building the image may take (default 600) |
| `healthcheck` | `command`, `interval_sec`, `timeout_sec`, `retries`, ...: the sandbox is ready when it exits 0 |
| `mcp_servers` | MCP servers the agent is given ([MCP tools](mcp-tools.md#tools-in-a-harbor-sandbox)) |

The **build context** is `environments/<name>/environment/` beside `configs/`:
whatever the Dockerfile copies lives there. The run copies the folder beside
its own config.

!!! warning "tmux"
    `terminus-2`, the default agent, works its terminal through `tmux`. An image
    without it fails every trial.

## A dataset over an environment

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
    harbor:
      environment: python              # an environments.yaml name
      agent: terminus-2
      instructions: Solve the problem; you may use Python.
      answer_file: /app/answer.txt     # the agent is asked to write its answer here
      timeout_sec: 300                 # per row
    reward: exact-match
```

| `harbor:` field | |
|---|---|
| `environment` | required here: the sandbox every row runs in |
| `agent` | the Harbor agent the model drives ([Harbor tasks](harbor-tasks.md#agents-and-models)) |
| `instructions` | said to the agent before every row's question |
| `answer_file` | an absolute path; the agent is told to write its final answer there |
| `timeout_sec` | how long the agent may work on one row |

The prediction is the agent's whole trajectory plus `answer`: the content of
`answer_file`, else the agent's last message. The reward reads `answer`, so any
reward works; for `exact_match`, have the instructions ask for the bare answer.

## Where it runs

`ARENA_HARBOR_SANDBOX` names the sandbox provider (`e2b`, `docker`, `daytona`,
`modal`, ...), `e2b` by default. E2B needs `E2B_API_KEY`; without it, trials
fall back to the local Docker. A provider that cannot give what the
environment asks for (GPUs, no network) is refused before the run starts.
