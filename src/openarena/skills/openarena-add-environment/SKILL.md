---
name: openarena-add-environment
description: Use when the rows of an ordinary openarena dataset (questions from Hugging Face, a file, ...) should be worked by an agent inside a sandbox rather than answered in one model call — defining a Harbor environment in environments.yaml (image or Dockerfile, its build context folder, resources, network, MCP tools) and a `harbor:` dataset over it whose answer file the dataset's reward scores.
---

# openarena-add-environment

An environment is a sandbox definition. A dataset with a `harbor:` block
naming one turns each of its rows into a Harbor task in that sandbox: the
row's question (after `instructions`) is the task's instruction, the model
under test drives the agent, and the agent's answer is scored by the
dataset's own reward. Use task folders instead (openarena-create-harbor-task)
when each task needs its own files and tests.

## The environment

```yaml
environments:
  python:
    description: Python 3.12 with tmux.
    tags: [python]
    dockerfile: |                      # or `docker_image: python:3.12-slim`, one of the two
      FROM python:3.12-slim
      RUN apt-get update && apt-get install -y --no-install-recommends tmux && rm -rf /var/lib/apt/lists/*
      WORKDIR /app
      COPY requirements.txt /tmp/requirements.txt
      RUN pip install --no-cache-dir -r /tmp/requirements.txt
    workdir: /app                      # where the agent starts; the image must have it
    cpus: 1                            # resources; leave out for the provider's default
    memory_mb: 2048
    # storage_mb: 4096
    # gpus: 1
    # gpu_types: [H100]
    network_mode: public               # public | no-network | allowlist
    # allowed_hosts: [pypi.org]        # with allowlist
    # env: {PYTHONUNBUFFERED: "1"}
    # build_timeout_sec: 600
    # healthcheck: {command: "test -f /app/ready", interval_sec: 5, retries: 3}
    # mcp_servers: [fetch]             # mcp_servers.yaml keys handed to the agent
```

Its **build context** is `environments/<name>/environment/` beside
`configs/`: whatever the Dockerfile `COPY`s lives there
(`environments/python/environment/requirements.txt` above). The run copies
the folder beside its config. `terminus-2`, the default agent, needs `tmux`
in the image; an image without it fails every trial.

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
      agent: terminus-2
      instructions: Solve the problem; you may use Python.   # said before every row's question
      answer_file: /app/answer.txt     # an absolute path; the agent is told to write its answer there
      timeout_sec: 300                 # per row
    reward: exact-match                # scores `answer` against the output_template's reference
```

The prediction is the agent's trajectory plus `answer`: the content of
`answer_file`, else its last message. The reward reads `answer` alone, so
any reward of openarena-add-reward works; `exact_match` wants the file to hold
the bare answer, which `instructions` should say.

## Steps

1. Write the environment; put what the Dockerfile copies under
   `environments/<name>/environment/`.
2. Write the dataset with `harbor.environment: <name>`, templates and reward
   (openarena-add-dataset for the templates).
3. `uv run openarena check my-arena/configs`.
4. A run needs `E2B_API_KEY` (E2B, the default `ARENA_HARBOR_SANDBOX`) or a
   local Docker, which it falls back to (openarena-run-evaluation); a
   provider that cannot give what the environment asks (GPUs, no network)
   is refused at launch; read the trials when rows score 0
   (openarena-read-results).
