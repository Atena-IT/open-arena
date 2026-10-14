# Harbor tasks

A [Harbor](https://harborframework.com/docs/tasks) task is a folder: what the
agent is told, the sandbox it works in, the tests that score it, and a
reference solution. openarena runs task folders exactly as Harbor defines them,
one trial per task, with the model under test driving a Harbor agent.

```
tasks/<name>/
  task.toml                    the task's settings
  instruction.md               what the agent is told
  environment/Dockerfile       the sandbox, built from environment/
  tests/test.sh                runs after the agent, writes the score
  solution/solve.sh            what a correct agent does
```

`tasks/` sits beside `configs/`; `openarena init` writes a working one,
`tasks/hello-world/`. A task's name uses letters, digits, `_` and `-`.

## The files

### task.toml

```toml
[metadata]
difficulty = "easy"
category = "general"

[agent]
timeout_sec = 300.0            # how long the agent may work

[verifier]
timeout_sec = 60.0             # how long tests/test.sh may take

[environment]
build_timeout_sec = 600.0
cpus = 1
memory_mb = 1024
# network_mode = "no-network"
# docker_image = "python:3.12-slim"  # a prebuilt image instead of environment/Dockerfile
```

### instruction.md

Plain instructions for the agent: what to do, where the inputs are, where the
result must go. Name exact paths and formats: the tests check them.

### environment/Dockerfile

```dockerfile
FROM python:3.12-slim
# terminus-2 works its terminal through tmux.
RUN apt-get update && apt-get install -y --no-install-recommends tmux && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY data/ /app/data/
```

`COPY` paths are relative to `environment/`. The tests and the solution do not
go in the image: Harbor brings them in after the agent is done.

### tests/test.sh

Harbor's verifier runs it in the sandbox after the agent. It writes the score:

- `/logs/verifier/reward.txt`: one number, `1` solved, `0` not (fractions allowed);
- or `/logs/verifier/reward.json`: several named scores, `{"reward": 1, "style": 0.5}`.

```bash
#!/bin/bash
mkdir -p /logs/verifier
if [ "$(cat /app/hello.txt 2>/dev/null)" = "Hello, world!" ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
```

Test the outcome, not the method.

### solution/solve.sh

What a correct agent does. Harbor's `oracle` agent runs it, and the tests must
then give 1: the proof that the task can be solved and the tests are right.

Both scripts must be executable: `chmod +x tests/test.sh solution/solve.sh`.

## A dataset of tasks

```yaml
datasets:
  my-tasks:
    type: harbor_tasks
    config:
      path: tasks                  # relative to the workspace; the run copies it
      tasks: [hello-world]         # only these folders; none runs every task under path
    harbor:
      agent: terminus-2
      timeout_sec: 600             # per task, over task.toml's
    reward: harbor-tests
rewards:
  harbor-tests:
    type: harbor_verifier          # reads what tests/test.sh wrote
    # config: {key: style}         # a reward.json key other than `reward`
```

No templates and no `harbor.environment`: each task brings its own.

## Agents and models

| agent | runs | calls |
|---|---|---|
| `terminus-2` (the default) | on your machine | any provider |
| `claude-code` | in the sandbox | Anthropic models only |
| `codex` | in the sandbox | OpenAI models only |
| `gemini-cli` | in the sandbox | Google models only |
| `goose`, `opencode`, `openhands`, `mini-swe-agent`, `swe-agent`, `cline-cli`, `cursor-cli` | in the sandbox | any provider |

An agent running in the sandbox cannot reach a model on your machine's
`localhost` (`ollama/...`, `vllm/...`): give it a hosted model. `check` and
the run refuse a pairing that cannot work, and say why.

## Trials

Each trial's logs land under the run's `harbor/trials/<trial>/`, with one line
per trial in `harbor/trials.jsonl`. See [Reading results](results.md#harbor-runs).
