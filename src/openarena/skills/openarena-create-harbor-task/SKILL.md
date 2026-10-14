---
name: openarena-create-harbor-task
description: Use when an agentic task — something an agent must do in a terminal or a codebase, checked by a script — has to be written for openarena as a Harbor task folder (task.toml, instruction.md, environment/Dockerfile, tests/test.sh, solution/solve.sh), and evaluated through a `harbor_tasks` dataset scored by the task's own tests. Covers each file's format, the verifier's reward file, and wiring the folder into datasets and evaluations.
---

# openarena-create-harbor-task

A Harbor task is a folder: what the agent is told, the sandbox it works in,
the tests that score it, and a reference solution. openarena runs task
folders exactly as [Harbor](https://harborframework.com/docs/tasks) defines
them, one trial per task, the model under test driving a Harbor agent.

## The folder

```
tasks/<name>/                  <name>: letters, digits, _ and -, starting with a letter or digit
  task.toml                    the task's settings
  instruction.md               what the agent is told, and nothing else
  environment/Dockerfile       the sandbox, built from environment/ (its build context)
  tests/test.sh                the verifier: runs after the agent, writes the score
  solution/solve.sh            what a correct agent does (Harbor's `oracle` agent runs it)
```

`tasks/` sits beside `configs/`. `uv run openarena init` writes a working
one, `tasks/hello-world/`: copy it and change it.

### task.toml

```toml
[metadata]
difficulty = "easy"            # free labels
category = "general"

[agent]
timeout_sec = 300.0            # how long the agent may work

[verifier]
timeout_sec = 60.0             # how long tests/test.sh may take

[environment]
build_timeout_sec = 600.0
cpus = 1
memory_mb = 1024
# storage_mb = 4096
# network_mode = "no-network"        # "public" by default; cut the network when the task should not need it
# docker_image = "python:3.12-slim"  # a prebuilt image instead of environment/Dockerfile
```

### instruction.md

Plain instructions, written for the agent: what to do, where the inputs
are, where the result must go. Name exact paths (`/app/report.json`) and
exact formats; the tests check those.

### environment/Dockerfile

```dockerfile
FROM python:3.12-slim
# terminus-2 (the default agent) works its terminal through tmux: install it.
RUN apt-get update && apt-get install -y --no-install-recommends tmux && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY data/ /app/data/          # files the task needs, from environment/data/
```

Paths in `COPY` are relative to `environment/`. Do not put the tests or the
solution in the image: Harbor brings them in after the agent is done.

### tests/test.sh — the score

Harbor runs it in the sandbox after the agent. It must write the score:

- `/logs/verifier/reward.txt`: one number from 0 to 1, `1` solved, `0` not, fractions allowed, or
- `/logs/verifier/reward.json`: several named scores, `{"reward": 1, "style": 0.5}`.

```bash
#!/bin/bash
mkdir -p /logs/verifier
if [ "$(cat /app/hello.txt 2>/dev/null)" = "Hello, world!" ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
```

A score outside 0–1 is clamped to it, and one that is not a number scores 0, each with a
warning in the run's log. Test the outcome, not the method; never fail on something the instruction
did not ask for. A `pytest` file can sit beside it (`tests/test_outputs.py`)
and `test.sh` run it, then write the score from its exit code.

### solution/solve.sh

The commands a correct agent would run. Harbor's `oracle` agent runs it, and
`test.sh` must then give 1: that is the proof the task is solvable and the
tests are right.

Both scripts must be executable: `chmod +x tests/test.sh solution/solve.sh`.

## Evaluate the tasks

A `harbor_tasks` dataset is a folder of tasks, one row each:

```yaml
datasets:
  my-tasks:
    type: harbor_tasks
    config:
      path: tasks                  # relative to configs/'s parent; the run copies it beside its config
      tasks: [hello-world]         # optional: only these folders; none runs every task under path
    harbor:
      agent: terminus-2            # terminus-2, claude-code, codex, gemini-cli, goose, opencode,
                                   # openhands, mini-swe-agent, swe-agent, cline-cli, cursor-cli
      timeout_sec: 600             # optional: per task, over task.toml's
    reward: harbor-tests
rewards:
  harbor-tests:
    type: harbor_verifier          # reads what tests/test.sh wrote
    # config: {key: style}         # a reward.json key other than `reward`
```

No templates, no `harbor.environment`: each task brings its own. Then an
evaluation naming the dataset and the models that drive the agent
(openarena-run-evaluation). Pair agents and models with care:

- `claude-code` calls Anthropic models only, `codex` OpenAI, `gemini-cli`
  Google; the others take any provider.
- `terminus-2` runs on this machine and can drive a local `ollama/` or
  `vllm/` model; every other agent runs inside the sandbox, where a model
  on this machine's localhost is out of reach: give them hosted models.

`check` and the run refuse a pairing that cannot work, saying why.

## Where it runs

`ARENA_HARBOR_SANDBOX` names the sandbox provider (`e2b`, `docker`,
`daytona`, `modal`, ...), `e2b` by default; E2B needs `E2B_API_KEY`, and
without it trials fall back to this machine's Docker. With neither, the run
refuses to start.
Each trial's logs land under the run's `harbor/trials/<trial>/`, one line per
trial in `harbor/trials.jsonl` (openarena-read-results).

## Steps

1. Copy `tasks/hello-world/` to `tasks/<name>/`; write the five files.
2. `chmod +x tasks/<name>/tests/test.sh tasks/<name>/solution/solve.sh`.
3. Add or reuse the `harbor_tasks` dataset and the `harbor_verifier` reward;
   `uv run openarena check my-arena/configs`.
4. Run a small evaluation (openarena-run-evaluation) and read the trials.
