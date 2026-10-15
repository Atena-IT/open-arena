---
name: openarena-create-harbor-task
description: Use when an agentic task — something an agent must do in a terminal or a codebase, checked by a script — has to be written for openarena as a Harbor task folder (task.toml, instruction.md, environment/Dockerfile, tests/test.sh, solution/solve.sh), and evaluated through a `harbor_tasks` dataset scored by the task's own tests. Covers each file's format, the verifier's reward file, scoring a task with the arena's rewards, and wiring the folder into datasets and evaluations.
---

# openarena-create-harbor-task

A Harbor task is a folder: what the agent is told, the sandbox it works in,
the tests that score it, and a reference solution. openarena runs task
folders exactly as [Harbor](https://harborframework.com/docs/tasks) defines
them, one trial per task, the model under test driving a Harbor agent.
In depth: [Harbor tasks](https://atena-it.github.io/open-arena/guides/harbor-tasks/).

## The folder

```
tasks/<name>/                  the folder's name is the task's name
  task.toml                    the task's settings; a folder without one is not a task
  instruction.md               what the agent is told, and nothing else
  environment/Dockerfile       the sandbox, built from environment/ (its build context)
  tests/test.sh                the verifier: runs after the agent, writes the score
  solution/solve.sh            what a correct agent does
```

`tasks/` sits beside `configs/`. `uv run openarena init` writes a working
one, `tasks/hello-world/`: copy it and change it.

### task.toml

openarena runs it as written and adds nothing. On the local Docker a task
without `cpus` and `memory_mb` gets the whole machine (the 2 CPU / 2048 MB
defaults are for environment datasets only): always set them.

```toml
[metadata]
difficulty = "easy"            # free labels
category = "general"

[agent]
timeout_sec = 300.0            # how long the agent may work

[verifier]
timeout_sec = 60.0             # how long tests/test.sh may take; be generous

[environment]
build_timeout_sec = 600.0
cpus = 1
memory_mb = 1024
# storage_mb = 4096
# network_mode = "no-network"        # "public" by default; then only terminus-2 works (see below)
# docker_image = "python:3.12-slim"  # a prebuilt image instead of environment/Dockerfile
```

### instruction.md

Plain instructions, written for the agent: what to do, where the inputs
are, where the result must go. Name exact paths (`/app/report.json`) and
exact formats, and say what the tests check; the agent sees nothing else.

### environment/Dockerfile

```dockerfile
FROM python:3.12-slim
# terminus-2 (the default agent) works its terminal through tmux: install it.
RUN apt-get update && apt-get install -y --no-install-recommends tmux && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY data/ /app/data/          # files the task needs, from environment/data/
```

Without tmux in the image, terminus-2 installs it at the start of every
trial, which needs a package manager and the network, and fails without
them. Paths in `COPY` are relative to `environment/`. Install what the
tests need too (`python3`, `pytest`): they run in the same sandbox. Pin
versions. Do not put the tests or the solution in the image: Harbor brings
the tests in after the agent is done.

### tests/test.sh — the score

Harbor runs it in the sandbox after the agent. It must write the score:

- `/logs/verifier/reward.txt`: one number from 0 to 1, `1` solved, `0` not, fractions allowed (read as `reward`), or
- `/logs/verifier/reward.json`: several named numbers, `{"reward": 1, "style": 0.5}`. Every value a number; always write `reward`.

```bash
#!/bin/bash
mkdir -p /logs/verifier
if [ "$(cat /app/hello.txt 2>/dev/null)" = "Hello, world!" ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
```

A score outside 0–1 is clamped to it, `nan`/`inf` scores 0, each with a
warning in the run's log; no file or no `reward` key scores 0. Write `0`
first and overwrite it, so a crash still leaves a score. Test the outcome,
not the method; never fail on something the instruction did not ask for;
keep it deterministic (no network, no clock). A `pytest` or Python file can
sit beside it (`tests/check.py`) and `test.sh` run it.

### Scoring with the arena's rewards (optional)

When no script can check the result (a summary, an explanation), let the
arena's rewards score the agent's answer, a judge for example:

- `tests/rewards.yaml`: rewards written like the workspace's `rewards.yaml`,
  under `rewards:`, models named by their `models.yaml` key.
- `tests/expected.txt` (optional): the expected answer they compare with.
- `tests/test.sh` copies the answer to `/logs/verifier/answer.txt`; without
  that file the answer is the agent's last message.

```yaml
# tests/rewards.yaml (in the task folder, not in configs/)
rewards:
  judge:
    type: lm_as_judge
    config:
      language_model: gpt-4o-mini
      instructions: Score 1 when the summary names the root cause and the fix, else 0.
```

```bash
#!/bin/bash
mkdir -p /logs/verifier
cp /app/summary.md /logs/verifier/answer.txt 2>/dev/null
echo '{"answer_collected": 1}' > /logs/verifier/reward.json
```

Each reward becomes one named score; the first is also `reward` (unless one
is named `reward`) and replaces the verifier's. The dataset's reward stays
`harbor_verifier`. An invalid file, unknown reward type or unknown model
stops the run; one that fails while scoring gives `reward` 0, with why in
the trial's `error`.

### solution/solve.sh

The commands a correct agent would run; running it, then the tests, must
give 1, and doing nothing must give 0. openarena does not run it (its
agents do not include Harbor's `oracle`): check it with Harbor itself.

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
      # limit: 3                   # optional: at most this many tasks
      # repeat: 3                  # optional: each task this many times; the mean is the success rate
    harbor:
      agent: terminus-2            # terminus-2, claude-code, codex, gemini-cli, goose, opencode,
                                   # openhands, mini-swe-agent, swe-agent, cline-cli, cursor-cli
      timeout_sec: 600             # optional: per task, over task.toml's [agent] timeout_sec
    reward: harbor-tests
    metrics:                       # optional: reward.json sub-scores as extra columns
      - {class: harbor_verifier, alias: style, key: style}
rewards:
  harbor-tests:
    type: harbor_verifier          # reads what tests/test.sh wrote
    # config: {key: style}         # a reward.json key other than `reward`
```

No templates. Set only `agent` and `timeout_sec` in `harbor:`:
`environment` is refused (each task brings its own), and so are
`instructions` and `answer_file`, which are for environment datasets (what
the agent is told is each task's `instruction.md`; its tests read what it
left in the sandbox). A name in
`config.tasks` without a `task.toml` stops the run. A Harbor dataset needs
its own evaluation, naming the models that drive the agent
(openarena-run-evaluation). Pair agents and models with care:

- `claude-code` calls Anthropic models only, `codex` OpenAI, `gemini-cli`
  Google; the others take any provider.
- `terminus-2` runs on this machine and can drive a local `ollama/` or
  `vllm/` model; every other agent runs inside the sandbox, where a model
  on this machine's localhost is out of reach: give them hosted models.
- A `network_mode = "no-network"` task works with `terminus-2` only: the
  others call their model from inside the sandbox. `check` does not read a
  task's `task.toml`, so it will not catch this.

`check` and the run refuse a provider or local-model pairing that cannot
work, saying why.

## Where it runs

`ARENA_HARBOR_SANDBOX` names the sandbox provider (`e2b`, `docker`,
`daytona`, `modal`, ...), `e2b` by default; E2B needs `E2B_API_KEY`, and
without it trials fall back to this machine's Docker. With neither, the run
refuses to start. On the local Docker, trials run one at a time unless
`ARENA_HARBOR_LOCAL_TRIALS` says otherwise.
Each trial's logs land under the run's `harbor/trials/<trial>/`, one line per
trial in `harbor/trials.jsonl` (openarena-read-results). A failed trial is
not retried: it scores 0 (a cell where no trial answered fails).

## Steps

1. Copy `tasks/hello-world/` to `tasks/<name>/`; write the five files, with
   `cpus`, `memory_mb` and the timeouts in `task.toml`.
2. `chmod +x tasks/<name>/tests/test.sh tasks/<name>/solution/solve.sh`.
3. Add or reuse the `harbor_tasks` dataset and the `harbor_verifier` reward;
   `uv run openarena check configs`.
4. Run one task with one model (openarena-run-evaluation), read the trial,
   then scale up. After editing a task, rerun with
   `uv run openarena evaluations run <name> --no-cache`: finished cells are
   otherwise reused.
