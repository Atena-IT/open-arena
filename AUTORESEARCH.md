# autoresearch

This is an experiment to have the LLM do its own research — autonomously
developing **general reward functions** and validating them at scale on
the (model × dataset) sweep.

## What "good" looks like

The project's goal: a small library of *general* rewards
(`lm_as_judge`, `recursive_lm_as_judge`, `multi_judge_panel`, …) that are not
task-specific yet still rank models on a given dataset the same way
that dataset's *primary* (task-specific) reward does. If a general
reward agrees with the primary reward on **which model wins each
dataset**, it can be trusted to pick the best model for new datasets
that don't have a hand-written reward yet.

So the loss the agent minimizes is **disagreement between the candidate
general reward and the per-dataset primary reward**, across the
(model, dataset) matrix. Two concrete summary stats:

- **Best-model agreement** (0–1): fraction of datasets where
  `argmax_model(candidate)` == `argmax_model(primary)`. Higher is
  better.
- **Mean per-dataset Spearman** (-1 to 1): for each dataset, rank the
  models by candidate and by primary, take Spearman correlation,
  average across datasets. Higher is better.

Use best-model agreement as the headline; use Spearman as a tiebreaker
when agreement is saturated or the model list is short.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g.
   `apr30`). The branch `autoresearch/<tag>` must not already exist —
   this is a fresh run.
2. **Create the branch** from current master:
   ```bash
   git checkout master && git pull --ff-only
   git checkout -b autoresearch/<tag>
   ```
3. **Read the in-scope files**: The repo is small. Read these for full
   context:
   - `README.md` — repository overview.
   - `AGENTS.md` / `CLAUDE.md` — notes for AI coding agents (read this).
   - `src/evaluate.py` — sweep entrypoint and harness (oracle, trial
     loop, TSV / JSON output). Do not modify.
   - `src/program.py` — editable `build_program()` / `build_agent()`
     graph builders. Edit when changing the trial program shape.
   - `src/config.py` — Pydantic schema + validators for `config.yaml`.
     Do not modify unless changing the YAML surface.
   - `.open-arena/last_run.tsv` — long-form sweep output you read directly to
     score iterations (no `analyze.py` script — you compute whatever
     agreement / correlation / trade-off stats fit the change).
   - `prepare_data.py` — data-prep entrypoint. Do not modify
     autonomously; edits require explicit human approval first.
   - `src/datasets/` — dataset loaders. Do not modify.
   - `src/rewards/__init__.py` — the reward registry. Read it; the only
     edit allowed here is registering a new project-local reward you
     just added.
   - `src/rewards/recursive_language_model_reward.py`,
     `src/rewards/multi_judge_panel.py` — existing project-local rewards.
     **These are the files you iterate on**, plus any new
     `src/rewards/<name>.py` you add.
   - `REWARDS_BUILDING.md` — how-to for adding a new reward
     (base classes, the `LMAsJudgeProgram` pattern, registration,
     YAML wiring, common pitfalls). Read before writing any reward.
   - `config.example.yaml` — full menu of provider / reward options.
     Reference only.
   - `config.yaml` — read-mostly; the only edits allowed are inside
     the top-level `metrics:` block (wiring in / tuning the candidate
     reward — any reward identifier listed there is auto-wrapped in
     `MeanMetricWrapper` and rides the primary `evaluate()` pass).
     Do not touch the `datasets:` block or `experiments.language_models`
     / `experiments.datasets` — those define the validation harness and
     must stay fixed across the run.
4. **Verify data is reachable**: smoke-test that every dataset in
   `experiments.datasets` actually loads (HF cache at
   `~/.cache/huggingface/`, `.env` populated for any cloud provider
   used). One quick way:
   ```bash
   uv run python -c "
   import yaml
   from src.datasets import load_dataset_from_yaml
   cfg = yaml.safe_load(open('config.yaml'))
   for n in cfg['experiments']['datasets']:
       it = iter(load_dataset_from_yaml('config.yaml', name=n))
       next(it); print('ok', n)
   "
   ```
   If a provider needs auth and credentials are missing, tell the human.
5. **Initialize `results.tsv`** with the header row only:
   ```bash
   printf 'commit\tcandidate\ttop1\tpairwise\tsp_min\tsp_med\tsp_max\tn_usable\tstatus\tdescription\n' > results.tsv
   ```
6. **Confirm and go**.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs the full sweep on a single host. The runner is
the `arena` console script (installed by `uv sync`). Plain
`python -m src.evaluate` will fail with `ModuleNotFoundError:
keras_tuner` outside the project venv, so always invoke it via
`uv run arena` (or activate the venv first). Cap runtime by tightening
per-dataset `limit:` so
one full sweep finishes in **~5 minutes** wall clock. The grid (models × datasets) and the
per-dataset primary rewards are fixed for the duration of the run —
that's the validation harness; changing it would invalidate the
comparison against earlier results.

**What you CAN do:**
- Modify existing project-local rewards under `src/rewards/`
  (`recursive_language_model_reward.py`, `multi_judge_panel.py`).
- Add new project-local rewards as `src/rewards/<name>.py` and register
  them in `src/rewards/__init__.py:_LOCAL_REWARDS`.
- Edit the top-level `metrics:` block in `config.yaml`: which candidate
  rewards to score, their hyperparameters (judge model, panel members,
  agreement threshold, instructions, max_iterations, max_llm_calls,
  in_mask / out_mask, etc.), and the `alias:` used as the column
  header. Reward identifiers there are auto-wrapped in
  `synalinks.metrics.MeanMetricWrapper` so they ride the primary
  `evaluate()` pass — no extra LM calls per "candidate reward".

**What you CANNOT do:**
- Modify `src/evaluate.py`, `src/config.py`, or `src/datasets/`. The
  harness is fixed.
- Modify `prepare_data.py` autonomously. It is editable, but only with
  prior agreement from the human — pause the loop, propose the change,
  and wait for explicit approval before touching it.
- Modify the `datasets:` block in `config.yaml` or the per-dataset
  primary `reward:` entries. Those are ground truth.
- Modify `experiments.language_models` or `experiments.datasets`.
  Changing the matrix mid-run breaks comparison with prior results.
- Modify `config.example.yaml` (reference only).
- Install new packages or add dependencies. Use only what's in
  `pyproject.toml`.
- Modify upstream synalinks built-ins (the reward bases, judge
  programs, etc.). Subclass and add locally instead.

**VRAM / cost** is a soft constraint. Candidate rewards listed under
top-level `metrics:` are auto-wrapped in `MeanMetricWrapper` and ride
the primary `evaluate()` pass — *no* extra model calls per candidate.
The cost worry is only inside the candidate reward itself (e.g.
`recursive_lm_as_judge` makes many LM calls per example to score that
one cell); against a long matrix that still adds up. If you're using
cloud LMs through litellm, watch token spend.

**Cache invalidation**: whenever the HP space changes (you add or rename
a candidate reward under `metrics:`, change its alias, change the model
list, or change the dataset list) the on-disk trial state in
`.open-arena/<dataset>/` is stale. Either pass `--no-cache` or
`rm -rf .open-arena/<dataset>` before the run. Don't delete `.open-arena/`
casually otherwise — completed trials are reused on resume.

**Simplicity criterion**: All else being equal, simpler is better. A
reward that lifts agreement by 0.01 but adds 200 lines of
hand-engineered prompt scaffolding probably isn't worth it. A simpler
reward that matches a complex one is a clear win. Removing knobs and
keeping the score is a great outcome.

**The first run**: your very first run establishes the baseline. Wire
in one or two cheap candidate rewards (e.g. a plain `lm_as_judge` with
default instructions) under the top-level `metrics:` block and run the
sweep unmodified. Compute baseline best-model agreement and mean
Spearman.

## Output format

`uv run arena` writes the sweep to `.open-arena/last_run.tsv` — long format, one
row per `(model, dataset, metric, value)` cell, plus a `direction`
column for the metric's optimization direction:

```
model	dataset	metric	value	direction
ollama/mistral	mmlu_test	reward	0.440000	max
ollama/mistral	mmlu_test	panel_judge	0.612000	max
ollama/llama3.2	mmlu_test	reward	0.520000	max
...
```

For multi-objective datasets, the Pareto frontier is printed to stdout
as a markdown table and written to `.open-arena/frontier.tsv`. Pass `--json
<path>` to additionally dump the full result (meta + rows) as JSON, or
`--json -` to emit JSON on stdout and skip the TSV / markdown entirely.

Pivot the TSV to matrices in your head (or `python -c`) and compute:

- per-dataset argmax for `reward` (primary) and for the candidate
  alias; the agreement is `(matches / num_datasets)`,
- per-dataset rank correlation (Spearman) between the two columns;
  average across datasets.

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT
comma-separated — commas break in descriptions and cells).

The TSV has a header row and 10 columns:

```
commit    candidate    top1    pairwise    sp_min    sp_med    sp_max    n_usable    status    description
```

1. git commit hash (short, 7 chars).
2. candidate alias being evaluated (the `alias:` from the top-level
   `metrics:` entry, e.g. `panel_judge`). If the run scored multiple
   candidates, log one row per candidate with the same commit.
3. **top1** — fraction of usable datasets where the candidate's #1
   model matches the primary's #1, in [0,1], 6 decimals. `0.000000`
   for crashes.
4. **pairwise** — Kendall-style pairwise agreement averaged across
   datasets (fraction of model pairs where primary and candidate agree
   on which is stronger; ties on the same side count as agreement,
   ties on only one side count as 0.5). In [0,1], 6 decimals. With 2
   models pairwise == top1; with 3+ models pairwise is the smoother
   signal. `0.000000` for crashes.
5. **sp_min** — minimum per-dataset Spearman ρ vs primary. Signed,
   6 decimals. The "worst case" datapoint — a strongly negative value
   means the candidate actively inverts ranking on at least one
   dataset, even if other stats look fine. `0.000000` for crashes.
6. **sp_med** — median per-dataset Spearman ρ vs primary. Signed,
   6 decimals. Robust to single-dataset outliers (which an arithmetic
   mean would smooth away). `0.000000` for crashes.
7. **sp_max** — maximum per-dataset Spearman ρ vs primary. Signed,
   6 decimals. `0.000000` for crashes.
8. **n_usable** — integer count of datasets contributing to the stats
   (those with both primary and candidate scores for the same model
   set). `0` for crashes. Lets you see at a glance whether failed
   trials shrank the comparison.
9. status: `keep`, `discard`, or `crash`.
10. short text description of what this experiment tried.

Example:

```
commit    candidate    agreement    spearman    status    description
a1b2c3d    lm_judge    0.500000    0.612000    keep    baseline lm_as_judge with default instructions
b2c3d4e    lm_judge    0.750000    0.781000    keep    sharpen judge instructions, in_mask=[content]
c3d4e5f    panel_judge    0.750000    0.793000    keep    3-judge panel with mistral/llama3.2/qwen, threshold 0.2
d4e5f6g    panel_judge    0.500000    0.611000    discard    raise threshold to 0.5 (smart-LM rarely fires)
e5f6g7h    rlm_judge    0.000000    0.000000    crash    add code-tool reward (timeout > 10 min)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/apr30` or
`autoresearch/apr30-host0`).

LOOP FOREVER:

1. **Check git state**:
   ```bash
   git status
   git log --oneline -5
   ```

2. **Pick an experimental idea and apply it.** Examples:
   - tweak instructions inside an existing project-local reward,
   - swap the judge LM,
   - change panel composition or `agreement_threshold`,
   - bump `max_iterations` / `max_llm_calls` on the recursive judge,
   - add a new project-local reward in `src/rewards/<name>.py` and
     register it in `src/rewards/__init__.py:_LOCAL_REWARDS`,
   - re-wire the top-level `metrics:` block in `config.yaml` to score
     the candidate (reward identifiers there are auto-wrapped in
     `MeanMetricWrapper`).

3. **Commit the change**:
   ```bash
   git add -A && git commit -m "<short description>"
   ```

4. **Clear the stale tuner cache** when the metric set changed (added
   / renamed / removed a candidate reward under `metrics:`). Either
   pass `--no-cache` to the sweep, or delete the per-dataset caches:
   ```bash
   rm -rf .open-arena/*/
   ```

5. **Run the sweep** — redirect everything; do NOT use `tee` or let
   raw output flood your context:
   ```bash
   uv run arena > .open-arena/run.log 2>&1
   ```

6. **Score the run.** Read `.open-arena/last_run.tsv` directly — it's the
   long-form contract (`model<TAB>dataset<TAB>metric<TAB>value<TAB>direction`).
   Compute whatever lens fits the change you made: per-candidate
   argmax-model agreement with the primary `reward` column,
   per-dataset Spearman ρ, primary-reward deltas per dataset, cost
   or token trade-offs, disagreement breakdowns, etc. There is no
   prescribed scoring script — you choose the statistic that
   actually tests the hypothesis behind the iteration.

   For multi-objective datasets, `.open-arena/frontier.tsv` lists the
   on-Pareto-frontier models per dataset.

   The TSV is small and human-readable — `cat`ing it into your
   reasoning is often enough; pivot with a quick Python snippet
   when you need real stats (Spearman ρ with tie handling, etc.).

7. **Crash check.** If `.open-arena/last_run.tsv` is missing or empty, the run
   crashed:
   ```bash
   test -s .open-arena/last_run.tsv || tail -n 80 .open-arena/run.log
   ```
   If the failure is a dumb fix (typo, YAML indent, missing import,
   missing env var) fix and re-run. If the idea is fundamentally
   broken, log `crash` and revert (step 10).

8. **Record the iteration.** Append a row to `results.tsv`
   summarizing what you changed and how it scored. Schema is your
   call — pick columns that let *future you* compare iterations
   without re-running. Typical shape:
   ```bash
   COMMIT=$(git rev-parse --short HEAD)
   printf '%s\t%s\t%s\t%s\n' \
     "$COMMIT" "<headline-stat>" "keep" "<one-line description>" >> results.tsv
   ```
   For crashes, use `crash` as the outcome and `0` (or `N/A`) for
   the headline stat. Do NOT `git add` `results.tsv` — it stays
   untracked across iterations as a local research log.

9. **If the change improved the sweep**, advance the branch — keep
   the commit, do nothing. "Improved" is a judgment call from the
   TSV: did the primary `reward` go up where it matters? Did costs
   not regress badly? Did rankings shift sensibly? Lean
   conservative — noise-level improvements should revert.

10. **If equal or worse**, revert:
    ```bash
    git reset --hard HEAD~1
    ```

You are a completely autonomous researcher trying things out. If they
work, keep. If not, discard. Advancing the branch is how you compound
gains.

**Timeout**: each experiment should take ~5 minutes total (+ a few
seconds for startup). If a run exceeds 10 minutes, kill it, treat it
as a failure, and lower the dataset `limit:` or drop a heavy candidate
from the top-level `metrics:` block. To kill a runaway sweep:
```bash
pkill -f 'src.evaluate'
```

**Crashes**: use judgment as above.

**NEVER STOP**: once the experiment loop has begun (after the initial
setup), do NOT pause to ask the human if you should continue. Do NOT
ask "should I keep going?" or "is this a good stopping point?". The
human might be asleep, or away, and expects you to continue
*indefinitely* until manually stopped. You are autonomous. If you run
out of ideas, think harder — re-read `config.example.yaml` for
unexplored knobs (mask config, schema toggles, embedding-based
rewards), revisit existing project-local rewards you haven't touched,
combine previous near-misses, try more radical reward designs (e.g. a
panel that vetoes on disagreement instead of escalating). The loop
runs until the human interrupts you, period.

As an example use case, a user might leave you running while they
sleep. ~5 minutes per run = ~12/hour = ~100 over a typical sleep —
they wake up to a fully populated `results.tsv` and a branch full of
reward iterations.
