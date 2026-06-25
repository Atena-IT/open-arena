---
name: arena-autoresearch
description: Start and operate the autonomous reward-R&D experiment loop (setup, confirm, then run until interrupted).
---

Read `AUTORESEARCH.md` end-to-end before starting. Read `AGENTS.md` / `CLAUDE.md` for the trigger phrases. This skill summarizes the protocol; the source doc is authoritative.

## Trigger phrases

Any of these (or an obvious paraphrase) kicks off setup:
- "start the research loop"
- "begin autoresearch"
- "run autoresearch"
- "kick off the autoresearch loop"

## Setup (do with the user before running anything)

1. **Agree on a run tag** — propose `<monthday>` (e.g. `jun23`). The branch `autoresearch/<tag>` must not already exist.

2. **Create the branch**:
   ```bash
   git checkout master && git pull --ff-only
   git checkout -b autoresearch/<tag>
   ```

3. **Read in-scope files** — do NOT skip this step:
   - `README.md`, `AGENTS.md` / `CLAUDE.md`
   - `src/evaluate.py` (harness — do not modify)
   - `src/program.py` (editable program graph)
   - `src/config.py` (Pydantic schema — do not modify)
   - `src/rewards/__init__.py`, `src/rewards/multi_judge_panel.py`, `src/rewards/rlm_as_judge.py`
   - `REWARDS_BUILDING.md`
   - `config.yaml` (read-mostly; only `metrics:` block is editable)
   - `config.example.yaml` (reference only)

4. **Smoke-test datasets**:
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
   If credentials are missing for a cloud provider, report to the human.

5. **Initialize `results.tsv`**:
   ```bash
   printf 'commit\tcandidate\ttop1\tpairwise\tsp_min\tsp_med\tsp_max\tn_usable\tstatus\tdescription\n' > results.tsv
   ```

6. **Confirm and go** — wait for explicit human confirmation, then enter the loop.

## The experiment loop (NEVER STOP once confirmed)

Loop indefinitely:

1. `git status && git log --oneline -5`

2. **Pick an idea and apply it**:
   - Tweak instructions in an existing project-local reward
   - Swap the judge LM
   - Change panel composition or `agreement_threshold`
   - Bump `max_iterations` / `max_llm_calls` on the recursive judge
   - Add a new reward in `src/rewards/<name>.py` and register in `src/rewards/__init__.py:_LOCAL_REWARDS`
   - Re-wire the top-level `metrics:` block in `config.yaml`

3. `git add -A && git commit -m "<short description>"`

4. **Clear stale cache** when the metric set changed:
   ```bash
   rm -rf .open-arena/*/
   ```

5. **Run the sweep** (redirect everything):
   ```bash
   uv run arena > .open-arena/run.log 2>&1
   ```
   Cap dataset `limit:` so one full sweep finishes in ~5 minutes. If a run exceeds 10 minutes, kill it: `pkill -f 'src.evaluate'`

6. **Score the run** by reading `.open-arena/last_run.tsv` directly:
   - Long-format: `model<TAB>dataset<TAB>metric<TAB>value<TAB>direction`
   - Compute per-dataset argmax agreement (primary `reward` vs candidate alias)
   - Compute per-dataset Spearman ρ, average across datasets
   - For multi-objective datasets, check `.open-arena/frontier.tsv`

7. **Crash check**:
   ```bash
   test -s .open-arena/last_run.tsv || tail -n 80 .open-arena/run.log
   ```

8. **Log the result** to `results.tsv` (tab-separated, 10 columns):
   ```
   commit  candidate  top1  pairwise  sp_min  sp_med  sp_max  n_usable  status  description
   ```
   `status` is `keep`, `discard`, or `crash`. Do NOT `git add results.tsv` — it stays untracked.
   ```bash
   COMMIT=$(git rev-parse --short HEAD)
   printf '%s\t%s\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%d\t%s\t%s\n' \
     "$COMMIT" "alias" top1 pairwise sp_min sp_med sp_max n_usable "keep" "description" >> results.tsv
   ```

9. **If improved** — keep the commit, advance the branch.

10. **If equal or worse** — revert:
    ```bash
    git reset --hard HEAD~1
    ```

## What you CAN do

- Modify `src/rewards/multi_judge_panel.py`, `src/rewards/rlm_as_judge.py`
- Add `src/rewards/<name>.py` and register in `src/rewards/__init__.py:_LOCAL_REWARDS`
- Edit the top-level `metrics:` block in `config.yaml`

## What you CANNOT do

- Modify `src/evaluate.py`, `src/config.py`, or `src/datasets/`
- Modify `prepare_data.py` autonomously (requires human approval)
- Modify `datasets:` block or `experiments.language_models` / `experiments.datasets` in `config.yaml`
- Modify `config.example.yaml`
- Install new packages

## NEVER STOP rule

Once the loop has begun (after human confirmation), do NOT pause to ask "should I keep going?". Run autonomously until the human interrupts you. If you run out of ideas, re-read `config.example.yaml` for unexplored knobs (mask config, schema toggles, embedding-based rewards), revisit existing rewards you haven't touched, or try more radical reward designs.
