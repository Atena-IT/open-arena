# Open Arena Evaluation GUI Design

## Goal
Build a polished, realistic GUI for the Open Arena show-me-how flow so a user can configure a benchmark run, choose showcase models, launch evaluations, watch the ranking evolve during the run, and inspect the final results without leaving the browser.

The GUI must stay faithful to the notebook story:

**dataset → experiments → evaluation → Langfuse → summary**

The first version should be semi-real rather than fully mocked: it must reuse the existing demo CSV, YAML configs, execution engine, and Langfuse integration, while allowing lightweight visual synthesis for event narration and chart animation where the backend does not already expose incremental events.

## Constraints
- Keep the GUI aligned with the show-me-how notebook in `demo/show_me_how_open_arena/open_arena_show_me_how.ipynb`.
- Reuse the existing demo dataset and YAMLs instead of inventing a parallel benchmark definition format.
- Prefill model choices from the showcase and runnable configs.
- Use a visually strong frontend stack with Tailwind-based components.
- Be honest about liveness: show partial results as partial, and do not fabricate scores or imply token streaming that does not exist.
- Support a simple custom `llm_as_judge` setup in the UI, including prompts and score name.
- Treat Langfuse as the source for traces and final inspectability.
- Avoid coupling the GUI to Jupyter execution; the notebook explains the flow, but the GUI should use the underlying project code directly.

## Existing Inputs
- Notebook narrative: `demo/show_me_how_open_arena/open_arena_show_me_how.ipynb`
- Showcase config: `demo/show_me_how_open_arena/configs/business_qa_showcase.yaml`
- Runnable config: `demo/show_me_how_open_arena/configs/business_qa_runnable.yaml`
- Demo dataset: `demo/show_me_how_open_arena/data/business_qa_demo.csv`
- CLI orchestration reference: `src/main_cli.py`
- Dataset upload/attach logic: `src/datasets/langfuse_upload.py`
- Concurrent experiment execution: `src/execution/executor.py`
- Concurrent evaluation primitives: `src/evaluation/base.py`
- Judge evaluator behavior: `src/evaluation/evaluators/llm_as_judge.py`
- Config schema: `src/config/types.py`

## Recommended Architecture
Use a split application:

1. **Frontend:** React + Vite + TypeScript + Tailwind + shadcn/ui
2. **Backend:** FastAPI service under `demo/gui/`
3. **Runtime source of truth:** existing CSV, YAML, Open Arena Python modules, and Langfuse

This is the right trade-off for the requested MVP:
- React + Tailwind gives a polished, demo-ready interface.
- FastAPI keeps the runtime close to the existing Python orchestration code.
- The backend can expose a clean GUI-oriented API without forcing the frontend to know about Langfuse or project internals.

## Scope Boundaries
### In scope
- A four-step GUI flow: configuration, evaluation setup, live run, final results
- Prefilled dataset/model configuration from the existing demo assets
- Real run launch via Python backend orchestration
- Real Langfuse-backed final summary and trace links
- Real concurrency for experiment execution
- Real concurrency for evaluation where the evaluator type supports it
- A simple custom judge experience for `llm_as_judge`
- Leaderboard and charts that update from real snapshots during the run

### Out of scope for v1
- Replacing the notebook as the authoritative written walkthrough
- Editing YAML files in place from the browser
- Full workflow designer for arbitrary benchmark topologies
- Arbitrary plugin metrics beyond the current evaluator model
- Token-level streaming or agent-thought streaming
- Cross-metric composite scores that hide how individual metrics behave

## User Experience
The GUI should feel like a compact control room for the notebook demo rather than a generic admin dashboard.

### Step 1: Configuration
Purpose: mirror the notebook setup phase and let the user confirm what will run.

Content:
- dataset card showing the CSV path, row count, sample limit, and runtime dataset name
- model selection grid with showcase model labels and runtime backend labels
- hero mission preview using the first dataset row
- environment readiness badges for Langfuse and provider keys

Behavior:
- prefill the sample limit to `20`, matching the notebook demo
- preselect the models listed in the showcase/runnable configs
- show the showcase label and runtime backend together so the mapping stays explicit

Primary action: **Continue to evaluation setup**

### Step 2: Evaluation setup
Purpose: let the user choose the metric story without overwhelming them.

Content:
- metric picker for `LLM as Judge`, `LLM as Verifier`, and `Custom Judge`
- a simple judge editor with fields for score name, judge model, system prompt, system prompt without reference, concurrency, retries, and timeout
- a summary box explaining whether the current judge uses `input + output + expected_output` or `input + output`
- an advanced accordion for verifier criteria and less-common options

Behavior:
- prefill defaults from the runnable config and notebook-friendly prompts
- allow multiple metrics, but require one metric to be marked as the live leaderboard metric
- keep the custom judge editor intentionally simple so the UI stays demo-friendly

Primary action: **Start run**

### Step 3: Live run
Purpose: show credible progress, ranking changes, and observability while the run is happening.

Layout:
- run header with dataset runtime name, selected models, selected metrics, elapsed time, and overall phase
- live leaderboard sorted by the currently selected metric
- progress panel with per-model and per-metric counts
- event feed with concise operational updates

Behavior:
- the leaderboard updates from real score snapshots
- ranking changes animate when order changes
- partial score coverage is always labeled with counts such as `4/20 scored`
- event text can be synthesized from state changes, but the underlying counts and scores must be real

Primary action: none required; secondary actions include switching metric tabs and opening partial details

### Step 4: Final results
Purpose: turn the run into an inspectable, presentation-ready summary.

Content:
- winner hero card
- final ranking table
- metric evolution chart for the active metric
- lowest-scored examples table
- details drawer with model config, score summary, prompt details, run names, and Langfuse links

Primary actions:
- **Open in Langfuse**
- **Run again**
- optional summary export if trivial to support

## Runtime Design
The backend should create a runtime configuration from the existing demo assets plus UI overrides, rather than editing source files.

### Runtime inputs
- showcase config for labels and visible model choices
- runnable config for executable backend model mapping
- CSV for row count, hero mission preview, and dataset rows
- UI overrides for selected models, selected metrics, sample limit, runtime dataset name, and custom prompts

### Runtime config strategy
Create a runtime config object in memory and, if needed, write a temporary YAML file only for compatibility boundaries. The source YAMLs remain unchanged.

This mirrors the notebook pattern where a runtime config is derived from the runnable template with a sample limit and runtime dataset name.

## Backend Components
Create backend modules under `demo/gui/backend/` with focused responsibilities.

### 1. Config loader
Responsibilities:
- load the showcase and runnable YAMLs
- extract the showcase-to-backend model mapping
- inspect the CSV to produce row count and hero mission preview
- evaluate environment readiness for required keys and Langfuse reachability

### 2. Run request builder
Responsibilities:
- validate the frontend request
- build the runtime config for selected models and metrics
- assign a run id
- derive the runtime dataset name

### 3. Run manager
Responsibilities:
- own in-memory `RunState` records
- start background tasks for experiment and evaluation work
- expose current phase, progress counters, errors, leaderboard snapshots, and links
- support polling-based clients cleanly

### 4. Experiment orchestrator
Responsibilities:
- upload or attach the runtime dataset in Langfuse
- run each selected experiment concurrently
- preserve per-model progress and errors
- emit completed row results into downstream evaluation queues

### 5. Evaluation orchestrator
Responsibilities:
- run selected metrics concurrently where possible
- maintain per-metric score snapshots
- update Langfuse scores through the existing evaluator logic or thin adapters that preserve the same semantics
- expose leaderboard-ready aggregates continuously

### 6. Langfuse summary reader
Responsibilities:
- collect final run summaries, trace links, score tables, and lowest-scored examples
- align the final GUI summary with the notebook’s final summary logic

## Parallelism Model
The GUI should improve on the current CLI behavior by parallelizing across experiments as well as within each experiment.

### Experiment concurrency
Each selected model runs as its own background task. Inside each task, the existing executor concurrency remains in force.

This gives real overlap across models, which is essential for a live leaderboard that changes during the run.

### Metric concurrency
Support concurrent metrics with realistic limits:
- pointwise metrics can score items as results become available for a model
- group metrics can score a dataset row once all required model outputs for that row exist
- each metric keeps its own concurrency settings and error counters

### Honest live behavior
The frontend may look dynamic, but the numbers must reflect one of these real states:
- completed outputs
- completed evaluations
- current average from completed evaluations only
- final Langfuse-backed results

No score should appear before its underlying evaluation finishes.

## Incremental Evaluation Strategy
To make the live page believable, the backend should support incremental scoring rather than waiting for all experiments to finish before evaluating.

### Pointwise metrics
For `llm_as_judge` and future pointwise metrics:
- each experiment worker pushes completed `ExecutionResult` items into a metric queue
- the metric worker scores items as they arrive
- the backend updates per-model averages and scored item counts after each completed evaluation

This yields genuine leaderboard movement during the run.

### Group metrics
For `llm_as_verifier` and future group evaluators:
- maintain a per-row buffer keyed by `lf_item_id`
- when all selected models have produced output for a row, dispatch the group evaluation for that row
- update each model’s score aggregates when the group result completes

This keeps verifier-style metrics realistic without pretending that they can score partial model sets.

## RunState Data Model
The backend should maintain an in-memory run record that can be serialized for polling.

Recommended fields:
- `run_id`
- `phase` (`configuring`, `uploading`, `running`, `evaluating`, `completed`, `failed`)
- `dataset_name`
- `sample_limit`
- `selected_models`
- `selected_metrics`
- `started_at`, `finished_at`
- `items_total`
- `items_completed_by_model`
- `evaluations_completed_by_metric_by_model`
- `current_scores_by_metric_by_model`
- `leaderboard_snapshots`
- `event_feed`
- `errors`
- `langfuse_links`

The frontend only talks to this API model; it should not call Langfuse directly.

## API Surface
Expose a small set of GUI-specific endpoints.

### `GET /api/demo/config`
Returns:
- showcase config summary
- runnable config summary
- model mapping
- dataset row count
- hero mission preview
- environment readiness
- default sample limit and runtime dataset name pattern

### `POST /api/runs`
Creates and starts a run.

Request includes:
- selected models
- sample limit
- dataset runtime name override
- selected metrics
- custom judge definitions

Response includes:
- run id
- initial run state

### `GET /api/runs/{run_id}`
Returns the current run state, including phase, counts, active metric, current leaderboard, and top-level errors.

### `GET /api/runs/{run_id}/leaderboard`
Returns the latest leaderboard for a chosen metric, including per-model score, scored item counts, and trend deltas.

### `GET /api/runs/{run_id}/events`
Returns the event feed for the live page.

### `GET /api/runs/{run_id}/results`
Returns the final summary payload, including ranking table, lowest-scored examples, Langfuse links, and per-model details.

## Frontend Component Model
Create the frontend under `demo/gui/frontend/` with these major components.

### Shell and navigation
- `AppShell`
- `RunStepper`
- `TopStatusBar`

### Configuration step
- `DatasetConfigCard`
- `ModelSelectionGrid`
- `ModelCard`
- `HeroMissionPreview`
- `EnvReadinessPanel`

### Evaluation step
- `MetricSelector`
- `MetricBuilder`
- `JudgeEditor`
- `VerifierEditor`
- `ReferenceModeSummary`

### Live step
- `RunHeader`
- `LiveLeaderboard`
- `RunProgressPanel`
- `MetricTabs`
- `EventFeed`

### Results step
- `WinnerHeroCard`
- `FinalRankingTable`
- `MetricEvolutionChart`
- `LowestScoredExamplesTable`
- `TraceDrawer`

## UI Style Direction
Use a professional, observability-inspired interface rather than a marketing microsite.

### Visual principles
- strong card hierarchy
- restrained motion
- clear status badges
- compact but readable dense tables
- charts that emphasize rank changes and stability rather than decorative effects

### Libraries
- Tailwind CSS
- shadcn/ui for cards, tables, dialogs, drawers, badges, tabs, accordion, and form primitives
- TanStack Query for polling and server state
- Recharts for score evolution charts

## Real vs Synthesized UX
The GUI should explicitly separate what is real from what is presentation polish.

### Real
- config defaults and model mapping
- dataset preview and hero mission
- experiment execution
- evaluation results
- counts, scores, errors, and rankings
- Langfuse trace and run links

### Synthesized but acceptable
- event-feed phrasing derived from state transitions
- animated rank changes between leaderboard snapshots
- sparkline interpolation between real snapshots
- partial progress narration when the backend only emits coarse-grained milestones

### Not acceptable
- fake scores before evaluation completes
- fake token streaming
- fake traces or links
- a “final” leaderboard before the run actually completes

## Error Handling and Empty States
The GUI should degrade gracefully and remain credible.

### Hard blockers
Prevent run start when:
- required env vars are missing
- Langfuse is unreachable
- the CSV or YAML files are unreadable

### Soft failures
Allow the run to continue when:
- one model errors on some items
- one metric times out on some items
- one evaluator definition is malformed after the run starts

Display soft failures with badges and counts rather than collapsing the entire experience.

### Empty and loading states
- show skeletons during initial config load
- show placeholders in the leaderboard until the first scores arrive
- show counts such as `0/20 scored` instead of blank score cells
- delay chart rendering until enough points exist to make the chart meaningful

## Verification
The GUI is complete when these checks pass.

### Notebook alignment
- the backend reads the same showcase and runnable YAMLs as the notebook
- the default sample limit is `20`
- the GUI exposes the showcase-to-backend mapping clearly
- the final summary fields match the notebook summary concepts

### Golden path
- the app loads and shows prefilled configuration
- a user can select 2–3 models and at least one metric
- the run starts successfully
- live progress updates appear while work is in flight
- the leaderboard changes as real evaluations complete
- the final results page shows rankings, examples, and Langfuse links

### Failure-path behavior
- missing env state appears clearly before run start
- Langfuse failure is surfaced without a broken blank page
- a model-level failure does not erase healthy models from the ranking
- malformed custom judge settings produce actionable errors

## Design Decision Summary
- Use React + Vite + Tailwind + shadcn/ui for a polished frontend.
- Use FastAPI for a thin but real Python backend.
- Reuse the notebook assets and Open Arena engine directly.
- Parallelize experiments across models and keep internal worker concurrency.
- Evaluate incrementally so the live leaderboard moves on real data.
- Keep the GUI honest by labeling partial progress and never fabricating scores.
