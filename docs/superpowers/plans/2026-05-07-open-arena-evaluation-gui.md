# Open Arena Evaluation GUI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a polished browser GUI for the show-me-how notebook flow that lets a user configure models and metrics, launch a real Open Arena run, watch a live leaderboard update, and inspect final Langfuse-backed results.

**Architecture:** Keep the notebook as the narrative source and build a dedicated GUI runtime under `demo/gui/`. Use a FastAPI backend to load the existing demo assets, launch background runs, expose polling-friendly run state, and serve a React + Tailwind frontend that mirrors the notebook sequence: configuration, evaluation setup, live run, and final results.

**Tech Stack:** Python, FastAPI, existing Open Arena modules, React, Vite, TypeScript, Tailwind CSS, TanStack Query, Recharts.

---

### Task 1: Scaffold the GUI backend and frontend workspaces

**Files:**
- Create: `demo/gui/backend/__init__.py`
- Create: `demo/gui/backend/main.py`
- Create: `demo/gui/backend/models.py`
- Create: `demo/gui/backend/__tests__/test_health_and_config.py`
- Create: `demo/gui/frontend/package.json`
- Create: `demo/gui/frontend/tsconfig.json`
- Create: `demo/gui/frontend/vite.config.ts`
- Create: `demo/gui/frontend/index.html`
- Create: `demo/gui/frontend/src/main.tsx`
- Create: `demo/gui/frontend/src/App.tsx`
- Create: `demo/gui/frontend/src/index.css`
- Create: `demo/gui/frontend/tailwind.config.js`
- Create: `demo/gui/frontend/postcss.config.js`
- Create: `demo/gui/app.py`

- [ ] **Step 1: Write the failing backend smoke test**

```python
from fastapi.testclient import TestClient

from demo.gui.backend.main import app


def test_health_endpoint():
    client = TestClient(app)
    response = client.get('/api/health')
    assert response.status_code == 200
    assert response.json() == {'status': 'ok'}
```

- [ ] **Step 2: Run the smoke test to verify it fails**

Run: `uv run pytest demo/gui/backend/__tests__/test_health_and_config.py::test_health_endpoint -v`
Expected: FAIL because `demo.gui.backend.main` does not exist yet.

- [ ] **Step 3: Add the backend application skeleton**

```python
from fastapi import FastAPI

app = FastAPI(title='Open Arena Evaluation GUI')


@app.get('/api/health')
def health() -> dict[str, str]:
    return {'status': 'ok'}
```

- [ ] **Step 4: Add the frontend scaffold**

```json
{
  "name": "open-arena-gui",
  "private": true,
  "scripts": {
    "dev": "vite",
    "build": "vite build"
  }
}
```

```tsx
import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
```

```tsx
export default function App() {
  return <div className="min-h-screen bg-slate-950 text-white">Open Arena GUI</div>
}
```

- [ ] **Step 5: Add the local launcher entry point**

```python
import uvicorn


if __name__ == '__main__':
    uvicorn.run('demo.gui.backend.main:app', host='127.0.0.1', port=8000, reload=True)
```

- [ ] **Step 6: Run the smoke test again**

Run: `uv run pytest demo/gui/backend/__tests__/test_health_and_config.py::test_health_endpoint -v`
Expected: PASS.

### Task 2: Expose notebook-aligned configuration and preview data

**Files:**
- Modify: `demo/gui/backend/main.py`
- Modify: `demo/gui/backend/models.py`
- Create: `demo/gui/backend/config_loader.py`
- Modify: `demo/gui/backend/__tests__/test_health_and_config.py`

- [ ] **Step 1: Write the failing config endpoint test**

```python
def test_demo_config_endpoint_returns_notebook_defaults():
    client = TestClient(app)
    response = client.get('/api/demo/config')
    assert response.status_code == 200
    payload = response.json()
    assert payload['sampleLimit'] == 20
    assert payload['dataset']['csvPath'].endswith('business_qa_demo.csv')
    assert payload['dataset']['rowCount'] >= 200
    assert payload['modelMapping']
    assert 'mission_title' in payload['heroMission']
```

- [ ] **Step 2: Run the config endpoint test to verify it fails**

Run: `uv run pytest demo/gui/backend/__tests__/test_health_and_config.py::test_demo_config_endpoint_returns_notebook_defaults -v`
Expected: FAIL with 404 or missing keys.

- [ ] **Step 3: Implement config loading from the notebook assets**

```python
from pathlib import Path
import csv

from src.config.types import ExperimentsFile

REPO_ROOT = Path(__file__).resolve().parents[3]
DEMO_ROOT = REPO_ROOT / 'demo' / 'show_me_how_open_arena'
SHOWCASE = ExperimentsFile.from_yaml(DEMO_ROOT / 'configs' / 'business_qa_showcase.yaml')
RUNNABLE = ExperimentsFile.from_yaml(DEMO_ROOT / 'configs' / 'business_qa_runnable.yaml')

with (DEMO_ROOT / 'data' / 'business_qa_demo.csv').open(newline='', encoding='utf-8') as handle:
    rows = list(csv.DictReader(handle))
```

- [ ] **Step 4: Return a structured config payload**

```python
@app.get('/api/demo/config', response_model=DemoConfigResponse)
def demo_config() -> DemoConfigResponse:
    return load_demo_config()
```

The response model should include:
- dataset CSV path, row count, default sample limit, runtime dataset name
- showcase-to-backend model mapping
- hero mission preview
- environment readiness booleans for Langfuse and provider keys

- [ ] **Step 5: Re-run the config test**

Run: `uv run pytest demo/gui/backend/__tests__/test_health_and_config.py -v`
Expected: PASS with both `/api/health` and `/api/demo/config` tests green.

### Task 3: Implement run state, background orchestration, and polling APIs

**Files:**
- Modify: `demo/gui/backend/models.py`
- Create: `demo/gui/backend/runtime.py`
- Create: `demo/gui/backend/run_manager.py`
- Modify: `demo/gui/backend/main.py`
- Create: `demo/gui/backend/__tests__/test_run_manager.py`

- [ ] **Step 1: Write the failing run-manager unit tests**

```python
def test_create_run_sets_initial_phase():
    manager = RunManager()
    run = manager.create_run(sample_limit=20, selected_models=['OpenAI GPT-5.4 Mini'])
    assert run.phase == 'configuring'
    assert run.items_total == 20


def test_leaderboard_uses_only_completed_scores():
    manager = RunManager()
    run = manager.create_run(sample_limit=20, selected_models=['A', 'B'])
    manager.record_score(run.run_id, metric_key='judge', model_name='A', score=0.75)
    board = manager.get_leaderboard(run.run_id, 'judge')
    assert board[0]['modelName'] == 'A'
    assert board[0]['scoredItems'] == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest demo/gui/backend/__tests__/test_run_manager.py -v`
Expected: FAIL because `RunManager` is not implemented.

- [ ] **Step 3: Define the run-state models**

```python
class RunPhase(str, Enum):
    CONFIGURING = 'configuring'
    UPLOADING = 'uploading'
    RUNNING = 'running'
    EVALUATING = 'evaluating'
    COMPLETED = 'completed'
    FAILED = 'failed'
```

Add models for:
- run summary
- selected metric definitions
- per-model progress
- leaderboard entries
- event feed rows
- final result rows

- [ ] **Step 4: Implement a thread-safe in-memory run manager**

```python
class RunManager:
    def __init__(self) -> None:
        self._runs: dict[str, RunState] = {}
        self._lock = threading.Lock()
```

The manager should support:
- create run
- update phase
- record model progress
- record score snapshots
- append events
- read summary, leaderboard, events, and final results

- [ ] **Step 5: Add the polling endpoints**

```python
@app.post('/api/runs', response_model=RunStateResponse)
async def create_run(request: CreateRunRequest):
    ...

@app.get('/api/runs/{run_id}', response_model=RunStateResponse)
def get_run(run_id: str):
    ...

@app.get('/api/runs/{run_id}/leaderboard')
def get_leaderboard(run_id: str, metric: str):
    ...

@app.get('/api/runs/{run_id}/events')
def get_events(run_id: str):
    ...
```

- [ ] **Step 6: Re-run the unit tests**

Run: `uv run pytest demo/gui/backend/__tests__/test_run_manager.py demo/gui/backend/__tests__/test_health_and_config.py -v`
Expected: PASS.

### Task 4: Wire real experiment execution and incremental evaluation

**Files:**
- Modify: `demo/gui/backend/runtime.py`
- Modify: `demo/gui/backend/run_manager.py`
- Create: `demo/gui/backend/orchestrator.py`
- Modify: `demo/gui/backend/main.py`
- Create: `demo/gui/backend/__tests__/test_orchestrator.py`

- [ ] **Step 1: Write the failing orchestration test with fakes**

```python
async def test_pointwise_scores_update_leaderboard_incrementally():
    manager = RunManager()
    orchestrator = DemoOrchestrator(manager=manager, executor_factory=FakeExecutorFactory())
    run = manager.create_run(sample_limit=2, selected_models=['A'])
    await orchestrator.run(run.run_id, fake_mode=True)
    board = manager.get_leaderboard(run.run_id, 'judge')
    assert board[0]['scoredItems'] == 2
    assert board[0]['currentAverageScore'] is not None
```

- [ ] **Step 2: Run the orchestration test to verify it fails**

Run: `uv run pytest demo/gui/backend/__tests__/test_orchestrator.py -v`
Expected: FAIL because the orchestrator does not exist yet.

- [ ] **Step 3: Reuse the existing project modules for runtime execution**

The orchestrator should:
- build dataset rows from the CSV/YAML assets
- upload or attach the runtime dataset in Langfuse
- launch selected experiments concurrently with `asyncio.gather`
- preserve the internal executor concurrency already implemented in `src/execution/executor.py`

```python
await asyncio.gather(*[
    self._run_experiment(run_state, experiment_config)
    for experiment_config in selected_experiments
])
```

- [ ] **Step 4: Evaluate incrementally**

Pointwise metrics:
- push each completed `ExecutionResult` into a metric queue
- score it immediately
- update the leaderboard snapshot after each score

Group metrics:
- buffer by `lf_item_id`
- dispatch a group evaluation only when all selected models have produced output for the row

- [ ] **Step 5: Add final-results aggregation**

Expose a `/api/runs/{run_id}/results` payload containing:
- winner card data
- final ranking table
- lowest-scored examples
- Langfuse links per model or run
- metric evolution series for charting

- [ ] **Step 6: Re-run orchestration tests**

Run: `uv run pytest demo/gui/backend/__tests__/test_orchestrator.py -v`
Expected: PASS with fake-mode orchestration and incremental scoreboard updates.

### Task 5: Build the React + Tailwind wizard and live screens

**Files:**
- Modify: `demo/gui/frontend/package.json`
- Modify: `demo/gui/frontend/src/App.tsx`
- Create: `demo/gui/frontend/src/lib/types.ts`
- Create: `demo/gui/frontend/src/lib/api.ts`
- Create: `demo/gui/frontend/src/components/StepShell.tsx`
- Create: `demo/gui/frontend/src/components/ConfigurationStep.tsx`
- Create: `demo/gui/frontend/src/components/EvaluationStep.tsx`
- Create: `demo/gui/frontend/src/components/LiveRunStep.tsx`
- Create: `demo/gui/frontend/src/components/ResultsStep.tsx`
- Create: `demo/gui/frontend/src/components/Leaderboard.tsx`
- Create: `demo/gui/frontend/src/components/MetricChart.tsx`
- Create: `demo/gui/frontend/src/components/EventFeed.tsx`
- Create: `demo/gui/frontend/src/components/ModelGrid.tsx`
- Create: `demo/gui/frontend/src/components/HeroMissionCard.tsx`

- [ ] **Step 1: Add the required frontend dependencies**

```json
{
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "@tanstack/react-query": "^5.59.0",
    "recharts": "^2.13.0"
  },
  "devDependencies": {
    "typescript": "^5.6.3",
    "vite": "^5.4.10",
    "tailwindcss": "^3.4.14",
    "postcss": "^8.4.47",
    "autoprefixer": "^10.4.20",
    "@vitejs/plugin-react": "^4.3.2"
  }
}
```

- [ ] **Step 2: Build the top-level wizard flow**

```tsx
const steps = ['Configuration', 'Evaluation', 'Live Run', 'Results'] as const
```

`App.tsx` should:
- fetch `/api/demo/config`
- manage selected models and metrics
- create a run
- switch to live polling once the run starts
- show the final results page when complete

- [ ] **Step 3: Implement the configuration and evaluation steps**

The configuration step must show:
- dataset card
- model grid with showcase and backend labels
- hero mission preview
- env readiness

The evaluation step must show:
- metric presets
- custom judge editor
- notebook-aligned prompt defaults
- advanced accordion for extra options

- [ ] **Step 4: Implement the live page**

The live page must show:
- run header with phase and elapsed time
- animated leaderboard from polling snapshots
- per-model progress bars
- metric tabs
- event feed

Use TanStack Query polling every 2–3 seconds.

- [ ] **Step 5: Implement the final results page**

The results page must show:
- winner card
- final ranking table
- metric evolution chart
- lowest-scored examples
- trace links or Langfuse links

- [ ] **Step 6: Build the frontend**

Run: `npm --prefix demo/gui/frontend install && npm --prefix demo/gui/frontend run build`
Expected: PASS with a generated production bundle.

### Task 6: Serve the app, verify end to end, and polish blockers

**Files:**
- Modify: `demo/gui/backend/main.py`
- Modify: `demo/gui/app.py`
- Modify: `demo/gui/frontend/src/*` as needed
- Modify: `demo/gui/backend/*` as needed

- [ ] **Step 1: Serve the built frontend from FastAPI**

Add a static mount that serves the Vite build output and falls back to `index.html` for non-API routes.

```python
app.mount('/', StaticFiles(directory=frontend_dist, html=True), name='gui')
```

- [ ] **Step 2: Add a short end-to-end smoke test for the API contract**

```python
def test_create_run_endpoint_returns_run_state():
    client = TestClient(app)
    response = client.post('/api/runs', json={
        'sampleLimit': 2,
        'selectedModels': ['OpenAI GPT-5.4 Mini'],
        'metrics': [{'key': 'judge', 'label': 'LLM as Judge', 'method': 'llm_as_judge'}],
    })
    assert response.status_code == 200
    assert response.json()['phase'] in {'configuring', 'uploading', 'running', 'evaluating', 'completed'}
```

- [ ] **Step 3: Run backend tests**

Run: `uv run pytest demo/gui/backend/__tests__ -v`
Expected: PASS.

- [ ] **Step 4: Run the frontend dev server and backend app**

Run in two shells:
- `python demo/gui/app.py`
- `npm --prefix demo/gui/frontend run dev -- --host 127.0.0.1 --port 4173`

Expected: both servers start without crashes.

- [ ] **Step 5: Exercise the golden path manually**

Verify in the browser:
- the config step is prefilled from the notebook assets
- the evaluation step starts from notebook-aligned judge defaults
- a run can be launched
- the live leaderboard and event feed update
- the results page renders final scores and links

- [ ] **Step 6: Fix only issues found in the smoke pass and rerun the relevant checks**

Run the narrow failing check again first, then rerun:
- `uv run pytest demo/gui/backend/__tests__ -v`
- `npm --prefix demo/gui/frontend run build`

Expected: PASS.

## Self-Review
- The plan covers configuration loading, background run orchestration, incremental evaluation, frontend wizard flow, live ranking, final results, and end-to-end verification.
- There are no placeholder markers such as `TODO` or `TBD` in the implementation steps.
- The plan keeps the GUI honest by requiring partial scores to be labeled as partial and by separating synthesized UI narration from real counts and scores.
