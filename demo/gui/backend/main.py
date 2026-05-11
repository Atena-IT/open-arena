from fastapi import FastAPI, HTTPException, Query

from demo.gui.backend.config_loader import load_demo_config
from demo.gui.backend.models import (
    CreateRunRequest,
    DemoConfigResponse,
    EventsResponse,
    HealthResponse,
    LeaderboardResponse,
    RunPhase,
    RunResultsResponse,
    RunState,
)
from demo.gui.backend.orchestrator import DemoOrchestrator
from demo.gui.backend.run_manager import RunManager

app = FastAPI(title="Open Arena GUI API")

_run_manager = RunManager()
_orchestrator = DemoOrchestrator(_run_manager)


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/api/demo/config", response_model=DemoConfigResponse)
def demo_config() -> DemoConfigResponse:
    return load_demo_config()


@app.post("/api/runs", response_model=RunState, status_code=201)
def create_run(req: CreateRunRequest) -> RunState:
    try:
        return _run_manager.create_run(req)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.get("/api/runs/{run_id}", response_model=RunState)
def get_run(run_id: str) -> RunState:
    state = _run_manager.get_run(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id!r}")
    return state


@app.post("/api/runs/{run_id}/start", response_model=RunState, status_code=202)
def start_run(run_id: str) -> RunState:
    try:
        state = _run_manager.start_run(run_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id!r}")
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    _orchestrator.start_run(run_id)
    return state


@app.get("/api/runs/{run_id}/leaderboard", response_model=LeaderboardResponse)
def get_leaderboard(run_id: str, metric: str = Query(...)) -> LeaderboardResponse:
    if _run_manager.get_run(run_id) is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id!r}")
    try:
        entries = _run_manager.get_leaderboard(run_id, metric)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return LeaderboardResponse(runId=run_id, metric=metric, entries=entries)


@app.get("/api/runs/{run_id}/events", response_model=EventsResponse)
def get_events(run_id: str, after: int | None = Query(default=None, ge=0)) -> EventsResponse:
    if _run_manager.get_run(run_id) is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id!r}")
    events = _run_manager.get_events(run_id, after=after)
    next_cursor = events[-1].sequence if events else after
    return EventsResponse(runId=run_id, events=events, nextCursor=next_cursor)


@app.get("/api/runs/{run_id}/results", response_model=RunResultsResponse)
def get_results(run_id: str) -> RunResultsResponse:
    state = _run_manager.get_run(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id!r}")
    if state.phase not in {RunPhase.COMPLETED, RunPhase.FAILED}:
        raise HTTPException(status_code=409, detail="Results are only available once the run reaches a terminal phase")
    return _run_manager.get_results(run_id)
