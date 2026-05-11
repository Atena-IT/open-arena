from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    status: str


class DatasetInfo(BaseModel):
    csvPath: str
    rowCount: int


class ModelMappingItem(BaseModel):
    experimentKey: str
    experimentName: str
    showcaseModel: str
    backendModel: str


class HeroMission(BaseModel):
    missionTitle: str
    researchDomain: str
    timeframeStart: str
    timeframeEnd: str
    allowedDomains: str
    focusSemantics: str
    outputType: str
    question: str
    expectedAnswer: str


class EnvStatus(BaseModel):
    LANGFUSE_SECRET_KEY: bool
    LANGFUSE_PUBLIC_KEY: bool
    LANGFUSE_HOST: bool
    OPENAI_API_KEY: bool
    GEMINI_API_KEY: bool
    ANTHROPIC_API_KEY: bool
    HUGGINGFACE_API_KEY: bool


class EvaluationDefaults(BaseModel):
    method: str
    label: str
    systemPrompt: str
    systemPromptNoReference: str


class DemoConfigResponse(BaseModel):
    sampleLimit: int
    dataset: DatasetInfo
    runtimeDatasetName: str
    modelMapping: list[ModelMappingItem]
    heroMission: HeroMission
    envStatus: EnvStatus
    evaluationDefaults: EvaluationDefaults


class RunPhase(str, Enum):
    CONFIGURING = "configuring"
    UPLOADING = "uploading"
    RUNNING = "running"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"


class ExecutionMode(str, Enum):
    REAL = "real"
    FAKE = "fake"


class MetricSpec(BaseModel):
    key: str = Field(..., min_length=1)
    label: str = Field(..., min_length=1)
    method: str = Field(..., min_length=1)
    systemPrompt: str | None = Field(default=None, min_length=1)
    systemPromptNoReference: str | None = Field(default=None, min_length=1)


class ResolvedModel(BaseModel):
    experimentKey: str
    experimentName: str
    showcaseModel: str
    backendModel: str


class ModelProgress(BaseModel):
    experimentKey: str
    experimentName: str
    showcaseModel: str
    backendModel: str
    completed: int = Field(..., ge=0)
    total: int = Field(..., ge=0)
    errors: int = Field(..., ge=0)


class EventRow(BaseModel):
    sequence: int = Field(..., ge=1)
    eventId: str
    timestamp: str
    kind: str
    payload: dict[str, Any]


class RunState(BaseModel):
    runId: str
    phase: RunPhase
    executionMode: ExecutionMode
    datasetName: str
    sampleLimit: int = Field(..., ge=1)
    itemsTotal: int = Field(..., ge=0)
    completedItems: int = Field(..., ge=0)
    errorCount: int = Field(..., ge=0)
    selectedModels: list[ResolvedModel]
    selectedMetrics: list[MetricSpec]
    activeMetricKey: str | None
    modelProgress: dict[str, ModelProgress]
    errors: list[str]
    recentEvents: list[EventRow]


class CreateRunRequest(BaseModel):
    sampleLimit: int = Field(..., ge=1)
    selectedModels: list[str] = Field(..., min_length=1)
    metrics: list[MetricSpec] = Field(..., min_length=1)
    runtimeDatasetName: str | None = None
    fakeMode: bool = False

    @field_validator("selectedModels")
    @classmethod
    def _require_unique_selected_models(cls, selected_models: list[str]) -> list[str]:
        if len(set(selected_models)) != len(selected_models):
            raise ValueError("selectedModels must not contain duplicates")
        return selected_models

    @field_validator("metrics")
    @classmethod
    def _require_unique_metric_keys(cls, metrics: list[MetricSpec]) -> list[MetricSpec]:
        keys = [metric.key for metric in metrics]
        if len(set(keys)) != len(keys):
            raise ValueError("metrics must not contain duplicate keys")
        return metrics


class LeaderboardEntry(BaseModel):
    experimentKey: str
    experimentName: str
    showcaseModel: str
    backendModel: str
    avgScore: float | None = None
    scoredCount: int = Field(..., ge=0)
    totalCount: int = Field(..., ge=0)


class LeaderboardResponse(BaseModel):
    runId: str
    metric: str
    entries: list[LeaderboardEntry]


class EventsResponse(BaseModel):
    runId: str
    events: list[EventRow]
    nextCursor: int | None = None


class MetricHistoryPoint(BaseModel):
    sequence: int = Field(..., ge=1)
    averages: dict[str, float]


class ResultExample(BaseModel):
    experimentKey: str
    experimentName: str
    showcaseModel: str
    backendModel: str
    itemReference: str
    input: str
    expectedOutput: str
    output: str | None = None
    score: float | None = None
    explanation: str | None = None
    executionError: str | None = None
    metricError: str | None = None
    traceId: str | None = None
    observationId: str | None = None


class MetricResults(BaseModel):
    key: str
    label: str
    method: str
    leaderboard: list[LeaderboardEntry]
    history: list[MetricHistoryPoint]
    lowestScored: list[ResultExample]


class RunResultsResponse(BaseModel):
    runId: str
    phase: RunPhase
    executionMode: ExecutionMode
    datasetName: str
    sampleLimit: int = Field(..., ge=1)
    itemsTotal: int = Field(..., ge=0)
    completedItems: int = Field(..., ge=0)
    errorCount: int = Field(..., ge=0)
    selectedModels: list[ResolvedModel]
    selectedMetrics: list[MetricSpec]
    activeMetricKey: str | None
    modelProgress: dict[str, ModelProgress]
    errors: list[str]
    recentEvents: list[EventRow]
    metrics: list[MetricResults]
