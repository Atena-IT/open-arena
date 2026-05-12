export type RunPhase =
  | "configuring"
  | "uploading"
  | "running"
  | "evaluating"
  | "completed"
  | "failed";

export type ExecutionMode = "real" | "fake";

export interface DatasetInfo {
  csvPath: string;
  rowCount: number;
}

export interface ModelMappingItem {
  experimentKey: string;
  experimentName: string;
  showcaseModel: string;
  backendModel: string;
}

export interface HeroMission {
  missionTitle: string;
  researchDomain: string;
  timeframeStart: string;
  timeframeEnd: string;
  allowedDomains: string;
  focusSemantics: string;
  outputType: string;
  question: string;
  expectedAnswer: string;
}

export interface EnvStatus {
  LANGFUSE_SECRET_KEY: boolean;
  LANGFUSE_PUBLIC_KEY: boolean;
  LANGFUSE_HOST: boolean;
  OPENAI_API_KEY: boolean;
  GEMINI_API_KEY: boolean;
  ANTHROPIC_API_KEY: boolean;
  HUGGINGFACE_API_KEY: boolean;
}

export interface EvaluationDefaults {
  method: string;
  label: string;
  systemPrompt: string;
  systemPromptNoReference: string;
}

export interface DemoConfigResponse {
  sampleLimit: number;
  dataset: DatasetInfo;
  runtimeDatasetName: string;
  modelMapping: ModelMappingItem[];
  heroMission: HeroMission;
  envStatus: EnvStatus;
  evaluationDefaults: EvaluationDefaults;
}

export interface MetricSpec {
  key: string;
  label: string;
  method: string;
  systemPrompt?: string | null;
  systemPromptNoReference?: string | null;
}

export interface ResolvedModel {
  experimentKey: string;
  experimentName: string;
  showcaseModel: string;
  backendModel: string;
}

export interface ModelProgress {
  experimentKey: string;
  experimentName: string;
  showcaseModel: string;
  backendModel: string;
  completed: number;
  total: number;
  errors: number;
}

export interface EventRow {
  sequence: number;
  eventId: string;
  timestamp: string;
  kind: string;
  payload: Record<string, unknown>;
}

export interface RunState {
  runId: string;
  phase: RunPhase;
  executionMode: ExecutionMode;
  datasetName: string;
  sampleLimit: number;
  itemsTotal: number;
  completedItems: number;
  errorCount: number;
  selectedModels: ResolvedModel[];
  selectedMetrics: MetricSpec[];
  activeMetricKey: string | null;
  modelProgress: Record<string, ModelProgress>;
  errors: string[];
  recentEvents: EventRow[];
}

export interface LeaderboardEntry {
  experimentKey: string;
  experimentName: string;
  showcaseModel: string;
  backendModel: string;
  avgScore: number | null;
  scoredCount: number;
  totalCount: number;
}

export interface LeaderboardResponse {
  runId: string;
  metric: string;
  entries: LeaderboardEntry[];
}

export interface MetricHistoryPoint {
  sequence: number;
  averages: Record<string, number>;
}

export interface ResultExample {
  experimentKey: string;
  experimentName: string;
  showcaseModel: string;
  backendModel: string;
  itemReference: string;
  input: string;
  expectedOutput: string;
  output: string | null;
  score: number | null;
  explanation?: string | null;
  executionError?: string | null;
  metricError?: string | null;
  traceId?: string | null;
  observationId?: string | null;
}

export interface MetricResults {
  key: string;
  label: string;
  method: string;
  leaderboard: LeaderboardEntry[];
  history: MetricHistoryPoint[];
  lowestScored: ResultExample[];
}

export interface RunResultsResponse {
  runId: string;
  phase: RunPhase;
  executionMode: ExecutionMode;
  datasetName: string;
  sampleLimit: number;
  itemsTotal: number;
  completedItems: number;
  errorCount: number;
  selectedModels: ResolvedModel[];
  selectedMetrics: MetricSpec[];
  activeMetricKey: string | null;
  modelProgress: Record<string, ModelProgress>;
  errors: string[];
  recentEvents: EventRow[];
  metrics: MetricResults[];
}

export interface EventsResponse {
  runId: string;
  events: EventRow[];
  nextCursor: number | null;
}

export interface CreateRunRequest {
  sampleLimit: number;
  selectedModels: string[];
  metrics: MetricSpec[];
  runtimeDatasetName?: string;
  fakeMode?: boolean;
}
