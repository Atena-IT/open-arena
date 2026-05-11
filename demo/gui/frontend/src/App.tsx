import { useEffect, useMemo, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";

import {
  createRun,
  fetchDemoConfig,
  fetchEvents,
  fetchLeaderboard,
  fetchResults,
  fetchRun,
  startRun,
} from "./lib/api";
import type {
  DemoConfigResponse,
  EventRow,
  LeaderboardEntry,
  MetricResults,
  MetricSpec,
  ModelMappingItem,
  ModelProgress,
  RunPhase,
  RunResultsResponse,
  RunState,
} from "./lib/types";

type WizardStep = "configuration" | "evaluation" | "live" | "results";

type MetricPreset = MetricSpec & { description: string };

const POLL_INTERVAL_MS = 2000;
const STEP_ORDER: Array<{ key: WizardStep; label: string; eyebrow: string }> = [
  { key: "configuration", label: "Configuration", eyebrow: "01" },
  { key: "evaluation", label: "Evaluation", eyebrow: "02" },
  { key: "live", label: "Live Run", eyebrow: "03" },
  { key: "results", label: "Results", eyebrow: "04" },
];
const SCORE_TICKS = [0, 0.25, 0.5, 0.75, 1];

export default function App() {
  const [step, setStep] = useState<WizardStep>(() => readResumeState().step);
  const [selectedModelKeys, setSelectedModelKeys] = useState<string[]>([]);
  const [sampleLimit, setSampleLimit] = useState(20);
  const [runtimeDatasetName, setRuntimeDatasetName] = useState("");
  const [fakeMode, setFakeMode] = useState(false);
  const [metrics, setMetrics] = useState<MetricSpec[]>([]);
  const [customMetricLabel, setCustomMetricLabel] = useState("Evidence Discipline");
  const [customSystemPrompt, setCustomSystemPrompt] = useState("");
  const [customSystemPromptNoReference, setCustomSystemPromptNoReference] = useState("");
  const [runId, setRunId] = useState<string | null>(() => readResumeState().runId);
  const [runSnapshot, setRunSnapshot] = useState<RunState | null>(null);
  const [eventStream, setEventStream] = useState<EventRow[]>([]);
  const [eventsCursor, setEventsCursor] = useState<number | null>(null);
  const [activeMetricKey, setActiveMetricKey] = useState<string>(() => readResumeState().metricKey);
  const [bootstrapped, setBootstrapped] = useState(false);
  const [launchError, setLaunchError] = useState<string | null>(null);
  const [isLaunching, setIsLaunching] = useState(false);
  const queryClient = useQueryClient();

  const configQuery = useQuery({
    queryKey: ["demo-config"],
    queryFn: fetchDemoConfig,
    staleTime: Infinity,
  });

  const currentRun = runSnapshot;

  const runQuery = useQuery({
    queryKey: ["run", runId],
    queryFn: () => fetchRun(runId as string),
    enabled: Boolean(runId),
    refetchInterval: currentRun && !isTerminalPhase(currentRun.phase) ? POLL_INTERVAL_MS : false,
  });

  const leaderboardQuery = useQuery({
    queryKey: ["leaderboard", runId, activeMetricKey],
    queryFn: () => fetchLeaderboard(runId as string, activeMetricKey),
    enabled: Boolean(runId && activeMetricKey),
    refetchInterval: currentRun && !isTerminalPhase(currentRun.phase) ? POLL_INTERVAL_MS : false,
  });

  const eventsQuery = useQuery({
    queryKey: ["events", runId, eventsCursor],
    queryFn: () => fetchEvents(runId as string, eventsCursor),
    enabled: Boolean(runId),
    refetchInterval: currentRun && !isTerminalPhase(currentRun.phase) ? POLL_INTERVAL_MS : false,
  });

  const resultsQuery = useQuery({
    queryKey: ["results", runId],
    queryFn: () => fetchResults(runId as string),
    enabled: Boolean(runId && currentRun && isTerminalPhase(currentRun.phase)),
  });

  useEffect(() => {
    if (!configQuery.data || bootstrapped) {
      return;
    }
    const config = configQuery.data;
    const readyForReal = canRunRealMode(config);
    const notebookMetric = buildNotebookMetric(config);
    setSelectedModelKeys(config.modelMapping.map((model) => model.experimentKey));
    setSampleLimit(config.sampleLimit);
    setRuntimeDatasetName(config.runtimeDatasetName);
    setFakeMode(!readyForReal);
    setMetrics([notebookMetric]);
    setActiveMetricKey(notebookMetric.key);
    setCustomSystemPrompt(config.evaluationDefaults.systemPrompt);
    setCustomSystemPromptNoReference(config.evaluationDefaults.systemPromptNoReference);
    setBootstrapped(true);
  }, [bootstrapped, configQuery.data]);

  useEffect(() => {
    if (!configQuery.data || !bootstrapped) {
      return;
    }
    setRuntimeDatasetName((current) => syncRuntimeDatasetName(current, configQuery.data.runtimeDatasetName, sampleLimit));
  }, [bootstrapped, configQuery.data, sampleLimit]);

  useEffect(() => {
    syncResumeUrl({ runId, step, metricKey: activeMetricKey });
  }, [activeMetricKey, runId, step]);

  useEffect(() => {
    if (runQuery.data) {
      setRunSnapshot(runQuery.data);
    }
  }, [runQuery.data]);

  useEffect(() => {
    if (!eventsQuery.data) {
      return;
    }
    setEventStream((previous) => mergeEvents(previous, eventsQuery.data.events));
    const nextCursor = eventsQuery.data.nextCursor;
    if (nextCursor != null) {
      setEventsCursor((previous) => {
        if (previous == null) {
          return nextCursor;
        }
        return Math.max(previous, nextCursor);
      });
    }
  }, [eventsQuery.data]);

  useEffect(() => {
    if (!currentRun) {
      return;
    }
    if (!activeMetricKey && currentRun.selectedMetrics.length > 0) {
      setActiveMetricKey(currentRun.selectedMetrics[0].key);
      return;
    }
    const metricKeys = new Set(currentRun.selectedMetrics.map((metric) => metric.key));
    if (!metricKeys.has(activeMetricKey) && currentRun.selectedMetrics.length > 0) {
      setActiveMetricKey(currentRun.selectedMetrics[0].key);
    }
  }, [activeMetricKey, currentRun]);

  useEffect(() => {
    if (!runId || !currentRun || !isTerminalPhase(currentRun.phase)) {
      return;
    }
    void fetchEvents(runId, eventsCursor).then((response) => {
      setEventStream((previous) => mergeEvents(previous, response.events));
      const nextCursor = response.nextCursor;
      if (nextCursor != null) {
        setEventsCursor((previous) => {
          if (previous == null) {
            return nextCursor;
          }
          return Math.max(previous, nextCursor);
        });
      }
    });
  }, [currentRun, eventsCursor, runId]);

  useEffect(() => {
    if (!runId || !activeMetricKey || !currentRun || !isTerminalPhase(currentRun.phase)) {
      return;
    }
    void queryClient.invalidateQueries({ queryKey: ["leaderboard", runId, activeMetricKey] });
  }, [activeMetricKey, currentRun?.phase, queryClient, runId]);

  useEffect(() => {
    if (!runId || !currentRun || !isTerminalPhase(currentRun.phase)) {
      return;
    }
    void queryClient.invalidateQueries({ queryKey: ["results", runId] });
  }, [currentRun?.phase, queryClient, runId]);

  const config = configQuery.data;
  const metricPresets = useMemo(() => (config ? buildMetricPresets(config) : []), [config]);
  const selectedModels = useMemo(
    () =>
      config?.modelMapping.filter((model) => selectedModelKeys.includes(model.experimentKey)) ?? [],
    [config, selectedModelKeys],
  );
  const availableMetricTabs = currentRun?.selectedMetrics ?? metrics;
  const leaderboardEntries = leaderboardQuery.data?.entries ?? [];
  const progressRows = useMemo(() => {
    if (!currentRun) {
      return [];
    }
    return Object.values(currentRun.modelProgress).sort((left, right) => left.experimentName.localeCompare(right.experimentName));
  }, [currentRun]);
  const resultsData = resultsQuery.data ?? null;
  const resultsMetricTabs = resultsData?.metrics.map((metric) => ({
    key: metric.key,
    label: metric.label,
    method: metric.method,
  })) ?? availableMetricTabs;
  const activeResultsMetric = useMemo(
    () => resultsData?.metrics.find((metric) => metric.key === activeMetricKey) ?? resultsData?.metrics[0] ?? null,
    [activeMetricKey, resultsData],
  );

  if (configQuery.isLoading || !bootstrapped) {
    return <LoadingState />;
  }

  if (configQuery.isError || !config) {
    return (
      <Shell>
        <div className="rounded-3xl border border-red-500/30 bg-red-500/10 p-8 text-red-50">
          <p className="text-sm uppercase tracking-[0.3em] text-red-200/80">Backend unavailable</p>
          <h1 className="mt-3 text-3xl font-semibold">The GUI could not load its notebook defaults.</h1>
          <p className="mt-4 text-sm text-red-100/80">
            {configQuery.error instanceof Error ? configQuery.error.message : "Unknown error"}
          </p>
        </div>
      </Shell>
    );
  }

  const canMoveToEvaluation = selectedModelKeys.length > 0;
  const canOpenLive = Boolean(currentRun);
  const canOpenResults = Boolean(currentRun && isTerminalPhase(currentRun.phase));
  const compactLayout = true;
  const showSidebar = false;

  return (
    <Shell>
      <div className={showSidebar ? "grid gap-6 xl:grid-cols-[1.35fr_0.65fr]" : "space-y-4"}>
        <div className="space-y-4">
          <HeroHeader config={config} currentRun={currentRun} isLaunching={isLaunching} compact={compactLayout} />
          <StepRail
            step={step}
            compact={compactLayout}
            onSelect={(candidate) => {
              if (candidate === "evaluation" && canMoveToEvaluation) {
                setStep(candidate);
              }
              if (candidate === "configuration") {
                setStep(candidate);
              }
              if (candidate === "live" && canOpenLive) {
                setStep(candidate);
              }
              if (candidate === "results" && canOpenResults) {
                setStep(candidate);
              }
            }}
            canOpenLive={canOpenLive}
            canOpenResults={canOpenResults}
          />

          {step === "configuration" && (
            <ConfigurationStep
              config={config}
              sampleLimit={sampleLimit}
              setSampleLimit={setSampleLimit}
              runtimeDatasetName={runtimeDatasetName}
              setRuntimeDatasetName={setRuntimeDatasetName}
              selectedModelKeys={selectedModelKeys}
              setSelectedModelKeys={setSelectedModelKeys}
              onContinue={() => setStep("evaluation")}
            />
          )}

          {step === "evaluation" && (
            <EvaluationStep
              config={config}
              metrics={metrics}
              selectedModels={selectedModels}
              metricPresets={metricPresets}
              customMetricLabel={customMetricLabel}
              setCustomMetricLabel={setCustomMetricLabel}
              customSystemPrompt={customSystemPrompt}
              setCustomSystemPrompt={setCustomSystemPrompt}
              customSystemPromptNoReference={customSystemPromptNoReference}
              setCustomSystemPromptNoReference={setCustomSystemPromptNoReference}
              fakeMode={fakeMode}
              setFakeMode={setFakeMode}
              launchError={launchError}
              isLaunching={isLaunching}
              onAddPreset={(preset) => {
                setMetrics((previous) => {
                  if (previous.some((metric) => metric.key === preset.key)) {
                    return previous;
                  }
                  return [...previous, preset];
                });
                setActiveMetricKey(preset.key);
              }}
              onRemoveMetric={(metricKey) => {
                setMetrics((previous) => previous.filter((metric) => metric.key !== metricKey));
                setActiveMetricKey((previous) => (previous === metricKey ? "" : previous));
              }}
              onAddCustomMetric={() => {
                if (!customMetricLabel.trim()) {
                  return;
                }
                const metric = buildCustomMetric(
                  customMetricLabel,
                  customSystemPrompt,
                  customSystemPromptNoReference,
                  metrics,
                );
                setMetrics((previous) => [...previous, metric]);
                setActiveMetricKey(metric.key);
              }}
              onBack={() => setStep("configuration")}
              onLaunch={async () => {
                setLaunchError(null);
                setIsLaunching(true);
                try {
                  const created = await createRun({
                    sampleLimit,
                    selectedModels: selectedModelKeys,
                    metrics,
                    runtimeDatasetName,
                    fakeMode,
                  });
                  setRunId(created.runId);
                  setRunSnapshot(created);
                  setEventsCursor(null);
                  setEventStream([]);
                  setActiveMetricKey(created.selectedMetrics[0]?.key ?? metrics[0]?.key ?? "");
                  const started = await startRun(created.runId);
                  setRunSnapshot(started);
                  setStep("live");
                } catch (error) {
                  setLaunchError(asErrorMessage(error));
                } finally {
                  setIsLaunching(false);
                }
              }}
            />
          )}

          {step === "live" && (
            <LiveStep
              currentRun={currentRun}
              leaderboardEntries={leaderboardEntries}
              progressRows={progressRows}
              activeMetricKey={activeMetricKey}
              setActiveMetricKey={setActiveMetricKey}
              metricTabs={availableMetricTabs}
              eventStream={eventStream}
              onOpenResults={() => setStep("results")}
            />
          )}

          {step === "results" && (
            <ResultsStep
              results={resultsData}
              resultsError={resultsQuery.isError ? asErrorMessage(resultsQuery.error) : null}
              isLoadingResults={resultsQuery.isLoading || resultsQuery.isFetching}
              activeMetric={activeResultsMetric}
              activeMetricKey={activeMetricKey}
              metricTabs={resultsMetricTabs}
              setActiveMetricKey={setActiveMetricKey}
              onBackToLive={() => setStep("live")}
            />
          )}
        </div>

        {showSidebar && (
          <aside className="space-y-4 xl:space-y-3">
            <SidePanelCard title="Notebook Source of Truth" eyebrow="Dataset">
              <div className="space-y-4 text-sm text-slate-200/80">
                <div className="rounded-2xl border border-white/10 bg-slate-950/70 p-4">
                  <p className="text-xs uppercase tracking-[0.28em] text-slate-400">Runtime dataset</p>
                  <p className="mt-2 text-lg font-medium text-white">{runtimeDatasetName}</p>
                  <p className="mt-2 text-slate-300">{config.dataset.csvPath}</p>
                  <p className="mt-1 text-slate-400">{config.dataset.rowCount} rows available in the CSV demo asset.</p>
                </div>
                <div className="rounded-2xl border border-white/10 bg-slate-950/70 p-4">
                  <p className="text-xs uppercase tracking-[0.28em] text-slate-400">Hero mission</p>
                  <h2 className="mt-2 text-lg font-medium text-white">{config.heroMission.missionTitle}</h2>
                  <p className="mt-2 text-slate-300">{config.heroMission.question}</p>
                  <p className="mt-3 text-xs text-slate-400">
                    {config.heroMission.researchDomain} · {config.heroMission.timeframeStart} → {config.heroMission.timeframeEnd}
                  </p>
                </div>
              </div>
            </SidePanelCard>

            <SidePanelCard title="Environment readiness" eyebrow="Ops">
              <div className="grid gap-3">
                {Object.entries(config.envStatus).map(([key, value]) => (
                  <div
                    key={key}
                    className={`flex items-center justify-between rounded-2xl border px-4 py-3 text-sm ${
                      value
                        ? "border-emerald-400/20 bg-emerald-400/10 text-emerald-100"
                        : "border-amber-400/20 bg-amber-400/10 text-amber-100"
                    }`}
                  >
                    <span className="font-medium">{key}</span>
                    <span>{value ? "Ready" : "Missing"}</span>
                  </div>
                ))}
              </div>
            </SidePanelCard>

            <SidePanelCard title="Live snapshot" eyebrow="Telemetry">
              {currentRun ? (
                <div className="space-y-4 text-sm text-slate-200/80">
                  <StatRow label="Run ID" value={currentRun.runId.slice(0, 8)} />
                  <StatRow label="Phase" value={titleCase(currentRun.phase)} />
                  <StatRow label="Mode" value={titleCase(currentRun.executionMode)} />
                  <StatRow label="Completed" value={`${currentRun.completedItems}/${currentRun.itemsTotal}`} />
                  <StatRow label="Errors" value={String(currentRun.errorCount)} />
                </div>
              ) : (
                <p className="text-sm text-slate-300/70">Create and launch a run to start polling progress, events, and the leaderboard.</p>
              )}
            </SidePanelCard>
          </aside>
        )}
      </div>
    </Shell>
  );
}

function ConfigurationStep(props: {
  config: DemoConfigResponse;
  sampleLimit: number;
  setSampleLimit: (value: number) => void;
  runtimeDatasetName: string;
  setRuntimeDatasetName: (value: string) => void;
  selectedModelKeys: string[];
  setSelectedModelKeys: (value: string[]) => void;
  onContinue: () => void;
}) {
  const {
    config,
    sampleLimit,
    setSampleLimit,
    runtimeDatasetName,
    setRuntimeDatasetName,
    selectedModelKeys,
    setSelectedModelKeys,
    onContinue,
  } = props;

  return (
    <Panel title="Notebook-aligned configuration" eyebrow="Step 1" subtitle="Choose the runnable dataset envelope and the model roster you want to race before defining metrics.">
      <div className="space-y-4 xl:space-y-3">
        <div className="grid gap-4 xl:grid-cols-[0.95fr_1.1fr_0.95fr] xl:gap-3">
          <div className="rounded-3xl border border-white/10 bg-slate-950/70 p-4 xl:p-3.5">
            <div className="flex items-center justify-between gap-3">
              <div>
                <p className="text-xs uppercase tracking-[0.28em] text-slate-400">Dataset cap</p>
                <h3 className="mt-2 text-xl font-semibold text-white xl:text-lg">{sampleLimit} sampled missions</h3>
              </div>
              <span className="rounded-full border border-cyan-400/20 bg-cyan-400/10 px-3 py-1 text-xs font-medium text-cyan-100">
                {config.dataset.rowCount} rows
              </span>
            </div>
            <input
              className="mt-4 h-2 w-full cursor-pointer appearance-none rounded-full bg-slate-800"
              type="range"
              min={1}
              max={config.dataset.rowCount}
              value={sampleLimit}
              onChange={(event) => setSampleLimit(Number(event.target.value))}
            />
            <div className="mt-2 flex items-center justify-between text-[11px] text-slate-500">
              <span>1</span>
              <span>{sampleLimit} selected</span>
              <span>{config.dataset.rowCount}</span>
            </div>
            <p className="mt-3 text-xs leading-5 text-slate-400">Use the full dataset when you want a real ranking signal, or keep it lower for a faster demo pass.</p>
          </div>

          <div className="rounded-3xl border border-white/10 bg-slate-950/70 p-4 xl:p-3.5">
            <label className="block text-sm text-slate-200/80">
              <span className="block text-xs uppercase tracking-[0.28em] text-slate-500">Runtime dataset name</span>
              <input
                className="mt-3 w-full rounded-2xl border border-white/10 bg-slate-900/80 px-4 py-3 text-white outline-none transition focus:border-violet-400/60 xl:py-2.5"
                value={runtimeDatasetName}
                onChange={(event) => setRuntimeDatasetName(event.target.value)}
              />
            </label>
            <div className="mt-3 flex flex-wrap items-center justify-between gap-3 text-xs text-slate-400">
              <span>{config.dataset.csvPath}</span>
              <button
                type="button"
                className="rounded-2xl border border-white/10 bg-white/5 px-4 py-2 text-sm font-medium text-slate-100 transition hover:bg-white/10"
                onClick={() => {
                  setSampleLimit(config.sampleLimit);
                  setRuntimeDatasetName(config.runtimeDatasetName);
                }}
              >
                Reset defaults
              </button>
            </div>
          </div>

          <div className="rounded-3xl border border-white/10 bg-gradient-to-br from-cyan-500/10 via-slate-950/80 to-violet-500/10 p-4 xl:p-3.5">
            <p className="text-xs uppercase tracking-[0.28em] text-cyan-100/70">Launch summary</p>
            <div className="mt-3 grid gap-3 sm:grid-cols-3 xl:grid-cols-1 xl:gap-2">
              <SummaryPill label="Models" value={String(selectedModelKeys.length)} />
              <SummaryPill label="Metric pack" value="Notebook next" />
              <SummaryPill label="Mode" value="Chosen in step 2" />
            </div>
            <button
              type="button"
              onClick={onContinue}
              disabled={selectedModelKeys.length === 0}
              className="mt-4 inline-flex w-full items-center justify-center rounded-2xl bg-white px-5 py-3 text-sm font-semibold text-slate-950 transition hover:bg-slate-100 disabled:cursor-not-allowed disabled:bg-slate-300 xl:py-2.5"
            >
              Continue to evaluation setup
            </button>
          </div>
        </div>

        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-8 xl:gap-2">
          {config.modelMapping.map((model) => {
            const selected = selectedModelKeys.includes(model.experimentKey);
            return (
              <button
                key={model.experimentKey}
                type="button"
                onClick={() => {
                  setSelectedModelKeys(toggleSelection(selectedModelKeys, model.experimentKey));
                }}
                className={`rounded-3xl border p-4 text-left transition xl:p-3 ${
                  selected
                    ? "border-violet-400/60 bg-violet-500/15 shadow-[0_0_0_1px_rgba(167,139,250,0.4)]"
                    : "border-white/10 bg-slate-950/70 hover:border-white/20 hover:bg-white/5"
                }`}
              >
                <div className="flex items-start justify-between gap-2">
                  <h3 className="text-sm font-semibold text-white xl:text-[13px] xl:leading-5">{model.experimentName}</h3>
                  <span
                    className={`rounded-full px-2 py-1 text-[10px] font-medium ${
                      selected ? "bg-violet-200 text-violet-950" : "bg-slate-800 text-slate-200"
                    }`}
                  >
                    {selected ? "On" : "Off"}
                  </span>
                </div>
                <div className="mt-2 space-y-1 text-[11px] leading-4 text-slate-300/80 xl:text-[10px]">
                  <p>{model.showcaseModel}</p>
                  <p className="text-slate-500">{model.backendModel}</p>
                </div>
              </button>
            );
          })}
        </div>
      </div>
    </Panel>
  );
}

function EvaluationStep(props: {
  config: DemoConfigResponse;
  metrics: MetricSpec[];
  selectedModels: ModelMappingItem[];
  metricPresets: MetricPreset[];
  customMetricLabel: string;
  setCustomMetricLabel: (value: string) => void;
  customSystemPrompt: string;
  setCustomSystemPrompt: (value: string) => void;
  customSystemPromptNoReference: string;
  setCustomSystemPromptNoReference: (value: string) => void;
  fakeMode: boolean;
  setFakeMode: (value: boolean) => void;
  launchError: string | null;
  isLaunching: boolean;
  onAddPreset: (preset: MetricPreset) => void;
  onRemoveMetric: (metricKey: string) => void;
  onAddCustomMetric: () => void;
  onBack: () => void;
  onLaunch: () => Promise<void>;
}) {
  const {
    config,
    metrics,
    selectedModels,
    metricPresets,
    customMetricLabel,
    setCustomMetricLabel,
    customSystemPrompt,
    setCustomSystemPrompt,
    customSystemPromptNoReference,
    setCustomSystemPromptNoReference,
    fakeMode,
    setFakeMode,
    launchError,
    isLaunching,
    onAddPreset,
    onRemoveMetric,
    onAddCustomMetric,
    onBack,
    onLaunch,
  } = props;

  const readyForReal = canRunRealMode(config);

  return (
    <Panel title="Metric design and launch" eyebrow="Step 2" subtitle="Start from the notebook judge, add optional rubric variants, then launch the run in real or fake mode.">
      <div className="grid gap-4 xl:grid-cols-[1.08fr_0.92fr]">
        <div className="space-y-4">
          <div className="rounded-3xl border border-white/10 bg-slate-950/70 p-5">
            <div className="flex items-center justify-between gap-4">
              <div>
                <p className="text-xs uppercase tracking-[0.28em] text-slate-400">Quick metric presets</p>
                <h3 className="mt-2 text-lg font-semibold text-white">Notebook judge + focused rubric variants</h3>
              </div>
              <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs font-medium text-slate-200">
                {metrics.length} selected
              </span>
            </div>
            <div className="mt-4 grid gap-3 md:grid-cols-3">
              {metricPresets.map((preset) => (
                <button
                  key={preset.key}
                  type="button"
                  onClick={() => onAddPreset(preset)}
                  className="rounded-3xl border border-white/10 bg-white/[0.03] p-3 text-left transition hover:border-violet-400/40 hover:bg-violet-500/10"
                >
                  <p className="text-sm font-semibold text-white">{preset.label}</p>
                  <p className="mt-2 text-xs leading-5 text-slate-300/75">{preset.description}</p>
                </button>
              ))}
            </div>
          </div>

          <div className="rounded-3xl border border-white/10 bg-slate-950/70 p-5">
            <div className="flex items-center justify-between gap-4">
              <div>
                <p className="text-xs uppercase tracking-[0.28em] text-slate-400">Custom judge metric</p>
                <h3 className="mt-2 text-lg font-semibold text-white">Create a rubric that stays inside the notebook flow</h3>
              </div>
              <button
                type="button"
                onClick={() => {
                  setCustomMetricLabel(config.evaluationDefaults.label);
                  setCustomSystemPrompt(config.evaluationDefaults.systemPrompt);
                  setCustomSystemPromptNoReference(config.evaluationDefaults.systemPromptNoReference);
                }}
                className="rounded-2xl border border-white/10 bg-white/5 px-4 py-2 text-xs font-medium text-slate-100 transition hover:bg-white/10"
              >
                Reset judge prompts
              </button>
            </div>
            <div className="mt-4 grid gap-3 xl:grid-cols-2">
              <label className="block space-y-2 text-sm text-slate-200/80 xl:col-span-2">
                <span className="text-xs uppercase tracking-[0.28em] text-slate-500">Metric label</span>
                <input
                  className="w-full rounded-2xl border border-white/10 bg-slate-900/80 px-4 py-3 text-white outline-none transition focus:border-violet-400/60"
                  value={customMetricLabel}
                  onChange={(event) => setCustomMetricLabel(event.target.value)}
                />
              </label>
              <label className="block space-y-2 text-sm text-slate-200/80">
                <span className="text-xs uppercase tracking-[0.28em] text-slate-500">Judge prompt with reference answer</span>
                <textarea
                  className="min-h-[104px] w-full rounded-3xl border border-white/10 bg-slate-900/80 px-4 py-3 text-sm leading-5 text-white outline-none transition focus:border-violet-400/60"
                  value={customSystemPrompt}
                  onChange={(event) => setCustomSystemPrompt(event.target.value)}
                />
              </label>
              <label className="block space-y-2 text-sm text-slate-200/80">
                <span className="text-xs uppercase tracking-[0.28em] text-slate-500">Judge prompt without reference answer</span>
                <textarea
                  className="min-h-[104px] w-full rounded-3xl border border-white/10 bg-slate-900/80 px-4 py-3 text-sm leading-5 text-white outline-none transition focus:border-violet-400/60"
                  value={customSystemPromptNoReference}
                  onChange={(event) => setCustomSystemPromptNoReference(event.target.value)}
                />
              </label>
            </div>
            <button
              type="button"
              onClick={onAddCustomMetric}
              className="mt-4 inline-flex items-center justify-center rounded-2xl border border-violet-400/30 bg-violet-500/15 px-4 py-3 text-sm font-medium text-violet-50 transition hover:bg-violet-500/25"
            >
              Add custom metric
            </button>
          </div>
        </div>

        <div className="space-y-4">
          <div className="rounded-3xl border border-white/10 bg-slate-950/70 p-5">
            <p className="text-xs uppercase tracking-[0.28em] text-slate-400">Launch mode</p>
            <div className="mt-4 grid gap-3 md:grid-cols-2">
              <button
                type="button"
                onClick={() => setFakeMode(false)}
                disabled={!readyForReal}
                className={`rounded-3xl border p-4 text-left transition ${
                  !fakeMode
                    ? "border-emerald-400/50 bg-emerald-500/15"
                    : "border-white/10 bg-white/[0.03]"
                } ${!readyForReal ? "cursor-not-allowed opacity-60" : "hover:border-emerald-400/30"}`}
              >
                <p className="text-sm font-semibold text-white">Real mode</p>
                <p className="mt-2 text-xs leading-5 text-slate-300/75">
                  Upload to Langfuse, run the selected backends, then score with the chosen metrics.
                </p>
              </button>
              <button
                type="button"
                onClick={() => setFakeMode(true)}
                className={`rounded-3xl border p-4 text-left transition ${
                  fakeMode
                    ? "border-cyan-400/50 bg-cyan-500/15"
                    : "border-white/10 bg-white/[0.03] hover:border-cyan-400/30"
                }`}
              >
                <p className="text-sm font-semibold text-white">Fake showcase mode</p>
                <p className="mt-2 text-xs leading-5 text-slate-300/75">
                  Keep the same API contract, but generate deterministic progress and scoring updates.
                </p>
              </button>
            </div>
            {!readyForReal && (
              <p className="mt-3 text-xs leading-5 text-amber-200/90">
                Real mode needs Langfuse credentials plus the OpenAI key because the runnable demo config uses GPT-5.4 backends.
              </p>
            )}
          </div>

          <div className="rounded-3xl border border-white/10 bg-slate-950/70 p-5">
            <p className="text-xs uppercase tracking-[0.28em] text-slate-400">Selected stack</p>
            <div className="mt-4 rounded-2xl border border-white/10 bg-white/[0.03] p-4">
              <p className="text-[11px] uppercase tracking-[0.24em] text-slate-500">Models</p>
              <div className="mt-3 flex flex-wrap gap-2">
                {selectedModels.map((model) => (
                  <span key={model.experimentKey} className="rounded-full border border-white/10 bg-white/[0.04] px-3 py-2 text-xs text-slate-100">
                    {model.experimentName}
                  </span>
                ))}
              </div>
            </div>
            <div className="mt-3 rounded-2xl border border-white/10 bg-white/[0.03] p-4">
              <p className="text-[11px] uppercase tracking-[0.24em] text-slate-500">Metrics</p>
              <div className="mt-3 flex max-h-[180px] flex-wrap gap-2 overflow-auto pr-1">
                {metrics.map((metric) => (
                  <div key={metric.key} className="inline-flex items-center gap-2 rounded-full border border-violet-400/25 bg-violet-500/10 px-3 py-2 text-xs text-violet-50">
                    <span>{metric.label}</span>
                    {metrics.length > 1 && (
                      <button
                        type="button"
                        onClick={() => onRemoveMetric(metric.key)}
                        className="rounded-full bg-white/10 px-2 py-0.5 text-[10px] text-violet-100 transition hover:bg-white/20"
                      >
                        Remove
                      </button>
                    )}
                  </div>
                ))}
              </div>
            </div>
            {launchError && (
              <div className="mt-3 rounded-2xl border border-red-500/25 bg-red-500/10 px-4 py-3 text-sm text-red-100">
                {launchError}
              </div>
            )}
            <div className="mt-4 flex flex-wrap gap-3">
              <button
                type="button"
                onClick={onBack}
                className="rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm font-medium text-slate-100 transition hover:bg-white/10"
              >
                Back to configuration
              </button>
              <button
                type="button"
                disabled={metrics.length === 0 || selectedModels.length === 0 || isLaunching}
                onClick={() => {
                  void onLaunch();
                }}
                className="rounded-2xl bg-white px-5 py-3 text-sm font-semibold text-slate-950 transition hover:bg-slate-100 disabled:cursor-not-allowed disabled:bg-slate-300"
              >
                {isLaunching ? "Launching run…" : fakeMode ? "Launch fake showcase" : "Launch real evaluation"}
              </button>
            </div>
          </div>
        </div>
      </div>
    </Panel>
  );
}

function LiveStep(props: {
  currentRun: RunState | null;
  leaderboardEntries: LeaderboardEntry[];
  progressRows: ModelProgress[];
  activeMetricKey: string;
  setActiveMetricKey: (value: string) => void;
  metricTabs: MetricSpec[];
  eventStream: EventRow[];
  onOpenResults: () => void;
}) {
  const [expandedPanel, setExpandedPanel] = useState<"histogram" | "progress" | "events" | null>(null);
  const {
    currentRun,
    leaderboardEntries,
    progressRows,
    activeMetricKey,
    setActiveMetricKey,
    metricTabs,
    eventStream,
    onOpenResults,
  } = props;

  if (!currentRun) {
    return (
      <Panel title="Live telemetry" eyebrow="Step 3" subtitle="Launch a run first to watch the leaderboard move.">
        <p className="text-sm text-slate-300/70">No run is active yet.</p>
      </Panel>
    );
  }

  const processedEvaluations = progressRows.reduce((total, progress) => total + progress.completed + progress.errors, 0);
  const totalPlannedEvaluations = progressRows.reduce((total, progress) => total + progress.total, 0);
  const finishedModels = progressRows.filter((progress) => progress.completed + progress.errors >= progress.total).length;
  const overallProgress = totalPlannedEvaluations === 0 ? 0 : processedEvaluations / totalPlannedEvaluations;
  const activeMetricLabel = metricTabs.find((metric) => metric.key === activeMetricKey)?.label ?? "Selected metric";

  const renderHistogramContent = (expanded: boolean) => {
    if (leaderboardEntries.length === 0) {
      return (
        <div className="flex h-full min-h-[240px] items-center justify-center rounded-3xl border border-dashed border-white/10 bg-white/[0.03] text-sm text-slate-400">
          Waiting for the first scored sample on {activeMetricLabel}…
        </div>
      );
    }

    return (
      <div className="grid grid-cols-[40px_minmax(0,1fr)] gap-3">
        <div className={`relative ${expanded ? "h-[420px]" : "h-[240px]"}`}>
          {SCORE_TICKS.slice().reverse().map((tick) => (
            <span
              key={tick}
              className="absolute right-0 -translate-y-1/2 text-[11px] font-medium text-slate-500"
              style={{ bottom: `${tick * 100}%` }}
            >
              {tick.toFixed(2)}
            </span>
          ))}
        </div>
        <div>
          <div className={`relative ${expanded ? "h-[420px]" : "h-[240px]"}`}>
            {SCORE_TICKS.map((tick) => (
              <div
                key={tick}
                className="absolute inset-x-0 border-t border-dashed border-white/10"
                style={{ bottom: `${tick * 100}%` }}
              />
            ))}
            <div className="absolute inset-0 flex items-end gap-3">
              {leaderboardEntries.map((entry, index) => (
                <div key={entry.experimentKey} className="flex h-full min-w-0 flex-1 items-end justify-center">
                  <div className="relative flex h-full w-full items-end justify-center">
                    <div className="absolute inset-x-0 -top-1 flex flex-col items-center gap-1 text-center">
                      <span
                        className={`inline-flex h-7 min-w-7 items-center justify-center rounded-full px-2 text-[11px] font-semibold ${
                          index === 0 ? "bg-emerald-200 text-emerald-950" : "bg-white/10 text-slate-200"
                        }`}
                      >
                        #{index + 1}
                      </span>
                      <span className="rounded-full border border-white/10 bg-slate-950/90 px-2.5 py-1 text-[11px] font-semibold text-white">
                        {formatScore(entry.avgScore)}
                      </span>
                    </div>
                    <div
                      className={`relative flex h-full w-full max-w-[88px] items-end overflow-hidden rounded-t-[1.4rem] border border-white/10 bg-slate-900/85 ${
                        entry.avgScore == null ? "opacity-60" : "shadow-[0_20px_60px_rgba(15,23,42,0.35)]"
                      }`}
                    >
                      <div
                        className={`w-full rounded-t-[1.4rem] transition-[height] duration-700 ${
                          entry.avgScore == null
                            ? "bg-white/10"
                            : index === 0
                              ? "bg-gradient-to-t from-emerald-400 via-cyan-400 to-violet-400"
                              : "bg-gradient-to-t from-violet-400 to-cyan-400"
                        }`}
                        style={{ height: scoreBarHeight(entry.avgScore) }}
                      />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
          <div className="mt-4 flex gap-3">
            {leaderboardEntries.map((entry) => (
              <div key={`${entry.experimentKey}-label`} className="min-w-0 flex-1 text-center">
                <p className="truncate text-xs font-semibold text-white">{entry.experimentName}</p>
                <p className="mt-1 text-[11px] text-slate-400">{entry.scoredCount}/{entry.totalCount} scored</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  };

  const renderProgressContent = (compact: boolean) => (
    <div className={`space-y-2 ${compact ? "" : "pr-1"}`}>
      {progressRows.map((progress) => (
        <div key={progress.experimentKey} className="rounded-3xl border border-white/10 bg-white/[0.03] p-3">
          <div className="flex items-center justify-between gap-3">
            <div>
              <h4 className="text-sm font-semibold text-white">{progress.experimentName}</h4>
              <p className="text-[11px] text-slate-400">{progress.completed + progress.errors}/{progress.total} processed</p>
            </div>
            <p className="text-[11px] text-slate-400">{progress.errors} errors</p>
          </div>
          <div className="mt-2 h-2 overflow-hidden rounded-full bg-slate-800">
            <div
              className="h-full rounded-full bg-gradient-to-r from-emerald-400 to-cyan-400 transition-[width] duration-700"
              style={{ width: scoreBarWidth(progressRatio(progress)) }}
            />
          </div>
        </div>
      ))}
    </div>
  );

  const renderEventContent = (compact: boolean) => (
    <div className={`space-y-2 ${compact ? "" : "pr-1"}`}>
      {eventStream.slice().reverse().map((event) => (
        <EventCard key={event.eventId} event={event} compact={compact} />
      ))}
    </div>
  );

  return (
    <>
      <Panel title="Live evaluation feed" eyebrow="Step 3" subtitle="A true column histogram on a fixed 0–1 axis, plus compact widgets you can expand on demand.">
        <div className="grid gap-3 xl:grid-cols-[1.08fr_0.92fr]">
          <div className="rounded-3xl border border-white/10 bg-slate-950/70 p-4 xl:h-[430px] xl:min-h-0 xl:overflow-hidden">
            <div className="flex h-full flex-col">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <p className="text-xs uppercase tracking-[0.28em] text-slate-400">Live ranking histogram</p>
                  <h3 className="mt-1 text-lg font-semibold text-white">One vertical column per model on the active metric</h3>
                  <p className="mt-1 text-xs text-slate-400">Scores stay on the backend-aligned normalized axis from 0.00 to 1.00.</p>
                </div>
                <div className="flex flex-wrap gap-2">
                  {metricTabs.map((metric) => (
                    <button
                      key={metric.key}
                      type="button"
                      onClick={() => setActiveMetricKey(metric.key)}
                      className={`rounded-full px-3 py-2 text-xs font-medium transition ${
                        metric.key === activeMetricKey
                          ? "bg-white text-slate-950"
                          : "border border-white/10 bg-white/[0.04] text-slate-200 hover:bg-white/[0.08]"
                      }`}
                    >
                      {metric.label}
                    </button>
                  ))}
                </div>
              </div>

              <div className="mt-3 grid gap-2 sm:grid-cols-5">
                <SummaryPill label="Phase" value={titleCase(currentRun.phase)} />
                <SummaryPill label="Mode" value={titleCase(currentRun.executionMode)} />
                <SummaryPill label="Metric" value={activeMetricLabel} />
                <SummaryPill label="Processed" value={`${processedEvaluations}/${totalPlannedEvaluations}`} />
                <SummaryPill label="Errors" value={String(currentRun.errorCount)} />
              </div>

              <div className="mt-3 rounded-3xl border border-white/10 bg-white/[0.03] p-4">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <p className="text-[11px] uppercase tracking-[0.24em] text-slate-500">Overall completion</p>
                    <p className="mt-1 text-sm text-slate-300/75">{processedEvaluations}/{totalPlannedEvaluations} evaluations processed · {finishedModels}/{progressRows.length} models finished</p>
                  </div>
                  <p className="text-lg font-semibold text-white">{Math.round(overallProgress * 100)}%</p>
                </div>
                <div className="mt-3 h-3 overflow-hidden rounded-full bg-slate-800">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-emerald-400 via-cyan-400 to-violet-400 transition-[width] duration-700"
                    style={{ width: scoreBarWidth(overallProgress) }}
                  />
                </div>
              </div>

              <div className="mt-3 min-h-0 flex-1 rounded-3xl border border-white/10 bg-white/[0.03] p-4 xl:overflow-hidden">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <p className="text-[11px] uppercase tracking-[0.24em] text-slate-500">Chart view</p>
                    <p className="mt-1 text-sm text-slate-300/75">Compact in the page, expandable when you want the full read.</p>
                  </div>
                  <button
                    type="button"
                    aria-label="Expand live ranking histogram"
                    onClick={() => setExpandedPanel("histogram")}
                    className="rounded-full border border-white/10 bg-white/[0.04] px-3 py-1.5 text-xs font-medium text-slate-200 transition hover:bg-white/[0.08]"
                  >
                    Expand
                  </button>
                </div>
                <div className="mt-4 h-full min-h-0">{renderHistogramContent(false)}</div>
              </div>
            </div>
          </div>

          <div className="space-y-3">
            <div className="rounded-3xl border border-white/10 bg-slate-950/70 p-4 xl:h-[208px] xl:min-h-0 xl:overflow-hidden">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <p className="text-xs uppercase tracking-[0.28em] text-slate-400">Per-model completion</p>
                  <h3 className="mt-1 text-base font-semibold text-white">How far each model is from the finish line</h3>
                </div>
                <button
                  type="button"
                  aria-label="Expand per-model completion"
                  onClick={() => setExpandedPanel("progress")}
                  className="rounded-full border border-white/10 bg-white/[0.04] px-3 py-1.5 text-xs font-medium text-slate-200 transition hover:bg-white/[0.08]"
                >
                  Expand
                </button>
              </div>
              <div className="mt-3 min-h-0 overflow-auto pr-1 xl:h-[132px]">{renderProgressContent(true)}</div>
            </div>

            <div className="rounded-3xl border border-white/10 bg-slate-950/70 p-4 xl:h-[208px] xl:min-h-0 xl:overflow-hidden">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <p className="text-xs uppercase tracking-[0.28em] text-slate-400">Event feed</p>
                  <h3 className="mt-1 text-base font-semibold text-white">Latest backend updates</h3>
                </div>
                <div className="flex items-center gap-2">
                  {isTerminalPhase(currentRun.phase) && (
                    <button
                      type="button"
                      onClick={onOpenResults}
                      className="rounded-2xl bg-white px-4 py-2 text-sm font-semibold text-slate-950 transition hover:bg-slate-100"
                    >
                      Open results
                    </button>
                  )}
                  <button
                    type="button"
                    aria-label="Expand latest backend updates"
                    onClick={() => setExpandedPanel("events")}
                    className="rounded-full border border-white/10 bg-white/[0.04] px-3 py-1.5 text-xs font-medium text-slate-200 transition hover:bg-white/[0.08]"
                  >
                    Expand
                  </button>
                </div>
              </div>
              <div className="mt-3 min-h-0 overflow-auto pr-1 xl:h-[132px]">{renderEventContent(true)}</div>
            </div>
          </div>
        </div>
      </Panel>

      {expandedPanel === "histogram" && (
        <OverlayDialog
          title="Live ranking histogram"
          subtitle="Expanded chart view with the same fixed 0.00–1.00 backend score scale."
          onClose={() => setExpandedPanel(null)}
        >
          {renderHistogramContent(true)}
        </OverlayDialog>
      )}

      {expandedPanel === "progress" && (
        <OverlayDialog
          title="Per-model completion"
          subtitle="Expanded progress view for the compact side widget."
          onClose={() => setExpandedPanel(null)}
        >
          {renderProgressContent(false)}
        </OverlayDialog>
      )}

      {expandedPanel === "events" && (
        <OverlayDialog
          title="Latest backend updates"
          subtitle="Expanded event stream with full payload cards."
          onClose={() => setExpandedPanel(null)}
        >
          {renderEventContent(false)}
        </OverlayDialog>
      )}
    </>
  );
}

function ResultsStep(props: {
  results: RunResultsResponse | null;
  resultsError: string | null;
  isLoadingResults: boolean;
  activeMetric: MetricResults | null;
  activeMetricKey: string;
  metricTabs: Array<{ key: string; label: string; method: string }>;
  setActiveMetricKey: (value: string) => void;
  onBackToLive: () => void;
}) {
  const [expandedPanel, setExpandedPanel] = useState<"leaderboard" | "examples" | "timeline" | null>(null);
  const {
    results,
    resultsError,
    isLoadingResults,
    activeMetric,
    activeMetricKey,
    metricTabs,
    setActiveMetricKey,
    onBackToLive,
  } = props;

  if (isLoadingResults && !results) {
    return (
      <Panel title="Final results" eyebrow="Step 4" subtitle="The backend is assembling the final ranking and scored examples.">
        <p className="text-sm text-slate-300/70">Preparing final results payload…</p>
      </Panel>
    );
  }

  if (resultsError) {
    return (
      <Panel title="Final results" eyebrow="Step 4" subtitle="The run finished, but the final payload could not be loaded.">
        <div className="rounded-2xl border border-red-500/25 bg-red-500/10 px-4 py-3 text-sm text-red-100">{resultsError}</div>
      </Panel>
    );
  }

  if (!results) {
    return (
      <Panel title="Final results" eyebrow="Step 4" subtitle="Once the run is complete, the backend publishes a dedicated final payload here.">
        <p className="text-sm text-slate-300/70">No run has completed yet.</p>
      </Panel>
    );
  }

  const progressRows = Object.values(results.modelProgress).sort((left, right) => left.experimentName.localeCompare(right.experimentName));
  const leaderboardEntries = activeMetric?.leaderboard ?? [];
  const highlightEntry = leaderboardEntries[0] ?? null;
  const lowestScored = activeMetric?.lowestScored ?? [];
  const processedEvaluations = progressRows.reduce((total, progress) => total + progress.completed + progress.errors, 0);
  const totalPlannedEvaluations = progressRows.reduce((total, progress) => total + progress.total, 0);
  const leaderScore = highlightEntry?.avgScore ?? null;
  const activeMetricLabel = activeMetric?.label ?? metricTabs.find((metric) => metric.key === activeMetricKey)?.label ?? "Selected metric";

  const renderLeaderboardContent = () => (
    <div className="space-y-3">
      {leaderboardEntries.length === 0 ? (
        <div className="flex min-h-[240px] items-center justify-center rounded-3xl border border-dashed border-white/10 bg-white/[0.03] text-sm text-slate-400">
          No final ranking data is available for this metric yet.
        </div>
      ) : (
        leaderboardEntries.map((entry, index) => {
          const rankTone =
            index === 0
              ? "border-emerald-400/30 bg-emerald-400/10"
              : index === 1
                ? "border-violet-400/25 bg-violet-500/10"
                : index === 2
                  ? "border-cyan-400/25 bg-cyan-500/10"
                  : "border-white/10 bg-white/[0.03]";
          return (
            <div key={entry.experimentKey} className={`rounded-3xl border p-4 ${rankTone}`}>
              <div className="grid gap-3 xl:grid-cols-[64px_minmax(0,1.35fr)_110px_minmax(0,1fr)_72px_96px] xl:items-center">
                <div>
                  <p className="text-[11px] uppercase tracking-[0.24em] text-slate-500">Rank</p>
                  <p className="mt-1 text-2xl font-semibold text-white">#{index + 1}</p>
                </div>
                <div className="min-w-0">
                  <h4 className="truncate text-base font-semibold text-white">{entry.experimentName}</h4>
                  <p className="mt-1 truncate text-xs text-slate-400">{entry.backendModel}</p>
                </div>
                <div className="xl:text-right">
                  <p className="text-[11px] uppercase tracking-[0.24em] text-slate-500">Score</p>
                  <p className="mt-1 text-xl font-semibold text-white">{formatScore(entry.avgScore)}</p>
                </div>
                <div>
                  <div className="h-3 overflow-hidden rounded-full bg-slate-800/90">
                    <div
                      className={`h-full rounded-full transition-[width] duration-700 ${
                        index === 0
                          ? "bg-gradient-to-r from-emerald-400 via-cyan-400 to-violet-400"
                          : "bg-gradient-to-r from-violet-400 to-cyan-400"
                      }`}
                      style={{ width: scoreBarWidth(entry.avgScore) }}
                    />
                  </div>
                </div>
                <div className="xl:text-right">
                  <p className="text-[11px] uppercase tracking-[0.24em] text-slate-500">Delta</p>
                  <p className="mt-1 text-sm font-medium text-slate-200">{formatScoreDelta(entry.avgScore, leaderScore, index === 0)}</p>
                </div>
                <div className="xl:text-right">
                  <p className="text-[11px] uppercase tracking-[0.24em] text-slate-500">Coverage</p>
                  <p className="mt-1 text-sm font-medium text-slate-200">{entry.scoredCount}/{entry.totalCount}</p>
                </div>
              </div>
            </div>
          );
        })
      )}
    </div>
  );

  const renderExamplesContent = () => (
    <div className="space-y-2">
      {lowestScored.length === 0 ? (
        <div className="rounded-3xl border border-dashed border-white/10 bg-white/[0.03] px-4 py-5 text-sm text-slate-400">
          No scored examples are available for this metric yet.
        </div>
      ) : (
        lowestScored.map((example) => (
          <div key={`${example.experimentKey}-${example.itemReference}`} className="rounded-3xl border border-white/10 bg-white/[0.03] p-3">
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                <p className="truncate text-sm font-semibold text-white">{example.experimentName}</p>
                <p className="mt-1 truncate text-[11px] text-slate-400">Item {example.itemReference}</p>
              </div>
              <div className="text-right">
                <p className="text-sm font-semibold text-white">{formatScore(example.score)}</p>
                <p className="mt-1 text-[11px] text-slate-500">{example.backendModel}</p>
              </div>
            </div>
            <div className="mt-2 grid gap-2 xl:grid-cols-2">
              <ExampleTextBlock label="Input" value={example.input} />
              <ExampleTextBlock label="Output" value={example.output ?? example.executionError ?? "No output captured"} />
            </div>
          </div>
        ))
      )}
    </div>
  );

  const renderTimelineContent = (compact: boolean) => (
    <div className={`space-y-2 ${compact ? "" : "pr-1"}`}>
      {results.recentEvents.slice().reverse().map((event) => (
        <EventCard key={event.eventId} event={event} compact={compact} />
      ))}
    </div>
  );

  return (
    <>
      <Panel title="Final readout" eyebrow="Step 4" subtitle="The leaderboard stays compact in the page, while every dense widget can expand on demand.">
        <div className="space-y-3">
          <div className="rounded-[2rem] border border-white/10 bg-slate-950/70 p-4">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <p className="text-xs uppercase tracking-[0.28em] text-slate-400">Winner snapshot</p>
                <h3 className="mt-1 text-xl font-semibold text-white">{highlightEntry?.experimentName ?? "No winner yet"}</h3>
                <p className="mt-1 text-xs leading-5 text-slate-300/75">
                  {highlightEntry
                    ? `${highlightEntry.backendModel} · ${formatScore(highlightEntry.avgScore)} normalized score on ${activeMetricLabel}`
                    : "Waiting for final ranking data."}
                </p>
              </div>
              <button
                type="button"
                onClick={onBackToLive}
                className="rounded-2xl border border-white/10 bg-white/5 px-4 py-2 text-sm font-medium text-slate-100 transition hover:bg-white/10"
              >
                Back to live view
              </button>
            </div>

            <div className="mt-3 flex flex-wrap gap-2">
              {metricTabs.map((metric) => (
                <button
                  key={metric.key}
                  type="button"
                  onClick={() => setActiveMetricKey(metric.key)}
                  className={`rounded-full px-3 py-2 text-xs font-medium transition ${
                    metric.key === activeMetricKey
                      ? "bg-white text-slate-950"
                      : "border border-white/10 bg-white/[0.04] text-slate-200 hover:bg-white/[0.08]"
                  }`}
                >
                  {metric.label}
                </button>
              ))}
            </div>

            <div className="mt-3 grid gap-2 sm:grid-cols-5">
              <SummaryPill label="Metric" value={activeMetricLabel} />
              <SummaryPill label="Sample limit" value={String(results.sampleLimit)} />
              <SummaryPill label="Processed" value={`${processedEvaluations}/${totalPlannedEvaluations}`} />
              <SummaryPill label="Errors" value={String(results.errorCount)} />
              <SummaryPill label="Models" value={String(results.selectedModels.length)} />
            </div>
          </div>

          <div className="grid gap-3 xl:grid-cols-[1.12fr_0.88fr]">
            <div className="rounded-3xl border border-white/10 bg-slate-950/70 p-4 xl:h-[430px] xl:min-h-0 xl:overflow-hidden">
              <div className="flex h-full flex-col">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <p className="text-xs uppercase tracking-[0.28em] text-slate-400">Final leaderboard</p>
                    <h3 className="mt-1 text-lg font-semibold text-white">Who won, by how much, and how stable the finish looked</h3>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="rounded-full border border-white/10 bg-white/[0.04] px-3 py-1 text-xs text-slate-300">
                      {leaderboardEntries.length} models ranked
                    </span>
                    <button
                      type="button"
                      aria-label="Expand final leaderboard"
                      onClick={() => setExpandedPanel("leaderboard")}
                      className="rounded-full border border-white/10 bg-white/[0.04] px-3 py-1.5 text-xs font-medium text-slate-200 transition hover:bg-white/[0.08]"
                    >
                      Expand
                    </button>
                  </div>
                </div>

                <div className="mt-3 rounded-2xl border border-white/10 bg-white/[0.03] px-4 py-2 text-[11px] font-medium text-slate-400">
                  Normalized score scale: 0.00 → 1.00
                </div>

                <div className="mt-3 min-h-0 flex-1 overflow-auto pr-2">{renderLeaderboardContent()}</div>
              </div>
            </div>

            <div className="space-y-3">
              <div className="rounded-3xl border border-white/10 bg-slate-950/70 p-4 xl:h-[208px] xl:min-h-0 xl:overflow-hidden">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <p className="text-xs uppercase tracking-[0.28em] text-slate-400">Lowest-scored examples</p>
                    <h3 className="mt-1 text-base font-semibold text-white">Where the active metric found the weakest outputs</h3>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-slate-400">{lowestScored.length} items</span>
                    <button
                      type="button"
                      aria-label="Expand lowest-scored examples"
                      onClick={() => setExpandedPanel("examples")}
                      className="rounded-full border border-white/10 bg-white/[0.04] px-3 py-1.5 text-xs font-medium text-slate-200 transition hover:bg-white/[0.08]"
                    >
                      Expand
                    </button>
                  </div>
                </div>
                <div className="mt-3 min-h-0 overflow-auto pr-1 xl:h-[132px]">{renderExamplesContent()}</div>
              </div>

              <div className="rounded-3xl border border-white/10 bg-slate-950/70 p-4 xl:h-[208px] xl:min-h-0 xl:overflow-hidden">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <p className="text-xs uppercase tracking-[0.28em] text-slate-400">Final backend timeline</p>
                    <h3 className="mt-1 text-base font-semibold text-white">Last events before the run closed</h3>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-slate-400">{results.recentEvents.length} events</span>
                    <button
                      type="button"
                      aria-label="Expand final backend timeline"
                      onClick={() => setExpandedPanel("timeline")}
                      className="rounded-full border border-white/10 bg-white/[0.04] px-3 py-1.5 text-xs font-medium text-slate-200 transition hover:bg-white/[0.08]"
                    >
                      Expand
                    </button>
                  </div>
                </div>
                <div className="mt-3 min-h-0 overflow-auto pr-1 xl:h-[132px]">{renderTimelineContent(true)}</div>
              </div>
            </div>
          </div>
        </div>
      </Panel>

      {expandedPanel === "leaderboard" && (
        <OverlayDialog
          title="Final leaderboard"
          subtitle="Expanded ranking view for the compact leaderboard widget."
          onClose={() => setExpandedPanel(null)}
        >
          <div className="rounded-2xl border border-white/10 bg-white/[0.03] px-4 py-2 text-[11px] font-medium text-slate-400">
            Normalized score scale: 0.00 → 1.00
          </div>
          <div className="mt-4">{renderLeaderboardContent()}</div>
        </OverlayDialog>
      )}

      {expandedPanel === "examples" && (
        <OverlayDialog
          title="Lowest-scored examples"
          subtitle="Expanded example view with full input and output cards."
          onClose={() => setExpandedPanel(null)}
        >
          {renderExamplesContent()}
        </OverlayDialog>
      )}

      {expandedPanel === "timeline" && (
        <OverlayDialog
          title="Final backend timeline"
          subtitle="Expanded event timeline for the compact widget."
          onClose={() => setExpandedPanel(null)}
        >
          {renderTimelineContent(false)}
        </OverlayDialog>
      )}
    </>
  );
}

function LoadingState() {
  return (
    <Shell>
      <div className="rounded-[2rem] border border-white/10 bg-slate-950/80 p-10">
        <p className="text-xs uppercase tracking-[0.3em] text-slate-400">Open Arena GUI</p>
        <h1 className="mt-4 text-4xl font-semibold text-white">Loading notebook-aligned defaults…</h1>
        <p className="mt-4 text-sm text-slate-300/75">The frontend is waiting for the FastAPI backend to expose the demo config, model mapping, and judge defaults.</p>
      </div>
    </Shell>
  );
}

function HeroHeader(props: { config: DemoConfigResponse; currentRun: RunState | null; isLaunching: boolean; compact?: boolean }) {
  const { config, currentRun, isLaunching, compact = false } = props;
  if (compact) {
    return (
      <div className="rounded-[2rem] border border-white/10 bg-gradient-to-br from-violet-500/20 via-slate-950/95 to-cyan-500/15 p-3 shadow-[0_30px_120px_rgba(15,23,42,0.45)]">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="max-w-4xl">
            <p className="text-[10px] uppercase tracking-[0.35em] text-violet-100/70">Notebook-driven evaluation cockpit</p>
            <h1 className="mt-1.5 text-xl font-semibold leading-tight text-white">
              Run the show-me-how demo like a product, not a notebook cell.
            </h1>
            <p className="mt-1 text-[11px] leading-4 text-slate-200/75">
              Notebook defaults, runnable models, and a GUI-first launch path.
            </p>
          </div>
          <div className="grid gap-2 text-[11px] text-slate-200/80 sm:grid-cols-3">
            <StatRow label="Prefilled models" value={String(config.modelMapping.length)} />
            <StatRow label="Default sample limit" value={String(config.sampleLimit)} />
            <StatRow label="Current phase" value={currentRun ? titleCase(currentRun.phase) : isLaunching ? "Launching" : "Draft"} />
          </div>
        </div>
      </div>
    );
  }
  return (
    <div className="rounded-[2rem] border border-white/10 bg-gradient-to-br from-violet-500/20 via-slate-950/95 to-cyan-500/15 p-6 shadow-[0_30px_120px_rgba(15,23,42,0.45)] xl:p-5">
      <div className="flex flex-wrap items-start justify-between gap-4 xl:items-center">
        <div className="max-w-3xl">
          <p className="text-xs uppercase tracking-[0.35em] text-violet-100/70">Notebook-driven evaluation cockpit</p>
          <h1 className="mt-3 text-3xl font-semibold leading-tight text-white md:text-4xl xl:text-[2rem]">
            Run the show-me-how demo like a product, not a notebook cell.
          </h1>
          <p className="mt-3 max-w-2xl text-sm leading-6 text-slate-200/75 xl:max-w-3xl">
            Preloaded models from the runnable YAML, a notebook-aligned judge, real or fake execution modes, and a live race board built on the backend event cursor.
          </p>
        </div>
        <div className="grid min-w-[220px] gap-2 text-sm text-slate-200/80 xl:min-w-[250px]">
          <StatRow label="Prefilled models" value={String(config.modelMapping.length)} />
          <StatRow label="Default sample limit" value={String(config.sampleLimit)} />
          <StatRow label="Current phase" value={currentRun ? titleCase(currentRun.phase) : isLaunching ? "Launching" : "Draft"} />
        </div>
      </div>
    </div>
  );
}

function StepRail(props: {
  step: WizardStep;
  onSelect: (step: WizardStep) => void;
  canOpenLive: boolean;
  canOpenResults: boolean;
  compact?: boolean;
}) {
  const { step, onSelect, canOpenLive, canOpenResults, compact = false } = props;
  if (compact) {
    return (
      <div className="flex flex-wrap gap-2">
        {STEP_ORDER.map((item) => {
          const interactive =
            item.key === "configuration" ||
            item.key === "evaluation" ||
            (item.key === "live" && canOpenLive) ||
            (item.key === "results" && canOpenResults);
          return (
            <button
              key={item.key}
              type="button"
              onClick={() => interactive && onSelect(item.key)}
              disabled={!interactive}
              aria-disabled={!interactive}
              className={`rounded-full border px-3 py-1.5 text-xs font-medium transition ${
                step === item.key
                  ? "border-violet-400/50 bg-violet-500/15 text-white"
                  : "border-white/10 bg-slate-950/70 text-slate-300 hover:border-white/20 hover:bg-white/[0.03]"
              } ${interactive ? "cursor-pointer" : "cursor-not-allowed opacity-60"}`}
            >
              {item.label}
            </button>
          );
        })}
      </div>
    );
  }
  return (
    <div className="grid gap-3 md:grid-cols-4 xl:gap-2">
      {STEP_ORDER.map((item) => {
        const interactive =
          item.key === "configuration" ||
          item.key === "evaluation" ||
          (item.key === "live" && canOpenLive) ||
          (item.key === "results" && canOpenResults);
        return (
          <button
            key={item.key}
            type="button"
            onClick={() => interactive && onSelect(item.key)}
              disabled={!interactive}
              aria-disabled={!interactive}
            className={`rounded-3xl border p-4 text-left transition xl:p-3 ${
              step === item.key
                ? "border-violet-400/50 bg-violet-500/15"
                : "border-white/10 bg-slate-950/70 hover:border-white/20 hover:bg-white/[0.03]"
            } ${interactive ? "cursor-pointer" : "cursor-not-allowed opacity-60"}`}
          >
            <p className="text-xs uppercase tracking-[0.28em] text-slate-500 xl:text-[10px]">{item.eyebrow}</p>
            <h2 className="mt-2 text-lg font-semibold text-white xl:mt-1 xl:text-base">{item.label}</h2>
          </button>
        );
      })}
    </div>
  );
}

function Panel(props: { title: string; eyebrow: string; subtitle: string; children: React.ReactNode }) {
  const { title, eyebrow, subtitle, children } = props;
  return (
    <section className="rounded-[2rem] border border-white/10 bg-slate-950/80 p-5 shadow-[0_18px_60px_rgba(15,23,42,0.35)] xl:p-4">
      <p className="text-[11px] uppercase tracking-[0.3em] text-slate-500">{eyebrow}</p>
      <div className="mt-1.5 flex flex-wrap items-end justify-between gap-3">
        <div>
          <h2 className="text-xl font-semibold text-white xl:text-[1.45rem]">{title}</h2>
          <p className="mt-1.5 max-w-3xl text-xs leading-5 text-slate-300/75">{subtitle}</p>
        </div>
      </div>
      <div className="mt-4">{children}</div>
    </section>
  );
}

function SidePanelCard(props: { title: string; eyebrow: string; children: React.ReactNode }) {
  const { title, eyebrow, children } = props;
  return (
    <section className="rounded-[2rem] border border-white/10 bg-slate-950/80 p-5 shadow-[0_18px_60px_rgba(15,23,42,0.3)] xl:p-4">
      <p className="text-xs uppercase tracking-[0.28em] text-slate-500">{eyebrow}</p>
      <h2 className="mt-2 text-xl font-semibold text-white">{title}</h2>
      <div className="mt-4">{children}</div>
    </section>
  );
}

function SummaryPill(props: { label: string; value: string }) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/[0.04] px-3 py-2.5">
      <p className="text-[11px] uppercase tracking-[0.24em] text-slate-500">{props.label}</p>
      <p className="mt-1 text-base font-semibold text-white">{props.value}</p>
    </div>
  );
}

function StatRow(props: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-4 rounded-2xl border border-white/10 bg-white/[0.04] px-4 py-3">
      <span className="text-slate-400">{props.label}</span>
      <span className="font-medium text-white">{props.value}</span>
    </div>
  );
}

function EventCard(props: { event: EventRow; compact?: boolean }) {
  const { event, compact = false } = props;
  return (
    <div className="rounded-3xl border border-white/10 bg-white/[0.03] p-4">
      <div className="flex items-center justify-between gap-3">
        <div>
          <p className="text-xs uppercase tracking-[0.28em] text-slate-500">{event.kind.replace(/_/g, " ")}</p>
          <p className="mt-1 text-sm font-medium text-white">Event #{event.sequence}</p>
        </div>
        <span className="text-xs text-slate-500">{formatTimestamp(event.timestamp)}</span>
      </div>
      <pre className={`mt-3 overflow-x-auto rounded-2xl bg-slate-950/80 p-3 text-xs text-slate-300/80 ${compact ? "max-h-[96px]" : "max-h-[150px]"}`}>
        {JSON.stringify(event.payload, null, 2)}
      </pre>
    </div>
  );
}

function ExampleTextBlock(props: { label: string; value: string }) {
  return (
    <div className="rounded-2xl border border-white/10 bg-slate-950/70 p-3">
      <p className="text-[11px] uppercase tracking-[0.24em] text-slate-500">{props.label}</p>
      <p className="mt-2 whitespace-pre-wrap text-sm leading-6 text-slate-200/85">{props.value}</p>
    </div>
  );
}

function OverlayDialog(props: { title: string; subtitle: string; onClose: () => void; children: React.ReactNode }) {
  const { title, subtitle, onClose, children } = props;
  const dialogRef = useRef<HTMLDivElement | null>(null);
  const closeButtonRef = useRef<HTMLButtonElement | null>(null);
  const lastFocusedElementRef = useRef<HTMLElement | null>(null);
  const headingId = `${slugify(title)}-dialog-title`;

  useEffect(() => {
    lastFocusedElementRef.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    closeButtonRef.current?.focus();

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        onClose();
        return;
      }
      if (event.key !== "Tab") {
        return;
      }
      const dialog = dialogRef.current;
      if (!dialog) {
        return;
      }
      const focusableElements = Array.from(
        dialog.querySelectorAll<HTMLElement>(
          'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])',
        ),
      ).filter((element) => !element.hasAttribute("disabled") && element.getAttribute("aria-hidden") !== "true");
      if (focusableElements.length === 0) {
        event.preventDefault();
        closeButtonRef.current?.focus();
        return;
      }
      const firstElement = focusableElements[0];
      const lastElement = focusableElements[focusableElements.length - 1];
      if (event.shiftKey && document.activeElement === firstElement) {
        event.preventDefault();
        lastElement.focus();
      } else if (!event.shiftKey && document.activeElement === lastElement) {
        event.preventDefault();
        firstElement.focus();
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
      lastFocusedElementRef.current?.focus();
    };
  }, [onClose]);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 p-4 backdrop-blur-sm"
      onClick={(event) => {
        if (event.target === event.currentTarget) {
          onClose();
        }
      }}
    >
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby={headingId}
        className="flex h-[calc(100vh-32px)] max-h-[880px] w-full max-w-6xl flex-col rounded-[2rem] border border-white/10 bg-slate-950 shadow-[0_30px_120px_rgba(15,23,42,0.55)]"
      >
        <div className="flex items-start justify-between gap-4 border-b border-white/10 p-5">
          <div>
            <p className="text-[11px] uppercase tracking-[0.3em] text-slate-500">Expanded widget</p>
            <h2 id={headingId} className="mt-1 text-xl font-semibold text-white">{title}</h2>
            <p className="mt-1 text-sm text-slate-300/75">{subtitle}</p>
          </div>
          <button
            ref={closeButtonRef}
            type="button"
            onClick={onClose}
            className="rounded-full border border-white/10 bg-white/[0.04] px-3 py-1.5 text-xs font-medium text-slate-200 transition hover:bg-white/[0.08]"
          >
            Close
          </button>
        </div>
        <div className="min-h-0 flex-1 overflow-auto p-5">{children}</div>
      </div>
    </div>
  );
}

function Shell(props: { children: React.ReactNode }) {
  return (
    <div
      className="min-h-screen bg-slate-950 px-6 py-4 text-slate-100 lg:px-8 xl:px-10"
      style={{
        backgroundImage:
          "radial-gradient(circle at top, rgba(124, 58, 237, 0.22), transparent 28%), radial-gradient(circle at bottom right, rgba(6, 182, 212, 0.18), transparent 22%)",
      }}
    >
      <div className="mx-auto max-w-[1600px]">{props.children}</div>
    </div>
  );
}

function buildNotebookMetric(config: DemoConfigResponse): MetricSpec {
  return {
    key: "notebook-judge",
    label: config.evaluationDefaults.label,
    method: config.evaluationDefaults.method,
    systemPrompt: config.evaluationDefaults.systemPrompt,
    systemPromptNoReference: config.evaluationDefaults.systemPromptNoReference,
  };
}

function buildMetricPresets(config: DemoConfigResponse): MetricPreset[] {
  return [
    {
      ...buildNotebookMetric(config),
      description: "The exact notebook-aligned quality rubric used by the runnable demo config.",
    },
    {
      key: "decision-usefulness",
      label: "Decision Usefulness",
      method: "llm_as_judge",
      systemPrompt: buildFocusedJudgePrompt("decision usefulness", true),
      systemPromptNoReference: buildFocusedJudgePrompt("decision usefulness", false),
      description: "Rewards memos that surface concrete implications, trade-offs, and next actions.",
    },
    {
      key: "evidence-discipline",
      label: "Evidence Discipline",
      method: "llm_as_judge",
      systemPrompt: buildFocusedJudgePrompt("evidence discipline", true),
      systemPromptNoReference: buildFocusedJudgePrompt("evidence discipline", false),
      description: "Rewards careful uncertainty handling, source discipline, and explicit caveats.",
    },
  ];
}

function buildFocusedJudgePrompt(focus: string, withReference: boolean): string {
  const referenceLine = withReference
    ? "Use the expected_output to judge whether the memo preserved the important facts and scope."
    : "Judge from the input and output only, without assuming a hidden gold answer.";
  return [
    `You are scoring a demo research memo on a 1-5 scale with a focus on ${focus}.`,
    referenceLine,
    "Reward topic discipline, actionable clarity, and explicit uncertainty.",
    "1 = unusable, 2 = weak, 3 = acceptable, 4 = solid, 5 = unusually strong.",
    "Return ONLY a JSON object with this format:",
    '{"thinking": "Short reasoning.", "score": 4}',
    "The score must be an integer from 1 to 5.",
  ].join(" ");
}

function buildCustomMetric(
  label: string,
  systemPrompt: string,
  systemPromptNoReference: string,
  existingMetrics: MetricSpec[],
): MetricSpec {
  const baseKey = slugify(label);
  let nextKey = baseKey;
  let counter = 2;
  const existingKeys = new Set(existingMetrics.map((metric) => metric.key));
  while (existingKeys.has(nextKey)) {
    nextKey = `${baseKey}-${counter}`;
    counter += 1;
  }
  return {
    key: nextKey,
    label,
    method: "llm_as_judge",
    systemPrompt,
    systemPromptNoReference,
  };
}

function readResumeState(): { step: WizardStep; runId: string | null; metricKey: string } {
  if (typeof window === "undefined") {
    return { step: "configuration", runId: null, metricKey: "" };
  }
  const searchParams = new URLSearchParams(window.location.search);
  const runId = searchParams.get("runId")?.trim() || null;
  const persistedStep = searchParams.get("step");
  const step = runId && isWizardStep(persistedStep) ? persistedStep : "configuration";
  return {
    step,
    runId,
    metricKey: runId ? searchParams.get("metric")?.trim() || "" : "",
  };
}

function syncResumeUrl(state: { step: WizardStep; runId: string | null; metricKey: string }): void {
  if (typeof window === "undefined") {
    return;
  }
  const url = new URL(window.location.href);
  if (state.runId) {
    url.searchParams.set("runId", state.runId);
    url.searchParams.set("step", state.step);
    if (state.metricKey) {
      url.searchParams.set("metric", state.metricKey);
    } else {
      url.searchParams.delete("metric");
    }
  } else {
    url.searchParams.delete("runId");
    url.searchParams.delete("step");
    url.searchParams.delete("metric");
  }
  const nextLocation = `${url.pathname}${url.search}${url.hash}`;
  const currentLocation = `${window.location.pathname}${window.location.search}${window.location.hash}`;
  if (nextLocation !== currentLocation) {
    window.history.replaceState(null, "", nextLocation);
  }
}

function isWizardStep(value: string | null): value is WizardStep {
  return value === "configuration" || value === "evaluation" || value === "live" || value === "results";
}

function syncRuntimeDatasetName(current: string, defaultRuntimeDatasetName: string, sampleLimit: number): string {
  const syncedName = `${stripSampleSuffix(defaultRuntimeDatasetName)} - sample ${sampleLimit}`;
  const trimmedCurrent = current.trim();
  if (!trimmedCurrent) {
    return syncedName;
  }
  return stripSampleSuffix(trimmedCurrent) === stripSampleSuffix(defaultRuntimeDatasetName) ? syncedName : current;
}

function stripSampleSuffix(value: string): string {
  return value.replace(/\s*-\s*sample\s+\d+\s*$/i, "").trim();
}

function mergeEvents(previous: EventRow[], next: EventRow[]): EventRow[] {
  const byId = new Map(previous.map((event) => [event.eventId, event]));
  for (const event of next) {
    byId.set(event.eventId, event);
  }
  return Array.from(byId.values()).sort((left, right) => left.sequence - right.sequence);
}

function toggleSelection(current: string[], value: string): string[] {
  if (current.includes(value)) {
    return current.filter((item) => item !== value);
  }
  return [...current, value];
}

function canRunRealMode(config: DemoConfigResponse): boolean {
  return (
    config.envStatus.LANGFUSE_SECRET_KEY &&
    config.envStatus.LANGFUSE_PUBLIC_KEY &&
    config.envStatus.LANGFUSE_HOST &&
    config.envStatus.OPENAI_API_KEY
  );
}

function progressRatio(progress: ModelProgress): number {
  if (progress.total === 0) {
    return 0;
  }
  return Math.min((progress.completed + progress.errors) / progress.total, 1);
}

function isTerminalPhase(phase: RunPhase): boolean {
  return phase === "completed" || phase === "failed";
}

function formatScore(score: number | null): string {
  return score == null ? "—" : clampUnit(score).toFixed(2);
}

function formatScoreDelta(score: number | null, leaderScore: number | null, isLeader: boolean): string {
  if (isLeader) {
    return "Leader";
  }
  if (score == null || leaderScore == null) {
    return "—";
  }
  return `-${Math.max(leaderScore - score, 0).toFixed(2)}`;
}

function scoreBarWidth(score: number | null): string {
  if (score == null) {
    return "0%";
  }
  return `${clampUnit(score) * 100}%`;
}

function scoreBarHeight(score: number | null): string {
  if (score == null) {
    return "6%";
  }
  return `${clampUnit(score) * 100}%`;
}

function clampUnit(value: number): number {
  return Math.min(Math.max(value, 0), 1);
}

function titleCase(value: string): string {
  return value.replace(/_/g, " ").replace(/\b\w/g, (letter: string) => letter.toUpperCase());
}

function slugify(value: string): string {
  return value
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/(^-|-$)/g, "") || "custom-metric";
}

function asErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  return "Unknown error";
}

function formatTimestamp(timestamp: string): string {
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) {
    return timestamp;
  }
  return new Intl.DateTimeFormat(undefined, {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(date);
}
