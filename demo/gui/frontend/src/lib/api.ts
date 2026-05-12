import type {
  CreateRunRequest,
  DemoConfigResponse,
  EventsResponse,
  LeaderboardResponse,
  RunResultsResponse,
  RunState,
} from "./types";

async function apiFetch<T>(input: RequestInfo, init?: RequestInit): Promise<T> {
  const response = await fetch(input, {
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
    ...init,
  });

  if (!response.ok) {
    let detail = response.statusText;
    try {
      const payload = (await response.json()) as { detail?: string };
      detail = payload.detail ?? detail;
    } catch {
      // ignore JSON parsing failures and fall back to status text
    }
    throw new Error(detail || `Request failed with status ${response.status}`);
  }

  return (await response.json()) as T;
}

export function fetchDemoConfig(): Promise<DemoConfigResponse> {
  return apiFetch<DemoConfigResponse>("/api/demo/config");
}

export function createRun(payload: CreateRunRequest): Promise<RunState> {
  return apiFetch<RunState>("/api/runs", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function startRun(runId: string): Promise<RunState> {
  return apiFetch<RunState>(`/api/runs/${runId}/start`, {
    method: "POST",
  });
}

export function fetchRun(runId: string): Promise<RunState> {
  return apiFetch<RunState>(`/api/runs/${runId}`);
}

export function fetchLeaderboard(runId: string, metric: string): Promise<LeaderboardResponse> {
  return apiFetch<LeaderboardResponse>(`/api/runs/${runId}/leaderboard?metric=${encodeURIComponent(metric)}`);
}

export function fetchEvents(runId: string, after?: number | null): Promise<EventsResponse> {
  const query = after == null ? "" : `?after=${encodeURIComponent(String(after))}`;
  return apiFetch<EventsResponse>(`/api/runs/${runId}/events${query}`);
}

export function fetchResults(runId: string): Promise<RunResultsResponse> {
  return apiFetch<RunResultsResponse>(`/api/runs/${runId}/results`);
}
