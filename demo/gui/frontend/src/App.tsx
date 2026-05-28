import { useEffect, useMemo, useState } from 'react'
import './App.css'

type HealthResponse = { status: string }
type DiscoveryItem = { id: string; label?: string | null; description?: string | null }
type DiscoveryResponse = { items: DiscoveryItem[] }
type LeaderboardItem = {
  id: string
  name: string
  visibility: string
  description?: string | null
}
type LeaderboardResponse = { items: LeaderboardItem[]; next_cursor?: string | null }
type RunItem = {
  id: string
  mode: string
  status: string
  cache_status: string
  created_at: string
  completed_at?: string | null
}
type RunResponse = { items: RunItem[]; next_cursor?: string | null }

type DashboardData = {
  health: HealthResponse
  metricKinds: DiscoveryItem[]
  leaderboards: LeaderboardItem[]
  runs: RunItem[]
}

const DEFAULT_BASE_URL = 'http://127.0.0.1:8000'
const DEFAULT_TOKEN = 'open-arena-dev-token'

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init)
  if (!response.ok) {
    const text = await response.text()
    throw new Error(text || `Request failed with status ${response.status}`)
  }
  return (await response.json()) as T
}

function formatDate(value?: string | null): string {
  if (!value) {
    return '—'
  }
  return new Date(value).toLocaleString()
}

function App() {
  const [baseUrl, setBaseUrl] = useState(DEFAULT_BASE_URL)
  const [token, setToken] = useState(DEFAULT_TOKEN)
  const [data, setData] = useState<DashboardData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [refreshTick, setRefreshTick] = useState(0)

  useEffect(() => {
    let cancelled = false

    async function loadDashboard() {
      setLoading(true)
      setError(null)
      try {
        const headers = { Authorization: ['Bearer', token].join(' ') }
        const [health, metricKinds, leaderboards, runs] = await Promise.all([
          fetchJson<HealthResponse>(`${baseUrl}/healthz`),
          fetchJson<DiscoveryResponse>(`${baseUrl}/v1/metric-kinds`, { headers }),
          fetchJson<LeaderboardResponse>(`${baseUrl}/v1/leaderboards?limit=4`, { headers }),
          fetchJson<RunResponse>(`${baseUrl}/v1/runs?limit=5`, { headers }),
        ])
        if (!cancelled) {
          setData({
            health,
            metricKinds: metricKinds.items,
            leaderboards: leaderboards.items,
            runs: runs.items,
          })
        }
      } catch (loadError) {
        if (!cancelled) {
          setData(null)
          setError(loadError instanceof Error ? loadError.message : 'Unknown error')
        }
      } finally {
        if (!cancelled) {
          setLoading(false)
        }
      }
    }

    void loadDashboard()
    return () => {
      cancelled = true
    }
  }, [baseUrl, refreshTick, token])

  const runStatusSummary = useMemo(() => {
    return data?.runs.reduce<Record<string, number>>((summary, run) => {
      summary[run.status] = (summary[run.status] ?? 0) + 1
      return summary
    }, {}) ?? {}
  }, [data])

  return (
    <div className="shell">
      <header className="hero-card">
        <div>
          <p className="eyebrow">Open Arena demo</p>
          <h1>Web UI skeleton for the persistent evaluation API.</h1>
          <p className="lede">
            This draft frontend is intentionally small: it verifies connectivity, surfaces discovery data,
            and sketches the leaderboard / run-control panels a fuller product UI can grow from.
          </p>
        </div>
        <div className="actions">
          <button type="button" onClick={() => setRefreshTick((value) => value + 1)}>
            Refresh data
          </button>
          <a href="https://vite.dev/guide/" target="_blank" rel="noreferrer">
            Vite docs
          </a>
        </div>
      </header>

      <section className="panel connection-panel">
        <div>
          <p className="section-label">API connection</p>
          <h2>Point the dashboard at a local Open Arena server.</h2>
        </div>
        <div className="field-grid">
          <label>
            Base URL
            <input value={baseUrl} onChange={(event) => setBaseUrl(event.target.value)} />
          </label>
          <label>
            API token
            <input value={token} onChange={(event) => setToken(event.target.value)} type="password" />
          </label>
        </div>
      </section>

      {error ? (
        <section className="panel error-panel">
          <p className="section-label">Connection problem</p>
          <p>{error}</p>
          <code>arena serve</code>
        </section>
      ) : null}

      <section className="stats-grid">
        <article className="panel stat-card">
          <p className="section-label">Health</p>
          <strong className="stat-value">{loading ? '…' : data?.health.status ?? 'offline'}</strong>
          <span className="muted">Unauthenticated readiness probe</span>
        </article>
        <article className="panel stat-card">
          <p className="section-label">Metric kinds</p>
          <strong className="stat-value">{loading ? '…' : data?.metricKinds.length ?? 0}</strong>
          <span className="muted">Discovered from /v1/metric-kinds</span>
        </article>
        <article className="panel stat-card">
          <p className="section-label">Leaderboards</p>
          <strong className="stat-value">{loading ? '…' : data?.leaderboards.length ?? 0}</strong>
          <span className="muted">Latest private + public boards</span>
        </article>
        <article className="panel stat-card">
          <p className="section-label">Runs</p>
          <strong className="stat-value">{loading ? '…' : data?.runs.length ?? 0}</strong>
          <span className="muted">Recent executions sampled from the API</span>
        </article>
      </section>

      <section className="content-grid">
        <article className="panel">
          <div className="section-heading">
            <div>
              <p className="section-label">Discovery</p>
              <h2>Metric palette</h2>
            </div>
            <span className="pill">{data?.metricKinds.length ?? 0} available</span>
          </div>
          <div className="chip-cloud">
            {(data?.metricKinds ?? []).slice(0, 12).map((metric) => (
              <span key={metric.id} className="chip">
                {metric.id}
              </span>
            ))}
            {!loading && (data?.metricKinds.length ?? 0) === 0 ? <p className="muted">No metrics returned yet.</p> : null}
          </div>
        </article>

        <article className="panel">
          <div className="section-heading">
            <div>
              <p className="section-label">Execution snapshot</p>
              <h2>Run statuses</h2>
            </div>
            <span className="pill">{Object.keys(runStatusSummary).length} states</span>
          </div>
          <ul className="summary-list">
            {Object.entries(runStatusSummary).map(([status, count]) => (
              <li key={status}>
                <span>{status}</span>
                <strong>{count}</strong>
              </li>
            ))}
            {!loading && Object.keys(runStatusSummary).length === 0 ? <li className="muted">No runs available.</li> : null}
          </ul>
        </article>
      </section>

      <section className="content-grid">
        <article className="panel">
          <div className="section-heading">
            <div>
              <p className="section-label">Leaderboards</p>
              <h2>Recent boards</h2>
            </div>
            <code>GET /v1/leaderboards</code>
          </div>
          <div className="stack">
            {(data?.leaderboards ?? []).map((board) => (
              <div key={board.id} className="list-row">
                <div>
                  <strong>{board.name}</strong>
                  <p className="muted">{board.description || 'No description yet.'}</p>
                </div>
                <span className="pill">{board.visibility}</span>
              </div>
            ))}
            {!loading && (data?.leaderboards.length ?? 0) === 0 ? <p className="muted">Create a leaderboard to populate this panel.</p> : null}
          </div>
        </article>

        <article className="panel">
          <div className="section-heading">
            <div>
              <p className="section-label">Runs</p>
              <h2>Recent jobs</h2>
            </div>
            <code>GET /v1/runs</code>
          </div>
          <div className="stack">
            {(data?.runs ?? []).map((run) => (
              <div key={run.id} className="list-row run-row">
                <div>
                  <strong>{run.mode}</strong>
                  <p className="muted">Started {formatDate(run.created_at)}</p>
                  <p className="muted">Completed {formatDate(run.completed_at)}</p>
                </div>
                <div className="run-meta">
                  <span className="pill">{run.status}</span>
                  <span className="pill secondary">cache: {run.cache_status}</span>
                </div>
              </div>
            ))}
            {!loading && (data?.runs.length ?? 0) === 0 ? <p className="muted">Imported or submitted runs will appear here.</p> : null}
          </div>
        </article>
      </section>

      <section className="panel workflow-panel">
        <div className="section-heading">
          <div>
            <p className="section-label">Suggested next steps</p>
            <h2>Turn this skeleton into a richer operator console.</h2>
          </div>
          <code>POST /v1/import-config</code>
        </div>
        <ol>
          <li>Wire create / edit forms for verifiers, environments, and leaderboards.</li>
          <li>Add a YAML editor + upload flow that posts config contents to <code>/v1/import-config</code>.</li>
          <li>Attach polling or SSE for long-running jobs and render leaderboard entry drilldowns.</li>
        </ol>
        <pre>{`arena serve\ncd demo/gui/frontend\nnpm install\nnpm run dev`}</pre>
      </section>
    </div>
  )
}

export default App
