# API compliance matrix

OpenAPI contract source: `openapi.yaml`

Current backend implementation:

- API layer: `src/api/app.py`
- Persistence + services: `src/api/service.py`
- OpenAPI-derived schemas: `packages/open-arena-core/open_arena_core/models.py`
- Adapter factory: `src/api/registry.py` / `src/api/settings.py`
- CLI bridge: `src/evaluate.py`

---

## M1–M5 milestones + cross-cutting workstreams

| # | Workstream | Status | Port ABC | Default adapter | Alternative adapter | Selecting env var | PR |
|---|---|---|---|---|---|---|---|
| WS-PORTS | Repository-pattern ports layer | **Done** | All port ABCs in `src/api/ports/` | — | — | — | #50 |
| WS1 | Store — Postgres/SQLAlchemy + Alembic | **Done** | `Store` (`src/api/ports/store.py`) | `SQLiteStore` | `SQLAlchemyStore` (Postgres + JSONB) | `OPEN_ARENA_STORE=postgres` + `DATABASE_URL` | #56 |
| WS2 | EnvironmentBackend — Gitea/Git | **Done** | `EnvironmentBackend` (`src/api/ports/environment_backend.py`) | `InlineEnvironmentBackend` | `GitEnvironmentBackend` (Gitea + GitHub) | `OPEN_ARENA_ENV_BACKEND=git` + `GITEA_BASE_URL` / `GITEA_TOKEN` / `GITEA_ORG` | #55 |
| WS3 | Environment versioning + `GET /v1/environments/{id}/versions` | **Done** | `EnvironmentBackend` | `InlineEnvironmentBackend` | — | — | #60 |
| WS4 | DatasetResolver — Unity Catalog | **Done** | `DatasetResolver` (`src/api/ports/dataset_resolver.py`) | `LegacyDatasetResolver` | `UnityCatalogDataset` (Delta + Parquet over S3) | `OPEN_ARENA_DATASET_RESOLVER=unity_catalog` + `UNITY_CATALOG_API_URL` / `UC_TOKEN` | #49 |
| WS5 | ResultsSink — MLflow | **Done** | `ResultsSink` (`src/api/ports/results_sink.py`) | `StoreResultsSink` | `MlflowResultsSink` | `OPEN_ARENA_RESULTS_SINK=mlflow` + `MLFLOW_TRACKING_URI` | #52 |
| WS6 | SandboxProvider — E2B | **Done** | `SandboxProvider` (`src/api/ports/sandbox_provider.py`) | `LocalSandboxProvider` | `E2BSandboxProvider` | `OPEN_ARENA_SANDBOX=e2b` + `E2B_API_KEY` | #51 |
| WS7 | AuthProvider — Keycloak OIDC JWT | **Done** | `AuthProvider` (`src/api/ports/auth_provider.py`) | `StaticBearerAuthProvider` (`OPEN_ARENA_API_TOKEN`) | `KeycloakAuthProvider` | `OPEN_ARENA_AUTH=keycloak` + `OIDC_ISSUER` / `OIDC_CLIENT_ID` / `OIDC_CLIENT_SECRET` | #54 |
| WS8 | Deployment — Docker Compose + Helm | **Done** | — | — | Dockerfile + `docker-compose.yml` + `helm/open-arena/` (per-org release ready) | — | #47 |
| WS9 | CLI resource sub-groups + local run mode | **Done** | — | — | `arena env/verifier/leaderboard/run/discover` + `arena serve` / `arena request` | — | #53 |
| WS10 | Single-tenant by design — each deployment runs its own OA instance; Keycloak auth is optional (`OPEN_ARENA_AUTH=keycloak`) | **N/A (by design)** | `AuthProvider` → `Principal.org` | — | `KeycloakAuthProvider` derives org from JWT `groups` claim | `OPEN_ARENA_AUTH=keycloak` + `OIDC_ISSUER` / `OIDC_CLIENT_ID` / `OIDC_CLIENT_SECRET` | — |
| WS11 | Docs + API compliance matrix (this file) | **Done** | — | — | — | — | — |
| WS12 | 3-package uv workspace (standalone CLI) | **Done** | — | — | `open-arena-core` / `open-arena-cli` / `open-arena` | — | #58 |

---

## API route compliance

| Workstream | Paths | Status | Notes |
|---|---|---|---|
| Verifiers | `GET /v1/verifiers`, `POST /v1/verifiers`, `GET/PATCH/DELETE /v1/verifiers/{id}` | **Done** | CRUD + aggregation / metric-kind validation |
| Environments | `GET /v1/environments`, `POST /v1/environments`, `GET/PATCH/DELETE /v1/environments/{id}` | **Done** | Inline + git-backed resolution |
| Environment versions | `GET /v1/environments/{id}/versions` | **Done** | WS3 (#60); inline and git-backed version descriptors |
| Leaderboards | `GET /v1/leaderboards`, `POST /v1/leaderboards`, `GET/PATCH/DELETE /v1/leaderboards/{id}` | **Done** | CRUD + ranking policy |
| Model catalog | `GET/PUT/PATCH /v1/leaderboards/{id}/model-catalog` | **Done** | Replace + patch supported |
| Models | `GET/POST /v1/leaderboards/{id}/models`, `GET/PATCH/DELETE /v1/leaderboards/{id}/models/{mid}` | **Done** | Leaderboard-scoped CRUD |
| Environment membership | `GET/POST /v1/leaderboards/{id}/environments`, `GET/PATCH/DELETE /v1/leaderboards/{id}/environments/{eid}` | **Done** | Reusable refs and inline memberships |
| Leaderboard entries | `GET /v1/leaderboards/{id}/entries` | **Done** | Built from persisted run results |
| Discovery | `GET /v1/metric-kinds`, `GET /v1/aggregations`, `GET /v1/model-providers`, `GET /v1/dataset-providers` | **Done** | Unknown identifiers rejected with 400 |
| Runs | `GET/POST /v1/runs`, `GET /v1/runs/{id}`, `GET /v1/runs/{id}/results` | **Done** | Persistent run records + background execution + cached subject reuse |

---

## Known gaps — upstream Hub contract

The following fields are present in the upstream Hub OpenAPI contract but not yet returned by the Open Arena implementation.

### Leaderboard entry fields

| Field | Path | Status |
|---|---|---|
| `rank_delta` | `LeaderboardEntry.rank_delta` | Not implemented — requires history comparison across snapshots |
| `last_evaluated` | `LeaderboardEntry.last_evaluated` | Not implemented — timestamp of most recent completed run per (model × environment) |
| `run_count` | `LeaderboardEntry.run_count` | Not implemented — count of completed runs aggregated per entry |

### Leaderboard object fields

| Field | Path | Status |
|---|---|---|
| `modalities` | `Leaderboard.modalities` | Not implemented — enum list (text / code / vision / ...) |
| `industries` | `Leaderboard.industries` | Not implemented — free-form tag list |
| `mode` | `Leaderboard.mode` | Not implemented — `public` / `private` / `org-private` |
| `owner_org` | `Leaderboard.owner_org` | Not applicable — stack is single-tenant by design; org identity is implicit in the deployment |

### Model fields

| Field | Path | Status |
|---|---|---|
| `slug` | `ModelDefinition.slug` | Not implemented — URL-safe model identifier for Hub permalinks |
| `owner` | `ModelDefinition.owner` | Not implemented — org/user who registered the model |

---

## Current execution scope

- Generator and agent run submission are both modeled and persisted.
- Executable runs currently require inline environments because the existing
  runner still needs dataset/verifier materialization locally.
- The existing `.open-arena/` sweep cache is preserved as the execution cache,
  while `.open-arena/api.db` is the source of truth for API-visible state (SQLite)
  or `DATABASE_URL` when `OPEN_ARENA_STORE=postgres`.

_Part of #45 (WS11: docs + API compliance matrix)_
