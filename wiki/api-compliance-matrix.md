# API compliance matrix

OpenAPI contract source: `openapi.yaml`

Current backend implementation:

- API layer: `src/api/app.py`
- Persistence + services: `src/api/service.py`
- OpenAPI-derived schemas: `src/api/models.py`
- CLI bridge: `src/evaluate.py`

| Workstream | Paths | Status | Notes |
|---|---|---|---|
| Verifiers | `/v1/verifiers`, `/v1/verifiers/{verifier_id}` | Implemented | CRUD + aggregation / metric-kind validation |
| Environments | `/v1/environments`, `/v1/environments/{environment_id}` | Implemented | Inline environment validation + registry persistence |
| Leaderboards | `/v1/leaderboards`, `/v1/leaderboards/{leaderboard_id}` | Implemented | CRUD + ranking policy persistence |
| Model catalog | `/v1/leaderboards/{leaderboard_id}/model-catalog` | Implemented | Replace + patch supported |
| Models | `/v1/leaderboards/{leaderboard_id}/models`, `/v1/leaderboards/{leaderboard_id}/models/{model_id}` | Implemented | Leaderboard-scoped model CRUD |
| Environment membership | `/v1/leaderboards/{leaderboard_id}/environments`, `/v1/leaderboards/{leaderboard_id}/environments/{environment_id}` | Implemented | Reusable refs and inline memberships supported |
| Leaderboard entries | `/v1/leaderboards/{leaderboard_id}/entries` | Implemented | Built from persisted run results |
| Discovery | `/v1/metric-kinds`, `/v1/aggregations`, `/v1/model-providers`, `/v1/dataset-providers` | Implemented | Unknown discovery identifiers rejected with 400 |
| Runs | `/v1/runs`, `/v1/runs/{run_id}`, `/v1/runs/{run_id}/results` | Implemented | Persistent run records + background execution + cached subject reuse |

## Current execution scope

- Generator and agent run submission are both modeled and persisted.
- Executable runs currently require inline environments, because the existing
  runner still needs dataset/verifier materialization locally.
- The existing `.open-arena/` sweep cache is preserved as the execution cache,
  while `.open-arena/api.db` is now the source of truth for API-visible state.
