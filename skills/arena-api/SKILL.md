---
name: arena-api
description: Operate the Open Arena REST API: start the server and make authenticated requests.
---

Read `src/api/app.py` for the live route list and `openapi.yaml` for the full resource model and schemas. This skill covers the server startup and the CLI client. Run `arena --help` for the current command set.

## Start the API server

```bash
arena serve                          # binds to 127.0.0.1:8000 (default)
arena serve --host 0.0.0.0           # expose on all interfaces
arena serve --host 0.0.0.0 --port 9000
```

The server is a FastAPI app (`src/api/app.py`) served via uvicorn. It starts synchronously; Ctrl-C to stop.

## Authentication

All `/v1/...` routes require a Bearer token in the `Authorization` header.

Set the token via the environment variable:

```bash
export OPEN_ARENA_API_TOKEN=my-secret-token
arena serve                          # server reads the same env var to validate
```

Default when the variable is unset: `open-arena-dev-token` (development only — set a real token in production).

## Make requests with `arena request`

```bash
arena request GET /v1/verifiers
arena request GET /v1/leaderboards
arena request POST /v1/environments --file payload.json
arena request PATCH /v1/leaderboards/<id> --file update.json
arena request DELETE /v1/leaderboards/<id>
```

Options:
- `--server URL` — server base URL (default: `http://127.0.0.1:8000`)
- `--token TOKEN` — override the bearer token (default: `OPEN_ARENA_API_TOKEN` env var or `open-arena-dev-token`)
- `--file PATH` — JSON file sent as the request body (required for POST/PATCH/PUT)

The command prints the JSON response to stdout and exits. Errors print the server's `Error` JSON object.

## Resource model

The API is organized around **leaderboards** and **environments**:

- **Leaderboard** — defines the comparison perimeter: a scoped model catalog plus one or more environments. Can be created with full payload or as an empty shell.
- **Environment** — atomic eval artifact: benchmark workload + dataset definition + verifier ensemble. Can be reused across leaderboards.
- **Verifier suite** — reusable set of verifiers (rewards/metrics) that an environment references.
- **Run** — evaluation submission. Targets a leaderboard, a subset of environments/models, or an explicit model × environment slice.

### Key endpoints

```
GET    /healthz                                    health check (no auth)

GET    /v1/verifiers                               list verifier suites
POST   /v1/verifiers                               create verifier suite
GET    /v1/verifiers/{id}                          get verifier suite
PATCH  /v1/verifiers/{id}                          update verifier suite
DELETE /v1/verifiers/{id}                          delete verifier suite

GET    /v1/environments                            list environments
POST   /v1/environments                            create environment
GET    /v1/environments/{id}                       get environment
PATCH  /v1/environments/{id}                       update environment
DELETE /v1/environments/{id}                       delete environment

GET    /v1/leaderboards                            list leaderboards
POST   /v1/leaderboards                            create leaderboard
GET    /v1/leaderboards/{id}                       get leaderboard
PATCH  /v1/leaderboards/{id}                       update leaderboard metadata/ranking
DELETE /v1/leaderboards/{id}                       delete leaderboard
GET    /v1/leaderboards/{id}/model-catalog         get scoped model catalog
PUT    /v1/leaderboards/{id}/model-catalog         replace model catalog
PATCH  /v1/leaderboards/{id}/model-catalog         patch model catalog
GET    /v1/leaderboards/{id}/models                list models in catalog
POST   /v1/leaderboards/{id}/models                add model to catalog
GET    /v1/leaderboards/{id}/models/{mid}          get model
PATCH  /v1/leaderboards/{id}/models/{mid}          update model
DELETE /v1/leaderboards/{id}/models/{mid}          remove model
GET    /v1/leaderboards/{id}/environments          list environment memberships
POST   /v1/leaderboards/{id}/environments          add environment to leaderboard
GET    /v1/leaderboards/{id}/environments/{eid}    get membership
PATCH  /v1/leaderboards/{id}/environments/{eid}    update membership
DELETE /v1/leaderboards/{id}/environments/{eid}    remove environment from leaderboard
GET    /v1/leaderboards/{id}/entries               get leaderboard entries (results matrix)

GET    /v1/runs                                    list runs
POST   /v1/runs                                    submit run (returns 202 Accepted)
GET    /v1/runs/{id}                               get run status
GET    /v1/runs/{id}/results                       get run results

GET    /v1/metric-kinds                            discovery: available metric kinds
GET    /v1/aggregations                            discovery: available aggregations
GET    /v1/model-providers                         discovery: available model providers
GET    /v1/dataset-providers                       discovery: available dataset providers
```

Discovery endpoints (`/v1/metric-kinds`, `/v1/aggregations`, etc.) return the valid identifiers. Requests that reference an unknown identifier are rejected with HTTP 400.

## Common request patterns

Health check (no auth):
```bash
arena request GET /healthz
```

List all leaderboards:
```bash
arena request GET /v1/leaderboards
```

Create a leaderboard from a JSON file:
```bash
arena request POST /v1/leaderboards --file leaderboard.json
```

Submit a run:
```bash
arena request POST /v1/runs --file run_payload.json
```

Poll run status:
```bash
arena request GET /v1/runs/<run_id>
```

## Notes

- `arena serve` and `arena request` are subcommands of the `arena` group. The default behavior (no subcommand) still runs the local sweep.
- Full resource sub-commands (`arena env`, `arena verifier`, `arena leaderboard`, `arena run`, `arena discover`) and local-run-mode are available in `open-arena` (WS9). Install with `pip install open-arena` or `uv sync` from the repo root. Run `arena --help` for the current command set.
- The full OpenAPI spec is in `openapi.yaml` at the repo root. For schema details (request/response shapes, enum values, pagination cursors), read that file directly.
