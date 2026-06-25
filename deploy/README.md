# Open Arena — deployment guide

This document covers:

1. [Local development with Docker Compose](#local-development-docker-compose)
2. [Kubernetes via Helm](#kubernetes-via-helm)
3. [Environment variable contract](#environment-variable-contract)

---

## Local development (Docker Compose)

### Prerequisites

- Docker Engine >= 24 with the Compose plugin (`docker compose version`)
- A copy of `.env.example` filled in as `.env`

### Quick start (lean stack — SQLite + Postgres)

```bash
cp .env.example .env
# Edit .env — at minimum set OPEN_ARENA_API_TOKEN

docker compose up --build
```

This starts:

| Service | URL | Notes |
|---|---|---|
| `api` | http://localhost:8000 | Open Arena FastAPI |
| `postgres` | localhost:5432 | forward-looking for WS1; api wired to it via `DATABASE_URL` |

The API state dir (`.open-arena/`) is stored in a named Docker volume
(`open_arena_state`) so trial caches survive container restarts.

### Full stack (+ MinIO + MLflow)

```bash
docker compose --profile full up --build
```

Additional services:

| Service | URL | Notes |
|---|---|---|
| `minio` | http://localhost:9000 (API) / :9001 (console) | S3-compatible object store |
| `mlflow` | http://localhost:5000 | Experiment tracking |

### Useful commands

```bash
# Build only (no start)
docker compose build

# Tail API logs
docker compose logs -f api

# Validate compose config (no runtime required)
docker compose config -q

# Stop and remove containers (volumes kept)
docker compose down

# Stop and remove containers + volumes (destructive)
docker compose down -v
```

---

## Kubernetes via Helm

### Prerequisites

- Helm >= 3.12
- A running Kubernetes cluster (minikube, k3s, EKS, AKS, …)
- `kubectl` configured for the target cluster

### Standalone install

```bash
# Lint first (static validation)
helm lint helm/open-arena

# Dry-run render to inspect manifests
helm template open-arena helm/open-arena \
  --set secrets.OPEN_ARENA_API_TOKEN=change-me

# Install into its own namespace
helm upgrade --install open-arena helm/open-arena \
  --namespace open-arena --create-namespace \
  --set secrets.OPEN_ARENA_API_TOKEN=your-token \
  --set ingress.enabled=true \
  --set org=myorg
```

The ingress hostname follows the pattern:
`arena.<org>.dev.reply-modelfactory.com`

Override with `ingress.hosts[0].host` and set
`ingress.useOrgHostPattern=false` if you need a custom host.

### Per-org values file

```yaml
# values-acme.yaml
org: acme
image:
  tag: "0.1.0"
ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
  tls:
    - secretName: open-arena-acme-tls
      hosts:
        - arena.acme.dev.reply-modelfactory.com
secrets:
  OPEN_ARENA_API_TOKEN: "..."
  GITEA_TOKEN: "..."
  AWS_ACCESS_KEY_ID: "..."
  AWS_SECRET_ACCESS_KEY: "..."
env:
  GITEA_BASE_URL: "https://gitea.acme.internal"
  GITEA_ORG: "acme"
  DATABASE_URL: "postgresql://arena:password@postgres:5432/arena"
  S3_ENDPOINT: "https://s3.fr-par.scw.cloud"
  S3_BUCKET: "arena-acme"
  MLFLOW_TRACKING_URI: "https://mlflow.acme.internal"
```

```bash
helm upgrade --install open-arena-acme helm/open-arena \
  --namespace open-arena-acme --create-namespace \
  --values values-acme.yaml
```

### ModelFactory org-node sub-chart

Embed open-arena as a sub-chart of the ModelFactory org-node chart.
See [`helm/open-arena/README.md`](../helm/open-arena/README.md) for the
full parent-chart integration pattern.

### Using an external secret

Instead of setting `secrets.*` in values (which stores plain text in
Helm releases), reference a pre-existing Kubernetes Secret:

```yaml
existingSecret: "open-arena-acme-ext-secret"
```

The secret must contain all keys listed in the [env var contract](#environment-variable-contract)
that your deployment needs. Typically managed via External Secrets Operator
or Vault Agent Injector.

---

## Environment variable contract

All variable names are exact — no aliases.

### API

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPEN_ARENA_API_TOKEN` | Yes | — | Bearer token for all API requests (`Authorization: Bearer <token>`). |

### Database

| Variable | Required | Default | Description |
|---|---|---|---|
| `DATABASE_URL` | No | SQLite at `.open-arena/api.db` | Postgres DSN. Forward-looking for WS1 (Postgres adapter). Leave empty to use SQLite. |

### Gitea

| Variable | Required | Default | Description |
|---|---|---|---|
| `GITEA_BASE_URL` | No | — | Base URL of the Gitea instance (e.g. `https://gitea.example.com`). |
| `GITEA_TOKEN` | No | — | Gitea personal access token. |
| `GITEA_ORG` | No | — | Default Gitea organization name. |

### Databricks Unity Catalog

| Variable | Required | Default | Description |
|---|---|---|---|
| `UNITY_CATALOG_API_URL` | No | — | Unity Catalog REST API base URL. |
| `UC_TOKEN` | No | — | Unity Catalog PAT / Databricks token. |

### MLflow

| Variable | Required | Default | Description |
|---|---|---|---|
| `MLFLOW_TRACKING_URI` | No | — | MLflow tracking server URI. |
| `MLFLOW_TRACKING_TOKEN` | No | — | Bearer token for a remote MLflow server. |

### S3 / object storage

| Variable | Required | Default | Description |
|---|---|---|---|
| `S3_ENDPOINT` | No | — | S3-compatible endpoint URL (e.g. `https://s3.fr-par.scw.cloud` or `http://minio:9000`). |
| `S3_BUCKET` | No | — | Default bucket name. |
| `AWS_ACCESS_KEY_ID` | No | — | S3 / AWS access key. Also used by MinIO. |
| `AWS_SECRET_ACCESS_KEY` | No | — | S3 / AWS secret key. |
| `AWS_DEFAULT_REGION` | No | `fr-par` | AWS / Scaleway region. |

### OIDC / SSO

| Variable | Required | Default | Description |
|---|---|---|---|
| `OIDC_ISSUER` | No | — | OIDC issuer URL (e.g. `https://auth.example.com`). |
| `OIDC_CLIENT_ID` | No | — | OIDC application client ID. |
| `OIDC_CLIENT_SECRET` | No | — | OIDC application client secret. |

### E2B sandbox

| Variable | Required | Default | Description |
|---|---|---|---|
| `E2B_API_KEY` | No | — | API key for the [E2B](https://e2b.dev) remote code sandbox. |
| `OPEN_ARENA_SANDBOX` | No | `local` | Sandbox backend: `local` (no isolation) or `e2b`. |

---

_Part of #42 (WS8: Docker Compose + Helm)_
