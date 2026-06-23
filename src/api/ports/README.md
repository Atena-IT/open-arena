# Open Arena — Port/Adapter Architecture

This directory defines the **port interfaces** for Open Arena's API layer.
Each port is an Abstract Base Class (ABC) with one or more default adapters.
The pattern deliberately separates the *what* (port ABC) from the *how*
(concrete adapter) so other workstreams can swap in new implementations without
touching `ArenaAPIService` or the HTTP layer.

---

## Ports at a glance

| Module                  | Port ABC              | Default adapter             | Env var                      | WS TODO |
|-------------------------|-----------------------|-----------------------------|------------------------------|---------|
| `store.py`              | `Store`               | `SQLiteStore`               | `OPEN_ARENA_STORE=sqlite`    | —       |
| `environment_backend.py`| `EnvironmentBackend`  | `InlineEnvironmentBackend`  | `OPEN_ARENA_ENV_BACKEND=inline` | WS2 (Gitea/GitHub) |
| `dataset_resolver.py`   | `DatasetResolver`     | `LegacyDatasetResolver`     | `OPEN_ARENA_DATASET_RESOLVER=legacy` | WS4 (Unity Catalog) |
| `results_sink.py`       | `ResultsSink`         | `StoreResultsSink`          | `OPEN_ARENA_RESULTS_SINK=store` | WS5 (MLflow) |
| `sandbox_provider.py`   | `SandboxProvider`     | `LocalSandboxProvider`      | `OPEN_ARENA_SANDBOX=local`   | WS6 (E2B-compatible) |
| `auth_provider.py`      | `AuthProvider`        | `StaticBearerAuthProvider`  | `OPEN_ARENA_AUTH=static`     | WS7 (Keycloak) |

---

## How to register a new adapter

Follow these four steps:

### 1. Implement the port ABC

Create your adapter in `src/api/` (or a sub-package if it has significant
code):

```python
# src/api/stores/dynamodb.py
from src.api.ports.store import Store
from src.api import models as api

class DynamoDBStore(Store):
    def save_verifier(self, verifier: api.VerifierSuite) -> None:
        ...  # DynamoDB put_item
    # ... implement every abstractmethod
```

### 2. Add a branch in `src/api/registry.py`

Find the builder function for your port (e.g. `_build_store`) and add an
`elif` branch:

```python
def _build_store(settings: ArenaSettings) -> Store:
    if settings.store == "sqlite":
        from src.api.stores.sqlite import SQLiteStore
        return SQLiteStore(path=settings.db_path)
    elif settings.store == "dynamodb":          # ← add this
        from src.api.stores.dynamodb import DynamoDBStore
        return DynamoDBStore(table=os.getenv("OPEN_ARENA_DYNAMO_TABLE", "open-arena"))
    raise ValueError(...)
```

### 3. Document the new env var in `src/api/settings.py`

Add a row to the table in the module docstring and (if a new configuration
knob is needed) a new field to `ArenaSettings`.

### 4. Update this README table

Add a row to the table above with the port, adapter name, env var key, and
the workstream that owns it.

---

## Wiring diagram

```
HTTP request
     │
     ▼
app.py  ──require_bearer──▶  AuthProvider (WS7: Keycloak)
     │
     ▼
ArenaAPIService.__init__(adapters=AdapterSet)
     │
     ├─── store               ──▶  Store           (WS: DynamoDB / Postgres …)
     ├─── env_backend         ──▶  EnvironmentBackend  (WS2: Gitea/GitHub)
     ├─── dataset_resolver    ──▶  DatasetResolver  (WS4: Unity Catalog)
     ├─── results_sink        ──▶  ResultsSink      (WS5: MLflow)
     └─── sandbox             ──▶  SandboxProvider  (WS6: E2B-compatible)
```

All adapters are instantiated once at startup by `build_adapters()` in
`src/api/registry.py` and injected into `ArenaAPIService` via
`ArenaAPIService(adapters=...)`.

---

## Adding a composite / chained adapter

Some ports (e.g. `ResultsSink`) benefit from *multicasting* — write to the
default `StoreResultsSink` **and** to MLflow.  Implement a wrapper:

```python
class MulticastResultsSink(ResultsSink):
    def __init__(self, sinks: list[ResultsSink]) -> None:
        self._sinks = sinks

    def write(self, run, result) -> None:
        for sink in self._sinks:
            sink.write(run, result)
```

Then wire it in `_build_results_sink`:

```python
elif settings.results_sink == "mlflow":
    store_sink = StoreResultsSink(store=store)
    mlflow_sink = MlflowResultsSink(tracking_uri=os.getenv("MLFLOW_TRACKING_URI"))
    return MulticastResultsSink([store_sink, mlflow_sink])
```
