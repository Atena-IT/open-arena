# License Apache 2.0: (c) 2026 Athena-Reply
"""``src.cli`` — Rich CLI sub-groups that expose every API resource.

All commands are registered under the :func:`~open_arena_cli.main.main` Click
group, which is the ``arena`` console-script entry-point.

Dispatch logic
--------------
Each command resolves to exactly one *backend*:

1. **Remote** (``--server`` given **and** ``--local`` not set):
   ``ArenaAPIClient`` issues HTTP requests to the given server URL.

2. **In-process / local** (``--local`` flag, **or** server is unreachable):
   ``ArenaAPIService`` is instantiated directly from
   ``build_adapters()``.  No HTTP hop.  The SQLite DB path is controlled
   by the ``OPEN_ARENA_DB_PATH`` environment variable (default:
   ``.open-arena/api.db``), so callers can point it at a temp file for
   isolated testing.

Precedence summary::

    explicit --server  →  remote client
    --local flag        →  in-process service (skips connectivity check)
    (default)           →  try remote; if unreachable → in-process service

Create / update commands accept a JSON ``--file`` argument.  The file is
parsed and forwarded as the request body (same semantics as
``arena request``).

All output is JSON on stdout.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from uuid import UUID

import click

from open_arena_core.constants import DEFAULT_API_TOKEN

# ---------------------------------------------------------------------------
# Backend dispatcher
# ---------------------------------------------------------------------------

DEFAULT_SERVER = "http://127.0.0.1:8000"
_TOKEN_ENVVAR = "OPEN_ARENA_API_TOKEN"


def _default_token() -> str:
    return os.getenv(_TOKEN_ENVVAR, DEFAULT_API_TOKEN)


def _load_file(file_path: str | None) -> dict[str, Any] | None:
    """Return parsed JSON from *file_path*, or ``None`` if not given."""
    if file_path is None:
        return None
    return json.loads(Path(file_path).read_text())


class _RemoteBackend:
    """Thin wrapper around :class:`ArenaAPIClient` for the remote path."""

    def __init__(self, server_url: str, token: str) -> None:
        from open_arena_core.client import ArenaAPIClient

        self._client = ArenaAPIClient(server_url, token=token)

    def get(self, path: str, **_params: Any) -> Any:
        """GET *path* with optional query params ignored (client builds its own URLs)."""
        return self._client.request("GET", path)

    def get_q(self, path: str, params: dict[str, Any]) -> Any:
        """GET with query string appended manually."""
        qs = "&".join(f"{k}={v}" for k, v in params.items() if v is not None)
        full_path = f"{path}?{qs}" if qs else path
        return self._client.request("GET", full_path)

    def post(self, path: str, payload: dict[str, Any] | None) -> Any:
        return self._client.request("POST", path, payload)

    def patch(self, path: str, payload: dict[str, Any]) -> Any:
        return self._client.request("PATCH", path, payload)

    def delete(self, path: str) -> None:
        self._client.request("DELETE", path)


class _LocalBackend:
    """Wraps :class:`ArenaAPIService` for direct in-process calls.

    Pydantic models are serialised back to dicts so the CLI layer sees
    the same JSON-like structure as the remote path.
    """

    def __init__(self) -> None:
        try:
            from src.api.registry import build_adapters
            from src.api.service import ArenaAPIService
        except ImportError as exc:
            msg = (
                "Local (in-process) mode requires the full Open Arena engine." + chr(10) +
                "Install it with:  pip install open-arena" + chr(10) +
                f"(original error: {exc})"
            )
            raise ImportError(msg) from exc

        self._svc = ArenaAPIService(adapters=build_adapters())

    @property
    def svc(self):
        return self._svc

    @staticmethod
    def _dump(obj: Any) -> Any:
        """Convert a Pydantic model (or list/dict of them) to JSON-safe dicts."""
        if obj is None:
            return None
        if hasattr(obj, "model_dump_json"):
            return json.loads(obj.model_dump_json())
        return obj


def _make_backend(
    server_url: str,
    token: str,
    *,
    local: bool,
) -> _RemoteBackend | _LocalBackend:
    """Pick and return the correct backend.

    Priority:
    - ``local=True``              → :class:`_LocalBackend`
    - server reachable via HTTP   → :class:`_RemoteBackend`
    - server unreachable          → :class:`_LocalBackend` (auto-fallback)
    """
    if local:
        return _LocalBackend()
    # Try a quick HEAD to check reachability.
    try:
        import httpx

        with httpx.Client(timeout=3.0) as c:
            c.head(server_url + "/healthz")
        return _RemoteBackend(server_url, token)
    except Exception:
        click.echo(
            f"[arena] server {server_url!r} unreachable — running in local (in-process) mode.",
            err=True,
        )
        return _LocalBackend()


# ---------------------------------------------------------------------------
# Shared options used by every sub-command
# ---------------------------------------------------------------------------

_SERVER_OPTION = click.option(
    "--server",
    "server_url",
    default=DEFAULT_SERVER,
    show_default=True,
    envvar="OPEN_ARENA_SERVER",
    help="Base URL of the Open Arena API server.",
)
_TOKEN_OPTION = click.option(
    "--token",
    default=_default_token,
    show_default=f"${_TOKEN_ENVVAR} or {DEFAULT_API_TOKEN}",
    help="Bearer token for API authentication.",
)
_LOCAL_OPTION = click.option(
    "--local",
    "local",
    is_flag=True,
    default=False,
    help=(
        "Run in-process without a server. "
        "Instantiates ArenaAPIService directly (no HTTP). "
        "Set OPEN_ARENA_DB_PATH to control the SQLite file."
    ),
)
_FILE_OPTION = click.option(
    "--file",
    "file_path",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    default=None,
    help="Path to a JSON file whose contents are sent as the request body.",
)
_LIMIT_OPTION = click.option("--limit", default=50, show_default=True, help="Page size.")
_CURSOR_OPTION = click.option("--cursor", default=None, help="Pagination cursor.")


def _out(obj: Any) -> None:
    """Print *obj* as pretty JSON on stdout."""
    if obj is None:
        return
    if hasattr(obj, "model_dump_json"):
        click.echo(obj.model_dump_json(indent=2))
    else:
        click.echo(json.dumps(obj, indent=2, default=str))


# ===========================================================================
# arena env
# ===========================================================================


@click.group("env")
def env_group():
    """Manage reusable evaluation environments."""


@env_group.command("list")
@_SERVER_OPTION
@_TOKEN_OPTION
@_LOCAL_OPTION
@_LIMIT_OPTION
@_CURSOR_OPTION
@click.option("--source-kind", default=None, help="Filter by source kind.")
@click.option("--mode", default=None, help="Filter by run mode (generator|agent).")
@click.option("--name", default=None, help="Exact environment name.")
def env_list(server_url, token, local, limit, cursor, source_kind, mode, name):
    """List reusable environments."""
    backend = _make_backend(server_url, token, local=local)
    if isinstance(backend, _LocalBackend):
        from open_arena_core import models as api

        result = backend.svc.list_environments(
            source_kind=api.EnvironmentSourceKind(source_kind) if source_kind else None,
            mode=api.RunMode(mode) if mode else None,
            name=name,
            limit=limit,
            cursor=cursor,
        )
        _out(result)
    else:
        params = {"limit": limit, "cursor": cursor, "source_kind": source_kind, "mode": mode, "name": name}
        _out(backend.get_q("/v1/environments", params))


@env_group.command("get")
@click.argument("environment_id")
@_SERVER_OPTION
@_TOKEN_OPTION
@_LOCAL_OPTION
def env_get(environment_id, server_url, token, local):
    """Get a reusable environment by ID."""
    backend = _make_backend(server_url, token, local=local)
    if isinstance(backend, _LocalBackend):
        result = backend.svc.get_environment(UUID(environment_id))
        _out(result)
    else:
        _out(backend.get(f"/v1/environments/{environment_id}"))


@env_group.command("create")
@_SERVER_OPTION
@_TOKEN_OPTION
@_LOCAL_OPTION
@_FILE_OPTION
def env_create(server_url, token, local, file_path):
    """Create a reusable environment (body from --file)."""
    payload = _load_file(file_path)
    if payload is None:
        raise click.UsageError("--file is required for env create")
    backend = _make_backend(server_url, token, local=local)
    if isinstance(backend, _LocalBackend):
        from open_arena_core import models as api

        result = backend.svc.create_environment(api.EnvironmentCreate.model_validate(payload))
        _out(result)
    else:
        _out(backend.post("/v1/environments", payload))


@env_group.command("delete")
@click.argument("environment_id")
@_SERVER_OPTION
@_TOKEN_OPTION
@_LOCAL_OPTION
def env_delete(environment_id, server_url, token, local):
    """Delete a reusable environment."""
    backend = _make_backend(server_url, token, local=local)
    if isinstance(backend, _LocalBackend):
        backend.svc.delete_environment(UUID(environment_id))
    else:
        backend.delete(f"/v1/environments/{environment_id}")
    click.echo(f"deleted {environment_id}")


# ===========================================================================
# arena verifier
# ===========================================================================


@click.group("verifier")
def verifier_group():
    """Manage reusable verifier suites."""


@verifier_group.command("list")
@_SERVER_OPTION
@_TOKEN_OPTION
@_LOCAL_OPTION
@_LIMIT_OPTION
@_CURSOR_OPTION
def verifier_list(server_url, token, local, limit, cursor):
    """List verifier suites."""
    backend = _make_backend(server_url, token, local=local)
    if isinstance(backend, _LocalBackend):
        result = backend.svc.list_verifiers(limit=limit, cursor=cursor)
        _out(result)
    else:
        params = {"limit": limit, "cursor": cursor}
        _out(backend.get_q("/v1/verifiers", params))


@verifier_group.command("get")
@click.argument("verifier_id")
@_SERVER_OPTION
@_TOKEN_OPTION
@_LOCAL_OPTION
def verifier_get(verifier_id, server_url, token, local):
    """Get a verifier suite by ID."""
    backend = _make_backend(server_url, token, local=local)
    if isinstance(backend, _LocalBackend):
        result = backend.svc.get_verifier(UUID(verifier_id))
        _out(result)
    else:
        _out(backend.get(f"/v1/verifiers/{verifier_id}"))


@verifier_group.command("create")
@_SERVER_OPTION
@_TOKEN_OPTION
@_LOCAL_OPTION
@_FILE_OPTION
def verifier_create(server_url, token, local, file_path):
    """Create a verifier suite (body from --file)."""
    payload = _load_file(file_path)
    if payload is None:
        raise click.UsageError("--file is required for verifier create")
    backend = _make_backend(server_url, token, local=local)
    if isinstance(backend, _LocalBackend):
        from open_arena_core import models as api

        result = backend.svc.create_verifier(api.VerifierSuiteCreate.model_validate(payload))
        _out(result)
    else:
        _out(backend.post("/v1/verifiers", payload))


@verifier_group.command("delete")
@click.argument("verifier_id")
@_SERVER_OPTION
@_TOKEN_OPTION
@_LOCAL_OPTION
def verifier_delete(verifier_id, server_url, token, local):
    """Delete a verifier suite."""
    backend = _make_backend(server_url, token, local=local)
    if isinstance(backend, _LocalBackend):
        backend.svc.delete_verifier(UUID(verifier_id))
    else:
        backend.delete(f"/v1/verifiers/{verifier_id}")
    click.echo(f"deleted {verifier_id}")


# ===========================================================================
# arena leaderboard
# ===========================================================================


@click.group("leaderboard")
def leaderboard_group():
    """Manage leaderboards, model catalogs, and environment memberships."""


@leaderboard_group.command("list")
@_SERVER_OPTION
@_TOKEN_OPTION
@_LOCAL_OPTION
@_LIMIT_OPTION
@_CURSOR_OPTION
@click.option("--visibility", default=None, help="Filter: private|organization|public.")
def leaderboard_list(server_url, token, local, limit, cursor, visibility):
    """List leaderboards."""
    backend = _make_backend(server_url, token, local=local)
    if isinstance(backend, _LocalBackend):
        from open_arena_core import models as api

        result = backend.svc.list_leaderboards(
            visibility=api.LeaderboardVisibility(visibility) if visibility else None,
            limit=limit,
            cursor=cursor,
        )
        _out(result)
    else:
        params = {"limit": limit, "cursor": cursor, "visibility": visibility}
        _out(backend.get_q("/v1/leaderboards", params))


@leaderboard_group.command("get")
@click.argument("leaderboard_id")
@_SERVER_OPTION
@_TOKEN_OPTION
@_LOCAL_OPTION
def leaderboard_get(leaderboard_id, server_url, token, local):
    """Get a leaderboard by ID."""
    backend = _make_backend(server_url, token, local=local)
    if isinstance(backend, _LocalBackend):
        result = backend.svc.get_leaderboard(UUID(leaderboard_id))
        _out(result)
    else:
        _out(backend.get(f"/v1/leaderboards/{leaderboard_id}"))


@leaderboard_group.command("create")
@_SERVER_OPTION
@_TOKEN_OPTION
@_LOCAL_OPTION
@_FILE_OPTION
def leaderboard_create(server_url, token, local, file_path):
    """Create a leaderboard (body from --file)."""
    payload = _load_file(file_path)
    if payload is None:
        raise click.UsageError("--file is required for leaderboard create")
    backend = _make_backend(server_url, token, local=local)
    if isinstance(backend, _LocalBackend):
        from open_arena_core import models as api

        result = backend.svc.create_leaderboard(api.LeaderboardCreate.model_validate(payload))
        _out(result)
    else:
        _out(backend.post("/v1/leaderboards", payload))


@leaderboard_group.command("delete")
@click.argument("leaderboard_id")
@_SERVER_OPTION
@_TOKEN_OPTION
@_LOCAL_OPTION
def leaderboard_delete(leaderboard_id, server_url, token, local):
    """Delete a leaderboard."""
    backend = _make_backend(server_url, token, local=local)
    if isinstance(backend, _LocalBackend):
        backend.svc.delete_leaderboard(UUID(leaderboard_id))
    else:
        backend.delete(f"/v1/leaderboards/{leaderboard_id}")
    click.echo(f"deleted {leaderboard_id}")


@leaderboard_group.command("models")
@click.argument("leaderboard_id")
@_SERVER_OPTION
@_TOKEN_OPTION
@_LOCAL_OPTION
@_LIMIT_OPTION
@_CURSOR_OPTION
def leaderboard_models(leaderboard_id, server_url, token, local, limit, cursor):
    """List models in a leaderboard's model catalog."""
    backend = _make_backend(server_url, token, local=local)
    if isinstance(backend, _LocalBackend):
        result = backend.svc.list_models(UUID(leaderboard_id), limit=limit, cursor=cursor)
        _out(result)
    else:
        params = {"limit": limit, "cursor": cursor}
        _out(backend.get_q(f"/v1/leaderboards/{leaderboard_id}/models", params))


@leaderboard_group.command("environments")
@click.argument("leaderboard_id")
@_SERVER_OPTION
@_TOKEN_OPTION
@_LOCAL_OPTION
@_LIMIT_OPTION
@_CURSOR_OPTION
def leaderboard_environments(leaderboard_id, server_url, token, local, limit, cursor):
    """List environment memberships in a leaderboard."""
    backend = _make_backend(server_url, token, local=local)
    if isinstance(backend, _LocalBackend):
        result = backend.svc.list_memberships(UUID(leaderboard_id), limit=limit, cursor=cursor)
        _out(result)
    else:
        params = {"limit": limit, "cursor": cursor}
        _out(backend.get_q(f"/v1/leaderboards/{leaderboard_id}/environments", params))


@leaderboard_group.command("entries")
@click.argument("leaderboard_id")
@_SERVER_OPTION
@_TOKEN_OPTION
@_LOCAL_OPTION
@_LIMIT_OPTION
@_CURSOR_OPTION
@click.option("--environment-id", default=None, help="Filter by environment UUID.")
@click.option("--model-id", default=None, help="Filter by model UUID.")
@click.option("--as-of", default=None, help="ISO-8601 datetime for point-in-time query.")
def leaderboard_entries(leaderboard_id, server_url, token, local, limit, cursor, environment_id, model_id, as_of):
    """List leaderboard entries (ranking)."""
    backend = _make_backend(server_url, token, local=local)
    if isinstance(backend, _LocalBackend):
        from datetime import datetime

        result = backend.svc.list_leaderboard_entries(
            UUID(leaderboard_id),
            environment_id=UUID(environment_id) if environment_id else None,
            model_id=UUID(model_id) if model_id else None,
            as_of=datetime.fromisoformat(as_of) if as_of else None,
            limit=limit,
            cursor=cursor,
        )
        _out(result)
    else:
        params = {
            "limit": limit,
            "cursor": cursor,
            "environment_id": environment_id,
            "model_id": model_id,
            "as_of": as_of,
        }
        _out(backend.get_q(f"/v1/leaderboards/{leaderboard_id}/entries", params))


# ===========================================================================
# arena run
# ===========================================================================


@click.group("run")
def run_group():
    """Submit and inspect evaluation runs.

    When --local is used (or the server is unreachable), runs execute
    in-process via ArenaAPIService — no HTTP server required.
    """


@run_group.command("submit")
@_SERVER_OPTION
@_TOKEN_OPTION
@_LOCAL_OPTION
@_FILE_OPTION
def run_submit(server_url, token, local, file_path):
    """Submit an evaluation run (body from --file).

    Local mode: ArenaAPIService is instantiated in-process and the
    run executes synchronously using the local SQLite store.
    """
    payload = _load_file(file_path)
    if payload is None:
        raise click.UsageError("--file is required for run submit")
    backend = _make_backend(server_url, token, local=local)
    if isinstance(backend, _LocalBackend):
        from open_arena_core import models as api

        result = backend.svc.create_run(api.RunCreate.model_validate(payload))
        _out(result)
    else:
        _out(backend.post("/v1/runs", payload))


@run_group.command("get")
@click.argument("run_id")
@_SERVER_OPTION
@_TOKEN_OPTION
@_LOCAL_OPTION
def run_get(run_id, server_url, token, local):
    """Get an evaluation run by ID."""
    backend = _make_backend(server_url, token, local=local)
    if isinstance(backend, _LocalBackend):
        result = backend.svc.get_run(UUID(run_id))
        _out(result)
    else:
        _out(backend.get(f"/v1/runs/{run_id}"))


@run_group.command("results")
@click.argument("run_id")
@_SERVER_OPTION
@_TOKEN_OPTION
@_LOCAL_OPTION
def run_results(run_id, server_url, token, local):
    """Get the results for a completed evaluation run."""
    backend = _make_backend(server_url, token, local=local)
    if isinstance(backend, _LocalBackend):
        result = backend.svc.get_run_result(UUID(run_id))
        _out(result)
    else:
        _out(backend.get(f"/v1/runs/{run_id}/results"))


@run_group.command("list")
@_SERVER_OPTION
@_TOKEN_OPTION
@_LOCAL_OPTION
@_LIMIT_OPTION
@_CURSOR_OPTION
@click.option("--leaderboard-id", default=None, help="Filter by leaderboard UUID.")
@click.option("--status", default=None, help="Filter by status (queued|running|succeeded|failed|cancelled).")
@click.option("--mode", default=None, help="Filter by mode (generator|agent).")
@click.option("--cache-status", default=None, help="Filter by cache status.")
def run_list(server_url, token, local, limit, cursor, leaderboard_id, status, mode, cache_status):
    """List evaluation runs."""
    backend = _make_backend(server_url, token, local=local)
    if isinstance(backend, _LocalBackend):
        from open_arena_core import models as api

        result = backend.svc.list_runs(
            leaderboard_id=UUID(leaderboard_id) if leaderboard_id else None,
            status=api.RunStatus(status) if status else None,
            mode=api.RunMode(mode) if mode else None,
            cache_status=api.CacheStatus(cache_status) if cache_status else None,
            limit=limit,
            cursor=cursor,
        )
        _out(result)
    else:
        params = {
            "limit": limit,
            "cursor": cursor,
            "leaderboard_id": leaderboard_id,
            "status": status,
            "mode": mode,
            "cache_status": cache_status,
        }
        _out(backend.get_q("/v1/runs", params))


# ===========================================================================
# arena discover
# ===========================================================================


@click.group("discover")
def discover_group():
    """Query server discovery endpoints (metric kinds, aggregations, providers)."""


@discover_group.command("metric-kinds")
@_SERVER_OPTION
@_TOKEN_OPTION
@_LOCAL_OPTION
def discover_metric_kinds(server_url, token, local):
    """List supported metric/reward identifiers."""
    backend = _make_backend(server_url, token, local=local)
    if isinstance(backend, _LocalBackend):
        _out(backend.svc.metric_kinds())
    else:
        _out(backend.get("/v1/metric-kinds"))


@discover_group.command("aggregations")
@_SERVER_OPTION
@_TOKEN_OPTION
@_LOCAL_OPTION
def discover_aggregations(server_url, token, local):
    """List supported aggregation identifiers."""
    backend = _make_backend(server_url, token, local=local)
    if isinstance(backend, _LocalBackend):
        _out(backend.svc.aggregations())
    else:
        _out(backend.get("/v1/aggregations"))


@discover_group.command("model-providers")
@_SERVER_OPTION
@_TOKEN_OPTION
@_LOCAL_OPTION
def discover_model_providers(server_url, token, local):
    """List supported model provider identifiers."""
    backend = _make_backend(server_url, token, local=local)
    if isinstance(backend, _LocalBackend):
        _out(backend.svc.model_providers())
    else:
        _out(backend.get("/v1/model-providers"))


@discover_group.command("dataset-providers")
@_SERVER_OPTION
@_TOKEN_OPTION
@_LOCAL_OPTION
def discover_dataset_providers(server_url, token, local):
    """List supported dataset provider identifiers."""
    backend = _make_backend(server_url, token, local=local)
    if isinstance(backend, _LocalBackend):
        _out(backend.svc.dataset_providers())
    else:
        _out(backend.get("/v1/dataset-providers"))


# ===========================================================================
# arena eval
# ===========================================================================


@click.group("eval")
def eval_group():
    """Submit evaluation jobs from an EvalEnvironment manifest (eval.yaml)."""


@eval_group.command("submit")
@_SERVER_OPTION
@_TOKEN_OPTION
@_LOCAL_OPTION
@click.option(
    "--file",
    "file_path",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    default=None,
    help="Path to an eval.yaml manifest file.",
)
@click.option(
    "--dir",
    "dir_path",
    type=click.Path(exists=True, file_okay=False, readable=True),
    default=None,
    help="Path to a directory containing an eval.yaml manifest.",
)
def eval_submit(server_url, token, local, file_path, dir_path):
    """Submit an evaluation job from an EvalEnvironment manifest.

    Accepts --file eval.yaml or --dir ./my-eval (must contain eval.yaml).
    The manifest is translated to a RunCreate payload and submitted via
    POST /v1/runs (no new endpoints).
    """
    if file_path is None and dir_path is None:
        raise click.UsageError("Either --file or --dir is required for eval submit")
    if file_path is not None and dir_path is not None:
        raise click.UsageError("Specify only one of --file or --dir, not both")

    manifest_path = file_path or dir_path
    # Manifest loader lives in the thin core package so this command works on
    # a standalone CLI install (no engine required) for both local & remote.
    from open_arena_core.manifest import load_manifest
    run_create = load_manifest(manifest_path)

    backend = _make_backend(server_url, token, local=local)
    if isinstance(backend, _LocalBackend):
        result = backend.svc.create_run(run_create)
        _out(result)
    else:
        payload = run_create.model_dump(mode="json")
        _out(backend.post("/v1/runs", payload))


# ---------------------------------------------------------------------------
# Public API: the list of groups to attach to the main Click group
# ---------------------------------------------------------------------------

CLI_GROUPS = [
    env_group,
    verifier_group,
    leaderboard_group,
    run_group,
    discover_group,
    eval_group,
]
