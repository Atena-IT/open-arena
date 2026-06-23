# License Apache 2.0: (c) 2026 Athena-Reply
"""Backend dispatcher for CLI commands.

Provides two backends:
- ``_RemoteBackend``: delegates to ArenaAPIClient (remote HTTP)
- ``_LocalBackend``: instantiates ArenaAPIService in-process (requires engine)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click

from open_arena_core.constants import DEFAULT_API_TOKEN

DEFAULT_SERVER = "http://127.0.0.1:8000"
_TOKEN_ENVVAR = "OPEN_ARENA_API_TOKEN"


def _default_token() -> str:
    import os
    return os.getenv(_TOKEN_ENVVAR, DEFAULT_API_TOKEN)


def _load_file(file_path: str | None) -> dict[str, Any] | None:
    """Return parsed JSON from *file_path*, or ``None`` if not given."""
    if file_path is None:
        return None
    return json.loads(Path(file_path).read_text())


class _RemoteBackend:
    """Thin wrapper around ArenaAPIClient for the remote path."""

    def __init__(self, server_url: str, token: str) -> None:
        from open_arena_core.client import ArenaAPIClient

        self._client = ArenaAPIClient(server_url, token=token)

    def get(self, path: str, **_params: Any) -> Any:
        return self._client.request("GET", path)

    def get_q(self, path: str, params: dict[str, Any]) -> Any:
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
    """Wraps ArenaAPIService for direct in-process calls.

    Requires ``open-arena`` (the engine package) to be installed.
    Raises a friendly ImportError with install instructions otherwise.
    """

    def __init__(self) -> None:
        try:
            from src.api.registry import build_adapters
            from src.api.service import ArenaAPIService
        except ImportError as exc:
            raise ImportError(
                "Local (in-process) mode requires the full Open Arena engine.\n"
                "Install it with:  pip install open-arena\n"
                f"(original error: {exc})"
            ) from exc

        self._svc = ArenaAPIService(adapters=build_adapters())

    @property
    def svc(self):
        return self._svc

    @staticmethod
    def _dump(obj: Any) -> Any:
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
    - ``local=True``              → _LocalBackend (requires engine)
    - server reachable via HTTP   → _RemoteBackend
    - server unreachable          → _LocalBackend (auto-fallback, requires engine)
    """
    if local:
        return _LocalBackend()
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


# Shared Click options

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
        "Instantiates ArenaAPIService directly (requires open-arena engine). "
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
