# License Apache 2.0: (c) 2026 Athena-Reply
"""Smoke tests for the ports / registry layer (WS-PORTS, issue #34).

These tests assert:
1. ``build_adapters()`` produces a valid ``AdapterSet`` with all six ports.
2. The ``AdapterSet`` instances are the expected default adapter types.
3. The FastAPI app has exactly the same number of routes as before the
   refactor.
4. ``ArenaAPIService`` can be instantiated with the registry-built adapters.
5. ``ArenaAPIService(store=<store>)`` (legacy kwarg) still works.
6. ``StaticBearerAuthProvider`` accepts the default dev token.
7. ``StaticBearerAuthProvider`` rejects an invalid token.
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Registry / AdapterSet
# ---------------------------------------------------------------------------

def test_build_adapters_returns_complete_adapter_set():
    from src.api.registry import build_adapters, AdapterSet

    adapters = build_adapters()
    assert isinstance(adapters, AdapterSet)


def test_default_adapters_are_correct_types(monkeypatch, tmp_path):
    # P2-1: default env_backend is now "dispatch" (DispatchingEnvironmentBackend)
    from src.api.registry import build_adapters
    from src.api.settings import ArenaSettings
    from src.api.stores.sqlite import SQLiteStore
    from src.api.ports.auth_provider import StaticBearerAuthProvider
    from src.api.environments.dispatching import DispatchingEnvironmentBackend
    from src.api.ports.dataset_resolver import LegacyDatasetResolver
    from src.api.ports.results_sink import StoreResultsSink
    from src.api.ports.sandbox_provider import LocalSandboxProvider

    monkeypatch.delenv("OPEN_ARENA_ENV_BACKEND", raising=False)
    settings = ArenaSettings(db_path=tmp_path / "test.db")
    adapters = build_adapters(settings)
    assert isinstance(adapters.store, SQLiteStore)
    assert isinstance(adapters.auth, StaticBearerAuthProvider)
    assert isinstance(adapters.env_backend, DispatchingEnvironmentBackend)
    assert isinstance(adapters.dataset_resolver, LegacyDatasetResolver)
    assert isinstance(adapters.results_sink, StoreResultsSink)
    assert isinstance(adapters.sandbox, LocalSandboxProvider)


def test_explicit_inline_env_backend(tmp_path):
    from src.api.registry import build_adapters
    from src.api.settings import ArenaSettings
    from src.api.ports.environment_backend import InlineEnvironmentBackend

    settings = ArenaSettings(env_backend="inline", db_path=tmp_path / "test.db")
    adapters = build_adapters(settings)
    assert isinstance(adapters.env_backend, InlineEnvironmentBackend)

# ---------------------------------------------------------------------------
# Route count
# ---------------------------------------------------------------------------

def test_app_route_count_unchanged():
    """Assert the expected route count after the ports refactor + WS3 versioning.

    ``len(app.routes)`` returns 43 because FastAPI adds 4 internal routes
    for the OpenAPI schema, docs UI, docs oauth2 redirect and redoc UI.
    Of those, 39 are proper ``APIRoute`` instances (our routes + /healthz).
    WS3 added one route (GET /v1/environments/{id}/versions), bringing the
    total from 42 → 43 and the API routes from 38 → 39.
    We assert both numbers to make accidental additions/removals obvious.
    """
    from src.api.app import app
    from fastapi.routing import APIRoute

    total_routes = len(app.routes)
    api_route_count = sum(1 for r in app.routes if isinstance(r, APIRoute))

    # Total (including FastAPI internal routes: openapi.json, docs, redoc, …)
    assert total_routes == 43, (
        f"Expected 43 total routes but found {total_routes}. "
        "WS3 added GET /v1/environments/{{id}}/versions (+1 route)."
    )
    # Pure API routes (our handlers + /healthz, excluding FastAPI internals)
    assert api_route_count == 39, (
        f"Expected 39 APIRoute instances but found {api_route_count}. "
        "WS3 added GET /v1/environments/{{id}}/versions (+1 route)."
    )


# ---------------------------------------------------------------------------
# ArenaAPIService construction
# ---------------------------------------------------------------------------

def test_service_builds_from_registry(tmp_path):
    from src.api.registry import build_adapters
    from src.api.settings import ArenaSettings
    from src.api.service import ArenaAPIService

    settings = ArenaSettings(db_path=tmp_path / "test.db")
    adapters = build_adapters(settings)
    svc = ArenaAPIService(adapters=adapters)
    assert svc.store is adapters.store


def test_service_legacy_store_kwarg(tmp_path):
    """``ArenaAPIService(store=<store>)`` must still work for back-compat."""
    from src.api.stores.sqlite import SQLiteStore
    from src.api.service import ArenaAPIService

    store = SQLiteStore(path=tmp_path / "legacy.db")
    svc = ArenaAPIService(store=store)
    assert svc.store is store


# ---------------------------------------------------------------------------
# AuthProvider
# ---------------------------------------------------------------------------

def test_static_bearer_accepts_default_token():
    import os
    from src.api.ports.auth_provider import StaticBearerAuthProvider
    from src.api.constants import DEFAULT_API_TOKEN

    # Ensure env var is not set so the default token is used.
    os.environ.pop("OPEN_ARENA_API_TOKEN", None)
    provider = StaticBearerAuthProvider()
    principal = provider.authenticate(f"Bearer {DEFAULT_API_TOKEN}")
    assert principal.subject == "static"
    assert principal.org is None


def test_static_bearer_rejects_invalid_token():
    import os
    from src.api.ports.auth_provider import StaticBearerAuthProvider
    from src.api.service import ApiError

    os.environ.pop("OPEN_ARENA_API_TOKEN", None)
    provider = StaticBearerAuthProvider()
    with pytest.raises(ApiError) as exc_info:
        provider.authenticate("Bearer wrong-token")
    assert exc_info.value.status_code == 401


def test_static_bearer_rejects_missing_header():
    import os
    from src.api.ports.auth_provider import StaticBearerAuthProvider
    from src.api.service import ApiError

    os.environ.pop("OPEN_ARENA_API_TOKEN", None)
    provider = StaticBearerAuthProvider()
    with pytest.raises(ApiError) as exc_info:
        provider.authenticate(None)
    assert exc_info.value.status_code == 401


# ---------------------------------------------------------------------------
# Store ABC compliance (SQLiteStore)
# ---------------------------------------------------------------------------

def test_sqlite_store_implements_store_abc(tmp_path):
    from src.api.ports.store import Store
    from src.api.stores.sqlite import SQLiteStore

    store = SQLiteStore(path=tmp_path / "abc.db")
    assert isinstance(store, Store)


# ---------------------------------------------------------------------------
# Back-compat: SQLiteStore importable from service module
# ---------------------------------------------------------------------------

def test_sqlite_store_importable_from_service():
    from src.api.service import SQLiteStore  # noqa: F401 — just checking importability
    assert SQLiteStore is not None
