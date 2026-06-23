# License Apache 2.0: (c) 2026 Athena-Reply
"""``src.api.app`` — FastAPI application and HTTP handlers.

This module is intentionally thin: all business logic lives in
:class:`~src.api.service.ArenaAPIService`; all external concerns are
handled by port adapters configured via :mod:`src.api.registry`.

Authentication is delegated to the :class:`~src.api.ports.auth_provider.AuthProvider`
port.  By default :class:`~src.api.ports.auth_provider.StaticBearerAuthProvider`
reproduces the original ``OPEN_ARENA_API_TOKEN`` comparison.

WS7: Keycloak — set ``OPEN_ARENA_AUTH=keycloak`` to activate
``KeycloakAuthProvider`` once it is registered in the registry.
"""
from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from typing import Annotated
from uuid import UUID

from fastapi import Depends, FastAPI, Header, Response
from fastapi.responses import JSONResponse

from open_arena_core import models as api
from src.api.service import ArenaAPIService, ApiError

app = FastAPI(title='Open Arena API', version='0.3.0')


@lru_cache
def _service() -> ArenaAPIService:
    """Build and cache the singleton :class:`ArenaAPIService`.

    Uses :func:`~src.api.registry.build_adapters` to wire all ports from
    the current environment settings.
    """
    from src.api.registry import build_adapters

    return ArenaAPIService(adapters=build_adapters())


def get_service() -> ArenaAPIService:
    return _service()


ServiceDep = Annotated[ArenaAPIService, Depends(get_service)]


@lru_cache
def _auth_provider():
    """Return the cached :class:`~src.api.ports.auth_provider.AuthProvider`."""
    from src.api.registry import build_adapters

    return build_adapters().auth


def require_bearer(authorization: str | None = Header(default=None)) -> None:
    """Authenticate the request using the configured AuthProvider.

    Raises :exc:`~src.api.service.ApiError` (401) when authentication fails.

    WS7: Keycloak — the same signature is retained; swap the adapter via
    ``OPEN_ARENA_AUTH=keycloak``.
    """
    _auth_provider().authenticate(authorization)


@app.exception_handler(ApiError)
async def handle_api_error(_request, exc: ApiError):
    return JSONResponse(status_code=exc.status_code, content=exc.error.model_dump(mode='json'))


@app.get('/healthz')
def healthz() -> dict[str, str]:
    return {'status': 'ok'}


@app.get('/v1/verifiers', response_model=api.VerifierSuiteListResponse, dependencies=[Depends(require_bearer)])
def list_verifiers(limit: int = 50, cursor: str | None = None, service: ServiceDep = None):
    return service.list_verifiers(limit=limit, cursor=cursor)


@app.post('/v1/verifiers', response_model=api.VerifierSuite, status_code=201, dependencies=[Depends(require_bearer)])
def create_verifier(payload: api.VerifierSuiteCreate, service: ServiceDep = None):
    return service.create_verifier(payload)


@app.get('/v1/verifiers/{verifier_id}', response_model=api.VerifierSuite, dependencies=[Depends(require_bearer)])
def get_verifier(verifier_id: UUID, service: ServiceDep = None):
    return service.get_verifier(verifier_id)


@app.patch('/v1/verifiers/{verifier_id}', response_model=api.VerifierSuite, dependencies=[Depends(require_bearer)])
def update_verifier(verifier_id: UUID, payload: api.VerifierSuitePatch, service: ServiceDep = None):
    return service.update_verifier(verifier_id, payload)


@app.delete('/v1/verifiers/{verifier_id}', status_code=204, dependencies=[Depends(require_bearer)])
def delete_verifier(verifier_id: UUID, service: ServiceDep = None):
    service.delete_verifier(verifier_id)
    return Response(status_code=204)


@app.get('/v1/environments', response_model=api.EnvironmentListResponse, dependencies=[Depends(require_bearer)])
def list_environments(source_kind: api.EnvironmentSourceKind | None = None, mode: api.RunMode | None = None, name: str | None = None, limit: int = 50, cursor: str | None = None, service: ServiceDep = None):
    return service.list_environments(source_kind=source_kind, mode=mode, name=name, limit=limit, cursor=cursor)


@app.post('/v1/environments', response_model=api.Environment, status_code=201, dependencies=[Depends(require_bearer)])
def create_environment(payload: api.EnvironmentCreate, service: ServiceDep = None):
    return service.create_environment(payload)


@app.get('/v1/environments/{environment_id}', response_model=api.Environment, dependencies=[Depends(require_bearer)])
def get_environment(environment_id: UUID, service: ServiceDep = None):
    return service.get_environment(environment_id)


@app.patch('/v1/environments/{environment_id}', response_model=api.Environment, dependencies=[Depends(require_bearer)])
def update_environment(environment_id: UUID, payload: api.EnvironmentPatch, service: ServiceDep = None):
    return service.update_environment(environment_id, payload)


@app.delete('/v1/environments/{environment_id}', status_code=204, dependencies=[Depends(require_bearer)])
def delete_environment(environment_id: UUID, service: ServiceDep = None):
    service.delete_environment(environment_id)
    return Response(status_code=204)


@app.get('/v1/leaderboards', response_model=api.LeaderboardListResponse, dependencies=[Depends(require_bearer)])
def list_leaderboards(visibility: api.LeaderboardVisibility | None = None, limit: int = 50, cursor: str | None = None, service: ServiceDep = None):
    return service.list_leaderboards(visibility=visibility, limit=limit, cursor=cursor)


@app.post('/v1/leaderboards', response_model=api.Leaderboard, status_code=201, dependencies=[Depends(require_bearer)])
def create_leaderboard(payload: api.LeaderboardCreate, service: ServiceDep = None):
    return service.create_leaderboard(payload)


@app.get('/v1/leaderboards/{leaderboard_id}', response_model=api.Leaderboard, dependencies=[Depends(require_bearer)])
def get_leaderboard(leaderboard_id: UUID, service: ServiceDep = None):
    return service.get_leaderboard(leaderboard_id)


@app.patch('/v1/leaderboards/{leaderboard_id}', response_model=api.Leaderboard, dependencies=[Depends(require_bearer)])
def update_leaderboard(leaderboard_id: UUID, payload: api.LeaderboardPatch, service: ServiceDep = None):
    return service.update_leaderboard(leaderboard_id, payload)


@app.delete('/v1/leaderboards/{leaderboard_id}', status_code=204, dependencies=[Depends(require_bearer)])
def delete_leaderboard(leaderboard_id: UUID, service: ServiceDep = None):
    service.delete_leaderboard(leaderboard_id)
    return Response(status_code=204)


@app.get('/v1/leaderboards/{leaderboard_id}/model-catalog', response_model=api.ModelCatalog, dependencies=[Depends(require_bearer)])
def get_model_catalog(leaderboard_id: UUID, service: ServiceDep = None):
    return service.get_model_catalog(leaderboard_id)


@app.put('/v1/leaderboards/{leaderboard_id}/model-catalog', response_model=api.ModelCatalog, dependencies=[Depends(require_bearer)])
def replace_model_catalog(leaderboard_id: UUID, payload: api.ModelCatalogReplace, service: ServiceDep = None):
    return service.replace_model_catalog(leaderboard_id, payload)


@app.patch('/v1/leaderboards/{leaderboard_id}/model-catalog', response_model=api.ModelCatalog, dependencies=[Depends(require_bearer)])
def patch_model_catalog(leaderboard_id: UUID, payload: api.ModelCatalogPatch, service: ServiceDep = None):
    return service.patch_model_catalog(leaderboard_id, payload)


@app.get('/v1/leaderboards/{leaderboard_id}/models', response_model=api.ModelDefinitionListResponse, dependencies=[Depends(require_bearer)])
def list_models(leaderboard_id: UUID, limit: int = 50, cursor: str | None = None, service: ServiceDep = None):
    return service.list_models(leaderboard_id, limit=limit, cursor=cursor)


@app.post('/v1/leaderboards/{leaderboard_id}/models', response_model=api.ModelDefinition, status_code=201, dependencies=[Depends(require_bearer)])
def add_model(leaderboard_id: UUID, payload: api.ModelDefinitionCreate, service: ServiceDep = None):
    return service.add_model(leaderboard_id, payload)


@app.get('/v1/leaderboards/{leaderboard_id}/models/{model_id}', response_model=api.ModelDefinition, dependencies=[Depends(require_bearer)])
def get_model(leaderboard_id: UUID, model_id: UUID, service: ServiceDep = None):
    return service.get_model(leaderboard_id, model_id)


@app.patch('/v1/leaderboards/{leaderboard_id}/models/{model_id}', response_model=api.ModelDefinition, dependencies=[Depends(require_bearer)])
def update_model(leaderboard_id: UUID, model_id: UUID, payload: api.ModelDefinitionPatch, service: ServiceDep = None):
    return service.update_model(leaderboard_id, model_id, payload)


@app.delete('/v1/leaderboards/{leaderboard_id}/models/{model_id}', status_code=204, dependencies=[Depends(require_bearer)])
def delete_model(leaderboard_id: UUID, model_id: UUID, service: ServiceDep = None):
    service.delete_model(leaderboard_id, model_id)
    return Response(status_code=204)


@app.get('/v1/leaderboards/{leaderboard_id}/environments', response_model=api.EnvironmentMembershipListResponse, dependencies=[Depends(require_bearer)])
def list_memberships(leaderboard_id: UUID, limit: int = 50, cursor: str | None = None, service: ServiceDep = None):
    return service.list_memberships(leaderboard_id, limit=limit, cursor=cursor)


@app.post('/v1/leaderboards/{leaderboard_id}/environments', response_model=api.EnvironmentMembership, status_code=201, dependencies=[Depends(require_bearer)])
def add_membership(leaderboard_id: UUID, payload: api.EnvironmentMembershipCreate, service: ServiceDep = None):
    return service.add_membership(leaderboard_id, payload)


@app.get('/v1/leaderboards/{leaderboard_id}/environments/{environment_id}', response_model=api.EnvironmentMembership, dependencies=[Depends(require_bearer)])
def get_membership(leaderboard_id: UUID, environment_id: UUID, service: ServiceDep = None):
    return service.get_membership(leaderboard_id, environment_id)


@app.patch('/v1/leaderboards/{leaderboard_id}/environments/{environment_id}', response_model=api.EnvironmentMembership, dependencies=[Depends(require_bearer)])
def update_membership(leaderboard_id: UUID, environment_id: UUID, payload: api.EnvironmentMembershipPatch, service: ServiceDep = None):
    return service.update_membership(leaderboard_id, environment_id, payload)


@app.delete('/v1/leaderboards/{leaderboard_id}/environments/{environment_id}', status_code=204, dependencies=[Depends(require_bearer)])
def delete_membership(leaderboard_id: UUID, environment_id: UUID, service: ServiceDep = None):
    service.delete_membership(leaderboard_id, environment_id)
    return Response(status_code=204)


@app.get('/v1/leaderboards/{leaderboard_id}/entries', response_model=api.LeaderboardEntryListResponse, dependencies=[Depends(require_bearer)])
def leaderboard_entries(leaderboard_id: UUID, environment_id: UUID | None = None, model_id: UUID | None = None, as_of: datetime | None = None, limit: int = 50, cursor: str | None = None, service: ServiceDep = None):
    return service.list_leaderboard_entries(leaderboard_id, environment_id=environment_id, model_id=model_id, as_of=as_of, limit=limit, cursor=cursor)

@app.get('/v1/metric-kinds', response_model=api.DiscoveryIdentifierListResponse, dependencies=[Depends(require_bearer)])
def metric_kinds(service: ServiceDep = None):
    return service.metric_kinds()


@app.get('/v1/aggregations', response_model=api.DiscoveryIdentifierListResponse, dependencies=[Depends(require_bearer)])
def aggregations(service: ServiceDep = None):
    return service.aggregations()


@app.get('/v1/model-providers', response_model=api.DiscoveryIdentifierListResponse, dependencies=[Depends(require_bearer)])
def model_providers(service: ServiceDep = None):
    return service.model_providers()


@app.get('/v1/dataset-providers', response_model=api.DiscoveryIdentifierListResponse, dependencies=[Depends(require_bearer)])
def dataset_providers(service: ServiceDep = None):
    return service.dataset_providers()


@app.get('/v1/runs', response_model=api.RunListResponse, dependencies=[Depends(require_bearer)])
def list_runs(leaderboard_id: UUID | None = None, status: api.RunStatus | None = None, mode: api.RunMode | None = None, cache_status: api.CacheStatus | None = None, limit: int = 50, cursor: str | None = None, service: ServiceDep = None):
    return service.list_runs(leaderboard_id=leaderboard_id, status=status, mode=mode, cache_status=cache_status, limit=limit, cursor=cursor)


@app.post('/v1/runs', response_model=api.Run, status_code=202, dependencies=[Depends(require_bearer)])
def create_run(payload: api.RunCreate, service: ServiceDep = None):
    return service.create_run(payload)


@app.get('/v1/runs/{run_id}', response_model=api.Run, dependencies=[Depends(require_bearer)])
def get_run(run_id: UUID, service: ServiceDep = None):
    return service.get_run(run_id)


@app.get('/v1/runs/{run_id}/results', response_model=api.RunResult, dependencies=[Depends(require_bearer)])
def get_run_results(run_id: UUID, service: ServiceDep = None):
    return service.get_run_result(run_id)
