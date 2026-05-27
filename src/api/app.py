from __future__ import annotations

import os
from datetime import datetime
from uuid import UUID

from fastapi import Depends, FastAPI, Header, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.api import models as api
from src.api.service import ArenaAPIService, ApiError, DEFAULT_API_TOKEN

service = ArenaAPIService()
app = FastAPI(title='Open Arena API', version='0.3.0')


def require_bearer(authorization: str | None = Header(default=None)) -> None:
    token = os.getenv('OPEN_ARENA_API_TOKEN', DEFAULT_API_TOKEN)
    expected = 'Bearer ' + token
    if authorization != expected:
        raise ApiError('unauthorized', 'Missing or invalid bearer token.', status_code=401)


@app.exception_handler(ApiError)
async def handle_api_error(_request, exc: ApiError):
    return JSONResponse(status_code=exc.status_code, content=exc.error.model_dump(mode='json'))


@app.get('/healthz')
def healthz() -> dict[str, str]:
    return {'status': 'ok'}


@app.get('/v1/verifiers', response_model=api.VerifierSuiteListResponse, dependencies=[Depends(require_bearer)])
def list_verifiers(limit: int = 50, cursor: str | None = None):
    return service.list_verifiers(limit=limit, cursor=cursor)


@app.post('/v1/verifiers', response_model=api.VerifierSuite, status_code=201, dependencies=[Depends(require_bearer)])
def create_verifier(payload: api.VerifierSuiteCreate):
    return service.create_verifier(payload)


@app.get('/v1/verifiers/{verifier_id}', response_model=api.VerifierSuite, dependencies=[Depends(require_bearer)])
def get_verifier(verifier_id: UUID):
    return service.get_verifier(verifier_id)


@app.patch('/v1/verifiers/{verifier_id}', response_model=api.VerifierSuite, dependencies=[Depends(require_bearer)])
def update_verifier(verifier_id: UUID, payload: api.VerifierSuitePatch):
    return service.update_verifier(verifier_id, payload)


@app.delete('/v1/verifiers/{verifier_id}', status_code=204, dependencies=[Depends(require_bearer)])
def delete_verifier(verifier_id: UUID):
    service.delete_verifier(verifier_id)
    return Response(status_code=204)


@app.get('/v1/environments', response_model=api.EnvironmentListResponse, dependencies=[Depends(require_bearer)])
def list_environments(source_kind: api.EnvironmentSourceKind | None = None, mode: api.RunMode | None = None, name: str | None = None, limit: int = 50, cursor: str | None = None):
    return service.list_environments(source_kind=source_kind, mode=mode, name=name, limit=limit, cursor=cursor)


@app.post('/v1/environments', response_model=api.Environment, status_code=201, dependencies=[Depends(require_bearer)])
def create_environment(payload: api.EnvironmentCreate):
    return service.create_environment(payload)


@app.get('/v1/environments/{environment_id}', response_model=api.Environment, dependencies=[Depends(require_bearer)])
def get_environment(environment_id: UUID):
    return service.get_environment(environment_id)


@app.patch('/v1/environments/{environment_id}', response_model=api.Environment, dependencies=[Depends(require_bearer)])
def update_environment(environment_id: UUID, payload: api.EnvironmentPatch):
    return service.update_environment(environment_id, payload)


@app.delete('/v1/environments/{environment_id}', status_code=204, dependencies=[Depends(require_bearer)])
def delete_environment(environment_id: UUID):
    service.delete_environment(environment_id)
    return Response(status_code=204)


@app.get('/v1/leaderboards', response_model=api.LeaderboardListResponse, dependencies=[Depends(require_bearer)])
def list_leaderboards(visibility: api.LeaderboardVisibility | None = None, limit: int = 50, cursor: str | None = None):
    return service.list_leaderboards(visibility=visibility, limit=limit, cursor=cursor)


@app.post('/v1/leaderboards', response_model=api.Leaderboard, status_code=201, dependencies=[Depends(require_bearer)])
def create_leaderboard(payload: api.LeaderboardCreate):
    return service.create_leaderboard(payload)


@app.get('/v1/leaderboards/{leaderboard_id}', response_model=api.Leaderboard, dependencies=[Depends(require_bearer)])
def get_leaderboard(leaderboard_id: UUID):
    return service.get_leaderboard(leaderboard_id)


@app.patch('/v1/leaderboards/{leaderboard_id}', response_model=api.Leaderboard, dependencies=[Depends(require_bearer)])
def update_leaderboard(leaderboard_id: UUID, payload: api.LeaderboardPatch):
    return service.update_leaderboard(leaderboard_id, payload)


@app.delete('/v1/leaderboards/{leaderboard_id}', status_code=204, dependencies=[Depends(require_bearer)])
def delete_leaderboard(leaderboard_id: UUID):
    service.delete_leaderboard(leaderboard_id)
    return Response(status_code=204)


@app.get('/v1/leaderboards/{leaderboard_id}/model-catalog', response_model=api.ModelCatalog, dependencies=[Depends(require_bearer)])
def get_model_catalog(leaderboard_id: UUID):
    return service.get_model_catalog(leaderboard_id)


@app.put('/v1/leaderboards/{leaderboard_id}/model-catalog', response_model=api.ModelCatalog, dependencies=[Depends(require_bearer)])
def replace_model_catalog(leaderboard_id: UUID, payload: api.ModelCatalogReplace):
    return service.replace_model_catalog(leaderboard_id, payload)


@app.patch('/v1/leaderboards/{leaderboard_id}/model-catalog', response_model=api.ModelCatalog, dependencies=[Depends(require_bearer)])
def patch_model_catalog(leaderboard_id: UUID, payload: api.ModelCatalogPatch):
    return service.patch_model_catalog(leaderboard_id, payload)


@app.get('/v1/leaderboards/{leaderboard_id}/models', response_model=api.ModelDefinitionListResponse, dependencies=[Depends(require_bearer)])
def list_models(leaderboard_id: UUID, limit: int = 50, cursor: str | None = None):
    return service.list_models(leaderboard_id, limit=limit, cursor=cursor)


@app.post('/v1/leaderboards/{leaderboard_id}/models', response_model=api.ModelDefinition, status_code=201, dependencies=[Depends(require_bearer)])
def add_model(leaderboard_id: UUID, payload: api.ModelDefinitionCreate):
    return service.add_model(leaderboard_id, payload)


@app.get('/v1/leaderboards/{leaderboard_id}/models/{model_id}', response_model=api.ModelDefinition, dependencies=[Depends(require_bearer)])
def get_model(leaderboard_id: UUID, model_id: UUID):
    return service.get_model(leaderboard_id, model_id)


@app.patch('/v1/leaderboards/{leaderboard_id}/models/{model_id}', response_model=api.ModelDefinition, dependencies=[Depends(require_bearer)])
def update_model(leaderboard_id: UUID, model_id: UUID, payload: api.ModelDefinitionPatch):
    return service.update_model(leaderboard_id, model_id, payload)


@app.delete('/v1/leaderboards/{leaderboard_id}/models/{model_id}', status_code=204, dependencies=[Depends(require_bearer)])
def delete_model(leaderboard_id: UUID, model_id: UUID):
    service.delete_model(leaderboard_id, model_id)
    return Response(status_code=204)


@app.get('/v1/leaderboards/{leaderboard_id}/environments', response_model=api.EnvironmentMembershipListResponse, dependencies=[Depends(require_bearer)])
def list_memberships(leaderboard_id: UUID, limit: int = 50, cursor: str | None = None):
    return service.list_memberships(leaderboard_id, limit=limit, cursor=cursor)


@app.post('/v1/leaderboards/{leaderboard_id}/environments', response_model=api.EnvironmentMembership, status_code=201, dependencies=[Depends(require_bearer)])
def add_membership(leaderboard_id: UUID, payload: api.EnvironmentMembershipCreate):
    return service.add_membership(leaderboard_id, payload)


@app.get('/v1/leaderboards/{leaderboard_id}/environments/{environment_id}', response_model=api.EnvironmentMembership, dependencies=[Depends(require_bearer)])
def get_membership(leaderboard_id: UUID, environment_id: UUID):
    return service.get_membership(leaderboard_id, environment_id)


@app.patch('/v1/leaderboards/{leaderboard_id}/environments/{environment_id}', response_model=api.EnvironmentMembership, dependencies=[Depends(require_bearer)])
def update_membership(leaderboard_id: UUID, environment_id: UUID, payload: api.EnvironmentMembershipPatch):
    return service.update_membership(leaderboard_id, environment_id, payload)


@app.delete('/v1/leaderboards/{leaderboard_id}/environments/{environment_id}', status_code=204, dependencies=[Depends(require_bearer)])
def delete_membership(leaderboard_id: UUID, environment_id: UUID):
    service.delete_membership(leaderboard_id, environment_id)
    return Response(status_code=204)


@app.get('/v1/leaderboards/{leaderboard_id}/entries', response_model=api.LeaderboardEntryListResponse, dependencies=[Depends(require_bearer)])
def leaderboard_entries(leaderboard_id: UUID, environment_id: UUID | None = None, model_id: UUID | None = None, as_of: datetime | None = None, limit: int = 50, cursor: str | None = None):
    return service.list_leaderboard_entries(leaderboard_id, environment_id=environment_id, model_id=model_id, as_of=as_of, limit=limit, cursor=cursor)


@app.get('/v1/public-leaderboard', response_model=api.PublicLeaderboard, dependencies=[Depends(require_bearer)])
def get_public_leaderboard():
    return service.get_public_leaderboard()


@app.get('/v1/public-leaderboard/entries', response_model=api.PublicLeaderboardEntryListResponse, dependencies=[Depends(require_bearer)])
def list_public_entries(environment_name: str | None = None, environment_version: str | None = None, model_name: str | None = None, model_version: str | None = None, limit: int = 50, cursor: str | None = None):
    return service.list_public_entries(environment_name=environment_name, environment_version=environment_version, model_name=model_name, model_version=model_version, limit=limit, cursor=cursor)


@app.get('/v1/metric-kinds', response_model=api.DiscoveryIdentifierListResponse, dependencies=[Depends(require_bearer)])
def metric_kinds():
    return service.metric_kinds()


@app.get('/v1/aggregations', response_model=api.DiscoveryIdentifierListResponse, dependencies=[Depends(require_bearer)])
def aggregations():
    return service.aggregations()


@app.get('/v1/model-providers', response_model=api.DiscoveryIdentifierListResponse, dependencies=[Depends(require_bearer)])
def model_providers():
    return service.model_providers()


@app.get('/v1/dataset-providers', response_model=api.DiscoveryIdentifierListResponse, dependencies=[Depends(require_bearer)])
def dataset_providers():
    return service.dataset_providers()


@app.get('/v1/runs', response_model=api.RunListResponse, dependencies=[Depends(require_bearer)])
def list_runs(leaderboard_id: UUID | None = None, status: api.RunStatus | None = None, mode: api.RunMode | None = None, cache_status: api.CacheStatus | None = None, limit: int = 50, cursor: str | None = None):
    return service.list_runs(leaderboard_id=leaderboard_id, status=status, mode=mode, cache_status=cache_status, limit=limit, cursor=cursor)


@app.post('/v1/runs', response_model=api.Run, status_code=202, dependencies=[Depends(require_bearer)])
def create_run(payload: api.RunCreate):
    return service.create_run(payload)


@app.get('/v1/runs/{run_id}', response_model=api.Run, dependencies=[Depends(require_bearer)])
def get_run(run_id: UUID):
    return service.get_run(run_id)


@app.get('/v1/runs/{run_id}/results', response_model=api.RunResult, dependencies=[Depends(require_bearer)])
def get_run_results(run_id: UUID):
    return service.get_run_result(run_id)


class ImportConfigRequest(BaseModel):
    config_path: str
    leaderboard_name: str | None = None
    create_run: bool = False


@app.post('/v1/import-config', dependencies=[Depends(require_bearer)])
def import_config(payload: ImportConfigRequest):
    result = service.import_config(payload.config_path, leaderboard_name=payload.leaderboard_name, create_run=payload.create_run)
    out = {}
    for key, value in result.items():
        if isinstance(value, list):
            out[key] = [item.model_dump(mode='json') for item in value]
        else:
            out[key] = value.model_dump(mode='json')
    return out
