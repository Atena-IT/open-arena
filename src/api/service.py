# License Apache 2.0: (c) 2026 Athena-Reply
"""``src.api.service`` — Business-logic layer for Open Arena API.

:class:`ArenaAPIService` is the single entry-point for every HTTP handler
in :mod:`src.api.app`.  It delegates all external concerns to a set of
port adapters (see :mod:`src.api.ports`) and contains no I/O of its own.

Backward-compatibility notes
-----------------------------
* ``SQLiteStore`` is still importable from this module (re-exported from
  :mod:`src.api.stores.sqlite`) so any code that did
  ``from src.api.service import SQLiteStore`` keeps working.
* ``ArenaAPIService(store=<store>)`` still works — the ``store`` kwarg is
  accepted and wired into the adapter set internally.
"""
from __future__ import annotations

import hashlib
import json
import logging
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import synalinks
import yaml
from synalinks.src.utils.naming import to_snake_case

from open_arena_core.constants import DEFAULT_API_TOKEN  # noqa: F401 — kept for back-compat
from open_arena_core import models as api
from src.api.ports.dataset_resolver import DatasetResolver
from src.api.ports.environment_backend import EnvironmentBackend
from src.api.ports.results_sink import ResultsSink
from src.api.ports.sandbox_provider import SandboxProvider
from src.api.ports.store import Store
from src.datasets import _DATASET_TYPES
from src.rewards import _REWARD_TYPES

# Re-export SQLiteStore for backward compatibility with any code that did
# ``from src.api.service import SQLiteStore``.
from src.api.stores.sqlite import SQLiteStore  # noqa: F401

logger = logging.getLogger(__name__)

DEFAULT_AGGREGATIONS = ('weighted_mean', 'mean', 'min', 'max')
DEFAULT_MODEL_PROVIDERS = (
    'anthropic',
    'aws',
    'azure',
    'bedrock',
    'cohere',
    'deepseek',
    'deployment',
    'gemini',
    'groq',
    'litellm_proxy',
    'mistral',
    'mlflow_gateway',
    'ollama',
    'openai',
    'openrouter',
    'together',
    'vllm',
    'xai',
)
PROVIDER_SOURCE_FIELDS = {
    'braintrust': 'dataset_name',
    'folder': 'path',
    'huggingface': 'path',
    'langfuse': 'dataset_name',
    'langsmith': 'dataset_name',
    'local': 'path',
    'opik': 'dataset_name',
    'phoenix': 'dataset_name',
}


class ApiError(Exception):
    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None, status_code: int = 400):
        super().__init__(message)
        self.error = api.Error(error=api.ErrorDetail(code=code, message=message, details=details))
        self.status_code = status_code


@dataclass(slots=True)
class PendingSubject:
    model: api.ModelDefinition
    environment: api.Environment
    fingerprint: str


class ArenaAPIService:
    """Business-logic façade for the Open Arena REST API.

    All external I/O is delegated to port adapters:

    * **store** — persistence (:class:`~src.api.ports.store.Store`)
    * **env_backend** — environment resolution
      (:class:`~src.api.ports.environment_backend.EnvironmentBackend`)
    * **dataset_resolver** — dataset binding translation
      (:class:`~src.api.ports.dataset_resolver.DatasetResolver`)
    * **results_sink** — result persistence / forwarding
      (:class:`~src.api.ports.results_sink.ResultsSink`)
    * **sandbox** — sweep execution
      (:class:`~src.api.ports.sandbox_provider.SandboxProvider`)

    Args:
        store: Back-compat kwarg.  When provided (and *adapters* is
            ``None``), the given store is used and all other ports fall
            back to their defaults.  Prefer passing a fully-built
            :class:`~src.api.registry.AdapterSet` via *adapters*.
        adapters: Pre-built adapter set from
            :func:`~src.api.registry.build_adapters`.  When ``None``,
            the registry is called with default settings.
    """

    def __init__(
        self,
        store: Store | None = None,
        *,
        adapters=None,  # AdapterSet | None — avoids circular import at module level
    ) -> None:
        if adapters is not None:
            # Fully-wired adapter set provided (preferred code path).
            self.store: Store = adapters.store
            self._env_backend: EnvironmentBackend = adapters.env_backend
            self._dataset_resolver: DatasetResolver = adapters.dataset_resolver
            self._results_sink: ResultsSink = adapters.results_sink
            self._sandbox: SandboxProvider = adapters.sandbox
        else:
            # Legacy / back-compat code path: build adapters from the
            # registry but allow the caller to override the store.
            from src.api.registry import build_adapters

            built = build_adapters()
            self.store = store if store is not None else built.store
            self._env_backend = built.env_backend
            self._dataset_resolver = built.dataset_resolver
            # ResultsSink must see the same store instance we are using.
            from src.api.ports.results_sink import StoreResultsSink

            self._results_sink = StoreResultsSink(store=self.store)
            self._sandbox = built.sandbox

        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='arena-api')

    # discovery -----------------------------------------------------------------
    def metric_kinds(self) -> api.DiscoveryIdentifierListResponse:
        metric_ids = set(_REWARD_TYPES)
        for name, value in vars(synalinks.metrics).items():
            if name.startswith('_'):
                continue
            if isinstance(value, type):
                metric_ids.add(to_snake_case(name))
        items = [api.DiscoveryIdentifier(id=item, display_name=item.replace('_', ' ')) for item in sorted(metric_ids)]
        return api.DiscoveryIdentifierListResponse(items=items)

    def aggregations(self) -> api.DiscoveryIdentifierListResponse:
        items = [api.DiscoveryIdentifier(id=item, display_name=item.replace('_', ' '), scopes=['verifiers', 'leaderboards', 'results']) for item in DEFAULT_AGGREGATIONS]
        return api.DiscoveryIdentifierListResponse(items=items)

    def model_providers(self) -> api.DiscoveryIdentifierListResponse:
        items = [api.DiscoveryIdentifier(id=item, display_name=item) for item in DEFAULT_MODEL_PROVIDERS]
        return api.DiscoveryIdentifierListResponse(items=items)

    def dataset_providers(self) -> api.DiscoveryIdentifierListResponse:
        items = [api.DiscoveryIdentifier(id=item, display_name=item) for item in sorted(_DATASET_TYPES)]
        return api.DiscoveryIdentifierListResponse(items=items)

    # verifiers -----------------------------------------------------------------
    def list_verifiers(self, *, limit: int = 50, cursor: str | None = None) -> api.VerifierSuiteListResponse:
        items, next_cursor = _paginate(self.store.list_verifiers(), limit=limit, cursor=cursor)
        return api.VerifierSuiteListResponse(items=items, next_cursor=next_cursor)

    def create_verifier(self, payload: api.VerifierSuiteCreate) -> api.VerifierSuite:
        self._validate_aggregation(payload.aggregation or 'weighted_mean')
        self._validate_metric_definitions(payload.metrics)
        now = _now()
        verifier = api.VerifierSuite(
            id=uuid4(),
            name=payload.name,
            description=payload.description,
            aggregation=payload.aggregation or 'weighted_mean',
            metrics=payload.metrics,
            metadata=payload.metadata,
            created_at=now,
            updated_at=now,
        )
        self.store.save_verifier(verifier)
        return verifier

    def get_verifier(self, verifier_id: UUID) -> api.VerifierSuite:
        verifier = self.store.get_verifier(verifier_id)
        if verifier is None:
            raise ApiError('verifier_not_found', f'Unknown verifier {verifier_id}.', status_code=404)
        return verifier

    def update_verifier(self, verifier_id: UUID, patch: api.VerifierSuitePatch) -> api.VerifierSuite:
        current = self.get_verifier(verifier_id)
        metrics = patch.metrics or current.metrics
        self._validate_metric_definitions(metrics)
        aggregation = patch.aggregation or current.aggregation or 'weighted_mean'
        self._validate_aggregation(aggregation)
        updated = current.model_copy(update={
            'name': patch.name or current.name,
            'description': patch.description if patch.description is not None else current.description,
            'aggregation': aggregation,
            'metrics': metrics,
            'metadata': patch.metadata if patch.metadata is not None else current.metadata,
            'updated_at': _now(),
        })
        self.store.save_verifier(updated)
        return updated

    def delete_verifier(self, verifier_id: UUID) -> None:
        if not self.store.delete('verifiers', verifier_id):
            raise ApiError('verifier_not_found', f'Unknown verifier {verifier_id}.', status_code=404)

    # environments ---------------------------------------------------------------
    def list_environments(self, *, source_kind: api.EnvironmentSourceKind | None = None, mode: api.RunMode | None = None, name: str | None = None, limit: int = 50, cursor: str | None = None) -> api.EnvironmentListResponse:
        items = self.store.list_environments()
        if source_kind is not None:
            items = [item for item in items if item.source.kind == source_kind]
        if name is not None:
            items = [item for item in items if item.source.name == name]
        if mode is not None:
            items = [item for item in items if self._supports_mode(item, mode)]
        page, next_cursor = _paginate(items, limit=limit, cursor=cursor)
        return api.EnvironmentListResponse(items=page, next_cursor=next_cursor)

    def create_environment(self, payload: api.EnvironmentCreate) -> api.Environment:
        self._validate_environment_payload(payload.source, payload.inline_definition)
        now = _now()
        env_id = uuid4()
        # WS3: build a provisional env to pass into _resolve_environment_identity.
        # The id doesn't affect resolution; we reuse it for the final object.
        provisional = api.Environment(
            id=env_id,
            source=payload.source,
            inline_definition=payload.inline_definition,
            metadata=payload.metadata,
            created_at=now,
            updated_at=now,
        )
        commit_sha, content_hash = self._resolve_environment_identity(provisional)
        env = provisional.model_copy(update={
            'commit_sha': commit_sha,
            'content_hash': content_hash,
        })
        self.store.save_environment(env)
        return env

    def get_environment(self, environment_id: UUID) -> api.Environment:
        env = self.store.get_environment(environment_id)
        if env is None:
            raise ApiError('environment_not_found', f'Unknown environment {environment_id}.', status_code=404)
        return env

    def update_environment(self, environment_id: UUID, patch: api.EnvironmentPatch) -> api.Environment:
        current = self.get_environment(environment_id)
        source = patch.source or current.source
        inline_definition = patch.inline_definition if patch.inline_definition is not None else current.inline_definition
        self._validate_environment_payload(source, inline_definition)
        # WS3: re-resolve identity when source changes.
        candidate = current.model_copy(update={
            'source': source,
            'inline_definition': inline_definition,
            'metadata': patch.metadata if patch.metadata is not None else current.metadata,
            'updated_at': _now(),
        })
        commit_sha, content_hash = self._resolve_environment_identity(candidate)
        updated = candidate.model_copy(update={
            'commit_sha': commit_sha,
            'content_hash': content_hash,
        })
        self.store.save_environment(updated)
        return updated

    def delete_environment(self, environment_id: UUID) -> None:
        if not self.store.delete('environments', environment_id):
            raise ApiError('environment_not_found', f'Unknown environment {environment_id}.', status_code=404)

    def list_environment_versions(self, environment_id: UUID) -> api.EnvironmentVersionListResponse:
        """Return known version descriptors for *environment_id* (WS3).

        For git-backed sources the backend is called to resolve the current
        ``commit_sha`` and ``content_hash``; a single :class:`~api.EnvironmentVersion`
        entry is returned for the pinned revision.  For inline environments a
        single entry is synthesised from the source ``version`` field.

        Future backends can enumerate tags/refs and return multiple entries;
        this implementation returns the single resolved/pinned version.
        """
        env = self.get_environment(environment_id)
        now = _now()

        if env.source.kind == api.EnvironmentSourceKind.inline:
            # Inline: return the source version; no VCS provenance.
            return api.EnvironmentVersionListResponse(items=[
                api.EnvironmentVersion(
                    version=env.source.version,
                    commit_sha=None,
                    content_hash=None,
                    git_ref=None,
                    created_at=env.created_at,
                )
            ])

        # Git-backed: resolve current commit and return as the single entry.
        try:
            resolved = self._env_backend.resolve(env)
            commit_sha = resolved.commit_sha
            content_hash = resolved.content_hash
        except (NotImplementedError, Exception):  # noqa: BLE001
            # Backend not configured or resolution failed; fall back to stored values.
            commit_sha = env.commit_sha
            content_hash = env.content_hash

        return api.EnvironmentVersionListResponse(items=[
            api.EnvironmentVersion(
                version=env.source.version,
                commit_sha=commit_sha,
                content_hash=content_hash,
                git_ref=env.source.git_ref,
                created_at=env.created_at,
            )
        ])

    # leaderboards ----------------------------------------------------------------
    def list_leaderboards(self, *, visibility: api.LeaderboardVisibility | None = None, limit: int = 50, cursor: str | None = None) -> api.LeaderboardListResponse:
        items = self.store.list_leaderboards()
        if visibility is not None:
            items = [item for item in items if item.visibility == visibility]
        page, next_cursor = _paginate(items, limit=limit, cursor=cursor)
        return api.LeaderboardListResponse(items=page, next_cursor=next_cursor)

    def create_leaderboard(self, payload: api.LeaderboardCreate) -> api.Leaderboard:
        ranking = payload.ranking or api.RankingPolicy(primary_metric='reward', aggregation='weighted_mean')
        self._validate_aggregation(ranking.aggregation or 'weighted_mean')
        now = _now()
        model_catalog = None
        env_memberships: list[api.EnvironmentMembership] = []
        if payload.bootstrap and payload.bootstrap.model_catalog:
            model_catalog = self._catalog_from_create(payload.bootstrap.model_catalog, now)
        if payload.bootstrap and payload.bootstrap.environments:
            env_memberships = [self._membership_from_create(m, now) for m in payload.bootstrap.environments]
        leaderboard = api.Leaderboard(
            id=uuid4(),
            name=payload.name,
            description=payload.description,
            visibility=payload.visibility or api.LeaderboardVisibility.private,
            ranking=ranking,
            model_catalog=model_catalog,
            environments=env_memberships or [],
            metadata=payload.metadata,
            created_at=now,
            updated_at=now,
        )
        self.store.save_leaderboard(leaderboard)
        return leaderboard

    def get_leaderboard(self, leaderboard_id: UUID) -> api.Leaderboard:
        leaderboard = self.store.get_leaderboard(leaderboard_id)
        if leaderboard is None:
            raise ApiError('leaderboard_not_found', f'Unknown leaderboard {leaderboard_id}.', status_code=404)
        return leaderboard

    def update_leaderboard(self, leaderboard_id: UUID, patch: api.LeaderboardPatch) -> api.Leaderboard:
        current = self.get_leaderboard(leaderboard_id)
        ranking = current.ranking
        if patch.ranking is not None:
            ranking = ranking.model_copy(update={
                'primary_metric': patch.ranking.primary_metric or ranking.primary_metric,
                'tie_breakers': patch.ranking.tie_breakers if patch.ranking.tie_breakers is not None else ranking.tie_breakers,
                'aggregation': patch.ranking.aggregation or ranking.aggregation,
            })
        self._validate_aggregation(ranking.aggregation or 'weighted_mean')
        updated = current.model_copy(update={
            'name': patch.name or current.name,
            'description': patch.description if patch.description is not None else current.description,
            'visibility': patch.visibility or current.visibility,
            'ranking': ranking,
            'metadata': patch.metadata if patch.metadata is not None else current.metadata,
            'updated_at': _now(),
        })
        self.store.save_leaderboard(updated)
        return updated

    def delete_leaderboard(self, leaderboard_id: UUID) -> None:
        if not self.store.delete('leaderboards', leaderboard_id):
            raise ApiError('leaderboard_not_found', f'Unknown leaderboard {leaderboard_id}.', status_code=404)

    def get_model_catalog(self, leaderboard_id: UUID) -> api.ModelCatalog:
        leaderboard = self.get_leaderboard(leaderboard_id)
        if leaderboard.model_catalog is None:
            raise ApiError('model_catalog_not_found', f'Leaderboard {leaderboard_id} has no model catalog.', status_code=404)
        return leaderboard.model_catalog

    def replace_model_catalog(self, leaderboard_id: UUID, payload: api.ModelCatalogReplace) -> api.ModelCatalog:
        leaderboard = self.get_leaderboard(leaderboard_id)
        catalog = self._catalog_from_create(payload, _now())
        leaderboard = leaderboard.model_copy(update={'model_catalog': catalog, 'updated_at': _now()})
        self.store.save_leaderboard(leaderboard)
        return catalog

    def patch_model_catalog(self, leaderboard_id: UUID, payload: api.ModelCatalogPatch) -> api.ModelCatalog:
        leaderboard = self.get_leaderboard(leaderboard_id)
        current = leaderboard.model_catalog or api.ModelCatalog(name='leaderboard-default', models=[], metadata=None, created_at=_now(), updated_at=_now())
        catalog = current.model_copy(update={
            'name': payload.name or current.name,
            'metadata': payload.metadata if payload.metadata is not None else current.metadata,
            'updated_at': _now(),
        })
        leaderboard = leaderboard.model_copy(update={'model_catalog': catalog, 'updated_at': _now()})
        self.store.save_leaderboard(leaderboard)
        return catalog

    def list_models(self, leaderboard_id: UUID, *, limit: int = 50, cursor: str | None = None) -> api.ModelDefinitionListResponse:
        catalog = self.get_model_catalog(leaderboard_id)
        items, next_cursor = _paginate(catalog.models, limit=limit, cursor=cursor)
        return api.ModelDefinitionListResponse(items=items, next_cursor=next_cursor)

    def add_model(self, leaderboard_id: UUID, payload: api.ModelDefinitionCreate) -> api.ModelDefinition:
        self._validate_model(payload)
        leaderboard = self.get_leaderboard(leaderboard_id)
        catalog = leaderboard.model_catalog or api.ModelCatalog(name='leaderboard-default', models=[], metadata=None, created_at=_now(), updated_at=_now())
        model = self._model_from_create(payload, _now())
        catalog = catalog.model_copy(update={'models': [*catalog.models, model], 'updated_at': _now()})
        leaderboard = leaderboard.model_copy(update={'model_catalog': catalog, 'updated_at': _now()})
        self.store.save_leaderboard(leaderboard)
        return model

    def get_model(self, leaderboard_id: UUID, model_id: UUID) -> api.ModelDefinition:
        catalog = self.get_model_catalog(leaderboard_id)
        for model in catalog.models:
            if model.id == model_id:
                return model
        raise ApiError('model_not_found', f'Unknown model {model_id} on leaderboard {leaderboard_id}.', status_code=404)

    def update_model(self, leaderboard_id: UUID, model_id: UUID, patch: api.ModelDefinitionPatch) -> api.ModelDefinition:
        leaderboard = self.get_leaderboard(leaderboard_id)
        catalog = leaderboard.model_catalog or api.ModelCatalog(name='leaderboard-default', models=[], metadata=None, created_at=_now(), updated_at=_now())
        new_models = []
        updated_model = None
        for model in catalog.models:
            if model.id != model_id:
                new_models.append(model)
                continue
            runtime = model.runtime
            if patch.runtime is not None:
                runtime = runtime.model_copy(update=patch.runtime.model_dump(exclude_none=True))
            candidate = model.model_copy(update={
                'name': patch.name or model.name,
                'display_name': patch.display_name if patch.display_name is not None else model.display_name,
                'family': patch.family if patch.family is not None else model.family,
                'tags': patch.tags if patch.tags is not None else model.tags,
                'runtime': runtime,
                'metadata': patch.metadata if patch.metadata is not None else model.metadata,
                'updated_at': _now(),
            })
            self._validate_model(candidate)
            updated_model = candidate
            new_models.append(candidate)
        if updated_model is None:
            raise ApiError('model_not_found', f'Unknown model {model_id} on leaderboard {leaderboard_id}.', status_code=404)
        catalog = catalog.model_copy(update={'models': new_models, 'updated_at': _now()})
        leaderboard = leaderboard.model_copy(update={'model_catalog': catalog, 'updated_at': _now()})
        self.store.save_leaderboard(leaderboard)
        return updated_model

    def delete_model(self, leaderboard_id: UUID, model_id: UUID) -> None:
        leaderboard = self.get_leaderboard(leaderboard_id)
        catalog = leaderboard.model_catalog or api.ModelCatalog(name='leaderboard-default', models=[], metadata=None, created_at=_now(), updated_at=_now())
        remaining = [model for model in catalog.models if model.id != model_id]
        if len(remaining) == len(catalog.models):
            raise ApiError('model_not_found', f'Unknown model {model_id} on leaderboard {leaderboard_id}.', status_code=404)
        catalog = catalog.model_copy(update={'models': remaining, 'updated_at': _now()})
        leaderboard = leaderboard.model_copy(update={'model_catalog': catalog, 'updated_at': _now()})
        self.store.save_leaderboard(leaderboard)

    def list_memberships(self, leaderboard_id: UUID, *, limit: int = 50, cursor: str | None = None) -> api.EnvironmentMembershipListResponse:
        leaderboard = self.get_leaderboard(leaderboard_id)
        items, next_cursor = _paginate(leaderboard.environments or [], limit=limit, cursor=cursor)
        return api.EnvironmentMembershipListResponse(items=items, next_cursor=next_cursor)

    def add_membership(self, leaderboard_id: UUID, payload: api.EnvironmentMembershipCreate) -> api.EnvironmentMembership:
        leaderboard = self.get_leaderboard(leaderboard_id)
        membership = self._membership_from_create(payload, _now())
        leaderboard = leaderboard.model_copy(update={'environments': [*(leaderboard.environments or []), membership], 'updated_at': _now()})
        self.store.save_leaderboard(leaderboard)
        return membership

    def get_membership(self, leaderboard_id: UUID, environment_id: UUID) -> api.EnvironmentMembership:
        leaderboard = self.get_leaderboard(leaderboard_id)
        for membership in leaderboard.environments or []:
            if membership.environment_id == environment_id:
                return membership
        raise ApiError('environment_membership_not_found', f'Unknown environment {environment_id} on leaderboard {leaderboard_id}.', status_code=404)

    def update_membership(self, leaderboard_id: UUID, environment_id: UUID, patch: api.EnvironmentMembershipPatch) -> api.EnvironmentMembership:
        leaderboard = self.get_leaderboard(leaderboard_id)
        updated_memberships = []
        updated = None
        for membership in leaderboard.environments or []:
            if membership.environment_id != environment_id:
                updated_memberships.append(membership)
                continue
            updated = membership.model_copy(update={'overrides': patch.overrides if patch.overrides is not None else membership.overrides})
            updated_memberships.append(updated)
        if updated is None:
            raise ApiError('environment_membership_not_found', f'Unknown environment {environment_id} on leaderboard {leaderboard_id}.', status_code=404)
        leaderboard = leaderboard.model_copy(update={'environments': updated_memberships, 'updated_at': _now()})
        self.store.save_leaderboard(leaderboard)
        return updated

    def delete_membership(self, leaderboard_id: UUID, environment_id: UUID) -> None:
        leaderboard = self.get_leaderboard(leaderboard_id)
        remaining = [membership for membership in leaderboard.environments or [] if membership.environment_id != environment_id]
        if len(remaining) == len(leaderboard.environments or []):
            raise ApiError('environment_membership_not_found', f'Unknown environment {environment_id} on leaderboard {leaderboard_id}.', status_code=404)
        leaderboard = leaderboard.model_copy(update={'environments': remaining, 'updated_at': _now()})
        self.store.save_leaderboard(leaderboard)

    # runs -----------------------------------------------------------------------
    def list_runs(self, *, leaderboard_id: UUID | None = None, status: api.RunStatus | None = None, mode: api.RunMode | None = None, cache_status: api.CacheStatus | None = None, limit: int = 50, cursor: str | None = None) -> api.RunListResponse:
        items = self.store.list_runs()
        if leaderboard_id is not None:
            items = [item for item in items if getattr(item.selection.root, 'leaderboard_id', None) == leaderboard_id]
        if status is not None:
            items = [item for item in items if item.status == status]
        if mode is not None:
            items = [item for item in items if item.mode == mode]
        if cache_status is not None:
            items = [item for item in items if item.cache_status == cache_status]
        page, next_cursor = _paginate(items, limit=limit, cursor=cursor)
        return api.RunListResponse(items=page, next_cursor=next_cursor)

    def create_run(self, payload: api.RunCreate) -> api.Run:
        run_create = payload.root
        self._validate_run_create(run_create)
        if run_create.idempotency_key:
            existing = self.store.get_run_by_idempotency(run_create.idempotency_key)
            if existing is not None:
                return existing
        selection = run_create.selection.root
        reuse_policy = run_create.reuse_policy or api.ReusePolicy()
        pending, cached_subjects, cache_hits = self._prepare_subjects(run_create.mode, selection, run_create.execution, reuse_policy)
        now = _now()
        status = api.RunStatus.succeeded if not pending else api.RunStatus.queued
        cache_status = _combine_cache_status(cache_hits, pending)
        run = api.Run(
            id=uuid4(),
            mode=api.RunMode(run_create.mode),
            selection=run_create.selection,
            status=status,
            cache_status=cache_status,
            cache_hits=cache_hits or None,
            labels=run_create.labels,
            created_at=now,
            completed_at=now if status == api.RunStatus.succeeded else None,
        )
        self.store.save_run(run, idempotency_key=run_create.idempotency_key)
        if status == api.RunStatus.succeeded:
            result = self._finalize_result(run, cached_subjects)
            self._results_sink.write(run, result)
            return run
        self._executor.submit(self._execute_run, run.id, run_create, pending, cached_subjects)
        return run

    def get_run(self, run_id: UUID) -> api.Run:
        run = self.store.get_run(run_id)
        if run is None:
            raise ApiError('run_not_found', f'Unknown run {run_id}.', status_code=404)
        return run

    def get_run_result(self, run_id: UUID) -> api.RunResult:
        result = self.store.get_run_result(run_id)
        if result is None:
            raise ApiError('run_result_not_found', f'Run {run_id} has no materialized results yet.', status_code=404)
        return result

    def list_leaderboard_entries(self, leaderboard_id: UUID, *, environment_id: UUID | None = None, model_id: UUID | None = None, as_of: datetime | None = None, limit: int = 50, cursor: str | None = None) -> api.LeaderboardEntryListResponse:
        leaderboard = self.get_leaderboard(leaderboard_id)
        entries = self._leaderboard_entries(leaderboard, as_of=as_of)
        if environment_id is not None:
            entries = [entry for entry in entries if entry.environment_id == environment_id]
        if model_id is not None:
            entries = [entry for entry in entries if entry.model_id == model_id]
        page, next_cursor = _paginate(entries, limit=limit, cursor=cursor)
        return api.LeaderboardEntryListResponse(items=page, next_cursor=next_cursor)

    # execution internals --------------------------------------------------------
    def _execute_run(self, run_id: UUID, run_create: api.GeneratorRunCreate | api.AgentRunCreate, pending: list[PendingSubject], cached_subjects: list[api.SubjectResult]) -> None:
        run = self.get_run(run_id)
        run = run.model_copy(update={'status': api.RunStatus.running, 'started_at': _now()})
        self.store.save_run(run, idempotency_key=run_create.idempotency_key)
        try:
            executed = self._run_pending_subjects(run.mode, run_create.execution, pending)
            subjects = [*cached_subjects, *executed]
            result = self._finalize_result(run, subjects)
            self._results_sink.write(run, result)
            done = self.get_run(run_id).model_copy(update={'status': api.RunStatus.succeeded, 'completed_at': _now()})
            self.store.save_run(done, idempotency_key=run_create.idempotency_key)
        except Exception as exc:  # noqa: BLE001
            failed = self.get_run(run_id).model_copy(update={
                'status': api.RunStatus.failed,
                'completed_at': _now(),
                'error': api.ErrorDetail(code='run_failed', message=str(exc)),
            })
            self.store.save_run(failed, idempotency_key=run_create.idempotency_key)

    def _prepare_subjects(self, mode: str, selection: api.RunSelection1 | api.RunSelection2, execution: api.GeneratorRunConfig | api.AgentRunConfig | None, reuse_policy: api.ReusePolicy) -> tuple[list[PendingSubject], list[api.SubjectResult], list[api.CacheHit]]:
        subjects = self._expand_selection(selection)
        pending: list[PendingSubject] = []
        cached_subjects: list[api.SubjectResult] = []
        cache_hits: list[api.CacheHit] = []
        for model, environment in subjects:
            fingerprint = self._run_fingerprint(model, environment, mode, execution, reuse_policy)
            cached = self.store.get_cached_subject(fingerprint) if reuse_policy.enabled else None
            if cached is None:
                pending.append(PendingSubject(model=model, environment=environment, fingerprint=fingerprint))
                continue
            prior_run_id, subject = cached
            cached_subjects.append(subject.model_copy(update={'cache_status': api.CacheStatus.hit, 'run_fingerprint': fingerprint}))
            cache_hits.append(api.CacheHit(
                model_name=model.name,
                model_version=model.runtime.model_version,
                environment_name=environment.source.name,
                environment_version=environment.source.version,
                status=api.CacheStatus.hit,
                prior_run_id=prior_run_id,
            ))
        if cache_hits and pending and not (reuse_policy.allow_partial_cache if reuse_policy.allow_partial_cache is not None else True):
            pending = [PendingSubject(model=model, environment=environment, fingerprint=self._run_fingerprint(model, environment, mode, execution, reuse_policy)) for model, environment in subjects]
            cached_subjects = []
            cache_hits = []
        return pending, cached_subjects, cache_hits

    def _run_pending_subjects(self, mode: api.RunMode, execution: api.GeneratorRunConfig | api.AgentRunConfig | None, pending: list[PendingSubject]) -> list[api.SubjectResult]:
        if not pending:
            return []
        # P2-1: collect sandbox policies from all pending subjects and resolve
        # a single policy to pass to the SandboxProvider.
        sandbox_policies: list[api.SandboxPolicy] = [
            item.environment.inline_definition.sandbox
            for item in pending
            if item.environment.inline_definition is not None
            and item.environment.inline_definition.sandbox is not None
        ]
        distinct_policies: list[api.SandboxPolicy] = []
        for p in sandbox_policies:
            if not any(p.model_dump() == d.model_dump() for d in distinct_policies):
                distinct_policies.append(p)
        if len(distinct_policies) > 1:
            # P2-2: per-task fan-out handles heterogeneous sandbox policies
            resolved_policy: api.SandboxPolicy | None = None
        else:
            resolved_policy = distinct_policies[0] if distinct_policies else None

        # P2-2: fan out when resolved_policy requests per-task sandboxing OR
        # when heterogeneous policies exist AND any of them requests it.
        # Heterogeneous policies (resolved_policy=None) are naturally handled
        # because each task carries its own policy from its environment.
        use_per_task = (
            (resolved_policy is not None and bool(resolved_policy.per_task_sandbox))
            or any(bool(p.per_task_sandbox) for p in distinct_policies)
        )
        if len(distinct_policies) > 1:
            if use_per_task:
                logger.warning(
                    "Multiple distinct sandbox policies detected across pending subjects — "
                    "routing to per-task fan-out. Each task will use its own policy. "
                    "Distinct policies: %s",
                    [p.model_dump(exclude_none=True) for p in distinct_policies],
                )
            else:
                logger.warning(
                    "Multiple distinct sandbox policies across pending subjects — "
                    "falling back to policy=None (unrestricted in-process). "
                    "Distinct policies: %s",
                    [p.model_dump(exclude_none=True) for p in distinct_policies],
                )
        if not use_per_task:
            # ----------------------------------------------------------------
            # Whole-run path (original — unchanged)
            # ----------------------------------------------------------------
            config_payload = self._config_for_pending(mode, execution, pending)
            with tempfile.TemporaryDirectory(prefix='open-arena-api-') as tmpdir:
                config_path = Path(tmpdir) / 'run.yaml'
                config_path.write_text(yaml.safe_dump(config_payload, sort_keys=False))
                # Delegate execution to the SandboxProvider port.
                # WS6: E2B-compatible — swap LocalSandboxProvider for E2BSandboxProvider.
                result = self._sandbox.run(config_path, policy=resolved_policy)
            rows = result['rows']
            by_dataset_model: dict[tuple[str, str], list[dict[str, Any]]] = {}
            for row in rows:
                by_dataset_model.setdefault((row['dataset'], row['model']), []).append(row)
            out: list[api.SubjectResult] = []
            for item in pending:
                dataset_name = str(item.environment.id)
                model_key = self._model_runtime_id(item.model)
                metric_rows = by_dataset_model.get((dataset_name, model_key))
                if not metric_rows:
                    raise ApiError('missing_result', f'No result rows produced for model {item.model.id} and environment {item.environment.id}.')
                metrics = [api.MetricResult(name=row['metric'], value=float(row['value']), direction=api.Direction(row['direction'])) for row in metric_rows if row['value'] is not None]
                trajectory = self._trajectory_summary(metric_rows, mode)
                subject = api.SubjectResult(
                    model=item.model,
                    environment=item.environment,
                    metrics=metrics,
                    cache_status=api.CacheStatus.miss,
                    trajectory_summary=trajectory,
                    run_fingerprint=item.fingerprint,
                )
                out.append(subject)
            return out

        # ----------------------------------------------------------------
        # P2-2: per-task fan-out path
        # ----------------------------------------------------------------
        return self._run_per_task_fan_out(mode, execution, pending)

    # P2-2 maximum concurrent tasks per provider (single-tenant).
    _PER_TASK_CONCURRENCY: int = 8

    def _run_per_task_fan_out(
        self,
        mode: api.RunMode,
        execution: api.GeneratorRunConfig | api.AgentRunConfig | None,
        pending: list[PendingSubject],
    ) -> list[api.SubjectResult]:
        """Execute each pending subject in its own sandbox concurrently.

        P2-2: fan-out implementation.  Builds a per-task YAML config for
        each pending subject, calls :meth:`SandboxProvider.run_task` for
        every item via a :class:`~concurrent.futures.ThreadPoolExecutor`,
        and assembles the results into :class:`~open_arena_core.models.SubjectResult`
        objects using the same logic as the whole-run path.

        Concurrency is bounded by :attr:`_PER_TASK_CONCURRENCY` via the
        ``max_workers`` cap on the :class:`~concurrent.futures.ThreadPoolExecutor`,
        single-tenant, no org-level fairness scheduling.

        Each task receives a unique ephemeral scratch directory tag of the
        form ``{env_id}.{model_key}.{trial_n}`` to prevent cross-task
        contamination.

        Heterogeneous policies are naturally handled: each task carries its
        own policy from its inline environment definition (or ``None``).
        """
        concurrency = self._PER_TASK_CONCURRENCY
        results_ordered: list[api.SubjectResult | BaseException] = [None] * len(pending)  # type: ignore[list-item]

        def _run_one(idx: int, item: PendingSubject) -> None:
            # Resolve the per-task sandbox policy from the item's own environment.
            task_policy: api.SandboxPolicy | None = (
                item.environment.inline_definition.sandbox
                if item.environment.inline_definition is not None
                else None
            )
            dataset_name = str(item.environment.id)
            model_key = self._model_runtime_id(item.model)
            scratch_tag = f"{dataset_name}.{model_key}.{idx}"

            # P2-4: Resolve the environment to detect pinned external packages.
            # When the resolved environment has a local_path (github_repo or
            # prime_environment_hub kind with a pinned commit), route execution
            # to the env-package runtimes (PrimeVerifiersRuntime / HarborTaskRuntime)
            # instead of the sweep-based run_task path.
            try:
                resolved_env = self._env_backend.resolve(item.environment)
            except Exception:  # noqa: BLE001
                resolved_env = None

            is_pinned_external = (
                resolved_env is not None
                and resolved_env.local_path is not None
                and item.environment.source.kind
                in (
                    api.EnvironmentSourceKind.github_repo,
                    api.EnvironmentSourceKind.prime_environment_hub,
                )
            )

            if is_pinned_external:
                # P2-4 path: execute the pinned env-package via runtime.
                from src.api.sandboxes.env_runtime import execute_env_package

                logger.debug(
                    "P2-4 env-package runtime: idx=%d scratch_tag=%r local_path=%r",
                    idx, scratch_tag, resolved_env.local_path,
                )
                with self._sandbox.open_session(task_policy) as session:
                    task_result = execute_env_package(
                        resolved_env.local_path,
                        session,
                        dataset_name=dataset_name,
                        model_key=model_key,
                        policy=task_policy,
                        scratch_tag=scratch_tag,
                    )
            else:
                # Pass the already-resolved env (from the P2-4 detection above)
                # so _config_for_pending skips a duplicate resolve() call.
                _pre_resolved = (
                    {str(item.environment.id): resolved_env}
                    if resolved_env is not None
                    else None
                )
                per_task_payload = self._config_for_pending(
                    mode, execution, [item], _resolved_envs=_pre_resolved
                )
                with tempfile.TemporaryDirectory(prefix=f'open-arena-task-{idx}-') as tmpdir:
                    config_path = Path(tmpdir) / 'run.yaml'
                    config_path.write_text(yaml.safe_dump(per_task_payload, sort_keys=False))
                    logger.debug(
                        "P2-2 per-task sandbox: idx=%d scratch_tag=%r policy=%r",
                        idx, scratch_tag,
                        task_policy.model_dump(exclude_none=True) if task_policy else None,
                    )
                    task_result = self._sandbox.run_task(
                        config_path,
                        policy=task_policy,
                        scratch_tag=scratch_tag,
                    )

            # Assemble into SubjectResult (same logic as the whole-run path).
            rows = task_result.rows
            by_model: dict[str, list[dict[str, Any]]] = {}
            for row in rows:
                by_model.setdefault(row['model'], []).append(row)
            metric_rows = by_model.get(model_key) or []
            if not metric_rows:
                results_ordered[idx] = ApiError(
                    'missing_result',
                    f'No result rows produced for model {item.model.id} and environment {item.environment.id}.',
                )
                return
            metrics = [
                api.MetricResult(
                    name=row['metric'],
                    value=float(row['value']),
                    direction=api.Direction(row['direction']),
                )
                for row in metric_rows if row['value'] is not None
            ]
            trajectory = self._trajectory_summary(metric_rows, mode)
            results_ordered[idx] = api.SubjectResult(
                model=item.model,
                environment=item.environment,
                metrics=metrics,
                cache_status=api.CacheStatus.miss,
                trajectory_summary=trajectory,
                run_fingerprint=item.fingerprint,
            )

        with ThreadPoolExecutor(
            max_workers=concurrency,
            thread_name_prefix='arena-pertask',
        ) as executor:
            futures = [executor.submit(_run_one, idx, item) for idx, item in enumerate(pending)]
            for fut in futures:
                exc = fut.exception()
                if exc is not None and not isinstance(exc, ApiError):
                    raise exc

        out: list[api.SubjectResult] = []
        for idx, result in enumerate(results_ordered):
            if isinstance(result, ApiError):
                raise result
            if isinstance(result, BaseException):
                raise result
            out.append(result)
        return out

    def _finalize_result(self, run: api.Run, subjects: list[api.SubjectResult]) -> api.RunResult:
        """Build the :class:`~src.api.models.RunResult` from *subjects*.

        Note: persistence is now delegated to the ResultsSink port.
        The caller (``create_run`` / ``_execute_run``) must call
        ``self._results_sink.write(run, result)`` after this method.
        """
        if not subjects:
            raise ApiError('empty_run', 'Run has no subjects to materialize.', status_code=500)
        aggregates = self._aggregate_subjects(subjects)
        return api.RunResult(run_id=run.id, mode=run.mode, subjects=subjects, aggregates=aggregates)

    def _expand_selection(self, selection: api.RunSelection1 | api.RunSelection2) -> list[tuple[api.ModelDefinition, api.Environment]]:
        if getattr(selection, 'leaderboard_id', None):
            leaderboard = self.get_leaderboard(selection.leaderboard_id)
            models = leaderboard.model_catalog.models if leaderboard.model_catalog else []
            environments = leaderboard.environments or []
            if selection.model_ids:
                models = [m for m in models if m.id in selection.model_ids]
            if selection.environment_ids:
                environments = [e for e in environments if e.environment_id in selection.environment_ids]
            return [(model, self._apply_membership_overrides(membership)) for model in models for membership in environments]
        return [(self._model_from_create(pair.model, _now()), self._environment_from_direct_pair(pair.environment)) for pair in selection.direct_pairs]

    def _apply_membership_overrides(self, membership: api.EnvironmentMembership) -> api.Environment:
        if not membership.overrides:
            return membership.environment
        data = membership.environment.model_dump(mode='json')
        data['metadata'] = {**(data.get('metadata') or {}), 'membership_overrides': membership.overrides}
        return api.Environment.model_validate(data)

    def _environment_from_direct_pair(self, environment_ref: api.EnvironmentRef | api.EnvironmentInlineMembership) -> api.Environment:
        if isinstance(environment_ref, api.EnvironmentRef):
            return self.get_environment(environment_ref.environment_id)
        now = _now()
        inline = environment_ref.inline_definition
        source = api.EnvironmentSource(kind=api.EnvironmentSourceKind.inline, name=inline.name, version=inline.version)
        return api.Environment(id=uuid4(), source=source, inline_definition=inline, created_at=now, updated_at=now)

    def _config_for_pending(
        self,
        mode: api.RunMode,
        execution: api.GeneratorRunConfig | api.AgentRunConfig | None,
        pending: list[PendingSubject],
        *,
        _resolved_envs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build the YAML config payload for *pending* subjects.

        Args:
            mode: The run mode (generator / agent).
            execution: Optional execution config with overrides.
            pending: Ordered list of pending subjects to configure.
            _resolved_envs: Optional pre-resolved environment map keyed by
                ``str(item.environment.id)``.  When provided, the per-item
                ``self._env_backend.resolve()`` call is skipped (avoids a
                duplicate network round-trip for the P2-4 per-task path which
                already resolved the environment in ``_run_one``).
        """
        datasets: dict[str, Any] = {}
        mcp_registry: dict[str, Any] = {}
        for item in pending:
            # Use the pre-resolved env when available (P2-4 per-task path),
            # otherwise resolve now (whole-run path and legacy callers).
            # WS2 (Gitea/GitHub): GitEnvironmentBackend will clone the repo and
            # return the resolved InlineEnvironmentDefinition here.
            env_key = str(item.environment.id)
            if _resolved_envs is not None and env_key in _resolved_envs:
                resolved = _resolved_envs[env_key]
            else:
                resolved = self._env_backend.resolve(item.environment)
            definition = resolved.definition

            if not self._supports_mode(item.environment, mode):
                raise ApiError('unsupported_mode', f'Environment {item.environment.id} does not support mode {mode.value}.')

            # Resolve dataset entry through the DatasetResolver port.
            # WS4: unity_catalog — UnityCatalogDatasetResolver will handle
            # type=unity_catalog bindings here.
            ds_entry = self._dataset_resolver.resolve(definition.dataset)

            verifier = self._resolve_verifier_binding(definition.verifier)
            reward, metrics = self._verifier_to_runner(verifier)
            ds_entry['reward'] = reward
            if metrics:
                ds_entry['metrics'] = metrics
            runtime_meta = definition.runtime.metadata or {}
            if mode == api.RunMode.generator:
                base_generator = dict(runtime_meta.get('generator', {}))
                overrides = (execution.generator_overrides if isinstance(execution, api.GeneratorRunConfig) and execution.generator_overrides else {})
                ds_entry['generator'] = {**base_generator, **overrides}
            else:
                base_agent = dict(runtime_meta.get('agent', {}))
                overrides = (execution.agent_overrides if isinstance(execution, api.AgentRunConfig) and execution.agent_overrides else {})
                ds_entry['agent'] = {**base_agent, **overrides}
                registry = runtime_meta.get('mcp_servers_registry') or {}
                mcp_registry.update(registry)
            datasets[str(item.environment.id)] = ds_entry
        return {
            'mcp_servers': mcp_registry or None,
            'datasets': datasets,
            'default': next(iter(datasets)),
            'experiments': {
                'language_models': [self._model_runtime_id(item.model) for item in pending],
                'datasets': list(datasets),
            },
        }

    def _resolve_verifier_binding(self, binding: api.VerifierSuiteBinding) -> api.VerifierSuite:
        inner = binding.root
        if isinstance(inner, api.VerifierSuiteRef):
            return self.get_verifier(inner.verifier_id)
        self._validate_aggregation(inner.aggregation or 'weighted_mean')
        self._validate_metric_definitions(inner.metrics)
        return api.VerifierSuite(
            id=uuid4(),
            name=inner.name,
            aggregation=inner.aggregation or 'weighted_mean',
            metrics=inner.metrics,
            created_at=_now(),
            updated_at=_now(),
        )

    def _verifier_to_runner(self, verifier: api.VerifierSuite) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        primary = verifier.metrics[0]
        reward = self._metric_to_runner(primary, primary=True)
        metrics = [self._metric_to_runner(metric, primary=False) for metric in verifier.metrics[1:]]
        return reward, metrics

    @staticmethod
    def _direction_value(direction: api.Direction | str | None) -> str:
        """Return the string value of *direction*, handling both enum and raw str."""
        if direction is None:
            return 'max'
        if isinstance(direction, str):
            return direction
        return direction.value

    def _metric_to_runner(self, metric: api.MetricDefinition, *, primary: bool) -> dict[str, Any]:
        payload = dict(metric.config or {})
        if metric.backbone:
            if metric.metric_kind == 'cosine_similarity':
                payload.setdefault('embedding_model', metric.backbone)
            else:
                payload.setdefault('language_model', metric.backbone)
        direction = self._direction_value(metric.direction)
        if metric.metric_kind in _REWARD_TYPES:
            if primary:
                return {'name': metric.metric_kind, **payload, 'direction': direction}
            return {
                'class': metric.metric_kind,
                'alias': metric.name,
                'direction': direction,
                'objective': bool(metric.objective),
                **payload,
            }
        if primary:
            raise ApiError('unsupported_primary_metric', f'Metric kind {metric.metric_kind!r} is not a reward and cannot be used as the primary verifier metric.')
        return {
            'class': metric.metric_kind,
            'alias': metric.name,
            'direction': direction,
            'objective': bool(metric.objective),
            **payload,
        }

    def _aggregate_subjects(self, subjects: list[api.SubjectResult]) -> list[api.AggregateMetric]:
        buckets: dict[str, list[float]] = {}
        for subject in subjects:
            for metric in subject.metrics:
                buckets.setdefault(metric.name, []).append(metric.value)
        return [api.AggregateMetric(name=name, value=sum(values) / len(values), aggregation='weighted_mean') for name, values in sorted(buckets.items()) if values]

    def _leaderboard_entries(self, leaderboard: api.Leaderboard, *, as_of: datetime | None = None) -> list[api.LeaderboardEntry]:
        entries: list[api.LeaderboardEntry] = []
        scored: list[tuple[api.ModelDefinition, api.EnvironmentMembership, float, UUID, list[api.MetricResult]]] = []
        for result in self.store.list_run_results():
            run = self.store.get_run(result.run_id)
            if run is None or run.status != api.RunStatus.succeeded:
                continue
            selection = run.selection.root
            if getattr(selection, 'leaderboard_id', None) != leaderboard.id:
                continue
            if as_of is not None and run.created_at > as_of:
                continue
            for subject in result.subjects:
                score = self._metric_value(subject.metrics, leaderboard.ranking.primary_metric)
                membership = next((m for m in leaderboard.environments or [] if m.environment_id == subject.environment.id), None)
                if score is None or membership is None:
                    continue
                scored.append((subject.model, membership, score, result.run_id, subject.metrics))
        scored.sort(key=lambda item: self._sort_key(item[2], leaderboard.ranking.primary_metric, descending=self._metric_direction(item[4], leaderboard.ranking.primary_metric, fallback=True)), reverse=True)
        for rank, (model, membership, score, run_id, _metrics) in enumerate(scored, start=1):
            entries.append(api.LeaderboardEntry(rank=rank, model_id=model.id, model_version=model.runtime.model_version, environment_id=membership.environment_id, environment_version=membership.environment.source.version, score=score, environment_breakdown={str(membership.environment_id): score}, last_run_id=run_id))
        return entries

    # validation -----------------------------------------------------------------
    def _validate_metric_definitions(self, metrics: list[api.MetricDefinition]) -> None:
        supported = {item.id for item in self.metric_kinds().items}
        for metric in metrics:
            if metric.metric_kind not in supported:
                raise ApiError('unknown_metric_kind', f'Unsupported metric kind {metric.metric_kind!r}.', details={'supported': sorted(supported)})
            if metric.weight < 0:
                raise ApiError('invalid_metric_weight', f'Metric {metric.name!r} has a negative weight.')

    def _validate_aggregation(self, aggregation: str) -> None:
        supported = {item.id for item in self.aggregations().items}
        if aggregation not in supported:
            raise ApiError('unknown_aggregation', f'Unsupported aggregation {aggregation!r}.', details={'supported': sorted(supported)})

    def _validate_model(self, payload: api.ModelDefinitionCreate | api.ModelDefinition) -> None:
        supported = {item.id for item in self.model_providers().items}
        if payload.runtime.provider not in supported:
            raise ApiError('unknown_model_provider', f'Unsupported model provider {payload.runtime.provider!r}.', details={'supported': sorted(supported)})

    def _validate_dataset_provider(self, provider: str) -> None:
        supported = {item.id for item in self.dataset_providers().items}
        if provider not in supported:
            raise ApiError('unknown_dataset_provider', f'Unsupported dataset provider {provider!r}.', details={'supported': sorted(supported)})

    def _validate_environment_payload(self, source: api.EnvironmentSource, inline_definition: api.InlineEnvironmentDefinition | None) -> None:
        if source.kind == api.EnvironmentSourceKind.inline and inline_definition is None:
            raise ApiError('inline_environment_required', '`inline_definition` is required when `source.kind` is `inline`.')
        if inline_definition is not None and source.name != inline_definition.name:
            raise ApiError('environment_name_mismatch', '`source.name` must match `inline_definition.name`.')
        if inline_definition is not None:
            self._validate_dataset_provider(inline_definition.dataset.provider)
            verifier = self._resolve_verifier_binding(inline_definition.verifier)
            self._validate_metric_definitions(verifier.metrics)
            self._validate_aggregation(verifier.aggregation or 'weighted_mean')

    def _validate_run_create(self, run_create: api.GeneratorRunCreate | api.AgentRunCreate) -> None:
        selection = run_create.selection.root
        has_leaderboard = getattr(selection, 'leaderboard_id', None) is not None
        has_direct = bool(getattr(selection, 'direct_pairs', None))
        if has_leaderboard == has_direct:
            raise ApiError('invalid_run_selection', 'Selection must choose either `leaderboard_id` or `direct_pairs`.')
        if has_leaderboard and getattr(selection, 'direct_pairs', None):
            raise ApiError('invalid_run_selection', '`direct_pairs` cannot be combined with `leaderboard_id`.')
        reuse_policy = run_create.reuse_policy or api.ReusePolicy()
        if reuse_policy.key_fields is not None and not reuse_policy.key_fields:
            raise ApiError('invalid_reuse_policy', '`reuse_policy.key_fields` cannot be empty.')

    # construction helpers -------------------------------------------------------
    def _membership_from_create(self, payload: api.EnvironmentMembershipCreate, now: datetime) -> api.EnvironmentMembership:
        environment = self._environment_from_direct_pair(payload.environment)
        return api.EnvironmentMembership(environment_id=environment.id, environment=environment, overrides=payload.overrides, created_at=now)

    def _catalog_from_create(self, payload: api.ModelCatalogReplace, now: datetime) -> api.ModelCatalog:
        models = [self._model_from_create(model, now) for model in payload.models]
        for model in models:
            self._validate_model(model)
        return api.ModelCatalog(name=payload.name or 'leaderboard-default', models=models, metadata=payload.metadata, created_at=now, updated_at=now)

    def _model_from_create(self, payload: api.ModelDefinitionCreate, now: datetime) -> api.ModelDefinition:
        self._validate_model(payload)
        return api.ModelDefinition(id=uuid4(), name=payload.name, display_name=payload.display_name, family=payload.family, tags=payload.tags, runtime=payload.runtime, metadata=payload.metadata, created_at=now, updated_at=now)

    # misc helpers ----------------------------------------------------------------
    def _resolve_environment_identity(self, env: api.Environment) -> tuple[str | None, str | None]:
        """Attempt to resolve ``(commit_sha, content_hash)`` for *env* (WS3).

        Called on environment create/update.  Returns ``(None, None)`` for
        inline environments or when the backend raises :exc:`NotImplementedError`
        (e.g. the default :class:`InlineEnvironmentBackend`).  Any other
        exception is silently swallowed so that environment creation never fails
        due to a transient VCS resolution error; the caller can retry via
        ``update_environment``.
        """
        if env.source.kind == api.EnvironmentSourceKind.inline:
            return None, None
        try:
            resolved = self._env_backend.resolve(env)
            return resolved.commit_sha, resolved.content_hash
        except NotImplementedError:
            return None, None
        except Exception:  # noqa: BLE001
            # Transient errors (network, auth) — return None so creation proceeds.
            return None, None

    def _run_fingerprint(self, model: api.ModelDefinition, environment: api.Environment, mode: str, execution: api.GeneratorRunConfig | api.AgentRunConfig | None, reuse_policy: api.ReusePolicy) -> str:
        """Compute a deterministic cache key for a model/environment pair (WS3).

        WS3 fingerprint change: when *environment* carries a ``commit_sha``
        (git-backed) the SHA is included in the payload instead of (or
        alongside) ``environment_version``.  This makes every leaderboard
        entry reproducible from a pinned commit.

        Back-compat: existing cached fingerprints were computed without
        ``commit_sha``.  Rather than silently invalidating all prior cache
        entries, ``commit_sha`` is added as an *additional* field alongside
        ``environment_version`` in ``field_values`` but is only included in
        the hash payload when present in ``key_fields`` (explicitly requested)
        OR when the environment has a ``commit_sha`` AND ``key_fields`` still
        contains ``environment_version`` (opt-in upgrade path).  Callers that
        need strict reproducibility should add ``"environment_commit_sha"`` to
        ``ReusePolicy.key_fields``.
        """
        key_fields = reuse_policy.key_fields or ['model_version', 'environment_version', 'mode', 'temperature', 'max_tokens']
        execution_data = execution.model_dump(mode='json', exclude_none=True) if execution is not None else {}
        field_values = {
            'model_name': model.name,
            'model_version': model.runtime.model_version,
            'environment_name': environment.source.name,
            'environment_version': environment.source.version,
            # WS3: include pinned commit identity when available.
            'environment_commit_sha': environment.commit_sha,
            'environment_content_hash': environment.content_hash,
            'mode': mode,
            'temperature': model.runtime.temperature,
            'max_tokens': model.runtime.max_tokens,
            **(model.runtime.hyperparameters or {}),
            **execution_data,
        }
        payload = {field: field_values.get(field) for field in key_fields}
        # When environment has a commit_sha and key_fields includes
        # environment_version (the legacy default), also fold commit_sha in so
        # that a cache hit on the same commit is guaranteed reproducible.
        if environment.commit_sha and 'environment_version' in key_fields and 'environment_commit_sha' not in key_fields:
            payload['environment_commit_sha'] = environment.commit_sha
        payload['provider'] = model.runtime.provider
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
        return digest

    def _model_runtime_id(self, model: api.ModelDefinition) -> str:
        return model.metadata.get('legacy_model_id') if model.metadata and model.metadata.get('legacy_model_id') else f'{model.runtime.provider}/{model.runtime.model_name}'

    def _metric_value(self, metrics: list[api.MetricResult], name: str) -> float | None:
        for metric in metrics:
            if metric.name == name:
                return metric.value
        return None

    def _metric_direction(self, metrics: list[api.MetricResult], name: str, *, fallback: bool = True) -> bool:
        for metric in metrics:
            if metric.name == name:
                return metric.direction != api.Direction.min
        return fallback

    def _supports_mode(self, environment: api.Environment, mode: api.RunMode) -> bool:
        if environment.inline_definition is None or environment.inline_definition.runtime.supported_modes is None:
            return True
        return mode in environment.inline_definition.runtime.supported_modes

    def _trajectory_summary(self, metric_rows: list[dict[str, Any]], mode: api.RunMode) -> api.TrajectorySummary | None:
        if mode != api.RunMode.agent:
            return None
        return api.TrajectorySummary(total_steps=len(metric_rows), tool_calls=0, failures=0)

    def _sort_key(self, score: float, _metric_name: str, *, descending: bool) -> float:
        return score if descending else -score


def _paginate(items: list[Any], *, limit: int, cursor: str | None) -> tuple[list[Any], str | None]:
    if limit < 1:
        raise ApiError('invalid_limit', '`limit` must be at least 1.')
    try:
        offset = int(cursor or 0)
    except (TypeError, ValueError) as exc:
        raise ApiError('invalid_cursor', '`cursor` must be a non-negative integer offset.') from exc
    if offset < 0:
        raise ApiError('invalid_cursor', '`cursor` must be a non-negative integer offset.')
    page = items[offset:offset + limit]
    next_cursor = str(offset + limit) if offset + limit < len(items) else None
    return page, next_cursor


def _now() -> datetime:
    return datetime.now(UTC)


def _iso(value: datetime) -> str:
    return value.isoformat()


def _combine_cache_status(cache_hits: list[api.CacheHit], pending: list[PendingSubject]) -> api.CacheStatus:
    if cache_hits and not pending:
        return api.CacheStatus.hit
    if cache_hits and pending:
        return api.CacheStatus.partial_hit
    if pending:
        return api.CacheStatus.miss
    return api.CacheStatus.bypassed


def _run_async(coro):
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    coro.close()
    raise RuntimeError('_run_async() cannot be called while an event loop is already running; await the coroutine instead.')
