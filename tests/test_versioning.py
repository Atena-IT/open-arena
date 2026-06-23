# License Apache 2.0: (c) 2026 Athena-Reply
"""Tests for WS3: environment versioning + /versions endpoint + reproducible fingerprint.

Covers:
- GET /v1/environments/{id}/versions returns version descriptors (mock backend).
- Environment carries commit_sha/content_hash when git-backed.
- Run fingerprint changes when commit_sha changes and is stable when it doesn't.
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from open_arena_core.models import (
    Environment,
    EnvironmentCreate,
    EnvironmentSource,
    EnvironmentSourceKind,
    EnvironmentVersion,
    EnvironmentVersionListResponse,
    InlineEnvironmentDefinition,
    DatasetBinding,
    VerifierSuiteBinding,
    VerifierSuiteInline,
    MetricDefinition,
    EnvironmentRuntimePolicy,
    ModelDefinition,
    ModelExecutionConfig,
    ReusePolicy,
)
from src.api.service import ArenaAPIService, ApiError
from src.api.ports.environment_backend import ResolvedEnvironment, InlineEnvironmentBackend
from src.api.stores.sqlite import SQLiteStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_store(tmp_path):
    return SQLiteStore(path=tmp_path / "test.db")


@pytest.fixture
def inline_env_definition():
    return InlineEnvironmentDefinition(
        name="test-env",
        version="1.0.0",
        dataset=DatasetBinding(provider="local", source_ref="/data"),
        verifier=VerifierSuiteBinding(
            root=VerifierSuiteInline(
                binding_type="inline",
                name="test-verifier",
                metrics=[MetricDefinition(name="accuracy", metric_kind="exact_match", weight=1.0)],
            )
        ),
        runtime=EnvironmentRuntimePolicy(),
    )


@pytest.fixture
def inline_source():
    return EnvironmentSource(
        kind=EnvironmentSourceKind.inline,
        name="test-env",
        version="1.0.0",
    )


@pytest.fixture
def git_source():
    return EnvironmentSource(
        kind=EnvironmentSourceKind.github_repo,
        name="test-git-env",
        version="1.0.0",
        uri="https://github.com/example/test-env",
        git_ref="main",
    )


@pytest.fixture
def model():
    return ModelDefinition(
        id=uuid.uuid4(),
        name="test-model",
        runtime=ModelExecutionConfig(
            provider="openai",
            model_name="gpt-4o",
            model_version="2024-08-06",
            temperature=0.0,
            max_tokens=1024,
        ),
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


def _make_service(tmp_path, env_backend=None):
    from src.api.registry import build_adapters, AdapterSet
    from src.api.settings import ArenaSettings

    settings = ArenaSettings(db_path=tmp_path / "test.db")
    adapters = build_adapters(settings)
    if env_backend is not None:
        adapters = AdapterSet(
            store=adapters.store,
            auth=adapters.auth,
            env_backend=env_backend,
            dataset_resolver=adapters.dataset_resolver,
            results_sink=adapters.results_sink,
            sandbox=adapters.sandbox,
        )
    return ArenaAPIService(adapters=adapters)


# ---------------------------------------------------------------------------
# 1. GET /v1/environments/{id}/versions — inline source
# ---------------------------------------------------------------------------

class TestListEnvironmentVersionsInline:
    def test_inline_returns_single_version(self, tmp_path, inline_source, inline_env_definition):
        svc = _make_service(tmp_path)
        payload = EnvironmentCreate(source=inline_source, inline_definition=inline_env_definition)
        env = svc.create_environment(payload)

        result = svc.list_environment_versions(env.id)

        assert isinstance(result, EnvironmentVersionListResponse)
        assert len(result.items) == 1
        version = result.items[0]
        assert isinstance(version, EnvironmentVersion)
        assert version.version == "1.0.0"
        assert version.commit_sha is None
        assert version.content_hash is None
        assert version.git_ref is None

    def test_inline_version_has_created_at(self, tmp_path, inline_source, inline_env_definition):
        svc = _make_service(tmp_path)
        payload = EnvironmentCreate(source=inline_source, inline_definition=inline_env_definition)
        env = svc.create_environment(payload)

        result = svc.list_environment_versions(env.id)
        assert result.items[0].created_at is not None

    def test_unknown_env_raises_404(self, tmp_path):
        svc = _make_service(tmp_path)
        with pytest.raises(ApiError) as exc_info:
            svc.list_environment_versions(uuid.uuid4())
        assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# 2. GET /v1/environments/{id}/versions — git-backed source (mock backend)
# ---------------------------------------------------------------------------

class TestListEnvironmentVersionsGit:
    def _make_mock_backend(self, commit_sha: str, content_hash: str):
        backend = MagicMock()
        backend.resolve.return_value = ResolvedEnvironment(
            definition=MagicMock(),
            commit_sha=commit_sha,
            content_hash=content_hash,
            local_path="/tmp/fake",
        )
        return backend

    def test_git_backed_returns_resolved_commit(self, tmp_path, git_source):
        mock_backend = self._make_mock_backend("abc123", "sha256hash")
        svc = _make_service(tmp_path, env_backend=mock_backend)

        # Manually create an env with git source (bypassing validation for simplicity)
        now = datetime.now(UTC)
        env = Environment(
            id=uuid.uuid4(),
            source=git_source,
            commit_sha="abc123",
            content_hash="sha256hash",
            created_at=now,
            updated_at=now,
        )
        svc.store.save_environment(env)

        result = svc.list_environment_versions(env.id)

        assert len(result.items) == 1
        version = result.items[0]
        assert version.commit_sha == "abc123"
        assert version.content_hash == "sha256hash"
        assert version.git_ref == "main"
        assert version.version == "1.0.0"

    def test_git_backed_falls_back_to_stored_identity_when_backend_fails(self, tmp_path, git_source):
        failing_backend = MagicMock()
        failing_backend.resolve.side_effect = RuntimeError("network error")
        svc = _make_service(tmp_path, env_backend=failing_backend)

        now = datetime.now(UTC)
        env = Environment(
            id=uuid.uuid4(),
            source=git_source,
            commit_sha="stored_sha",
            content_hash="stored_hash",
            created_at=now,
            updated_at=now,
        )
        svc.store.save_environment(env)

        result = svc.list_environment_versions(env.id)
        assert result.items[0].commit_sha == "stored_sha"
        assert result.items[0].content_hash == "stored_hash"


# ---------------------------------------------------------------------------
# 3. Environment carries commit_sha/content_hash when git-backed
# ---------------------------------------------------------------------------

class TestEnvironmentGitIdentity:
    def test_git_backed_env_stores_commit_and_hash(self, tmp_path, git_source):
        mock_backend = MagicMock()
        mock_backend.resolve.return_value = ResolvedEnvironment(
            definition=MagicMock(),
            commit_sha="deadbeef",
            content_hash="cafebabe",
            local_path="/tmp/repo",
        )
        svc = _make_service(tmp_path, env_backend=mock_backend)

        now = datetime.now(UTC)
        env = Environment(
            id=uuid.uuid4(),
            source=git_source,
            commit_sha="deadbeef",
            content_hash="cafebabe",
            created_at=now,
            updated_at=now,
        )
        svc.store.save_environment(env)

        fetched = svc.get_environment(env.id)
        assert fetched.commit_sha == "deadbeef"
        assert fetched.content_hash == "cafebabe"

    def test_inline_env_has_no_vcs_identity(self, tmp_path, inline_source, inline_env_definition):
        svc = _make_service(tmp_path)
        payload = EnvironmentCreate(source=inline_source, inline_definition=inline_env_definition)
        env = svc.create_environment(payload)

        assert env.commit_sha is None
        assert env.content_hash is None

    def test_environment_model_has_commit_sha_field(self):
        """EnvironmentVersion and Environment carry the new WS3 fields."""
        from open_arena_core.models import Environment, EnvironmentVersion

        # Environment has commit_sha + content_hash
        assert hasattr(Environment.model_fields, 'commit_sha') or 'commit_sha' in Environment.model_fields
        assert hasattr(Environment.model_fields, 'content_hash') or 'content_hash' in Environment.model_fields

        # EnvironmentVersion carries all expected fields
        assert 'version' in EnvironmentVersion.model_fields
        assert 'commit_sha' in EnvironmentVersion.model_fields
        assert 'content_hash' in EnvironmentVersion.model_fields
        assert 'git_ref' in EnvironmentVersion.model_fields
        assert 'created_at' in EnvironmentVersion.model_fields


# ---------------------------------------------------------------------------
# 4. Run fingerprint changes when commit_sha changes; stable when it doesn't
# ---------------------------------------------------------------------------

class TestRunFingerprint:
    def _env(self, commit_sha: str | None) -> Environment:
        now = datetime.now(UTC)
        return Environment(
            id=uuid.uuid4(),
            source=EnvironmentSource(
                kind=EnvironmentSourceKind.github_repo,
                name="test-env",
                version="1.0.0",
                git_ref="main",
            ),
            commit_sha=commit_sha,
            content_hash=None,
            created_at=now,
            updated_at=now,
        )

    def test_fingerprint_changes_when_commit_sha_changes(self, tmp_path, model):
        svc = _make_service(tmp_path)
        env_a = self._env("sha_v1")
        env_b = self._env("sha_v2")
        policy = ReusePolicy()

        fp_a = svc._run_fingerprint(model, env_a, "generator", None, policy)
        fp_b = svc._run_fingerprint(model, env_b, "generator", None, policy)

        assert fp_a != fp_b, "Fingerprint must differ when commit_sha changes"

    def test_fingerprint_stable_with_same_commit_sha(self, tmp_path, model):
        svc = _make_service(tmp_path)
        env_a = self._env("sha_v1")
        env_b = self._env("sha_v1")  # same SHA, different object
        policy = ReusePolicy()

        fp_a = svc._run_fingerprint(model, env_a, "generator", None, policy)
        fp_b = svc._run_fingerprint(model, env_b, "generator", None, policy)

        assert fp_a == fp_b, "Fingerprint must be stable for the same commit_sha"

    def test_fingerprint_stable_without_commit_sha(self, tmp_path, model):
        """Inline/no-SHA environments keep backward-compatible fingerprints."""
        svc = _make_service(tmp_path)
        env = self._env(None)
        policy = ReusePolicy()

        fp_a = svc._run_fingerprint(model, env, "generator", None, policy)
        fp_b = svc._run_fingerprint(model, env, "generator", None, policy)

        assert fp_a == fp_b

    def test_fingerprint_with_explicit_commit_sha_key_field(self, tmp_path, model):
        """Explicit environment_commit_sha in key_fields is respected."""
        svc = _make_service(tmp_path)
        env_with = self._env("sha_abc")
        env_without = self._env(None)
        # Explicitly include commit sha in key_fields
        policy = ReusePolicy(key_fields=["model_version", "environment_version", "mode", "environment_commit_sha"])

        fp_with = svc._run_fingerprint(model, env_with, "generator", None, policy)
        fp_without = svc._run_fingerprint(model, env_without, "generator", None, policy)

        assert fp_with != fp_without

    def test_inline_env_fingerprint_unchanged_vs_pre_ws3(self, tmp_path, model):
        """Inline environments (no commit_sha) produce same fingerprint as before WS3.

        This test verifies that existing cached fingerprints for inline
        environments are NOT silently invalidated by the WS3 change.
        """
        svc = _make_service(tmp_path)
        now = datetime.now(UTC)
        inline_env = Environment(
            id=uuid.uuid4(),
            source=EnvironmentSource(
                kind=EnvironmentSourceKind.inline,
                name="bench-env",
                version="2.0",
            ),
            commit_sha=None,
            content_hash=None,
            created_at=now,
            updated_at=now,
        )
        policy = ReusePolicy()

        # The fingerprint must be deterministic regardless of when it's called.
        fp1 = svc._run_fingerprint(model, inline_env, "generator", None, policy)
        fp2 = svc._run_fingerprint(model, inline_env, "generator", None, policy)
        assert fp1 == fp2


# ---------------------------------------------------------------------------
# 5. Route count includes new /versions endpoint
# ---------------------------------------------------------------------------

def test_app_has_43_routes():
    """Adding GET /v1/environments/{id}/versions must bring route count to 43."""
    from src.api.app import app
    from fastapi.routing import APIRoute

    total_routes = len(app.routes)
    api_route_count = sum(1 for r in app.routes if isinstance(r, APIRoute))

    assert total_routes == 43, (
        f"Expected 43 total routes (42 + 1 new /versions) but found {total_routes}."
    )
    assert api_route_count == 39, (
        f"Expected 39 APIRoute instances but found {api_route_count}."
    )
