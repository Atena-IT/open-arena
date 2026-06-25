# License Apache 2.0: (c) 2026 Athena-Reply
"""Tests for PrimeEnvHubBackend (P2-3: Prime Intellect Environment Hub, issue #65).

All HTTP calls are fully mocked via :mod:`unittest.mock` — no real network
requests are made.

Test coverage
-------------
* ``_parse_owner_slug``: parsing from URI, external_id, name (``owner/slug``
  and bare slug), and error on missing information.
* ``_resolved_version``: default to ``"latest"``; respect explicit version.
* ``PI_API_KEY`` missing → :exc:`ValueError` with a descriptive message.
* ``_resolve_latest_version``: happy-path (list of strings), dict-based list,
  404 raises :exc:`LookupError`, empty list raises :exc:`LookupError`.
* ``_fetch_metadata``: happy-path, 404 raises :exc:`LookupError`.
* ``PrimeEnvHubBackend.resolve``:
  - ``latest`` alias is pinned to a concrete version id.
  - Content hash (SHA-256) is computed from the downloaded artifact.
  - Cache key / path follows ``{owner}/{slug}/{resolved_version}/``.
  - ``ResolvedEnvironment`` fields are populated correctly (``commit_sha``,
    ``content_hash``, ``local_path``, ``definition``).
  - Cache hit skips the download.
  - ``inline`` kind is delegated to :class:`InlineEnvironmentBackend`.
  - Unsupported ``source.kind`` raises :exc:`NotImplementedError`.
* Registry wiring: ``OPEN_ARENA_ENV_BACKEND=prime_hub`` yields a
  :class:`PrimeEnvHubBackend` instance; existing ``inline`` and ``git``
  selections remain unaffected.
"""
from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from src.api import models as api
from src.api.environments.prime_hub_backend import (
    PrimeEnvHubBackend,
    _parse_owner_slug,
    _resolved_version,
    _require_api_key,
    _resolve_latest_version,
    _fetch_metadata,
    _compute_sha256,
    _cache_path,
    _infer_artifact_filename,
)
from src.api.ports.environment_backend import (
    InlineEnvironmentBackend,
    ResolvedEnvironment,
)


# ---------------------------------------------------------------------------
# Test-data helpers
# ---------------------------------------------------------------------------

def _make_source(
    kind: api.EnvironmentSourceKind = api.EnvironmentSourceKind.prime_environment_hub,
    name: str = "primeintellect/gsm8k-verifier",
    version: str = "1.0.0",
    uri: str | None = "https://hub.primeintellect.ai/primeintellect/gsm8k-verifier",
    external_id: str | None = None,
) -> api.EnvironmentSource:
    return api.EnvironmentSource(
        kind=kind,
        name=name,
        version=version,
        uri=uri,
        external_id=external_id,
    )


def _make_env(
    source: api.EnvironmentSource,
    inline_def: api.InlineEnvironmentDefinition | None = None,
) -> api.Environment:
    now = datetime.now(tz=timezone.utc)
    return api.Environment(
        id=uuid.uuid4(),
        source=source,
        inline_definition=inline_def,
        created_at=now,
        updated_at=now,
    )


def _make_inline_def(
    name: str = "gsm8k-verifier",
    version: str = "1.0.0",
) -> api.InlineEnvironmentDefinition:
    return api.InlineEnvironmentDefinition(
        name=name,
        version=version,
        dataset=api.DatasetBinding(provider="local", source_ref="/data"),
        verifier=api.VerifierSuiteBinding(
            root=api.VerifierSuiteInline(
                binding_type="inline",
                name="v",
                metrics=[
                    api.MetricDefinition(
                        name="acc", metric_kind="exact_match", weight=1.0
                    )
                ],
            )
        ),
        runtime=api.EnvironmentRuntimePolicy(),
    )


# ---------------------------------------------------------------------------
# _parse_owner_slug
# ---------------------------------------------------------------------------

class TestParseOwnerSlug:
    def test_uri_with_owner_and_slug(self):
        source = _make_source(
            uri="https://hub.primeintellect.ai/acme/my-env",
            external_id=None,
            name="acme/my-env",
        )
        owner, slug = _parse_owner_slug(source)
        assert owner == "acme"
        assert slug == "my-env"

    def test_uri_takes_priority_over_external_id(self):
        source = _make_source(
            uri="https://hub.primeintellect.ai/uri-owner/uri-slug",
            external_id="eid-owner/eid-slug",
            name="name-owner/name-slug",
        )
        owner, slug = _parse_owner_slug(source)
        assert owner == "uri-owner"
        assert slug == "uri-slug"

    def test_external_id_fallback(self):
        source = _make_source(
            uri=None,
            external_id="extowner/extslug",
            name="nameowner/nameslug",
        )
        owner, slug = _parse_owner_slug(source)
        assert owner == "extowner"
        assert slug == "extslug"

    def test_name_fallback_with_slash(self):
        source = _make_source(
            uri=None,
            external_id=None,
            name="nameowner/nameslug",
        )
        owner, slug = _parse_owner_slug(source)
        assert owner == "nameowner"
        assert slug == "nameslug"

    def test_name_bare_slug_returns_empty_owner(self):
        source = _make_source(
            uri=None,
            external_id=None,
            name="bareslug",
        )
        owner, slug = _parse_owner_slug(source)
        assert owner == ""
        assert slug == "bareslug"

    def test_uri_single_segment_falls_through_to_name(self):
        """When URI path has only one segment, fall through to external_id / name."""
        source = _make_source(
            uri="https://hub.primeintellect.ai/onlyone",
            external_id="eid-owner/eid-slug",
            name="name-owner/name-slug",
        )
        owner, slug = _parse_owner_slug(source)
        assert owner == "eid-owner"
        assert slug == "eid-slug"

    def test_all_empty_raises(self):
        source = api.EnvironmentSource(
            kind=api.EnvironmentSourceKind.prime_environment_hub,
            name="",
            version="1.0",
        )
        with pytest.raises(ValueError, match="Cannot determine owner/slug"):
            _parse_owner_slug(source)


# ---------------------------------------------------------------------------
# _resolved_version
# ---------------------------------------------------------------------------

class TestResolvedVersion:
    def test_explicit_version_returned_as_is(self):
        source = _make_source(version="2.3.1")
        assert _resolved_version(source) == "2.3.1"

    def test_empty_string_defaults_to_latest(self):
        source = _make_source(version="")
        assert _resolved_version(source) == "latest"

    def test_whitespace_only_defaults_to_latest(self):
        source = _make_source(version="   ")
        assert _resolved_version(source) == "latest"


# ---------------------------------------------------------------------------
# PI_API_KEY requirement
# ---------------------------------------------------------------------------

class TestRequireApiKey:
    def test_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("PI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="PI_API_KEY"):
            _require_api_key()

    def test_present_key_returned(self, monkeypatch):
        monkeypatch.setenv("PI_API_KEY", "sk-test-key-123")
        assert _require_api_key() == "sk-test-key-123"


# ---------------------------------------------------------------------------
# _resolve_latest_version
# ---------------------------------------------------------------------------

class TestResolveLatestVersion:
    def _mock_client(self, status_code: int, json_data) -> MagicMock:
        client = MagicMock()
        resp = MagicMock()
        resp.status_code = status_code
        resp.json.return_value = json_data
        resp.raise_for_status = MagicMock()
        client.get.return_value = resp
        return client

    def test_list_of_strings_returns_first(self):
        client = self._mock_client(200, ["v3", "v2", "v1"])
        result = _resolve_latest_version(client, "owner", "slug")
        assert result == "v3"
        client.get.assert_called_once_with("/owner/slug/versions")

    def test_list_of_dicts_with_version_key(self):
        client = self._mock_client(200, [{"version": "2024.1"}, {"version": "2023.12"}])
        result = _resolve_latest_version(client, "acme", "env")
        assert result == "2024.1"

    def test_list_of_dicts_with_id_key(self):
        client = self._mock_client(200, [{"id": "abc123", "name": "env-v1"}])
        result = _resolve_latest_version(client, "owner", "slug")
        assert result == "abc123"

    def test_404_raises_lookup_error(self):
        client = self._mock_client(404, {})
        with pytest.raises(LookupError, match="owner/slug"):
            _resolve_latest_version(client, "owner", "slug")

    def test_empty_list_raises_lookup_error(self):
        client = self._mock_client(200, [])
        with pytest.raises(LookupError, match="empty versions list"):
            _resolve_latest_version(client, "owner", "slug")


# ---------------------------------------------------------------------------
# _fetch_metadata
# ---------------------------------------------------------------------------

class TestFetchMetadata:
    def _mock_client(self, status_code: int, json_data) -> MagicMock:
        client = MagicMock()
        resp = MagicMock()
        resp.status_code = status_code
        resp.json.return_value = json_data
        resp.raise_for_status = MagicMock()
        client.get.return_value = resp
        return client

    def test_happy_path_returns_metadata(self):
        meta = {"download_url": "https://files.example.com/artifact.tar.gz", "name": "env"}
        client = self._mock_client(200, meta)
        result = _fetch_metadata(client, "owner", "slug", "v1.0")
        assert result == meta
        client.get.assert_called_once_with("/owner/slug/@v1.0")

    def test_404_raises_lookup_error(self):
        client = self._mock_client(404, {})
        with pytest.raises(LookupError, match="owner/slug@badver"):
            _fetch_metadata(client, "owner", "slug", "badver")

    def test_404_message_contains_version(self):
        client = self._mock_client(404, {})
        with pytest.raises(LookupError, match="nonexistent"):
            _fetch_metadata(client, "owner", "slug", "nonexistent")


# ---------------------------------------------------------------------------
# _compute_sha256 and _infer_artifact_filename helpers
# ---------------------------------------------------------------------------

class TestComputeSha256:
    def test_sha256_matches_expected(self, tmp_path):
        content = b"hello prime intellect"
        f = tmp_path / "artifact.tar.gz"
        f.write_bytes(content)
        expected = hashlib.sha256(content).hexdigest()
        assert _compute_sha256(f) == expected

    def test_sha256_differs_for_different_content(self, tmp_path):
        f1 = tmp_path / "a.gz"
        f2 = tmp_path / "b.gz"
        f1.write_bytes(b"aaaa")
        f2.write_bytes(b"bbbb")
        assert _compute_sha256(f1) != _compute_sha256(f2)


class TestInferArtifactFilename:
    def test_tar_gz_url(self):
        assert _infer_artifact_filename("https://files.example.com/env-v1.tar.gz") == "env-v1.tar.gz"

    def test_zip_url(self):
        assert _infer_artifact_filename("https://cdn.prime.ai/envs/pkg-1.0.zip") == "pkg-1.0.zip"

    def test_url_with_no_extension_fallback(self):
        name = _infer_artifact_filename("https://example.com/download")
        assert name == "download"

    def test_empty_path_fallback(self):
        name = _infer_artifact_filename("https://example.com")
        assert name == "artifact.tar.gz"


# ---------------------------------------------------------------------------
# _cache_path module-level helper
# ---------------------------------------------------------------------------

class TestCachePath:
    def test_cache_path_with_owner(self):
        from src.api.environments.prime_hub_backend import _CACHE_ROOT
        p = _cache_path("acme", "gsm8k", "v1.0")
        assert p == _CACHE_ROOT / "acme" / "gsm8k" / "v1.0"

    def test_cache_path_without_owner(self):
        from src.api.environments.prime_hub_backend import _CACHE_ROOT
        p = _cache_path("", "bareenv", "2.0")
        assert p == _CACHE_ROOT / "_" / "bareenv" / "2.0"


# ---------------------------------------------------------------------------
# PrimeEnvHubBackend.resolve — full flow
# ---------------------------------------------------------------------------

class TestPrimeEnvHubBackendResolve:
    """Integration-style tests with all HTTP and filesystem I/O mocked."""

    def _setup_backend(self, tmp_path: Path) -> PrimeEnvHubBackend:
        return PrimeEnvHubBackend(cache_root=tmp_path / "cache")

    def _artifact_bytes(self) -> bytes:
        return b"fake-artifact-content-xyz"

    def _expected_hash(self) -> str:
        return hashlib.sha256(self._artifact_bytes()).hexdigest()

    def _mock_httpx_client(
        self,
        versions_resp: list,
        metadata_resp: dict,
        pinned_version: str = "v1.2.3",
    ):
        """Return a context-manager mock for httpx.Client used by the backend."""
        mock_cm = MagicMock()
        client = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=client)
        mock_cm.__exit__ = MagicMock(return_value=False)

        def get_side_effect(url, **kwargs):
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            if url.endswith("/versions"):
                resp.status_code = 200
                resp.json.return_value = versions_resp
            elif f"@{pinned_version}" in url:
                resp.status_code = 200
                resp.json.return_value = metadata_resp
            else:
                resp.status_code = 404
                resp.json.return_value = {}
            return resp

        client.get.side_effect = get_side_effect
        return mock_cm

    def test_resolve_latest_pins_to_concrete_version(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PI_API_KEY", "sk-test")

        backend = self._setup_backend(tmp_path)
        pinned = "v1.2.3"
        dl_url = "https://cdn.prime.ai/artifact.tar.gz"
        artifact_bytes = self._artifact_bytes()

        source = _make_source(version="latest")
        env = _make_env(source)

        with patch("httpx.Client") as mock_cls:
            mock_cls.return_value = self._mock_httpx_client(
                versions_resp=[pinned, "v1.0.0"],
                metadata_resp={"download_url": dl_url},
                pinned_version=pinned,
            )
            with patch.object(
                PrimeEnvHubBackend,
                "_download_artifact",
                side_effect=lambda url, dest_dir, key: (
                    dest_dir.mkdir(parents=True, exist_ok=True),
                    (dest_dir / "artifact.tar.gz").write_bytes(artifact_bytes),
                ),
            ):
                result = backend.resolve(env)

        assert result.commit_sha == pinned
        assert result.content_hash == self._expected_hash()
        assert result.local_path is not None
        assert result.definition is not None

    def test_resolve_explicit_version_skips_versions_endpoint(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PI_API_KEY", "sk-test")

        backend = self._setup_backend(tmp_path)
        explicit_version = "2.0.0"
        dl_url = "https://cdn.prime.ai/pkg.tar.gz"
        artifact_bytes = self._artifact_bytes()

        source = _make_source(version=explicit_version)
        env = _make_env(source)

        client_calls: list[str] = []

        def get_side_effect(url, **kwargs):
            client_calls.append(url)
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            if f"@{explicit_version}" in url:
                resp.status_code = 200
                resp.json.return_value = {"download_url": dl_url}
            else:
                resp.status_code = 404
                resp.json.return_value = {}
            return resp

        mock_cm = MagicMock()
        client = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=client)
        mock_cm.__exit__ = MagicMock(return_value=False)
        client.get.side_effect = get_side_effect

        with patch("httpx.Client") as mock_cls:
            mock_cls.return_value = mock_cm
            with patch.object(
                PrimeEnvHubBackend,
                "_download_artifact",
                side_effect=lambda url, dest_dir, key: (
                    dest_dir.mkdir(parents=True, exist_ok=True),
                    (dest_dir / "pkg.tar.gz").write_bytes(artifact_bytes),
                ),
            ):
                result = backend.resolve(env)

        # /versions must NOT have been called
        assert not any("versions" in c for c in client_calls)
        assert result.commit_sha == explicit_version

    def test_resolve_cache_hit_skips_download(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PI_API_KEY", "sk-test")

        backend = self._setup_backend(tmp_path)
        pinned = "v3.0"
        artifact_bytes = self._artifact_bytes()

        source = _make_source(version=pinned)
        env = _make_env(source)

        # Pre-populate the cache
        owner, slug = _parse_owner_slug(source)
        cache_dir = backend._cache_path(owner, slug, pinned)
        cache_dir.mkdir(parents=True, exist_ok=True)
        artifact_file = cache_dir / "artifact.tar.gz"
        artifact_file.write_bytes(artifact_bytes)

        client_mock = MagicMock()
        client_mock.raise_for_status = MagicMock()
        client_mock.status_code = 200
        client_mock.json.return_value = {"download_url": "https://example.com/artifact.tar.gz"}

        mock_cm = MagicMock()
        inner_client = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=inner_client)
        mock_cm.__exit__ = MagicMock(return_value=False)
        inner_client.get.return_value = client_mock

        with patch("httpx.Client") as mock_cls, \
             patch.object(PrimeEnvHubBackend, "_download_artifact") as mock_dl:
            mock_cls.return_value = mock_cm
            result = backend.resolve(env)

        # Download must not be called — artifact already cached
        mock_dl.assert_not_called()
        assert result.content_hash == self._expected_hash()
        assert result.local_path == str(cache_dir)

    def test_resolve_populates_resolved_environment_fields(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PI_API_KEY", "sk-test")

        backend = self._setup_backend(tmp_path)
        pinned = "v9.1.0"
        artifact_bytes = b"some-artifact-data"
        expected_hash = hashlib.sha256(artifact_bytes).hexdigest()

        source = _make_source(
            uri="https://hub.primeintellect.ai/myorg/cool-env",
            version=pinned,
        )
        env = _make_env(source)

        meta = {"download_url": "https://cdn.example.com/cool.tar.gz"}
        mock_cm = MagicMock()
        inner_client = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=inner_client)
        mock_cm.__exit__ = MagicMock(return_value=False)
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = meta
        resp.raise_for_status = MagicMock()
        inner_client.get.return_value = resp

        with patch("httpx.Client") as mock_cls, \
             patch.object(
                 PrimeEnvHubBackend,
                 "_download_artifact",
                 side_effect=lambda url, dest_dir, key: (
                     dest_dir.mkdir(parents=True, exist_ok=True),
                     (dest_dir / "cool.tar.gz").write_bytes(artifact_bytes),
                 ),
             ):
            mock_cls.return_value = mock_cm
            result = backend.resolve(env)

        assert isinstance(result, ResolvedEnvironment)
        assert result.commit_sha == pinned
        assert result.content_hash == expected_hash
        assert result.local_path is not None
        # local_path should be the cache directory
        expected_cache = backend._cache_path("myorg", "cool-env", pinned)
        assert result.local_path == str(expected_cache)
        assert result.definition is not None
        assert result.definition.version == pinned

    def test_resolve_cache_key_includes_owner_slug_version(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PI_API_KEY", "sk-test")

        backend = self._setup_backend(tmp_path)
        source = _make_source(
            uri="https://hub.primeintellect.ai/owner123/slug456",
            version="7.8.9",
        )
        env = _make_env(source)
        artifact_bytes = b"artifact"

        mock_cm = MagicMock()
        inner = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=inner)
        mock_cm.__exit__ = MagicMock(return_value=False)
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"download_url": "https://cdn.example.com/a.tar.gz"}
        resp.raise_for_status = MagicMock()
        inner.get.return_value = resp

        with patch("httpx.Client") as mock_cls, \
             patch.object(
                 PrimeEnvHubBackend,
                 "_download_artifact",
                 side_effect=lambda url, dest_dir, key: (
                     dest_dir.mkdir(parents=True, exist_ok=True),
                     (dest_dir / "a.tar.gz").write_bytes(artifact_bytes),
                 ),
             ):
            mock_cls.return_value = mock_cm
            result = backend.resolve(env)

        assert "owner123" in result.local_path
        assert "slug456" in result.local_path
        assert "7.8.9" in result.local_path

    def test_missing_pi_api_key_raises_before_any_request(self, tmp_path, monkeypatch):
        monkeypatch.delenv("PI_API_KEY", raising=False)

        backend = self._setup_backend(tmp_path)
        source = _make_source()
        env = _make_env(source)

        with patch("httpx.Client") as mock_cls:
            with pytest.raises(ValueError, match="PI_API_KEY"):
                backend.resolve(env)

        mock_cls.assert_not_called()

    def test_404_on_versions_raises_lookup_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PI_API_KEY", "sk-test")

        backend = self._setup_backend(tmp_path)
        source = _make_source(version="latest")
        env = _make_env(source)

        mock_cm = MagicMock()
        inner = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=inner)
        mock_cm.__exit__ = MagicMock(return_value=False)
        resp = MagicMock()
        resp.status_code = 404
        resp.raise_for_status = MagicMock()
        inner.get.return_value = resp

        with patch("httpx.Client") as mock_cls:
            mock_cls.return_value = mock_cm
            with pytest.raises(LookupError, match="not found"):
                backend.resolve(env)

    def test_404_on_metadata_raises_lookup_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PI_API_KEY", "sk-test")

        backend = self._setup_backend(tmp_path)
        explicit_version = "0.0.1"
        source = _make_source(version=explicit_version)
        env = _make_env(source)

        mock_cm = MagicMock()
        inner = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=inner)
        mock_cm.__exit__ = MagicMock(return_value=False)
        resp = MagicMock()
        resp.status_code = 404
        resp.raise_for_status = MagicMock()
        inner.get.return_value = resp

        with patch("httpx.Client") as mock_cls:
            mock_cls.return_value = mock_cm
            with pytest.raises(LookupError, match="0.0.1"):
                backend.resolve(env)


# ---------------------------------------------------------------------------
# Inline delegation
# ---------------------------------------------------------------------------

class TestInlineDelegation:
    def test_inline_kind_delegated_to_inline_backend(self):
        inline_def = _make_inline_def()
        source = _make_source(kind=api.EnvironmentSourceKind.inline)
        env = _make_env(source, inline_def=inline_def)

        backend = PrimeEnvHubBackend()
        with patch.object(
            InlineEnvironmentBackend,
            "resolve",
            return_value=ResolvedEnvironment(definition=inline_def),
        ) as mock_resolve:
            result = backend.resolve(env)

        mock_resolve.assert_called_once_with(env)
        assert result.definition is inline_def
        assert result.commit_sha is None

    def test_inline_kind_no_definition_raises_api_error(self):
        from src.api.service import ApiError

        source = _make_source(kind=api.EnvironmentSourceKind.inline)
        env = _make_env(source, inline_def=None)

        backend = PrimeEnvHubBackend()
        with pytest.raises(ApiError):
            backend.resolve(env)


# ---------------------------------------------------------------------------
# Unsupported source.kind
# ---------------------------------------------------------------------------

class TestUnsupportedKind:
    def test_github_repo_raises_not_implemented(self):
        source = _make_source(kind=api.EnvironmentSourceKind.github_repo)
        env = _make_env(source)

        backend = PrimeEnvHubBackend()
        with pytest.raises(NotImplementedError, match="github_repo"):
            backend.resolve(env)

    def test_huggingface_hub_raises_not_implemented(self):
        source = _make_source(kind=api.EnvironmentSourceKind.huggingface_hub)
        env = _make_env(source)

        backend = PrimeEnvHubBackend()
        with pytest.raises(NotImplementedError, match="huggingface_hub"):
            backend.resolve(env)


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------

class TestRegistryWiring:
    def test_prime_hub_selector_returns_prime_env_hub_backend(self, tmp_path):
        from src.api.registry import _build_env_backend
        from src.api.settings import ArenaSettings

        settings = ArenaSettings(
            env_backend="prime_hub",
            db_path=tmp_path / "test.db",
        )
        backend = _build_env_backend(settings)
        assert isinstance(backend, PrimeEnvHubBackend)

    def test_inline_still_works_after_prime_hub_addition(self, tmp_path):
        from src.api.registry import _build_env_backend
        from src.api.settings import ArenaSettings

        settings = ArenaSettings(
            env_backend="inline",
            db_path=tmp_path / "test.db",
        )
        backend = _build_env_backend(settings)
        assert isinstance(backend, InlineEnvironmentBackend)

    def test_git_still_works_after_prime_hub_addition(self, tmp_path):
        from src.api.registry import _build_env_backend
        from src.api.settings import ArenaSettings
        from src.api.environments.git_backend import GitEnvironmentBackend

        settings = ArenaSettings(
            env_backend="git",
            db_path=tmp_path / "test.db",
        )
        backend = _build_env_backend(settings)
        assert isinstance(backend, GitEnvironmentBackend)

    def test_unknown_backend_raises_value_error_mentioning_prime_hub(self, tmp_path):
        from src.api.registry import _build_env_backend
        from src.api.settings import ArenaSettings

        settings = ArenaSettings(
            env_backend="nonexistent",
            db_path=tmp_path / "test.db",
        )
        with pytest.raises(ValueError, match="prime_hub"):
            _build_env_backend(settings)
