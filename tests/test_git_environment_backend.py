# License Apache 2.0: (c) 2026 Athena-Reply
"""Tests for GitEnvironmentBackend (WS2: Gitea/GitHub, issue #36).

All git operations and HTTP calls are fully mocked — no real repos are
cloned and no real Gitea API is contacted.

Test coverage
-------------
* Clone-URL construction (Gitea with token, Gitea without token, GitHub
  with token, GitHub without token, SHA-only ref via two-step fetch).
* ``_resolve_commit_sha`` and ``_compute_content_hash`` extraction from a
  mocked ``subprocess.run``.
* ``ResolvedEnvironment`` population (commit_sha, content_hash, local_path,
  definition).
* Inline delegation: ``kind=inline`` is forwarded to
  ``InlineEnvironmentBackend`` transparently.
* Unsupported ``source.kind`` raises ``NotImplementedError``.
* ``snapshot_inline``: Gitea repo creation + file commit via mocked httpx;
  asserts repo URL and commit SHA are returned correctly.
* Registry wiring: ``OPEN_ARENA_ENV_BACKEND=git`` yields a
  ``GitEnvironmentBackend`` instance.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

from src.api import models as api
from src.api.environments.git_backend import (
    GitEnvironmentBackend,
    _build_clone_url,
    _build_gitea_clone_url,
    _build_github_clone_url,
    _compute_content_hash,
    _is_gitea_source,
    _render_env_py,
    _render_pyproject_toml,
    _resolve_commit_sha,
)
from src.api.ports.environment_backend import (
    InlineEnvironmentBackend,
    ResolvedEnvironment,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_source(
    kind: api.EnvironmentSourceKind = api.EnvironmentSourceKind.github_repo,
    name: str = "my-env",
    version: str = "1.0.0",
    uri: str | None = "https://github.com/acme/my-env",
    git_ref: str | None = "main",
    external_id: str | None = None,
) -> api.EnvironmentSource:
    return api.EnvironmentSource(
        kind=kind,
        name=name,
        version=version,
        uri=uri,
        git_ref=git_ref,
        external_id=external_id,
    )


def _make_env(source: api.EnvironmentSource, inline_def: api.InlineEnvironmentDefinition | None = None) -> api.Environment:
    now = datetime.now(tz=timezone.utc)
    return api.Environment(
        id=uuid.uuid4(),
        source=source,
        inline_definition=inline_def,
        created_at=now,
        updated_at=now,
    )


def _make_inline_def(
    name: str = "my-env",
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
# Clone-URL construction
# ---------------------------------------------------------------------------

class TestBuildCloneUrl:
    """Tests for ``_build_clone_url``, ``_build_gitea_clone_url``, and
    ``_build_github_clone_url``."""

    def test_github_no_token(self, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GITEA_BASE_URL", raising=False)
        source = _make_source(uri="https://github.com/acme/my-env")
        url = _build_github_clone_url(source)
        assert url == "https://github.com/acme/my-env.git"

    def test_github_with_token(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_secret")
        monkeypatch.delenv("GITEA_BASE_URL", raising=False)
        source = _make_source(uri="https://github.com/acme/my-env")
        url = _build_github_clone_url(source)
        assert url == "https://oauth2:ghp_secret@github.com/acme/my-env.git"

    def test_github_uri_already_has_git_suffix(self, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GITEA_BASE_URL", raising=False)
        source = _make_source(uri="https://github.com/acme/my-env.git")
        url = _build_github_clone_url(source)
        assert url == "https://github.com/acme/my-env.git"

    def test_gitea_with_token(self, monkeypatch):
        monkeypatch.setenv("GITEA_BASE_URL", "https://gitea.example.com")
        monkeypatch.setenv("GITEA_ORG", "arena-org")
        monkeypatch.setenv("GITEA_TOKEN", "gitea_tok")
        source = _make_source(
            uri="https://gitea.example.com/arena-org/my-env",
            external_id="gitea:my-env",
        )
        url = _build_gitea_clone_url(source)
        assert url == "https://oauth2:gitea_tok@gitea.example.com/arena-org/my-env.git"

    def test_gitea_without_token(self, monkeypatch):
        monkeypatch.setenv("GITEA_BASE_URL", "https://gitea.example.com")
        monkeypatch.setenv("GITEA_ORG", "arena-org")
        monkeypatch.delenv("GITEA_TOKEN", raising=False)
        source = _make_source(
            uri="https://gitea.example.com/arena-org/my-env",
            external_id="gitea:my-env",
        )
        url = _build_gitea_clone_url(source)
        assert url == "https://gitea.example.com/arena-org/my-env.git"

    def test_gitea_missing_base_url_raises(self, monkeypatch):
        monkeypatch.delenv("GITEA_BASE_URL", raising=False)
        source = _make_source(external_id="gitea:my-env")
        with pytest.raises(ValueError, match="GITEA_BASE_URL"):
            _build_gitea_clone_url(source)

    def test_github_missing_uri_raises(self, monkeypatch):
        monkeypatch.delenv("GITEA_BASE_URL", raising=False)
        source = _make_source(uri=None)
        with pytest.raises(ValueError, match="no URI"):
            _build_github_clone_url(source)

    def test_is_gitea_source_by_external_id(self, monkeypatch):
        monkeypatch.delenv("GITEA_BASE_URL", raising=False)
        source = _make_source(external_id="gitea:my-repo")
        assert _is_gitea_source(source) is True

    def test_is_gitea_source_by_host(self, monkeypatch):
        monkeypatch.setenv("GITEA_BASE_URL", "https://gitea.example.com")
        source = _make_source(uri="https://gitea.example.com/org/repo")
        assert _is_gitea_source(source) is True

    def test_is_not_gitea_source_for_github(self, monkeypatch):
        monkeypatch.delenv("GITEA_BASE_URL", raising=False)
        source = _make_source(uri="https://github.com/acme/repo")
        assert _is_gitea_source(source) is False

    def test_build_clone_url_dispatches_to_gitea(self, monkeypatch):
        monkeypatch.setenv("GITEA_BASE_URL", "https://gitea.example.com")
        monkeypatch.setenv("GITEA_ORG", "org")
        monkeypatch.setenv("GITEA_TOKEN", "tok")
        source = _make_source(
            uri="https://gitea.example.com/org/my-env",
            external_id="gitea:my-env",
        )
        url = _build_clone_url(source)
        assert "gitea.example.com" in url
        assert "oauth2:tok@" in url

    def test_build_clone_url_dispatches_to_github(self, monkeypatch):
        monkeypatch.delenv("GITEA_BASE_URL", raising=False)
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        source = _make_source(uri="https://github.com/acme/my-env")
        url = _build_clone_url(source)
        assert url == "https://github.com/acme/my-env.git"


# ---------------------------------------------------------------------------
# subprocess helpers
# ---------------------------------------------------------------------------

class TestSubprocessHelpers:
    """Tests for ``_resolve_commit_sha`` and ``_compute_content_hash``."""

    def test_resolve_commit_sha(self, tmp_path):
        fake_sha = "abc123def456" * 3  # 36-char fake SHA
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=f"{fake_sha}\n", returncode=0)
            result = _resolve_commit_sha(tmp_path)
        assert result == fake_sha
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "rev-parse" in args
        assert "HEAD" in args

    def test_compute_content_hash_deterministic(self, tmp_path):
        ls_tree_output = (
            "100644 blob aabbcc  foo.py\n"
            "100644 blob ddeeff  bar.py\n"
        )
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=ls_tree_output, returncode=0)
            h1 = _compute_content_hash(tmp_path)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=ls_tree_output, returncode=0)
            h2 = _compute_content_hash(tmp_path)

        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex digest

    def test_compute_content_hash_differs_for_different_trees(self, tmp_path):
        output_a = "100644 blob aaaaaa  a.py\n"
        output_b = "100644 blob bbbbbb  b.py\n"
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=output_a, returncode=0)
            ha = _compute_content_hash(tmp_path)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=output_b, returncode=0)
            hb = _compute_content_hash(tmp_path)
        assert ha != hb

    def test_compute_content_hash_sorts_lines(self, tmp_path):
        """Order of ls-tree output must not affect the hash."""
        lines = [
            "100644 blob cccccc  z.py",
            "100644 blob aaaaaa  a.py",
            "100644 blob bbbbbb  m.py",
        ]
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="\n".join(lines) + "\n", returncode=0
            )
            h_unsorted = _compute_content_hash(tmp_path)

        # Expected: SHA-256 of the *sorted* lines
        sorted_lines = sorted(lines)
        expected = hashlib.sha256("\n".join(sorted_lines).encode()).hexdigest()
        assert h_unsorted == expected


# ---------------------------------------------------------------------------
# GitEnvironmentBackend.resolve — git path
# ---------------------------------------------------------------------------

def _make_subprocess_side_effect(commit_sha: str, ls_tree_output: str):
    """Return a side-effect function for ``subprocess.run`` calls in resolve."""
    call_count = {"n": 0}

    def side_effect(cmd, **kwargs):
        mock_result = MagicMock()
        mock_result.returncode = 0
        if "clone" in cmd:
            mock_result.stdout = ""
        elif "rev-parse" in cmd and "HEAD" in cmd:
            mock_result.stdout = commit_sha + "\n"
        elif "ls-tree" in cmd:
            mock_result.stdout = ls_tree_output
        else:
            mock_result.stdout = ""
        return mock_result

    return side_effect


class TestGitEnvironmentBackendResolve:
    def _patch_subprocess(self, commit_sha: str, ls_tree: str):
        """Context manager that patches subprocess.run."""
        return patch(
            "subprocess.run",
            side_effect=_make_subprocess_side_effect(commit_sha, ls_tree),
        )

    def _patch_clone(self):
        """Patch _shallow_clone so no filesystem writes happen."""
        return patch(
            "src.api.environments.git_backend._shallow_clone",
            return_value=None,
        )

    def _patch_load_inline(self, definition: api.InlineEnvironmentDefinition):
        return patch(
            "src.api.environments.git_backend._load_inline_definition_from_repo",
            return_value=definition,
        )

    def test_resolve_github_returns_resolved_environment(self, tmp_path, monkeypatch):
        monkeypatch.delenv("GITEA_BASE_URL", raising=False)
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)

        commit_sha = "deadbeef" * 5
        ls_tree = "100644 blob aabbcc  env.py\n"
        inline_def = _make_inline_def()

        source = _make_source(uri="https://github.com/acme/my-env", git_ref="v1.0")
        env = _make_env(source)

        backend = GitEnvironmentBackend(cache_dir=tmp_path)

        with self._patch_clone() as mock_clone, \
             self._patch_subprocess(commit_sha, ls_tree) as mock_sub, \
             self._patch_load_inline(inline_def):
            # The cache dir doesn't exist, so _shallow_clone will be called
            result = backend.resolve(env)

        assert isinstance(result, ResolvedEnvironment)
        assert result.commit_sha == commit_sha
        assert result.content_hash is not None
        assert len(result.content_hash) == 64
        assert result.local_path is not None
        assert result.definition is inline_def

    def test_resolve_uses_cache_when_directory_exists(self, tmp_path, monkeypatch):
        monkeypatch.delenv("GITEA_BASE_URL", raising=False)
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)

        commit_sha = "cafebabe" * 5
        ls_tree = "100644 blob 112233  data.json\n"
        inline_def = _make_inline_def()

        # source name is "my-env" (default), git_ref "main" → cache key "my-env__1.0.0__main"
        source = _make_source(uri="https://github.com/acme/my-env", git_ref="main")
        env = _make_env(source)

        backend = GitEnvironmentBackend(cache_dir=tmp_path)
        # Pre-create the cache dir so clone is skipped
        cache_key = "my-env__1.0.0__main"
        (tmp_path / cache_key).mkdir()

        with self._patch_clone() as mock_clone, \
             self._patch_subprocess(commit_sha, ls_tree), \
             self._patch_load_inline(inline_def):
            result = backend.resolve(env)

        mock_clone.assert_not_called()
        assert result.commit_sha == commit_sha

    def test_resolve_gitea_injects_credentials_in_clone_url(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GITEA_BASE_URL", "https://gitea.internal")
        monkeypatch.setenv("GITEA_ORG", "myorg")
        monkeypatch.setenv("GITEA_TOKEN", "secret_tok")

        commit_sha = "feedface" * 5
        ls_tree = "100644 blob 445566  env.py\n"
        inline_def = _make_inline_def()

        source = _make_source(
            uri="https://gitea.internal/myorg/my-env",
            git_ref="release-1",
            external_id="gitea:my-env",
        )
        env = _make_env(source)

        captured_urls: list[str] = []

        def fake_clone(url, ref, dest):
            captured_urls.append(url)
            # Create the dest dir so the rename trick succeeds
            dest.mkdir(parents=True, exist_ok=True)

        backend = GitEnvironmentBackend(cache_dir=tmp_path)

        with patch(
            "src.api.environments.git_backend._shallow_clone",
            side_effect=fake_clone,
        ), self._patch_subprocess(commit_sha, ls_tree), \
           self._patch_load_inline(inline_def):
            backend.resolve(env)

        assert len(captured_urls) == 1
        url = captured_urls[0]
        assert "oauth2:secret_tok@" in url
        assert "gitea.internal" in url
        assert "my-env.git" in url

    def test_resolve_prime_environment_hub_kind(self, tmp_path, monkeypatch):
        monkeypatch.delenv("GITEA_BASE_URL", raising=False)
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)

        commit_sha = "baadf00d" * 5
        ls_tree = "100644 blob aabbcc  env.py\n"
        inline_def = _make_inline_def()

        source = _make_source(
            kind=api.EnvironmentSourceKind.prime_environment_hub,
            uri="https://github.com/prime/env-pkg",
            git_ref="v2",
        )
        env = _make_env(source)

        backend = GitEnvironmentBackend(cache_dir=tmp_path)

        with self._patch_clone(), \
             self._patch_subprocess(commit_sha, ls_tree), \
             self._patch_load_inline(inline_def):
            result = backend.resolve(env)

        assert result.commit_sha == commit_sha


# ---------------------------------------------------------------------------
# Inline delegation
# ---------------------------------------------------------------------------

class TestInlineDelegation:
    def test_inline_kind_delegates_to_inline_backend(self):
        inline_def = _make_inline_def()
        source = _make_source(kind=api.EnvironmentSourceKind.inline, git_ref=None)
        env = _make_env(source, inline_def=inline_def)

        backend = GitEnvironmentBackend()
        with patch.object(
            InlineEnvironmentBackend,
            "resolve",
            return_value=ResolvedEnvironment(definition=inline_def),
        ) as mock_resolve:
            result = backend.resolve(env)

        mock_resolve.assert_called_once_with(env)
        assert result.definition is inline_def
        assert result.commit_sha is None

    def test_inline_kind_no_definition_raises(self):
        """ApiError from InlineEnvironmentBackend propagates unchanged."""
        from src.api.service import ApiError

        source = _make_source(kind=api.EnvironmentSourceKind.inline, git_ref=None)
        env = _make_env(source, inline_def=None)

        backend = GitEnvironmentBackend()
        with pytest.raises(ApiError):
            backend.resolve(env)


# ---------------------------------------------------------------------------
# Unsupported kind
# ---------------------------------------------------------------------------

class TestUnsupportedKind:
    def test_huggingface_hub_raises_not_implemented(self):
        source = _make_source(kind=api.EnvironmentSourceKind.huggingface_hub)
        env = _make_env(source)

        backend = GitEnvironmentBackend()
        with pytest.raises(NotImplementedError, match="huggingface_hub"):
            backend.resolve(env)


# ---------------------------------------------------------------------------
# snapshot_inline
# ---------------------------------------------------------------------------

class TestSnapshotInline:
    """Tests for ``GitEnvironmentBackend.snapshot_inline``."""

    def _make_httpx_response(self, status_code: int = 200, json_data: dict | None = None):
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.json.return_value = json_data or {}
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    def _setup_client_mock(self, mock_client_cls):
        """Configure the mock httpx.Client context manager."""
        client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        return client

    def test_snapshot_inline_creates_repo_and_commits_files(self, monkeypatch):
        monkeypatch.setenv("GITEA_BASE_URL", "https://gitea.example.com")
        monkeypatch.setenv("GITEA_ORG", "arena-org")
        monkeypatch.setenv("GITEA_TOKEN", "secret_tok")

        definition = _make_inline_def(name="my-env", version="1.0.0")
        backend = GitEnvironmentBackend()

        repo_url = "https://gitea.example.com/arena-org/arena-env-my-env-1-0-0.git"
        commit_sha = "abc123abc123abc123abc123abc123abc123abc1"

        # Mock responses
        create_repo_resp = self._make_httpx_response(
            200, {"clone_url": repo_url, "name": "arena-env-my-env-1-0-0"}
        )
        # GET for existing file check (file not found → 404)
        get_file_resp = self._make_httpx_response(404, {})
        # POST to create file — env.py
        post_env_resp = self._make_httpx_response(
            201,
            {
                "commit": {"sha": commit_sha},
                "content": {"name": "env.py"},
            },
        )
        # POST to create file — pyproject.toml
        post_pyproject_resp = self._make_httpx_response(
            201,
            {
                "commit": {"sha": "anothersha"},
                "content": {"name": "pyproject.toml"},
            },
        )

        with patch("httpx.Client") as mock_client_cls:
            client = self._setup_client_mock(mock_client_cls)
            client.post.side_effect = [create_repo_resp, post_env_resp, post_pyproject_resp]
            client.get.return_value = get_file_resp

            result_url, result_sha = backend.snapshot_inline(definition)

        assert result_url == repo_url
        # snapshot_inline must return the *last* write's commit (pyproject.toml) —
        # the tree HEAD containing BOTH files — not the earlier env.py commit.
        assert result_sha == "anothersha"

    def test_snapshot_inline_calls_create_repo_with_correct_payload(self, monkeypatch):
        monkeypatch.setenv("GITEA_BASE_URL", "https://gitea.example.com")
        monkeypatch.setenv("GITEA_ORG", "my-org")
        monkeypatch.setenv("GITEA_TOKEN", "tok")

        definition = _make_inline_def(name="special-env", version="2.5.0")
        backend = GitEnvironmentBackend()

        create_resp = self._make_httpx_response(
            200,
            {"clone_url": "https://gitea.example.com/my-org/custom-repo.git"},
        )
        get_file_resp = self._make_httpx_response(404)
        file_resp = self._make_httpx_response(
            201, {"commit": {"sha": "sha111"}, "content": {}}
        )

        with patch("httpx.Client") as mock_client_cls:
            client = self._setup_client_mock(mock_client_cls)
            client.post.side_effect = [create_resp, file_resp, file_resp]
            client.get.return_value = get_file_resp

            backend.snapshot_inline(
                definition,
                repo_name="custom-repo",
                gitea_base_url="https://gitea.example.com",
                gitea_org="my-org",
                gitea_token="tok",
            )

        # First POST should be create-repo
        first_post_call = client.post.call_args_list[0]
        assert "/api/v1/orgs/my-org/repos" in first_post_call[0][0]
        payload = first_post_call[1]["json"]
        assert payload["name"] == "custom-repo"
        assert payload["private"] is True

    def test_snapshot_inline_missing_base_url_raises(self, monkeypatch):
        monkeypatch.delenv("GITEA_BASE_URL", raising=False)
        monkeypatch.setenv("GITEA_ORG", "org")
        monkeypatch.setenv("GITEA_TOKEN", "tok")

        definition = _make_inline_def()
        backend = GitEnvironmentBackend()

        with pytest.raises(ValueError, match="GITEA_BASE_URL"):
            backend.snapshot_inline(definition)

    def test_snapshot_inline_missing_org_raises(self, monkeypatch):
        monkeypatch.setenv("GITEA_BASE_URL", "https://gitea.example.com")
        monkeypatch.delenv("GITEA_ORG", raising=False)
        monkeypatch.setenv("GITEA_TOKEN", "tok")

        definition = _make_inline_def()
        backend = GitEnvironmentBackend()

        with pytest.raises(ValueError, match="GITEA_ORG"):
            backend.snapshot_inline(definition)

    def test_snapshot_inline_missing_token_raises(self, monkeypatch):
        monkeypatch.setenv("GITEA_BASE_URL", "https://gitea.example.com")
        monkeypatch.setenv("GITEA_ORG", "org")
        monkeypatch.delenv("GITEA_TOKEN", raising=False)

        definition = _make_inline_def()
        backend = GitEnvironmentBackend()

        with pytest.raises(ValueError, match="GITEA_TOKEN"):
            backend.snapshot_inline(definition)

    def test_snapshot_inline_default_repo_name_derived_from_definition(self, monkeypatch):
        monkeypatch.setenv("GITEA_BASE_URL", "https://gitea.example.com")
        monkeypatch.setenv("GITEA_ORG", "org")
        monkeypatch.setenv("GITEA_TOKEN", "tok")

        definition = _make_inline_def(name="my-env", version="3.1.4")
        backend = GitEnvironmentBackend()

        create_resp = self._make_httpx_response(
            200,
            {"clone_url": "https://gitea.example.com/org/arena-env-my-env-3-1-4.git"},
        )
        get_file_resp = self._make_httpx_response(404)
        file_resp = self._make_httpx_response(
            201, {"commit": {"sha": "sha999"}, "content": {}}
        )

        with patch("httpx.Client") as mock_client_cls:
            client = self._setup_client_mock(mock_client_cls)
            client.post.side_effect = [create_resp, file_resp, file_resp]
            client.get.return_value = get_file_resp

            backend.snapshot_inline(definition)

        first_post = client.post.call_args_list[0]
        payload = first_post[1]["json"]
        assert payload["name"] == "arena-env-my-env-3-1-4"


# ---------------------------------------------------------------------------
# Scaffold renderers
# ---------------------------------------------------------------------------

class TestScaffoldRenderers:
    def test_render_env_py_contains_load_environment(self):
        definition = _make_inline_def()
        src = _render_env_py(definition)
        assert "def load_environment" in src
        assert definition.name in src

    def test_render_pyproject_toml_contains_project_name(self):
        definition = _make_inline_def(name="my-env", version="1.0.0")
        toml = _render_pyproject_toml(definition)
        assert "my-env" in toml
        assert "1.0.0" in toml
        assert "[project]" in toml

    def test_render_pyproject_toml_contains_entry_point(self):
        definition = _make_inline_def()
        toml = _render_pyproject_toml(definition)
        assert "open_arena.environments" in toml


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------

class TestRegistryWiring:
    def test_registry_git_backend_selector(self, tmp_path):
        from src.api.registry import _build_env_backend
        from src.api.settings import ArenaSettings
        from src.api.environments.git_backend import GitEnvironmentBackend

        settings = ArenaSettings(
            env_backend="git",
            db_path=tmp_path / "test.db",
        )
        backend = _build_env_backend(settings)
        assert isinstance(backend, GitEnvironmentBackend)

    def test_registry_inline_still_works(self, tmp_path):
        from src.api.registry import _build_env_backend
        from src.api.settings import ArenaSettings

        settings = ArenaSettings(
            env_backend="inline",
            db_path=tmp_path / "test.db",
        )
        backend = _build_env_backend(settings)
        assert isinstance(backend, InlineEnvironmentBackend)

    def test_registry_unknown_backend_raises(self, tmp_path):
        from src.api.registry import _build_env_backend
        from src.api.settings import ArenaSettings

        settings = ArenaSettings(
            env_backend="unknown_xyz",
            db_path=tmp_path / "test.db",
        )
        with pytest.raises(ValueError, match="OPEN_ARENA_ENV_BACKEND"):
            _build_env_backend(settings)
