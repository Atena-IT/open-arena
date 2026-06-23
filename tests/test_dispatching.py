# License Apache 2.0: (c) 2026 Athena-Reply
"""Tests for the DispatchingEnvironmentBackend (P2-1, issue #63).

Covers:
* inline kind -> delegates to InlineEnvironmentBackend
* github_repo kind -> delegates to GitEnvironmentBackend
* prime_environment_hub kind -> delegates to PrimeEnvHubBackend
* huggingface_hub with inline_definition -> delegates to InlineEnvironmentBackend
* huggingface_hub without inline_definition -> NotImplementedError
* unknown kind -> NotImplementedError
* DispatchingEnvironmentBackend is returned by registry with default settings
"""
from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from open_arena_core import models as api
from src.api.environments.dispatching import DispatchingEnvironmentBackend
from src.api.ports.environment_backend import ResolvedEnvironment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_env(kind: api.EnvironmentSourceKind, inline: bool = True) -> api.Environment:
    """Build a minimal Environment with the given source.kind."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    source = api.EnvironmentSource(kind=kind, name="test-env", version="1.0.0")
    inline_def = None
    if inline:
        inline_def = api.InlineEnvironmentDefinition(
            name="test-env",
            version="1.0.0",
            dataset=api.DatasetBinding(provider="local"),
            verifier=api.VerifierSuiteBinding(
                root=api.VerifierSuiteInline(
                    binding_type="inline",
                    name="test-verifier",
                    metrics=[
                        api.MetricDefinition(
                            name="acc", metric_kind="exact_match", weight=1.0
                        )
                    ],
                )
            ),
            runtime=api.EnvironmentRuntimePolicy(),
        )
    return api.Environment(
        id=uuid4(),
        source=source,
        inline_definition=inline_def,
        created_at=now,
        updated_at=now,
    )


def _make_backend(
    inline_backend=None,
    git_backend=None,
    prime_hub_backend=None,
) -> DispatchingEnvironmentBackend:
    return DispatchingEnvironmentBackend(
        inline_backend=inline_backend,
        git_backend=git_backend,
        prime_hub_backend=prime_hub_backend,
    )


def _mock_resolved() -> ResolvedEnvironment:
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    inline_def = api.InlineEnvironmentDefinition(
        name="resolved",
        version="1.0.0",
        dataset=api.DatasetBinding(provider="local"),
        verifier=api.VerifierSuiteBinding(
            root=api.VerifierSuiteInline(
                binding_type="inline",
                name="v",
                metrics=[api.MetricDefinition(name="m", metric_kind="exact_match", weight=1.0)],
            )
        ),
        runtime=api.EnvironmentRuntimePolicy(),
    )
    return ResolvedEnvironment(definition=inline_def)


# ---------------------------------------------------------------------------
# Tests: routing by kind
# ---------------------------------------------------------------------------

def test_dispatches_inline_to_inline_backend():
    env = _make_env(api.EnvironmentSourceKind.inline)
    mock_inline = MagicMock()
    mock_inline.resolve.return_value = _mock_resolved()
    backend = _make_backend(inline_backend=mock_inline)
    result = backend.resolve(env)
    mock_inline.resolve.assert_called_once_with(env)
    assert isinstance(result, ResolvedEnvironment)


def test_dispatches_github_repo_to_git_backend():
    env = _make_env(api.EnvironmentSourceKind.github_repo)
    mock_git = MagicMock()
    mock_git.resolve.return_value = _mock_resolved()
    backend = _make_backend(git_backend=mock_git)
    result = backend.resolve(env)
    mock_git.resolve.assert_called_once_with(env)
    assert isinstance(result, ResolvedEnvironment)


def test_dispatches_prime_hub_to_prime_hub_backend():
    env = _make_env(api.EnvironmentSourceKind.prime_environment_hub)
    mock_prime = MagicMock()
    mock_prime.resolve.return_value = _mock_resolved()
    backend = _make_backend(prime_hub_backend=mock_prime)
    result = backend.resolve(env)
    mock_prime.resolve.assert_called_once_with(env)
    assert isinstance(result, ResolvedEnvironment)


def test_dispatches_huggingface_with_inline_to_inline_backend():
    """HF hub kind with inline_definition falls through to inline backend."""
    env = _make_env(api.EnvironmentSourceKind.huggingface_hub, inline=True)
    mock_inline = MagicMock()
    mock_inline.resolve.return_value = _mock_resolved()
    backend = _make_backend(inline_backend=mock_inline)
    result = backend.resolve(env)
    mock_inline.resolve.assert_called_once_with(env)
    assert isinstance(result, ResolvedEnvironment)


def test_dispatches_huggingface_without_inline_raises():
    """HF hub kind without inline_definition raises NotImplementedError."""
    env = _make_env(api.EnvironmentSourceKind.huggingface_hub, inline=False)
    backend = _make_backend()
    with pytest.raises(NotImplementedError, match="huggingface_hub"):
        backend.resolve(env)


def test_unknown_kind_raises():
    """Unrecognised kind raises NotImplementedError."""
    from datetime import datetime, timezone

    # Build a real Environment but patch source.kind to an unrecognised value
    now = datetime.now(timezone.utc)
    source = api.EnvironmentSource(kind=api.EnvironmentSourceKind.inline, name="x", version="1.0.0")
    env = api.Environment(id=uuid4(), source=source, inline_definition=None, created_at=now, updated_at=now)

    # Monkeypatch the kind attribute to simulate an unrecognised value
    object.__setattr__(env.source, "kind", "totally_unknown_kind")

    backend = _make_backend()
    with pytest.raises(NotImplementedError):
        backend.resolve(env)


# ---------------------------------------------------------------------------
# Tests: lazy sub-backend instantiation
# ---------------------------------------------------------------------------

def test_git_backend_not_instantiated_until_needed():
    """Git backend is only imported/instantiated when a github_repo env is resolved."""
    backend = DispatchingEnvironmentBackend()
    assert backend._git is None  # not yet instantiated


def test_prime_hub_backend_not_instantiated_until_needed():
    backend = DispatchingEnvironmentBackend()
    assert backend._prime_hub is None


# ---------------------------------------------------------------------------
# Tests: registry integration
# ---------------------------------------------------------------------------

def test_dispatch_is_default_env_backend(monkeypatch, tmp_path):
    """With OPEN_ARENA_ENV_BACKEND unset, registry returns DispatchingEnvironmentBackend."""
    monkeypatch.delenv("OPEN_ARENA_ENV_BACKEND", raising=False)
    from src.api.registry import build_adapters
    from src.api.settings import ArenaSettings

    settings = ArenaSettings(db_path=tmp_path / "test.db")
    adapters = build_adapters(settings)
    assert isinstance(adapters.env_backend, DispatchingEnvironmentBackend)


def test_explicit_dispatch_setting(tmp_path):
    """OPEN_ARENA_ENV_BACKEND=dispatch returns DispatchingEnvironmentBackend."""
    from src.api.registry import build_adapters
    from src.api.settings import ArenaSettings

    settings = ArenaSettings(env_backend="dispatch", db_path=tmp_path / "test.db")
    adapters = build_adapters(settings)
    assert isinstance(adapters.env_backend, DispatchingEnvironmentBackend)