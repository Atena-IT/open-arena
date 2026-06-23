# License Apache 2.0: (c) 2026 Athena-Reply
"""``src.api.registry`` — Adapter factory.

:func:`build_adapters` reads :class:`~src.api.settings.ArenaSettings` and
returns a fully-wired :class:`AdapterSet` ready to be injected into
:class:`~src.api.service.ArenaAPIService`.

Adding a new adapter
--------------------
1. Implement the port ABC in ``src/api/ports/<port>.py``.
2. Add an ``elif settings.<port> == "<key>":`` branch in the matching
   builder function below.
3. Update ``OPEN_ARENA_<PORT>`` defaults in ``src/api/settings.py`` if
   appropriate.
4. Document the new adapter in ``src/api/ports/README.md``.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.api.ports.auth_provider import AuthProvider, StaticBearerAuthProvider
from src.api.ports.dataset_resolver import DatasetResolver, LegacyDatasetResolver
from src.api.ports.environment_backend import EnvironmentBackend, InlineEnvironmentBackend
from src.api.ports.results_sink import ResultsSink, StoreResultsSink
from src.api.ports.sandbox_provider import LocalSandboxProvider, SandboxProvider
from src.api.ports.store import Store
from src.api.settings import ArenaSettings, get_settings


@dataclass
class AdapterSet:
    """A fully-resolved set of port adapters.

    All fields are concrete implementations of their respective port ABCs.
    Pass an :class:`AdapterSet` to
    :meth:`~src.api.service.ArenaAPIService.__init__` to override defaults.
    """

    store: Store
    auth: AuthProvider
    env_backend: EnvironmentBackend
    dataset_resolver: DatasetResolver
    results_sink: ResultsSink
    sandbox: SandboxProvider


def _build_store(settings: ArenaSettings) -> Store:
    if settings.store == "sqlite":
        from src.api.stores.sqlite import SQLiteStore

        return SQLiteStore(path=settings.db_path)
    raise ValueError(
        f"Unknown OPEN_ARENA_STORE={settings.store!r}.  "
        "Supported values: 'sqlite'."
    )


def _build_auth(settings: ArenaSettings) -> AuthProvider:
    if settings.auth == "static":
        return StaticBearerAuthProvider()
    # WS7: Keycloak — add: elif settings.auth == "keycloak": return KeycloakAuthProvider()
    raise ValueError(
        f"Unknown OPEN_ARENA_AUTH={settings.auth!r}.  "
        "Supported values: 'static'."
    )


def _build_env_backend(settings: ArenaSettings) -> EnvironmentBackend:
    if settings.env_backend == "inline":
        return InlineEnvironmentBackend()
    # WS2 (Gitea/GitHub): elif settings.env_backend == "git": return GitEnvironmentBackend()
    raise ValueError(
        f"Unknown OPEN_ARENA_ENV_BACKEND={settings.env_backend!r}.  "
        "Supported values: 'inline'."
    )


def _build_dataset_resolver(settings: ArenaSettings) -> DatasetResolver:
    if settings.dataset_resolver == "legacy":
        return LegacyDatasetResolver()
    # WS4: unity_catalog — elif settings.dataset_resolver == "unity_catalog": ...
    raise ValueError(
        f"Unknown OPEN_ARENA_DATASET_RESOLVER={settings.dataset_resolver!r}.  "
        "Supported values: 'legacy'."
    )


def _build_results_sink(store: Store, settings: ArenaSettings) -> ResultsSink:
    if settings.results_sink == "store":
        return StoreResultsSink(store=store)
    # WS5: MLflow — elif settings.results_sink == "mlflow": return MlflowResultsSink(...)
    raise ValueError(
        f"Unknown OPEN_ARENA_RESULTS_SINK={settings.results_sink!r}.  "
        "Supported values: 'store'."
    )


def _build_sandbox(settings: ArenaSettings) -> SandboxProvider:
    if settings.sandbox == "local":
        return LocalSandboxProvider()
    elif settings.sandbox == "e2b":
        from src.api.sandboxes.e2b_provider import E2BSandboxProvider

        return E2BSandboxProvider()
    raise ValueError(
        f"Unknown OPEN_ARENA_SANDBOX={settings.sandbox!r}.  "
        "Supported values: 'local', 'e2b'."
    )


def build_adapters(settings: ArenaSettings | None = None) -> AdapterSet:
    """Build a complete :class:`AdapterSet` from *settings*.

    Args:
        settings: Settings snapshot to use.  When ``None``, calls
            :func:`~src.api.settings.get_settings` to read the current
            environment.

    Returns:
        An :class:`AdapterSet` with all ports wired to their concrete
        adapters.
    """
    if settings is None:
        settings = get_settings()

    store = _build_store(settings)
    return AdapterSet(
        store=store,
        auth=_build_auth(settings),
        env_backend=_build_env_backend(settings),
        dataset_resolver=_build_dataset_resolver(settings),
        results_sink=_build_results_sink(store, settings),
        sandbox=_build_sandbox(settings),
    )
