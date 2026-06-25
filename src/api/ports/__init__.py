# License Apache 2.0: (c) 2026 Athena-Reply
"""``src.api.ports`` — Abstract port interfaces for Open Arena.

Each sub-module defines one port (ABC) and its default adapter:

+---------------------------+----------------------------------+----------------------------------+
| Module                    | Port ABC                         | Default adapter                  |
+===========================+==================================+==================================+
| :mod:`.store`             | :class:`~.store.Store`           | :class:`~src.api.stores.sqlite.SQLiteStore` |
+---------------------------+----------------------------------+----------------------------------+
| :mod:`.environment_backend` | :class:`~.environment_backend.EnvironmentBackend` | :class:`~.environment_backend.InlineEnvironmentBackend` |
+---------------------------+----------------------------------+----------------------------------+
| :mod:`.dataset_resolver`  | :class:`~.dataset_resolver.DatasetResolver` | :class:`~.dataset_resolver.LegacyDatasetResolver` |
+---------------------------+----------------------------------+----------------------------------+
| :mod:`.results_sink`      | :class:`~.results_sink.ResultsSink` | :class:`~.results_sink.StoreResultsSink` |
+---------------------------+----------------------------------+----------------------------------+
| :mod:`.sandbox_provider`  | :class:`~.sandbox_provider.SandboxProvider` | :class:`~.sandbox_provider.LocalSandboxProvider` |
+---------------------------+----------------------------------+----------------------------------+
| :mod:`.auth_provider`     | :class:`~.auth_provider.AuthProvider` | :class:`~.auth_provider.StaticBearerAuthProvider` |
+---------------------------+----------------------------------+----------------------------------+

See ``src/api/ports/README.md`` for a walkthrough of the pattern and
instructions on registering a new adapter.
"""

from src.api.ports.auth_provider import AuthProvider, Principal, StaticBearerAuthProvider
from src.api.ports.dataset_resolver import DatasetResolver, LegacyDatasetResolver
from src.api.ports.environment_backend import (
    EnvironmentBackend,
    InlineEnvironmentBackend,
    ResolvedEnvironment,
)
from src.api.ports.results_sink import ResultsSink, StoreResultsSink
from src.api.ports.sandbox_provider import LocalSandboxProvider, SandboxProvider
from src.api.ports.store import Store

__all__ = [
    # store
    "Store",
    # environment_backend
    "EnvironmentBackend",
    "ResolvedEnvironment",
    "InlineEnvironmentBackend",
    # dataset_resolver
    "DatasetResolver",
    "LegacyDatasetResolver",
    # results_sink
    "ResultsSink",
    "StoreResultsSink",
    # sandbox_provider
    "SandboxProvider",
    "LocalSandboxProvider",
    # auth_provider
    "AuthProvider",
    "Principal",
    "StaticBearerAuthProvider",
]
