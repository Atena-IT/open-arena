# License Apache 2.0: (c) 2026 Athena-Reply
"""``src.api.settings`` — Environment-driven adapter selection.

Each port has a corresponding ``OPEN_ARENA_<PORT>`` environment variable
whose value selects which adapter the registry will instantiate.
All defaults reproduce the original behavior so no configuration change
is required for existing deployments.

+---------------------------------+------------------+-----------------------------+
| Environment variable            | Default          | Effect                      |
+=================================+==================+=============================+
| ``OPEN_ARENA_STORE``            | ``sqlite``       | Persistence backend          |
+---------------------------------+------------------+-----------------------------+
| ``OPEN_ARENA_AUTH``             | ``static``       | Bearer token validation      |
+---------------------------------+------------------+-----------------------------+
| ``OPEN_ARENA_ENV_BACKEND``      | ``inline``       | Environment resolution       |
+---------------------------------+------------------+-----------------------------+
| ``OPEN_ARENA_DATASET_RESOLVER`` | ``legacy``       | Dataset binding translation  |
+---------------------------------+------------------+-----------------------------+
| ``OPEN_ARENA_RESULTS_SINK``     | ``store``        | Result persistence           |
+---------------------------------+------------------+-----------------------------+
| ``OPEN_ARENA_SANDBOX``          | ``local``        | Sweep execution              |
+---------------------------------+------------------+-----------------------------+
| ``OPEN_ARENA_DB_PATH``          | ``.open-arena/api.db`` | SQLite file location   |
+---------------------------------+------------------+-----------------------------+
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ArenaSettings:
    """Immutable snapshot of adapter-selection settings.

    All values are read from environment variables at construction time;
    passing explicit values is supported for testing.
    """

    store: str = field(default_factory=lambda: os.getenv("OPEN_ARENA_STORE", "sqlite"))
    auth: str = field(default_factory=lambda: os.getenv("OPEN_ARENA_AUTH", "static"))
    env_backend: str = field(
        default_factory=lambda: os.getenv("OPEN_ARENA_ENV_BACKEND", "inline")
    )
    dataset_resolver: str = field(
        default_factory=lambda: os.getenv("OPEN_ARENA_DATASET_RESOLVER", "legacy")
    )
    results_sink: str = field(
        default_factory=lambda: os.getenv("OPEN_ARENA_RESULTS_SINK", "store")
    )
    sandbox: str = field(default_factory=lambda: os.getenv("OPEN_ARENA_SANDBOX", "local"))
    db_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("OPEN_ARENA_DB_PATH", ".open-arena/api.db")
        )
    )


def get_settings() -> ArenaSettings:
    """Return a fresh :class:`ArenaSettings` from the current environment."""
    return ArenaSettings()
