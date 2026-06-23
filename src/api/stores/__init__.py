# License Apache 2.0: (c) 2026 Athena-Reply
"""``src.api.stores`` — Concrete :class:`~src.api.ports.store.Store` adapters.

Currently ships one adapter:

* :class:`~src.api.stores.sqlite.SQLiteStore` — the default SQLite-backed
  implementation re-exported here for backward compatibility.
"""

from src.api.stores.sqlite import SQLiteStore

__all__ = ["SQLiteStore"]
