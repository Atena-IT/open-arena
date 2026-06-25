# License Apache 2.0: (c) 2026 Athena-Reply
"""``tests/conftest.py`` -- Session-wide test fixtures.

Ensures that no test reads from or writes to the real ``.open-arena/api.db``
by pointing ``OPEN_ARENA_DB_PATH`` at a temporary directory for the lifetime
of the test session.

Why this matters
----------------
``ArenaSettings`` reads ``OPEN_ARENA_DB_PATH`` at construction time.  Without
isolation, a test that runs while the developer also has the API server running
(or after a previous test polluted the DB) can see stale data and fail.

The fixture is *autouse* + *session*-scoped so every test in this suite gets
the isolated path without needing to opt-in individually.  Tests that need a
*per-test* fresh DB should use a ``tmp_path``-scoped fixture in addition to
(or instead of) this one.
"""
from __future__ import annotations

import os
import tempfile

import pytest


@pytest.fixture(autouse=True, scope="session")
def _isolate_db_path(tmp_path_factory):
    """Redirect OPEN_ARENA_DB_PATH to a temp directory for the test session.

    This prevents tests from touching ``.open-arena/api.db`` in the project
    directory.  Each test that needs a fresh SQLite file should request
    ``tmp_path`` and pass ``ArenaSettings(db_path=tmp_path / "test.db")``
    explicitly.
    """
    tmp_dir = tmp_path_factory.mktemp("arena_db")
    db_path = str(tmp_dir / "test_session.db")
    original = os.environ.get("OPEN_ARENA_DB_PATH")
    os.environ["OPEN_ARENA_DB_PATH"] = db_path
    yield db_path
    # Restore original value (or remove) after the session
    if original is None:
        os.environ.pop("OPEN_ARENA_DB_PATH", None)
    else:
        os.environ["OPEN_ARENA_DB_PATH"] = original