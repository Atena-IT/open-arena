# License Apache 2.0: (c) 2026 Athena-Reply
"""SQLite-backed :class:`~src.api.ports.store.Store` adapter.

This is the default persistence backend for Open Arena.  All data is
stored in a single SQLite database file (default:
``.open-arena/api.db``).

The class is the canonical implementation of
:class:`~src.api.ports.store.Store` and it exposes no SQLite-specific
public API beyond what the port ABC declares.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from pydantic import BaseModel

from open_arena_core import models as api
from src.api.ports.store import Store

STATE_DIR = Path(".open-arena")
DB_PATH = STATE_DIR / "api.db"


def _now() -> datetime:
    return datetime.now(UTC)


def _iso(value: datetime) -> str:
    return value.isoformat()


# Tables that may be referenced by name in raw SQL. The ``table`` argument is
# always supplied internally (a fixed set of entity names from the Store port,
# never request data), but it is validated against this explicit allowlist
# before being interpolated into a statement so the f-string SELECT/DELETE
# helpers below can never become a SQL-injection vector if a caller regresses.
_ALLOWED_TABLES = frozenset(
    {"verifiers", "environments", "leaderboards", "runs", "run_results", "subject_cache"}
)


def _check_table(table: str) -> str:
    """Validate *table* against :data:`_ALLOWED_TABLES`; return it unchanged.

    Raises:
        ValueError: if *table* is not a known table name.
    """
    if table not in _ALLOWED_TABLES:
        raise ValueError(
            f"Unknown table {table!r}; expected one of {sorted(_ALLOWED_TABLES)}."
        )
    return table


class SQLiteStore(Store):
    """SQLite-backed implementation of the :class:`~src.api.ports.store.Store` port.

    Thread-safe: all writes are serialised through an :class:`threading.RLock`.
    Reads acquire the same lock so that a read never sees a partial write.

    The database is initialised (tables created) on first instantiation.

    Args:
        path: Path to the SQLite database file.  The parent directory is
            created automatically.  Defaults to ``.open-arena/api.db``.
    """

    def __init__(self, path: Path = DB_PATH) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()

    # -- internal helpers --------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS verifiers (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    doc TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS environments (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    source_kind TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    doc TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS leaderboards (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    visibility TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    doc TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    mode TEXT NOT NULL,
                    status TEXT NOT NULL,
                    cache_status TEXT NOT NULL,
                    leaderboard_id TEXT,
                    idempotency_key TEXT UNIQUE,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    doc TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS run_results (
                    run_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    doc TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS subject_cache (
                    fingerprint TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    doc TEXT NOT NULL
                );
                """
            )

    def _save_doc(
        self,
        table: str,
        doc_id: str,
        model: BaseModel,
        *,
        name: str | None = None,
        source_kind: str | None = None,
        visibility: str | None = None,
        mode: str | None = None,
        status: str | None = None,
        cache_status: str | None = None,
        leaderboard_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> None:
        payload = json.dumps(model.model_dump(mode="json", exclude_none=True))
        now = _iso(_now())
        with self._lock, self._connect() as conn:
            if table == "verifiers":
                conn.execute(
                    "REPLACE INTO verifiers (id, name, created_at, updated_at, doc) "
                    "VALUES (?, ?, COALESCE((SELECT created_at FROM verifiers WHERE id = ?), ?), ?, ?)",
                    (doc_id, name or "", doc_id, now, now, payload),
                )
            elif table == "environments":
                conn.execute(
                    "REPLACE INTO environments (id, name, source_kind, created_at, updated_at, doc) "
                    "VALUES (?, ?, ?, COALESCE((SELECT created_at FROM environments WHERE id = ?), ?), ?, ?)",
                    (doc_id, name or "", source_kind or "", doc_id, now, now, payload),
                )
            elif table == "leaderboards":
                conn.execute(
                    "REPLACE INTO leaderboards (id, name, visibility, created_at, updated_at, doc) "
                    "VALUES (?, ?, ?, COALESCE((SELECT created_at FROM leaderboards WHERE id = ?), ?), ?, ?)",
                    (doc_id, name or "", visibility or "", doc_id, now, now, payload),
                )
            elif table == "runs":
                conn.execute(
                    "REPLACE INTO runs (id, mode, status, cache_status, leaderboard_id, "
                    "idempotency_key, created_at, updated_at, doc) "
                    "VALUES (?, ?, ?, ?, ?, ?, COALESCE((SELECT created_at FROM runs WHERE id = ?), ?), ?, ?)",
                    (
                        doc_id,
                        mode or "",
                        status or "",
                        cache_status or "",
                        leaderboard_id,
                        idempotency_key,
                        doc_id,
                        now,
                        now,
                        payload,
                    ),
                )
            else:
                raise ValueError(table)

    def _get_doc(
        self,
        table: str,
        doc_id: str,
        model_cls: type[BaseModel],
    ) -> BaseModel | None:
        _check_table(table)
        with self._lock, self._connect() as conn:
            row = conn.execute(
                f"SELECT doc FROM {table} WHERE id = ?",  # noqa: S608 — table validated above
                (doc_id,),
            ).fetchone()
        if not row:
            return None
        return model_cls.model_validate_json(row["doc"])

    def _list_docs(
        self,
        table: str,
        model_cls: type[BaseModel],
    ) -> list[BaseModel]:
        _check_table(table)
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                f"SELECT doc FROM {table} ORDER BY created_at DESC",  # noqa: S608 — table validated above
            ).fetchall()
        return [model_cls.model_validate_json(row["doc"]) for row in rows]

    # -- Store port implementation ------------------------------------------------

    def save_verifier(self, verifier: api.VerifierSuite) -> None:  # noqa: D102
        self._save_doc("verifiers", str(verifier.id), verifier, name=verifier.name)

    def get_verifier(self, verifier_id: UUID) -> api.VerifierSuite | None:  # noqa: D102
        return self._get_doc("verifiers", str(verifier_id), api.VerifierSuite)  # type: ignore[return-value]

    def list_verifiers(self) -> list[api.VerifierSuite]:  # noqa: D102
        return self._list_docs("verifiers", api.VerifierSuite)  # type: ignore[return-value]

    def save_environment(self, environment: api.Environment) -> None:  # noqa: D102
        self._save_doc(
            "environments",
            str(environment.id),
            environment,
            name=environment.source.name,
            source_kind=environment.source.kind.value,
        )

    def get_environment(self, environment_id: UUID) -> api.Environment | None:  # noqa: D102
        return self._get_doc("environments", str(environment_id), api.Environment)  # type: ignore[return-value]

    def list_environments(self) -> list[api.Environment]:  # noqa: D102
        return self._list_docs("environments", api.Environment)  # type: ignore[return-value]

    def save_leaderboard(self, leaderboard: api.Leaderboard) -> None:  # noqa: D102
        self._save_doc(
            "leaderboards",
            str(leaderboard.id),
            leaderboard,
            name=leaderboard.name,
            visibility=leaderboard.visibility.value,
        )

    def get_leaderboard(self, leaderboard_id: UUID) -> api.Leaderboard | None:  # noqa: D102
        return self._get_doc("leaderboards", str(leaderboard_id), api.Leaderboard)  # type: ignore[return-value]

    def list_leaderboards(self) -> list[api.Leaderboard]:  # noqa: D102
        return self._list_docs("leaderboards", api.Leaderboard)  # type: ignore[return-value]

    def save_run(self, run: api.Run, *, idempotency_key: str | None = None) -> None:  # noqa: D102
        selection = run.selection.root
        leaderboard_id = getattr(selection, "leaderboard_id", None)
        self._save_doc(
            "runs",
            str(run.id),
            run,
            mode=run.mode.value,
            status=run.status.value,
            cache_status=run.cache_status.value,
            leaderboard_id=str(leaderboard_id) if leaderboard_id else None,
            idempotency_key=idempotency_key,
        )

    def get_run(self, run_id: UUID) -> api.Run | None:  # noqa: D102
        return self._get_doc("runs", str(run_id), api.Run)  # type: ignore[return-value]

    def list_runs(self) -> list[api.Run]:  # noqa: D102
        return self._list_docs("runs", api.Run)  # type: ignore[return-value]

    def get_run_by_idempotency(self, idempotency_key: str) -> api.Run | None:  # noqa: D102
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT doc FROM runs WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
        if not row:
            return None
        return api.Run.model_validate_json(row["doc"])

    def save_run_result(self, result: api.RunResult) -> None:  # noqa: D102
        with self._lock, self._connect() as conn:
            conn.execute(
                "REPLACE INTO run_results (run_id, created_at, doc) VALUES (?, ?, ?)",
                (
                    str(result.run_id),
                    _iso(_now()),
                    json.dumps(result.model_dump(mode="json", exclude_none=True)),
                ),
            )

    def get_run_result(self, run_id: UUID) -> api.RunResult | None:  # noqa: D102
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT doc FROM run_results WHERE run_id = ?",
                (str(run_id),),
            ).fetchone()
        if not row:
            return None
        return api.RunResult.model_validate_json(row["doc"])

    def list_run_results(self) -> list[api.RunResult]:  # noqa: D102
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT doc FROM run_results ORDER BY created_at DESC"
            ).fetchall()
        return [api.RunResult.model_validate_json(row["doc"]) for row in rows]

    def save_cached_subject(
        self,
        fingerprint: str,
        subject: api.SubjectResult,
        run_id: UUID,
    ) -> None:  # noqa: D102
        with self._lock, self._connect() as conn:
            conn.execute(
                "REPLACE INTO subject_cache (fingerprint, run_id, created_at, doc) "
                "VALUES (?, ?, ?, ?)",
                (
                    fingerprint,
                    str(run_id),
                    _iso(_now()),
                    json.dumps(subject.model_dump(mode="json", exclude_none=True)),
                ),
            )

    def get_cached_subject(
        self,
        fingerprint: str,
    ) -> tuple[UUID, api.SubjectResult] | None:  # noqa: D102
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT run_id, doc FROM subject_cache WHERE fingerprint = ?",
                (fingerprint,),
            ).fetchone()
        if not row:
            return None
        return UUID(row["run_id"]), api.SubjectResult.model_validate_json(row["doc"])

    def delete(self, table: str, doc_id: UUID) -> bool:  # noqa: D102
        _check_table(table)
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                f"DELETE FROM {table} WHERE id = ?",  # noqa: S608 — table validated above
                (str(doc_id),),
            )
            return cur.rowcount > 0
