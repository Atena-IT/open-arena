# License Apache 2.0: (c) 2026 Athena-Reply
"""SQLAlchemy 2.0-backed :class:`~src.api.ports.store.Store` adapter.

This adapter targets **PostgreSQL** in production (via ``psycopg`` driver)
while remaining unit-testable against an in-memory SQLite database.

The JSON/JSONB duality
-----------------------
Full model payloads are stored in a ``doc`` column declared as::

    JSON().with_variant(JSONB, "postgresql")

SQLAlchemy's ``with_variant`` swaps the column type at DDL/bind-time:

* **PostgreSQL** -- the column becomes ``JSONB``, enabling GIN index support
  and binary storage with efficient operators (``@>``, ``->>``, etc.).
* **SQLite** (and any other dialect) -- the column falls back to plain
  ``JSON``, which SQLAlchemy stores as a ``TEXT`` field with automatic
  serialisation/deserialisation.

This means the **same Python code** works against both databases; only the
wire format differs.

Selecting the adapter
---------------------
Set ``OPEN_ARENA_STORE=postgres`` to activate this adapter.  The DSN is
read from ``DATABASE_URL`` (default: ``postgresql+psycopg://localhost/open-arena``).

    export OPEN_ARENA_STORE=postgres
    export DATABASE_URL=postgresql+psycopg://user:pass@host:5432/open-arena
"""
from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel
from sqlalchemy import (
    Column,
    DateTime,
    Index,
    MetaData,
    String,
    Table,
    UniqueConstraint,
    create_engine,
    select,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Engine
from sqlalchemy.types import JSON

from src.api import models as api
from src.api.ports.store import Store

_DEFAULT_DSN = "postgresql+psycopg://localhost/open-arena"

metadata = MetaData()


# ---------------------------------------------------------------------------
# JSON/JSONB variant column helper
# ---------------------------------------------------------------------------

def _json_col(name: str) -> Column:
    """Return a Column that is JSONB on Postgres and plain JSON elsewhere."""
    return Column(name, JSON().with_variant(JSONB, "postgresql"), nullable=False)


# ---------------------------------------------------------------------------
# Table definitions -- mirror SQLiteStore's schema exactly
# ---------------------------------------------------------------------------

verifiers_table = Table(
    "verifiers",
    metadata,
    Column("id", String, primary_key=True),
    Column("name", String, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    _json_col("doc"),
)

environments_table = Table(
    "environments",
    metadata,
    Column("id", String, primary_key=True),
    Column("name", String, nullable=False),
    Column("source_kind", String, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    _json_col("doc"),
)

leaderboards_table = Table(
    "leaderboards",
    metadata,
    Column("id", String, primary_key=True),
    Column("name", String, nullable=False),
    Column("visibility", String, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    _json_col("doc"),
)

runs_table = Table(
    "runs",
    metadata,
    Column("id", String, primary_key=True),
    Column("mode", String, nullable=False),
    Column("status", String, nullable=False),
    Column("cache_status", String, nullable=False),
    Column("leaderboard_id", String, nullable=True),
    Column("idempotency_key", String, nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    _json_col("doc"),
    UniqueConstraint("idempotency_key", name="uq_runs_idempotency_key"),
)

run_results_table = Table(
    "run_results",
    metadata,
    Column("run_id", String, primary_key=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    _json_col("doc"),
)

subject_cache_table = Table(
    "subject_cache",
    metadata,
    Column("fingerprint", String, primary_key=True),
    Column("run_id", String, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    _json_col("doc"),
)

# Indexes for common query patterns
Index("ix_runs_leaderboard_id", runs_table.c.leaderboard_id)
Index("ix_runs_created_at", runs_table.c.created_at)
Index("ix_verifiers_created_at", verifiers_table.c.created_at)
Index("ix_environments_created_at", environments_table.c.created_at)
Index("ix_leaderboards_created_at", leaderboards_table.c.created_at)
Index("ix_run_results_created_at", run_results_table.c.created_at)


# ---------------------------------------------------------------------------
# Table name -> table object mapping (used by delete())
# ---------------------------------------------------------------------------

_TABLE_MAP: dict[str, Table] = {
    "verifiers": verifiers_table,
    "environments": environments_table,
    "leaderboards": leaderboards_table,
    "runs": runs_table,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> datetime:
    return datetime.now(UTC)


def _dump(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(mode="json", exclude_none=True)


class SqlAlchemyStore(Store):
    """SQLAlchemy 2.0-backed implementation of the :class:`~src.api.ports.store.Store` port.

    Targets PostgreSQL in production; works identically against in-memory
    SQLite for unit tests (the ``doc`` column becomes plain JSON there).

    Args:
        dsn: SQLAlchemy database URL.  Defaults to the ``DATABASE_URL``
            environment variable, falling back to
            ``postgresql+psycopg://localhost/open-arena``.
    """

    def __init__(self, dsn: str | None = None) -> None:
        resolved_dsn = dsn or os.environ.get("DATABASE_URL", _DEFAULT_DSN)
        self._engine: Engine = create_engine(
            resolved_dsn,
            pool_pre_ping=True,
            future=True,
        )
        metadata.create_all(self._engine)

    # -- internal helpers --------------------------------------------------------

    def _upsert(self, table: Table, row: dict[str, Any]) -> None:
        """Insert or replace a row, preserving ``created_at`` on updates."""
        with self._engine.begin() as conn:
            existing = conn.execute(
                select(table.c.created_at).where(table.c.id == row["id"])
            ).fetchone()
            if existing is not None:
                row = {**row, "created_at": existing[0]}
            conn.execute(table.delete().where(table.c.id == row["id"]))
            conn.execute(table.insert().values(**row))

    def _get_doc(
        self,
        table: Table,
        doc_id: str,
        model_cls: type[BaseModel],
    ) -> BaseModel | None:
        with self._engine.connect() as conn:
            row = conn.execute(
                select(table.c.doc).where(table.c.id == doc_id)
            ).fetchone()
        if row is None:
            return None
        return model_cls.model_validate(row[0])

    def _list_docs(
        self,
        table: Table,
        model_cls: type[BaseModel],
    ) -> list[BaseModel]:
        with self._engine.connect() as conn:
            rows = conn.execute(
                select(table.c.doc).order_by(table.c.created_at.desc())
            ).fetchall()
        return [model_cls.model_validate(row[0]) for row in rows]

    # -- verifiers ---------------------------------------------------------------

    def save_verifier(self, verifier: api.VerifierSuite) -> None:  # noqa: D102
        now = _now()
        self._upsert(
            verifiers_table,
            {
                "id": str(verifier.id),
                "name": verifier.name,
                "created_at": now,
                "updated_at": now,
                "doc": _dump(verifier),
            },
        )

    def get_verifier(self, verifier_id: UUID) -> api.VerifierSuite | None:  # noqa: D102
        return self._get_doc(verifiers_table, str(verifier_id), api.VerifierSuite)  # type: ignore[return-value]

    def list_verifiers(self) -> list[api.VerifierSuite]:  # noqa: D102
        return self._list_docs(verifiers_table, api.VerifierSuite)  # type: ignore[return-value]

    # -- environments ------------------------------------------------------------

    def save_environment(self, environment: api.Environment) -> None:  # noqa: D102
        now = _now()
        self._upsert(
            environments_table,
            {
                "id": str(environment.id),
                "name": environment.source.name,
                "source_kind": environment.source.kind.value,
                "created_at": now,
                "updated_at": now,
                "doc": _dump(environment),
            },
        )

    def get_environment(self, environment_id: UUID) -> api.Environment | None:  # noqa: D102
        return self._get_doc(environments_table, str(environment_id), api.Environment)  # type: ignore[return-value]

    def list_environments(self) -> list[api.Environment]:  # noqa: D102
        return self._list_docs(environments_table, api.Environment)  # type: ignore[return-value]

    # -- leaderboards ------------------------------------------------------------

    def save_leaderboard(self, leaderboard: api.Leaderboard) -> None:  # noqa: D102
        now = _now()
        self._upsert(
            leaderboards_table,
            {
                "id": str(leaderboard.id),
                "name": leaderboard.name,
                "visibility": leaderboard.visibility.value,
                "created_at": now,
                "updated_at": now,
                "doc": _dump(leaderboard),
            },
        )

    def get_leaderboard(self, leaderboard_id: UUID) -> api.Leaderboard | None:  # noqa: D102
        return self._get_doc(leaderboards_table, str(leaderboard_id), api.Leaderboard)  # type: ignore[return-value]

    def list_leaderboards(self) -> list[api.Leaderboard]:  # noqa: D102
        return self._list_docs(leaderboards_table, api.Leaderboard)  # type: ignore[return-value]

    # -- runs --------------------------------------------------------------------

    def save_run(self, run: api.Run, *, idempotency_key: str | None = None) -> None:  # noqa: D102
        selection = run.selection.root
        leaderboard_id = getattr(selection, "leaderboard_id", None)
        now = _now()
        row: dict[str, Any] = {
            "id": str(run.id),
            "mode": run.mode.value,
            "status": run.status.value,
            "cache_status": run.cache_status.value,
            "leaderboard_id": str(leaderboard_id) if leaderboard_id else None,
            "idempotency_key": idempotency_key,
            "created_at": now,
            "updated_at": now,
            "doc": _dump(run),
        }
        with self._engine.begin() as conn:
            existing = conn.execute(
                select(runs_table.c.created_at).where(runs_table.c.id == row["id"])
            ).fetchone()
            if existing is not None:
                row = {**row, "created_at": existing[0]}
            conn.execute(runs_table.delete().where(runs_table.c.id == row["id"]))
            conn.execute(runs_table.insert().values(**row))

    def get_run(self, run_id: UUID) -> api.Run | None:  # noqa: D102
        return self._get_doc(runs_table, str(run_id), api.Run)  # type: ignore[return-value]

    def list_runs(self) -> list[api.Run]:  # noqa: D102
        return self._list_docs(runs_table, api.Run)  # type: ignore[return-value]

    def get_run_by_idempotency(self, idempotency_key: str) -> api.Run | None:  # noqa: D102
        with self._engine.connect() as conn:
            row = conn.execute(
                select(runs_table.c.doc).where(
                    runs_table.c.idempotency_key == idempotency_key
                )
            ).fetchone()
        if row is None:
            return None
        return api.Run.model_validate(row[0])

    # -- run results -------------------------------------------------------------

    def save_run_result(self, result: api.RunResult) -> None:  # noqa: D102
        with self._engine.begin() as conn:
            conn.execute(
                run_results_table.delete().where(
                    run_results_table.c.run_id == str(result.run_id)
                )
            )
            conn.execute(
                run_results_table.insert().values(
                    run_id=str(result.run_id),
                    created_at=_now(),
                    doc=_dump(result),
                )
            )

    def get_run_result(self, run_id: UUID) -> api.RunResult | None:  # noqa: D102
        with self._engine.connect() as conn:
            row = conn.execute(
                select(run_results_table.c.doc).where(
                    run_results_table.c.run_id == str(run_id)
                )
            ).fetchone()
        if row is None:
            return None
        return api.RunResult.model_validate(row[0])

    def list_run_results(self) -> list[api.RunResult]:  # noqa: D102
        with self._engine.connect() as conn:
            rows = conn.execute(
                select(run_results_table.c.doc).order_by(
                    run_results_table.c.created_at.desc()
                )
            ).fetchall()
        return [api.RunResult.model_validate(row[0]) for row in rows]

    # -- subject cache -----------------------------------------------------------

    def save_cached_subject(
        self,
        fingerprint: str,
        subject: api.SubjectResult,
        run_id: UUID,
    ) -> None:  # noqa: D102
        with self._engine.begin() as conn:
            conn.execute(
                subject_cache_table.delete().where(
                    subject_cache_table.c.fingerprint == fingerprint
                )
            )
            conn.execute(
                subject_cache_table.insert().values(
                    fingerprint=fingerprint,
                    run_id=str(run_id),
                    created_at=_now(),
                    doc=_dump(subject),
                )
            )

    def get_cached_subject(
        self,
        fingerprint: str,
    ) -> tuple[UUID, api.SubjectResult] | None:  # noqa: D102
        with self._engine.connect() as conn:
            row = conn.execute(
                select(subject_cache_table.c.run_id, subject_cache_table.c.doc).where(
                    subject_cache_table.c.fingerprint == fingerprint
                )
            ).fetchone()
        if row is None:
            return None
        return UUID(row[0]), api.SubjectResult.model_validate(row[1])

    # -- generic delete ----------------------------------------------------------

    def delete(self, table: str, doc_id: UUID) -> bool:  # noqa: D102
        tbl = _TABLE_MAP.get(table)
        if tbl is None:
            raise ValueError(f"Unknown table {table!r}")
        with self._engine.begin() as conn:
            result = conn.execute(
                tbl.delete().where(tbl.c.id == str(doc_id))
            )
            return result.rowcount > 0