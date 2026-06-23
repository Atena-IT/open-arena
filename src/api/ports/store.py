# License Apache 2.0: (c) 2026 Athena-Reply
"""Port 1 — Store

Abstract interface for all persistence operations performed by
:class:`~src.api.service.ArenaAPIService`.  Every concrete backend
(SQLite, DynamoDB, Postgres, …) must implement this ABC.

The default adapter is :class:`~src.api.stores.sqlite.SQLiteStore`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from uuid import UUID

from src.api import models as api


class Store(ABC):
    """Persistence port for Open Arena resources.

    A *Store* manages five resource kinds (verifiers, environments,
    leaderboards, runs, run_results) plus a subject-level cache keyed
    by fingerprint and idempotency look-ups on runs.

    All mutating methods are expected to be **idempotent on the primary
    key** — a second call with the same ID should overwrite, not raise.
    """

    # -- verifiers ----------------------------------------------------------------

    @abstractmethod
    def save_verifier(self, verifier: api.VerifierSuite) -> None:
        """Persist (create or overwrite) a verifier suite."""

    @abstractmethod
    def get_verifier(self, verifier_id: UUID) -> api.VerifierSuite | None:
        """Return the verifier with *verifier_id*, or ``None``."""

    @abstractmethod
    def list_verifiers(self) -> list[api.VerifierSuite]:
        """Return all verifiers, newest-first."""

    # -- environments -------------------------------------------------------------

    @abstractmethod
    def save_environment(self, environment: api.Environment) -> None:
        """Persist (create or overwrite) an environment."""

    @abstractmethod
    def get_environment(self, environment_id: UUID) -> api.Environment | None:
        """Return the environment with *environment_id*, or ``None``."""

    @abstractmethod
    def list_environments(self) -> list[api.Environment]:
        """Return all environments, newest-first."""

    # -- leaderboards -------------------------------------------------------------

    @abstractmethod
    def save_leaderboard(self, leaderboard: api.Leaderboard) -> None:
        """Persist (create or overwrite) a leaderboard."""

    @abstractmethod
    def get_leaderboard(self, leaderboard_id: UUID) -> api.Leaderboard | None:
        """Return the leaderboard with *leaderboard_id*, or ``None``."""

    @abstractmethod
    def list_leaderboards(self) -> list[api.Leaderboard]:
        """Return all leaderboards, newest-first."""

    # -- runs ---------------------------------------------------------------------

    @abstractmethod
    def save_run(self, run: api.Run, *, idempotency_key: str | None = None) -> None:
        """Persist (create or overwrite) a run.

        When *idempotency_key* is provided the store must enforce
        uniqueness — a second ``save_run`` with the same key must
        silently overwrite rather than insert a duplicate row.
        """

    @abstractmethod
    def get_run(self, run_id: UUID) -> api.Run | None:
        """Return the run with *run_id*, or ``None``."""

    @abstractmethod
    def list_runs(self) -> list[api.Run]:
        """Return all runs, newest-first."""

    @abstractmethod
    def get_run_by_idempotency(self, idempotency_key: str) -> api.Run | None:
        """Return the run associated with *idempotency_key*, or ``None``."""

    # -- run results --------------------------------------------------------------

    @abstractmethod
    def save_run_result(self, result: api.RunResult) -> None:
        """Persist (create or overwrite) a run result."""

    @abstractmethod
    def get_run_result(self, run_id: UUID) -> api.RunResult | None:
        """Return the result for *run_id*, or ``None``."""

    @abstractmethod
    def list_run_results(self) -> list[api.RunResult]:
        """Return all run results, newest-first."""

    # -- subject cache ------------------------------------------------------------

    @abstractmethod
    def save_cached_subject(
        self,
        fingerprint: str,
        subject: api.SubjectResult,
        run_id: UUID,
    ) -> None:
        """Persist a per-subject result keyed by *fingerprint*."""

    @abstractmethod
    def get_cached_subject(
        self,
        fingerprint: str,
    ) -> tuple[UUID, api.SubjectResult] | None:
        """Return ``(run_id, subject)`` for *fingerprint*, or ``None``."""

    # -- generic delete -----------------------------------------------------------

    @abstractmethod
    def delete(self, table: str, doc_id: UUID) -> bool:
        """Delete the document with *doc_id* from *table*.

        Returns ``True`` when a row was deleted, ``False`` when not found.

        The *table* parameter matches the resource kind string used
        internally (``'verifiers'``, ``'environments'``, ``'leaderboards'``,
        ``'runs'``).
        """
