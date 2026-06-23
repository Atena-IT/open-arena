# License Apache 2.0: (c) 2026 Athena-Reply
"""Tests for :class:`~src.api.stores.sqlalchemy_store.SqlAlchemyStore`.

All tests run against an in-memory SQLite database (``sqlite://``) so no
real PostgreSQL server is required.  They verify Store-conformance by
exercising every abstract method declared in the port ABC.

Run with::

    uv run pytest tests/test_sqlalchemy_store.py -q
"""
from __future__ import annotations

import inspect
from datetime import UTC, datetime
from uuid import uuid4, UUID

import pytest

from src.api import models as api
from src.api.ports.store import Store
from src.api.stores.sqlalchemy_store import SqlAlchemyStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def store() -> SqlAlchemyStore:
    """Ephemeral in-memory SQLite store, fresh per test."""
    return SqlAlchemyStore("sqlite://")


def _now() -> datetime:
    return datetime.now(UTC)


def _verifier() -> api.VerifierSuite:
    return api.VerifierSuite(
        id=uuid4(),
        name="test-verifier",
        metrics=[
            api.MetricDefinition(
                name="accuracy",
                metric_kind="accuracy",
                weight=1.0,
            )
        ],
        created_at=_now(),
        updated_at=_now(),
    )


def _env() -> api.Environment:
    return api.Environment(
        id=uuid4(),
        source=api.EnvironmentSource(
            kind=api.EnvironmentSourceKind.inline,
            name="test-env",
            version="1.0.0",
        ),
        created_at=_now(),
        updated_at=_now(),
    )


def _leaderboard() -> api.Leaderboard:
    return api.Leaderboard(
        id=uuid4(),
        name="test-lb",
        visibility=api.LeaderboardVisibility.private,
        ranking=api.RankingPolicy(primary_metric="accuracy"),
        created_at=_now(),
        updated_at=_now(),
    )


def _run(leaderboard_id: UUID | None = None) -> api.Run:
    if leaderboard_id is not None:
        selection = api.RunSelection(
            root=api.RunSelection1(leaderboard_id=leaderboard_id)
        )
    else:
        dataset = api.DatasetBinding(
            provider="local",
            source_ref="test-dataset",
        )
        verifier_binding = api.VerifierSuiteBinding(
            root=api.VerifierSuiteRef(
                binding_type="ref",
                verifier_id=uuid4(),
            )
        )
        inline_env = api.EnvironmentInlineMembership(
            inline_definition=api.InlineEnvironmentDefinition(
                name="inline-env",
                version="0.1",
                dataset=dataset,
                verifier=verifier_binding,
                runtime=api.EnvironmentRuntimePolicy(),
            )
        )
        selection = api.RunSelection(
            root=api.RunSelection2(
                direct_pairs=[
                    api.DirectRunPair(
                        model=api.ModelDefinitionCreate(
                            name="gpt-4o",
                            runtime=api.ModelExecutionConfig(
                                provider="openai",
                                model_name="gpt-4o",
                                model_version="2024-08-06",
                            ),
                        ),
                        environment=inline_env,
                    )
                ]
            )
        )
    return api.Run(
        id=uuid4(),
        mode=api.RunMode.generator,
        selection=selection,
        status=api.RunStatus.queued,
        cache_status=api.CacheStatus.pending,
        created_at=_now(),
    )


def _model_def() -> api.ModelDefinition:
    return api.ModelDefinition(
        id=uuid4(),
        name="gpt-4o",
        runtime=api.ModelExecutionConfig(
            provider="openai",
            model_name="gpt-4o",
            model_version="2024-08-06",
        ),
        created_at=_now(),
        updated_at=_now(),
    )


def _subject_result(env: api.Environment) -> api.SubjectResult:
    return api.SubjectResult(
        model=_model_def(),
        environment=env,
        metrics=[
            api.MetricResult(
                name="accuracy",
                value=0.87,
                direction=api.Direction.max,
            )
        ],
        cache_status=api.CacheStatus.miss,
    )


def _run_result(run: api.Run, env: api.Environment) -> api.RunResult:
    return api.RunResult(
        run_id=run.id,
        mode=run.mode,
        subjects=[_subject_result(env)],
    )


# ---------------------------------------------------------------------------
# ABC conformance
# ---------------------------------------------------------------------------

class TestStoreConformance:
    """Verify SqlAlchemyStore is a concrete subclass of Store."""

    def test_is_store_subclass(self):
        assert issubclass(SqlAlchemyStore, Store)

    def test_implements_all_abstract_methods(self):
        abstract_methods = {
            name
            for name, member in inspect.getmembers(Store)
            if getattr(member, "__isabstractmethod__", False)
        }
        store_methods = set(dir(SqlAlchemyStore))
        missing = abstract_methods - store_methods
        assert not missing, f"Missing abstract methods: {missing}"

    def test_instantiable(self, store):
        assert isinstance(store, Store)


# ---------------------------------------------------------------------------
# Verifiers
# ---------------------------------------------------------------------------

class TestVerifiers:
    def test_save_and_get(self, store: SqlAlchemyStore):
        v = _verifier()
        store.save_verifier(v)
        result = store.get_verifier(v.id)
        assert result is not None
        assert result.id == v.id
        assert result.name == v.name

    def test_get_missing_returns_none(self, store: SqlAlchemyStore):
        assert store.get_verifier(uuid4()) is None

    def test_list_empty(self, store: SqlAlchemyStore):
        assert store.list_verifiers() == []

    def test_list_returns_all(self, store: SqlAlchemyStore):
        v1 = _verifier()
        v2 = _verifier()
        store.save_verifier(v1)
        store.save_verifier(v2)
        result = store.list_verifiers()
        assert len(result) == 2
        ids = {r.id for r in result}
        assert ids == {v1.id, v2.id}

    def test_save_is_idempotent(self, store: SqlAlchemyStore):
        v = _verifier()
        store.save_verifier(v)
        store.save_verifier(v)  # second call should not raise
        assert len(store.list_verifiers()) == 1

    def test_save_overwrites(self, store: SqlAlchemyStore):
        v = _verifier()
        store.save_verifier(v)
        updated = v.model_copy(update={"name": "updated-name"})
        store.save_verifier(updated)
        result = store.get_verifier(v.id)
        assert result.name == "updated-name"

    def test_delete_existing(self, store: SqlAlchemyStore):
        v = _verifier()
        store.save_verifier(v)
        assert store.delete("verifiers", v.id) is True
        assert store.get_verifier(v.id) is None

    def test_delete_missing_returns_false(self, store: SqlAlchemyStore):
        assert store.delete("verifiers", uuid4()) is False


# ---------------------------------------------------------------------------
# Environments
# ---------------------------------------------------------------------------

class TestEnvironments:
    def test_save_and_get(self, store: SqlAlchemyStore):
        env = _env()
        store.save_environment(env)
        result = store.get_environment(env.id)
        assert result is not None
        assert result.id == env.id

    def test_get_missing_returns_none(self, store: SqlAlchemyStore):
        assert store.get_environment(uuid4()) is None

    def test_list_returns_all(self, store: SqlAlchemyStore):
        e1, e2 = _env(), _env()
        store.save_environment(e1)
        store.save_environment(e2)
        assert len(store.list_environments()) == 2

    def test_delete(self, store: SqlAlchemyStore):
        env = _env()
        store.save_environment(env)
        assert store.delete("environments", env.id) is True
        assert store.get_environment(env.id) is None


# ---------------------------------------------------------------------------
# Leaderboards
# ---------------------------------------------------------------------------

class TestLeaderboards:
    def test_save_and_get(self, store: SqlAlchemyStore):
        lb = _leaderboard()
        store.save_leaderboard(lb)
        result = store.get_leaderboard(lb.id)
        assert result is not None
        assert result.id == lb.id
        assert result.name == lb.name

    def test_get_missing_returns_none(self, store: SqlAlchemyStore):
        assert store.get_leaderboard(uuid4()) is None

    def test_list_returns_all(self, store: SqlAlchemyStore):
        lb1, lb2 = _leaderboard(), _leaderboard()
        store.save_leaderboard(lb1)
        store.save_leaderboard(lb2)
        assert len(store.list_leaderboards()) == 2

    def test_delete(self, store: SqlAlchemyStore):
        lb = _leaderboard()
        store.save_leaderboard(lb)
        assert store.delete("leaderboards", lb.id) is True
        assert store.get_leaderboard(lb.id) is None


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

class TestRuns:
    def test_save_and_get(self, store: SqlAlchemyStore):
        run = _run()
        store.save_run(run)
        result = store.get_run(run.id)
        assert result is not None
        assert result.id == run.id

    def test_get_missing_returns_none(self, store: SqlAlchemyStore):
        assert store.get_run(uuid4()) is None

    def test_list_returns_all(self, store: SqlAlchemyStore):
        r1, r2 = _run(), _run()
        store.save_run(r1)
        store.save_run(r2)
        assert len(store.list_runs()) == 2

    def test_idempotency_key_lookup(self, store: SqlAlchemyStore):
        run = _run()
        key = "test-key-abc"
        store.save_run(run, idempotency_key=key)
        result = store.get_run_by_idempotency(key)
        assert result is not None
        assert result.id == run.id

    def test_idempotency_missing_returns_none(self, store: SqlAlchemyStore):
        assert store.get_run_by_idempotency("no-such-key") is None

    def test_save_same_idempotency_key_twice_overwrites(self, store: SqlAlchemyStore):
        run1 = _run()
        key = "idempotent-key"
        store.save_run(run1, idempotency_key=key)
        # Save the same run again with same key -- must not raise
        store.save_run(run1, idempotency_key=key)
        assert len(store.list_runs()) == 1

    def test_delete(self, store: SqlAlchemyStore):
        run = _run()
        store.save_run(run)
        assert store.delete("runs", run.id) is True
        assert store.get_run(run.id) is None


# ---------------------------------------------------------------------------
# RunResults
# ---------------------------------------------------------------------------

class TestRunResults:
    def test_save_and_get(self, store: SqlAlchemyStore):
        run = _run()
        env = _env()
        store.save_run(run)
        rr = _run_result(run, env)
        store.save_run_result(rr)
        result = store.get_run_result(run.id)
        assert result is not None
        assert result.run_id == run.id

    def test_get_missing_returns_none(self, store: SqlAlchemyStore):
        assert store.get_run_result(uuid4()) is None

    def test_list_returns_all(self, store: SqlAlchemyStore):
        env = _env()
        r1, r2 = _run(), _run()
        store.save_run(r1)
        store.save_run(r2)
        store.save_run_result(_run_result(r1, env))
        store.save_run_result(_run_result(r2, env))
        assert len(store.list_run_results()) == 2

    def test_save_is_idempotent(self, store: SqlAlchemyStore):
        run = _run()
        env = _env()
        rr = _run_result(run, env)
        store.save_run(run)
        store.save_run_result(rr)
        store.save_run_result(rr)
        assert len(store.list_run_results()) == 1


# ---------------------------------------------------------------------------
# Subject cache
# ---------------------------------------------------------------------------

class TestSubjectCache:
    def test_save_and_get(self, store: SqlAlchemyStore):
        env = _env()
        run = _run()
        subject = _subject_result(env)
        fp = "fp-abc123"
        store.save_cached_subject(fp, subject, run.id)
        result = store.get_cached_subject(fp)
        assert result is not None
        rid, subj = result
        assert rid == run.id
        assert subj.cache_status == subject.cache_status

    def test_get_missing_returns_none(self, store: SqlAlchemyStore):
        assert store.get_cached_subject("no-such-fingerprint") is None

    def test_overwrite_same_fingerprint(self, store: SqlAlchemyStore):
        env = _env()
        run1 = _run()
        run2 = _run()
        subject = _subject_result(env)
        fp = "fp-overwrite"
        store.save_cached_subject(fp, subject, run1.id)
        store.save_cached_subject(fp, subject, run2.id)
        result = store.get_cached_subject(fp)
        assert result is not None
        rid, _ = result
        assert rid == run2.id