# License Apache 2.0: (c) 2026 Athena-Reply
"""Port 4 — ResultsSink

Writes the materialized result of a completed run to one or more
downstream systems.

The default adapter is :class:`StoreResultsSink` which persists results
(subject cache + run_results) via the :class:`~src.api.ports.store.Store`
port, exactly as the original ``_finalize_result`` did.

WS5: MLflow — add an ``MlflowResultsSink`` that additionally logs metrics
to an MLflow tracking server.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from open_arena_core import models as api
from src.api.ports.store import Store


class ResultsSink(ABC):
    """Port for persisting / forwarding run results.

    A sink is called once per subject after a run completes, and once
    more for the full :class:`~src.api.models.RunResult` record.

    WS5: MLflow — implement ``MlflowResultsSink`` that logs each
    :class:`~src.api.models.SubjectResult` metric to an MLflow experiment.
    """

    @abstractmethod
    def write(self, run: api.Run, result: api.RunResult) -> None:
        """Persist / forward *result* for the completed *run*.

        Args:
            run: The completed :class:`~src.api.models.Run`.
            result: The materialized :class:`~src.api.models.RunResult`
                containing all subject results and aggregate metrics.
        """


class StoreResultsSink(ResultsSink):
    """Default adapter — writes results back into the :class:`~src.api.ports.store.Store`.

    Reproduces the original ``_finalize_result`` subject-cache + run_result
    persistence.  The subject fingerprint is used as the cache key so that
    future runs with matching parameters can reuse the result.

    WS5: MLflow — chain an ``MlflowResultsSink`` after this one by
    wrapping both in a ``MulticastResultsSink``.
    """

    def __init__(self, store: Store) -> None:
        self._store = store

    def write(self, run: api.Run, result: api.RunResult) -> None:  # noqa: D102
        self._store.save_run_result(result)
        for subject in result.subjects:
            if subject.run_fingerprint:
                self._store.save_cached_subject(subject.run_fingerprint, subject, run.id)
