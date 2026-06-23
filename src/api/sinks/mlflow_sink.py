# License Apache 2.0: (c) 2026 Athena-Reply
"""``src.api.sinks.mlflow_sink`` — MLflow :class:`~src.api.ports.results_sink.ResultsSink` adapter.

WS5: MLflow Results Sink (issue #39).

:class:`MlflowResultsSink` **composes** :class:`~src.api.ports.results_sink.StoreResultsSink`
so that OA-DB persistence and leaderboard materialisation continue to work
unchanged, while additionally logging every run's metrics, params, and tags to
an MLflow Tracking Server.

Configuration (environment variables)
--------------------------------------
``MLFLOW_TRACKING_URI``
    URI of the MLflow tracking server, e.g. ``http://localhost:5000`` or a
    Databricks ``databricks://`` URI.  If unset the MLflow client falls back
    to its own default (``./mlruns`` local directory).

``MLFLOW_TRACKING_TOKEN``
    Optional bearer token for authenticated MLflow servers (e.g. Databricks
    personal access token).  When set it is forwarded to the MLflow client via
    ``MLFLOW_TRACKING_TOKEN`` — the SDK reads this automatically.

Experiment naming
-----------------
The MLflow experiment name is derived from the run labels:

* If ``run.labels`` contains ``"org"`` **and** ``"project"`` the name is
  ``"{org}/{project}"``.
* If only ``"project"`` is present the name is ``"open-arena/{project}"``.
* Otherwise the name falls back to ``"open-arena/default"``.

What gets logged per subject
-----------------------------
For every :class:`~src.api.models.SubjectResult` in the
:class:`~src.api.models.RunResult`:

* **Metrics** — one ``mlflow.log_metric`` call per
  :class:`~src.api.models.MetricResult` (name → value).
* **Params** — model provider, model name, model version, environment name,
  environment version, run mode, and (if present) temperature + max_tokens from
  the model's runtime hyperparameters.
* **Tags** — ``open_arena.run_id``, ``open_arena.run_fingerprint`` (when
  present), ``unity_catalog.full_name``, ``storage.uri``, and
  ``artifact.sha256`` when those values can be found in the subject/environment
  metadata.

Resilience
----------
Any exception raised by the MLflow client (network timeout, auth error, …) is
caught, a ``WARNING`` is emitted to the ``open-arena.mlflow_sink`` logger, and
the method returns normally so the composed store write is never blocked.
"""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from src.api import models as api
from src.api.ports.results_sink import ResultsSink, StoreResultsSink
from src.api.ports.store import Store

if TYPE_CHECKING:
    pass

_LOG = logging.getLogger("open-arena.mlflow_sink")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_UNSET = object()


def _experiment_name(run: api.Run) -> str:
    """Derive a ``{org}/{project}`` MLflow experiment name from *run.labels*."""
    labels = run.labels or {}
    org = labels.get("org", "")
    project = labels.get("project", "")
    if org and project:
        return f"{org}/{project}"
    if project:
        return f"open-arena/{project}"
    return "open-arena/default"


def _extract_tag(metadata: dict | None, key: str) -> str | None:
    """Return *key* from *metadata* if present and non-empty, else ``None``."""
    if not metadata:
        return None
    value = metadata.get(key)
    return str(value) if value is not None else None


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class MlflowResultsSink(ResultsSink):
    """MLflow adapter for :class:`~src.api.ports.results_sink.ResultsSink`.

    Composes :class:`~src.api.ports.results_sink.StoreResultsSink` so that
    OA-DB persistence always runs first, then logs to MLflow.  If MLflow
    is unreachable the warning is logged and the method still returns
    successfully.

    Args:
        store: The :class:`~src.api.ports.store.Store` used to build the
            composed :class:`~src.api.ports.results_sink.StoreResultsSink`.
        tracking_uri: MLflow tracking URI.  Defaults to the
            ``MLFLOW_TRACKING_URI`` environment variable (or MLflow's own
            default when the variable is unset).
    """

    def __init__(
        self,
        store: Store,
        tracking_uri: str | None = _UNSET,  # type: ignore[assignment]
    ) -> None:
        self._store_sink = StoreResultsSink(store=store)
        # Resolve tracking URI: explicit argument wins, then env var, then None.
        if tracking_uri is _UNSET:
            tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        self._tracking_uri: str | None = tracking_uri

    # ------------------------------------------------------------------
    # ResultsSink implementation
    # ------------------------------------------------------------------

    def write(self, run: api.Run, result: api.RunResult) -> None:
        """Persist to the OA store then log to MLflow.

        The store write always runs first.  If MLflow logging raises an
        exception it is swallowed and a ``WARNING`` is emitted so the
        run pipeline is not interrupted.

        Args:
            run: The completed :class:`~src.api.models.Run`.
            result: The materialized :class:`~src.api.models.RunResult`.
        """
        # --- 1. Always persist to the OA store (DB + subject cache). --------
        self._store_sink.write(run, result)

        # --- 2. Log to MLflow (best-effort; errors do not propagate). --------
        try:
            self._log_to_mlflow(run, result)
        except Exception as exc:  # noqa: BLE001
            _LOG.warning(
                "MLflow logging failed for run %s — store write already succeeded. "
                "Error: %s",
                run.id,
                exc,
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_mlflow_client(self):  # type: ignore[return]
        """Lazy-import ``mlflow`` and return a configured client.

        Importing inside the method means the module can be imported without
        the ``mlflow`` extra installed.
        """
        import mlflow  # noqa: PLC0415 — intentional lazy import

        if self._tracking_uri:
            mlflow.set_tracking_uri(self._tracking_uri)
        return mlflow

    def _log_to_mlflow(self, run: api.Run, result: api.RunResult) -> None:
        """Core MLflow logging logic (raises on any error)."""
        mlflow = self._get_mlflow_client()

        experiment_name = _experiment_name(run)
        # get_or_create the experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id

        run_id_str = str(run.id)

        for subject in result.subjects:
            self._log_subject(
                mlflow=mlflow,
                experiment_id=experiment_id,
                run=run,
                result=result,
                subject=subject,
                run_id_str=run_id_str,
            )

    def _log_subject(
        self,
        *,
        mlflow,  # noqa: ANN001
        experiment_id: str,
        run: api.Run,
        result: api.RunResult,
        subject: api.SubjectResult,
        run_id_str: str,
    ) -> None:
        """Start an MLflow run for a single subject and log everything."""
        model = subject.model
        environment = subject.environment

        # ---- params ----------------------------------------------------------
        runtime = model.runtime
        params: dict[str, str] = {
            "model.provider": runtime.provider,
            "model.name": runtime.model_name,
            "model.version": runtime.model_version,
            "environment.name": environment.source.name,
            "environment.version": environment.source.version,
            "run.mode": result.mode.value,
        }
        # Optional hyperparameters
        if runtime.temperature is not None:
            params["model.temperature"] = str(runtime.temperature)
        if runtime.max_tokens is not None:
            params["model.max_tokens"] = str(runtime.max_tokens)

        # ---- tags ------------------------------------------------------------
        tags: dict[str, str] = {
            "open_arena.run_id": run_id_str,
        }
        if subject.run_fingerprint:
            tags["open_arena.run_fingerprint"] = subject.run_fingerprint

        # Unity Catalog / storage / artifact tags — look in both environment
        # and model metadata for flexibility.
        _candidates = [
            (environment.metadata or {}),
            (model.metadata or {}),
            (environment.source.metadata or {}),
        ]
        for source_meta in _candidates:
            for tag_key in ("unity_catalog.full_name", "storage.uri", "artifact.sha256"):
                if tag_key not in tags:
                    value = _extract_tag(source_meta, tag_key)
                    if value is not None:
                        tags[tag_key] = value

        # ---- run name --------------------------------------------------------
        run_name = f"{model.name}@{environment.source.name}"

        # ---- log -------------------------------------------------------------
        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=run_name,
            tags=tags,
        ):
            mlflow.log_params(params)
            for metric in subject.metrics:
                mlflow.log_metric(metric.name, metric.value)
