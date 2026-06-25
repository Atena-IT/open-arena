# License Apache 2.0: (c) 2026 Athena-Reply
"""``src.api.sinks`` — Concrete :class:`~src.api.ports.results_sink.ResultsSink` adapters.

Currently ships two adapters:

* :class:`~src.api.ports.results_sink.StoreResultsSink` — the default
  adapter (defined in the port module itself) that persists results to the
  :class:`~src.api.ports.store.Store`.
* :class:`~src.api.sinks.mlflow_sink.MlflowResultsSink` — WS5 adapter that
  composes ``StoreResultsSink`` and additionally logs metrics/params/tags to
  an MLflow Tracking Server.  Requires the ``mlflow`` extra
  (``uv sync --extra mlflow``).
"""
