# License Apache 2.0: (c) 2026 Atena-Reply

"""A run traces only when the MLflow server answers; a server that is down costs a log line, not the run."""

import os

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import httpx

from openarena.programs import observability


def test_an_unreachable_server_is_named_and_a_healthy_one_passes(monkeypatch):
    def answer(request: httpx.Request) -> httpx.Response:
        if request.url.host == "up":
            return httpx.Response(200, text="OK")
        raise httpx.ConnectError("refused")

    real = httpx.get
    monkeypatch.setattr(observability.httpx if hasattr(observability, "httpx") else httpx, "get", lambda url, timeout: httpx.Client(transport=httpx.MockTransport(answer)).get(url, timeout=timeout))
    assert observability.mlflow_reachable("http://up:5000") is None
    why = observability.mlflow_reachable("http://down:5000")
    assert why and "nothing answered at http://down:5000" in why
    assert observability.mlflow_reachable("sqlite:///mlflow.db") is None  # no server to ask
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert "not set" in observability.mlflow_reachable()
    httpx.get = real


def test_tracing_stays_off_when_the_server_does_not_answer(monkeypatch):
    """The startup turns tracing on too: against a dead server it must not, or every call waits on retries."""
    enabled = []
    monkeypatch.setenv("ARENA_MLFLOW_TRACING", "1")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:9")
    monkeypatch.setattr(observability.synalinks, "enable_observability", lambda **kw: enabled.append(kw))
    assert observability.enable_observability() is False and enabled == []
    monkeypatch.setattr(observability, "mlflow_reachable", lambda uri=None, timeout=3.0: None)
    assert observability.enable_observability("exp") is True
    assert enabled == [{"tracking_uri": "http://127.0.0.1:9", "experiment_name": "exp"}]
