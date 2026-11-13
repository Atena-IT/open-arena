# License Apache 2.0: (c) 2026 Atena-Reply

"""The engine's tests: no backend, no network."""

import os

import pytest

# litellm fetches its pricing table from GitHub at import unless told to use
# the copy it ships with; tests must not depend on the network.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")


@pytest.fixture(autouse=True)
def _wide_console(monkeypatch):
    """The CLI's output on one line whatever the terminal: off a terminal rich wraps at 80 columns, and a temp path is long."""
    from openarena.cli.main import console

    monkeypatch.setattr(console, "width", 1000)


@pytest.fixture(autouse=True)
def _mlflow_tracking_uri(monkeypatch):
    """Each test starts from MLflow's tracking URI as the one before found it.

    `enable_observability` sets it process-wide, and monkeypatch restores the
    environment variable, not MLflow's own copy: a test pointing it at a dead
    server would leave every later test that traces waiting on that server.
    """
    from mlflow.tracking._tracking_service import utils

    monkeypatch.setattr(utils, "_tracking_uri", utils._tracking_uri)


@pytest.fixture(autouse=True)
def openrouter_models(monkeypatch):
    """OpenRouter's model list as the tests set it: `{model id: supported parameters}`, empty to start.

    For an `openrouter/...` id synalinks asks that list whether the model reasons, once per model
    (`LanguageModel.model_metadata`); no test reaches OpenRouter. A test that needs a reasoning model adds it
    with `"reasoning"` among its parameters. litellm's table is not asked for such an id.
    """
    import synalinks

    listed: dict[str, list[str]] = {}
    asked = synalinks.LanguageModel.model_metadata

    async def model_metadata(self):
        if not self.model.startswith("openrouter/"):
            return await asked(self)
        return {"supported_parameters": listed[self.model]} if self.model in listed else {}

    monkeypatch.setattr(synalinks.LanguageModel, "model_metadata", model_metadata)
    return listed
