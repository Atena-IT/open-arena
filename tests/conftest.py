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
