# License Apache 2.0: (c) 2026 Atena-Reply

"""A platform's dataset provider without its SDK says which extra of openarena brings it."""

import sys
import tomllib
from pathlib import Path

import pytest

from openarena.datasets import get

TEMPLATE = {"input_template": '{"question": {{ question | tojson }}}'}

# provider type, the module its SDK is imported as, the extra that installs it, the constructor's arguments
PROVIDERS = [
    ("langfuse", "langfuse", "langfuse", {"dataset_name": "bench"}),
    ("langsmith", "langsmith", "langsmith", {"dataset_name": "bench"}),
    ("opik", "opik", "opik", {"dataset_name": "bench"}),
    ("phoenix", "phoenix", "phoenix", {"dataset_name": "bench"}),
    ("braintrust", "braintrust", "braintrust", {"project": "p", "dataset_name": "bench"}),
]


@pytest.mark.parametrize(("provider", "module", "extra", "kwargs"), PROVIDERS, ids=[p[0] for p in PROVIDERS])
def test_a_missing_sdk_names_its_extra(provider, module, extra, kwargs, monkeypatch):
    # As if the SDK were not installed: an import of it raises ImportError.
    for name in [m for m in sys.modules if m == module or m.startswith(module + ".")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, module, None)
    with pytest.raises(ImportError, match=rf"openarena\[{extra}\]"):
        dataset = get(provider)(**kwargs, **TEMPLATE)
        next(iter(dataset()))


def test_every_extra_named_is_declared():
    pyproject = tomllib.loads((Path(__file__).parent.parent / "pyproject.toml").read_text())
    extras = pyproject["project"]["optional-dependencies"]
    for _, _, extra, _ in PROVIDERS:
        assert extra in extras
    assert "unity-catalog" in extras and "all" in extras
