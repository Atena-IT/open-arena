# License Apache 2.0: (c) 2026 Atena-Reply

"""The engine stands alone: nothing it imports comes from the backend around it."""

import importlib
import os
import pkgutil
import subprocess
import sys

import pytest

import openarena
from openarena import rewards


def test_every_module_imports_without_the_backend():
    # A fresh interpreter, so the backend's packages cannot already be loaded by another test.
    code = (
        "import importlib, pkgutil, sys, openarena\n"
        "for m in pkgutil.walk_packages(openarena.__path__, 'openarena.'):\n"
        "    importlib.import_module(m.name)\n"
        "leaked = sorted(n for n in sys.modules if n.split('.')[0] in {'libraries', 'services', 'data_models', 'routes', 'auth'})\n"
        "assert not leaked, leaked\n"
    )
    env = {**os.environ, "LITELLM_LOCAL_MODEL_COST_MAP": "True", "PYTHONPATH": ""}
    done = subprocess.run([sys.executable, "-c", code], cwd=os.path.dirname(__file__), env=env, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


@pytest.mark.parametrize(
    ("name", "category"),
    [
        ("exact_match", "deterministic"),
        ("cosine_similarity", "deterministic"),
        ("lm_as_judge", "lm_judge"),
    ],
)
def test_reward_category(name, category):
    assert rewards.category(name) == category


def test_every_reward_has_a_category():
    known = {"deterministic", "lm_judge", "rubrics_judge", "agent_judge"}
    assert {rewards.category(name) for name in rewards._REWARD_TYPES} <= known


def test_the_command_is_installed():
    assert importlib.util.find_spec("openarena.cli.main") is not None
    assert "cli" in {m.name for m in pkgutil.iter_modules(openarena.__path__)}
