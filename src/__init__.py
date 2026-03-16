"""Open Arena shared paths, configuration helpers, and packaged prompts."""

import os
from importlib.resources import files
from pathlib import Path

import yaml

REPOSITORY_LOCATION = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
RESOURCES_LOCATION = os.path.join(REPOSITORY_LOCATION, "resources")
DATA_LOCATION = os.path.join(RESOURCES_LOCATION, "data")
EVALUATION_RESULTS_LOCATION = os.path.join(RESOURCES_LOCATION, "evaluation_results")
EXECUTION_RESULTS_LOCATION = os.path.join(RESOURCES_LOCATION, "execution_results")
PROMPT_LOCATION = os.path.join(RESOURCES_LOCATION, "prompt")

try:
    _prompts_text = files("src").joinpath("prompts.default.yaml").read_text(encoding="utf-8")
    default_prompts = yaml.safe_load(_prompts_text)
except FileNotFoundError:
    default_prompts = {}


def load_config(path: str = os.path.join(REPOSITORY_LOCATION, "config.yaml")) -> dict:
    """Load the Open Arena YAML configuration file."""

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path.resolve()}")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


__all__ = [
    "DATA_LOCATION",
    "EVALUATION_RESULTS_LOCATION",
    "EXECUTION_RESULTS_LOCATION",
    "PROMPT_LOCATION",
    "REPOSITORY_LOCATION",
    "RESOURCES_LOCATION",
    "default_prompts",
    "load_config",
]
