import os, yaml
from pathlib import Path


""" REPOSITORY ROOT DIRECTORY """
REPOSITORY_LOCATION = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

""" RESOURCES ROOT DIRECTORY """
RESOURCES_LOCATION = os.path.join(REPOSITORY_LOCATION, "resources")

""" EXCEL FILE DIRECTORY """
DATA_LOCATION = os.path.join(RESOURCES_LOCATION, "data")

""" EVALUATION RESULTS DIRECTORY """
EVALUATION_RESULTS_LOCATION = os.path.join(RESOURCES_LOCATION, "evaluation_results")

""" EXECUTION RESULTS DIRECTORY """
EXECUTION_RESULTS_LOCATION = os.path.join(RESOURCES_LOCATION, "execution_results")

""" PROMPTS DIRECTORY """
PROMPT_LOCATION = os.path.join(RESOURCES_LOCATION, "prompt")


""" LOAD CONFIG FUNCTION"""
def load_config(path: str = os.path.join(REPOSITORY_LOCATION, "config.yaml")) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path.resolve()}")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
