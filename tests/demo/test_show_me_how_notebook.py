from __future__ import annotations

import json
from pathlib import Path

import yaml


DEMO_DIR = Path(__file__).resolve().parents[2] / "demo" / "show_me_how_open_arena"
NOTEBOOK_PATH = DEMO_DIR / "open_arena_show_me_how.ipynb"
SHOWCASE_CONFIG_PATH = DEMO_DIR / "configs" / "business_qa_showcase.yaml"
RUNNABLE_CONFIG_PATH = DEMO_DIR / "configs" / "business_qa_runnable.yaml"


def _code_sources() -> list[str]:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    return ["".join(cell.get("source", [])) for cell in notebook["cells"] if cell.get("cell_type") == "code"]


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_notebook_uses_showcase_and_runnable_configs():
    joined = "\n".join(_code_sources())

    assert "business_qa_showcase.yaml" in joined
    assert "business_qa_runnable.yaml" in joined
    assert "backend_model" in joined
    assert "SHOWCASE_SAMPLE_LIMIT = 20" in joined


def test_notebook_runs_workflow_and_builds_score_summary():
    joined = "\n".join(_code_sources())

    assert "subprocess" in joined
    assert "datasets.get_runs" in joined
    assert "score_summary" in joined
    assert "avg_score_1_to_5" in joined


def test_showcase_config_lists_cross_vendor_demo_models():
    config = _load_yaml(SHOWCASE_CONFIG_PATH)
    experiment_names = [experiment["name"].lower() for experiment in config["experiments"]]
    listed_models = [experiment["litellm"]["model"].lower() for experiment in config["experiments"]]

    assert config["dataset"]["limit"] == 20
    assert len(config["experiments"]) >= 8
    assert any("gemini" in name for name in experiment_names)
    assert any("claude" in name for name in experiment_names)
    assert any("huggingface" in name or "smollm" in name or "qwen" in name for name in experiment_names)
    assert any("gpt-5.4" in model for model in listed_models)


def test_runnable_config_keeps_openai_backends_for_all_experiments():
    config = _load_yaml(RUNNABLE_CONFIG_PATH)
    backend_models = {experiment["litellm"]["model"] for experiment in config["experiments"]}

    assert config["dataset"]["limit"] == 20
    assert len(config["experiments"]) >= 8
    assert backend_models <= {"gpt-5.4-mini", "gpt-5.4-nano"}
