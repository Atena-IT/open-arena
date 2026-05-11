"""Load showcase/runnable YAML configs and CSV dataset for the GUI demo."""

from __future__ import annotations

import csv
import os
import re
from pathlib import Path

from dotenv import load_dotenv

from demo.gui.backend.models import (
    DemoConfigResponse,
    EnvStatus,
    EvaluationDefaults,
    HeroMission,
    ModelMappingItem,
)
from src.config.types import ExperimentsFile

REPO_ROOT = Path(__file__).resolve().parents[3]
DEMO_ROOT = REPO_ROOT / "demo" / "show_me_how_open_arena"
SHOWCASE_CONFIG_PATH = DEMO_ROOT / "configs" / "business_qa_showcase.yaml"
RUNNABLE_CONFIG_PATH = DEMO_ROOT / "configs" / "business_qa_runnable.yaml"
DATASET_RELATIVE_PATH = Path("demo") / "show_me_how_open_arena" / "data" / "business_qa_demo.csv"
DATASET_PATH = REPO_ROOT / DATASET_RELATIVE_PATH
ENV_PATH = REPO_ROOT / ".env"

SAMPLE_LIMIT = 20

_ENV_KEYS = [
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_HOST",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "ANTHROPIC_API_KEY",
    "HUGGINGFACE_API_KEY",
]


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def load_demo_config() -> DemoConfigResponse:
    load_dotenv(ENV_PATH)
    showcase = ExperimentsFile.from_yaml(SHOWCASE_CONFIG_PATH)
    runnable = ExperimentsFile.from_yaml(RUNNABLE_CONFIG_PATH)

    with DATASET_PATH.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    runtime_dataset_name = f"{runnable.dataset.name} - sample {SAMPLE_LIMIT}"

    model_mapping: list[ModelMappingItem] = []
    for showcase_experiment, runnable_experiment in zip(showcase.experiments, runnable.experiments, strict=True):
        if showcase_experiment.name != runnable_experiment.name:
            raise ValueError(
                "Showcase and runnable experiment names must match to build a stable model mapping: "
                f"{showcase_experiment.name!r} != {runnable_experiment.name!r}"
            )
        model_mapping.append(
            ModelMappingItem(
                experimentKey=_slugify(showcase_experiment.name),
                experimentName=showcase_experiment.name,
                showcaseModel=showcase_experiment.litellm.model,
                backendModel=runnable_experiment.litellm.model,
            )
        )

    hero_row = rows[0]
    env_status = EnvStatus(**{key: bool(os.getenv(key)) for key in _ENV_KEYS})

    return DemoConfigResponse(
        sampleLimit=SAMPLE_LIMIT,
        dataset={
            "csvPath": str(DATASET_RELATIVE_PATH),
            "rowCount": len(rows),
        },
        runtimeDatasetName=runtime_dataset_name,
        modelMapping=model_mapping,
        heroMission=HeroMission(
            missionTitle=hero_row["mission_title"],
            researchDomain=hero_row["research_domain"],
            timeframeStart=hero_row["timeframe_start"],
            timeframeEnd=hero_row["timeframe_end"],
            allowedDomains=hero_row["allowed_domains"],
            focusSemantics=hero_row["focus_semantics"],
            outputType=hero_row["output_type"],
            question=hero_row["question"],
            expectedAnswer=hero_row["expected_answer"],
        ),
        envStatus=env_status,
        evaluationDefaults=EvaluationDefaults(
            method=runnable.evaluation.method,
            label="Notebook Judge",
            systemPrompt=runnable.evaluation.system_prompt,
            systemPromptNoReference=runnable.evaluation.system_prompt_no_reference,
        ),
    )
