from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config.types import EvaluationConfig, ExperimentConfig, ExperimentsFile


def test_experiment_config_forbids_unknown_keys():
    with pytest.raises(ValidationError, match="extra"):
        ExperimentConfig(
            name="exp",
            litellm={"model": "x"},
            timeout_s=1.0,
            unexpected_option=True,
        )


def test_evaluation_config_forbids_unknown_keys():
    with pytest.raises(ValidationError, match="extra"):
        EvaluationConfig(
            method="llm_as_judge",
            litellm={"model": "x"},
            system_prompt="judge with reference",
            system_prompt_no_reference="judge without reference",
            max_concurency=5,
        )


def test_experiments_file_forbids_unknown_top_level_keys():
    with pytest.raises(ValidationError, match="extra"):
        ExperimentsFile(
            dataset={
                "name": "dataset",
                "source": {"provider": "local", "path": "rows.csv", "format": "csv"},
                "input": "{{ question }}",
                "expected_output": "{{ answer }}",
            },
            system_prompt="system",
            experiments=[
                {
                    "name": "exp",
                    "litellm": {"model": "x"},
                }
            ],
            evaluation={
                "method": "llm_as_judge",
                "litellm": {"model": "x"},
                "system_prompt": "judge with reference",
                "system_prompt_no_reference": "judge without reference",
            },
            typo_root=True,
        )
