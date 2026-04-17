from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class DatasetConfig(BaseModel):
    """Global dataset configuration. Rows are fetched via a provider adapter
    and shaped into Records using Jinja2 `input` / `expected_output` templates."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, description="Logical dataset name (used for Langfuse dataset creation).")
    source: dict[str, Any] = Field(
        ...,
        description="Provider-specific source config. Must contain `provider` (e.g. 'local', 'huggingface').",
    )
    input: str = Field(..., min_length=1, description="Jinja2 template rendered per row to produce the user input.")
    expected_output: str = Field(..., min_length=1, description="Jinja2 template rendered per row to produce the ground truth.")
    limit: int | None = Field(default=None, ge=1, description="Optional cap on number of rows.")
    description: str | None = Field(default=None, description="Optional human-readable description.")


class MCPServer(BaseModel):
    name: str = Field(..., min_length=1)
    url: HttpUrl = Field(...)


class LiteLLMConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str = Field(..., min_length=1)
    reasoning_effort: str | None = Field(
        default=None,
        description=(
            "Reasoning/thinking budget for reasoning models (o1, deepseek-r1, "
            "qwen3, ...). Typical values: 'low', 'medium', 'high', 'disable'. "
            "See https://docs.litellm.ai/docs/reasoning_content."
        ),
    )


class ExperimentConfig(BaseModel):
    name: str = Field(..., min_length=1)
    litellm: LiteLLMConfig = Field(...)
    mcp: list[MCPServer] | None = Field(default=None)


class EvaluationConfig(BaseModel):
    method: str = Field(..., min_length=1)
    litellm: LiteLLMConfig = Field(...)
    score_name: str | None = Field(default="evaluation_score")
    max_concurrency: int | None = Field(default=10, ge=1)
    system_prompt: str | None = Field(
        default=None,
        description=(
            "Override the judge system prompt (reference mode). If omitted, "
            "the evaluator's default from src/prompts.default.yaml is used."
        ),
    )
    system_prompt_no_reference: str | None = Field(
        default=None,
        description=(
            "Override the judge system prompt used when a row has no "
            "expected_output. If omitted, the evaluator's default is used."
        ),
    )
    granularity: int | None = Field(
        default=None,
        ge=2,
        le=26,
        description=(
            "Verifier methods only ('llm_as_verifier', 'pairwise_verifier'): "
            "number of score letters (A..) used for logprob-based scoring. "
            "Typical: 8."
        ),
    )
    repeats: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Verifier methods only ('llm_as_verifier', 'pairwise_verifier'): "
            "number of repeated verification samples to average (K). "
            "Typical: 1-16."
        ),
    )
    criteria: list[str] | None = Field(
        default=None,
        min_length=1,
        description=(
            "Verifier method only ('llm_as_verifier'): criterion names to "
            "decompose scoring over (C). Each criterion is scored in its "
            "own verifier call; the final reward is the arithmetic mean "
            "across criteria. Omit for a single holistic score. "
            "Example: ['correctness', 'clarity', 'completeness']."
        ),
    )


class ExperimentsFile(BaseModel):
    dataset: DatasetConfig = Field(...)
    system_prompt: str = Field(..., min_length=1)
    experiments: list[ExperimentConfig] = Field(..., min_length=1)
    evaluation: EvaluationConfig = Field(...)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "ExperimentsFile":
        config_file = Path(yaml_path)
        if not config_file.exists():
            raise FileNotFoundError(f"File not found: {yaml_path}")
        with open(config_file, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
