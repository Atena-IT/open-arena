from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


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
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    litellm: LiteLLMConfig = Field(...)
    mcp: list[MCPServer] | None = Field(default=None)
    timeout_s: float | None = Field(
        default=None,
        gt=0,
        description="Per-row wall-clock timeout for LLM calls. Row fails with a timeout error if exceeded.",
    )


class CriterionConfig(BaseModel):
    """A single evaluation criterion as in Kwok et al. 2026.

    `name` is the short label (e.g. "Root Cause Analysis") shown in the
    prompt header. `description` is the long rubric paragraph the verifier
    reads to decide how to score along this dimension.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: str = Field(..., min_length=1)
    litellm: LiteLLMConfig = Field(...)
    score_name: str | None = Field(default="evaluation_score")
    max_concurrency: int | None = Field(default=10, ge=1)
    system_prompt: str = Field(
        ...,
        min_length=1,
        description=(
            "System prompt used in reference mode (when the row has a "
            "non-empty expected_output). Required — see config.example.yaml "
            "for ready-to-copy templates per evaluator method."
        ),
    )
    system_prompt_no_reference: str = Field(
        ...,
        min_length=1,
        description=(
            "System prompt used when the row has no expected_output. "
            "Required — see config.example.yaml for ready-to-copy templates."
        ),
    )
    granularity: int | None = Field(
        default=None,
        ge=2,
        le=26,
        description=(
            "'llm_as_verifier' only: number of score letters (A..) used "
            "for logprob-based scoring (G). Typical: 8."
        ),
    )
    repeats: int | None = Field(
        default=None,
        ge=1,
        description=(
            "'llm_as_verifier' only: number of repeated verification "
            "samples to average per pair-criterion call (K). Typical: 1-16."
        ),
    )
    criteria: list[CriterionConfig] | None = Field(
        default=None,
        min_length=1,
        description=(
            "'llm_as_verifier' only: list of criteria to decompose scoring "
            "over (C). Each item is {name, description} where `description` "
            "is the long rubric paragraph the verifier reads (Kwok et al. "
            "2026). Each criterion is scored in its own pairwise call and "
            "the per-pair reward is the mean across criteria. Omit for a "
            "single holistic call."
        ),
    )
    max_retries: int | None = Field(
        default=None,
        ge=1,
        description=(
            "'llm_as_judge' only: retry budget on structured-output parse "
            "failures (OutputParserException). Default: 3."
        ),
    )
    timeout_s: float | None = Field(
        default=None,
        gt=0,
        description="Per-item wall-clock timeout for evaluator calls. Item fails with a timeout error if exceeded.",
    )

    @field_validator("criteria", mode="before")
    @classmethod
    def _reject_string_criteria(cls, v):
        # Catch the common foot-gun of `criteria: ["correctness", "clarity"]`
        # left over from the pre-decomposition format and surface a clear
        # error instead of an opaque pydantic validation message.
        if isinstance(v, list) and any(isinstance(item, str) for item in v):
            raise ValueError(
                "criteria entries must be objects with `name` and `description` "
                "fields (not bare strings). See config.example.yaml."
            )
        return v


class ExperimentsFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

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
