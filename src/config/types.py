from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class DatasetFormat(str, Enum):
    """Allowed dataset file formats."""
    excel = "excel"
    csv = "csv"


class DatasetType(str, Enum):
    """Allowed dataset semantic types (how the dataset should be interpreted)."""
    QA = "QA"
    ToolScale = "ToolScale"
    ToolsExample = "ToolsExample"


class DatasetConfig(BaseModel):
    """Global dataset configuration shared by all experiments in the file."""

    name: str = Field(
        ...,
        description="Dataset identifier or display name.",
        min_length=1,
    )
    source: str = Field(
        ...,
        description="Path to the dataset file, relative to the project root folder.",
        min_length=1,
    )
    format: DatasetFormat = Field(
        ...,
        description="Dataset file format. Allowed values: 'excel' or 'csv'.",
    )
    type: DatasetType = Field(
        ...,
        description="Dataset content type. Allowed values: 'QA', 'ToolScale', 'ToolsExample'.",
    )


class MCPServer(BaseModel):
    """Definition of a single MCP server entry."""

    name: str = Field(
        ...,
        description="Human-readable name of the MCP server (unique within the experiment is recommended).",
        min_length=1,
    )
    url: HttpUrl = Field(
        ...,
        description="Base URL of the MCP server endpoint (must be a valid URL).",
    )


class LiteLLMConfig(BaseModel):
    """
    LiteLLM configuration.

    Notes:
    - Must contain at least 'model'.
    - Accepts any number of additional custom keys (provider settings, parameters, etc.).
    """

    model_config = ConfigDict(extra="allow")

    model: str = Field(
        ...,
        description="Model name or identifier (e.g., 'gpt-4o', 'gpt-4', 'bedrock:...').",
        min_length=1,
    )


class ExperimentConfig(BaseModel):
    """Single experiment definition."""

    name: str = Field(
        ...,
        description="Experiment name (used as identifier for runs/reports).",
        min_length=1,
    )
    litellm: LiteLLMConfig = Field(
        ...,
        description="LiteLLM configuration block (must include 'model' and may include extra custom keys).",
    )
    mcp: list[MCPServer] | None = Field(
        default=None,
        description="Optional list of MCP servers available to the experiment.",
    )


class EvaluationConfig(BaseModel):
    """Evaluation configuration for scoring experiment results."""

    method: str = Field(
        ...,
        description="Evaluation method name (e.g., 'llm_as_judge', 'exact_match').",
        min_length=1,
    )
    litellm: LiteLLMConfig = Field(
        ...,
        description="LiteLLM configuration for the evaluation model (e.g., judge LLM).",
    )
    score_name: str | None = Field(
        default="evaluation_score",
        description="Name of the score to write to Langfuse (default: 'evaluation_score').",
    )
    max_concurrency: int | None = Field(
        default=10,
        description="Maximum number of concurrent evaluations (default: 10).",
        ge=1,
    )


class ExperimentsFile(BaseModel):
    """Root YAML document model (global dataset + global system_prompt + experiments list)."""

    dataset: DatasetConfig = Field(
        ...,
        description="Global dataset configuration shared by all experiments.",
    )
    system_prompt: str = Field(
        ...,
        description="Global system prompt applied to all experiments (unless your runner overrides it).",
        min_length=1,
    )
    experiments: list[ExperimentConfig] = Field(
        ...,
        description="List of experiments to run against the global dataset and system prompt.",
        min_length=1,
    )
    evaluation: EvaluationConfig = Field(
        ...,
        description="Evaluation configuration for scoring experiment results.",
    )

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "ExperimentsFile":
        """
        Load and validate ExperimentsFile from a YAML file.

        Uses Pydantic's validation to ensure the YAML structure matches
        the expected schema.

        :param yaml_path: Path to the YAML configuration file
        :return: Validated ExperimentsFile instance
        :raises FileNotFoundError: If the YAML file doesn't exist
        :raises ValueError: If the YAML structure is invalid
        :raises yaml.YAMLError: If the file contains invalid YAML syntax
        """
        config_file = Path(yaml_path)

        if not config_file.exists():
            raise FileNotFoundError(f"File not found: {yaml_path}")

        with open(config_file, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        return cls(**config_dict)
