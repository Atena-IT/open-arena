# License Apache 2.0: (c) 2026 Athena-Reply

"""Pydantic validation for the open-arena YAML config.

Everything that used to live as `_resolve_agent_config` / `_parse_metrics` /
`_strip_dataset_reward_meta` / `_parse_reward_direction` / inline removed-key
checks in `evaluate.py` is consolidated here so the sweep file only has to
build programs and run trials. `Config.load(path)` is the single entry point;
it returns a validated tree with helper accessors for the bits the sweep
needs (`selected_dataset_names`, `resolved_agent`, `reward_spec`,
`reward_direction`, `generator_kwargs`, `dataset_metrics`).

Pydantic raises on structural / semantic problems before the first trial
starts; runtime resolution (instantiating `synalinks.metrics.Metric` /
`Reward` objects from validated specs) stays in `evaluate.py` because it
needs the synalinks runtime imported.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


Direction = Literal["max", "min"]


class MetricEntry(BaseModel):
    """A validated entry from a `metrics:` list (top-level or per-dataset).

    The YAML accepts either a bare string (`- total_tokens`) or a mapping
    with `class:` plus an optional `alias:` / `direction:` / `objective:` and
    any number of constructor kwargs. The string form is normalized to the
    mapping form in `_string_shortcut` so downstream code only sees one
    shape.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    class_name: str = Field(alias="class")
    alias: str | None = None
    direction: Direction = "max"
    objective: bool = False

    @model_validator(mode="before")
    @classmethod
    def _string_shortcut(cls, value: Any) -> Any:
        if isinstance(value, str):
            return {"class": value}
        return value

    @property
    def kwargs(self) -> dict[str, Any]:
        """Extra fields forwarded as constructor kwargs (incl. `name:`)."""
        return dict(self.__pydantic_extra__ or {})


class RewardBlock(BaseModel):
    """Validated `reward:` block.

    `name:` selects a registered reward (snake_case identifier);
    `direction:` is sweep metadata for keras-tuner ranking and is stripped
    before the spec is handed to `get_reward(...)`. Everything else flows
    through `__pydantic_extra__` as constructor kwargs (e.g.
    `language_model:`, `instructions:`, `in_mask:`, `out_mask:`).
    """

    model_config = ConfigDict(extra="allow")

    name: str
    direction: Direction = "max"

    def spec(self) -> dict[str, Any]:
        """`{name, ...kwargs}` ready for `src.rewards.get(...)`."""
        return {"name": self.name, **(self.__pydantic_extra__ or {})}


class AgentBlock(BaseModel):
    """Validated `agent:` block.

    `mcp_servers:` is a list of names referencing the top-level
    `mcp_servers:` registry — `Config` cross-checks them against that
    registry once both are parsed (see `Config._validate_selection_and_refs`).
    All other keys (`max_iterations`, `autonomous`, `temperature`,
    `instructions`, `final_instructions`, ...) flow to
    `synalinks.FunctionCallingAgent` via `__pydantic_extra__`.
    """

    model_config = ConfigDict(extra="allow")

    type: str = "function_calling"
    mcp_servers: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _require_mcp_servers(self) -> "AgentBlock":
        if not self.mcp_servers:
            raise ValueError(
                "`agent.mcp_servers:` is required and must list at least one "
                "server name from the top-level `mcp_servers:` registry."
            )
        if self.type != "function_calling":
            raise ValueError(
                f"`agent.type:` unsupported value {self.type!r}. Supported: "
                f"'function_calling'."
            )
        reserved = {"language_model", "tools"}
        overlap = reserved & set(self.__pydantic_extra__ or {})
        if overlap:
            raise ValueError(
                f"`agent:` keys {sorted(overlap)} are set automatically and "
                f"cannot be overridden in YAML."
            )
        return self


class DatasetEntry(BaseModel):
    """A `datasets.<name>:` entry.

    Arena-level keys (`generator`, `agent`, `reward`, `metrics`) are
    consumed by the sweep; every other key is a dataset-provider kwarg
    (forwarded by `load_dataset_from_yaml` to the matching `Dataset`
    subclass). `extra="allow"` keeps those provider kwargs untouched.
    """

    model_config = ConfigDict(extra="allow")

    type: str
    generator: dict[str, Any] | None = None
    agent: AgentBlock | None = None
    reward: RewardBlock
    metrics: list[MetricEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def _generator_xor_agent(self) -> "DatasetEntry":
        if self.generator is not None and self.agent is not None:
            raise ValueError(
                "`generator:` and `agent:` are mutually exclusive — pick one."
            )
        return self


class ExperimentsBlock(BaseModel):
    """`experiments:` block — the sweep axes."""

    model_config = ConfigDict(extra="forbid")

    language_models: list[str]
    datasets: list[str] | None = None

    @model_validator(mode="before")
    @classmethod
    def _reject_removed_keys(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if "rewards" in data:
            raise ValueError(
                "`experiments.rewards:` was removed — declare extra scoring "
                "functions under top-level `metrics:` instead. Any reward "
                "identifier (e.g. `lm_as_judge`, `cosine_similarity`) listed "
                "there is auto-wrapped in `MeanMetricWrapper` so it rides the "
                "primary evaluate() pass — no extra evaluate per reward."
            )
        if "primary_direction" in data:
            raise ValueError(
                "`experiments.primary_direction:` was removed — set "
                "`direction: max|min` per-dataset under that dataset's "
                "`reward:` block instead. Direction is now per-dataset (each "
                "dataset can have its own loss/reward orientation)."
            )
        return data


class Config(BaseModel):
    """The full validated `config.yaml`.

    `Config.load(path)` is the entry point — it parses YAML, validates,
    cross-checks MCP refs, and returns this object. Helper accessors
    (`selected_dataset_names`, `resolved_agent`, `reward_spec`,
    `reward_direction`, `generator_kwargs`, `dataset_metrics`) hide the
    per-dataset lookup boilerplate from the sweep code.
    """

    model_config = ConfigDict(extra="forbid")

    seed: int | None = None
    default_language_model: str | None = None
    default_embedding_model: str | None = None
    mcp_servers: dict[str, dict[str, Any]] = Field(default_factory=dict)
    datasets: dict[str, DatasetEntry]
    default: str | None = None
    metrics: list[MetricEntry] = Field(default_factory=list)
    experiments: ExperimentsBlock

    @model_validator(mode="after")
    def _validate_selection_and_refs(self) -> "Config":
        explicit = self.experiments.datasets
        if explicit:
            names = list(explicit)
        elif self.default:
            names = [self.default]
        else:
            raise ValueError(
                "no datasets selected — set `experiments.datasets:` or "
                "`default:`."
            )
        missing = [n for n in names if n not in self.datasets]
        if missing:
            raise ValueError(
                f"`experiments.datasets:` references unknown dataset(s) "
                f"{missing}; available: {sorted(self.datasets)}."
            )
        for ds_name, ds in self.datasets.items():
            if ds.agent is None:
                continue
            unknown = [
                s for s in ds.agent.mcp_servers if s not in self.mcp_servers
            ]
            if unknown:
                available = sorted(self.mcp_servers) or ["(empty registry)"]
                raise ValueError(
                    f"dataset {ds_name!r}: `agent.mcp_servers:` references "
                    f"unknown server(s) {unknown}. Declared in top-level "
                    f"`mcp_servers:` registry: {available}."
                )
        return self

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        """Parse + validate a YAML config file."""
        path = Path(path)
        with path.open() as f:
            raw = yaml.safe_load(f) or {}
        try:
            return cls.model_validate(raw)
        except ValidationError as e:
            raise ValueError(f"{path}: invalid config:\n{e}") from e

    # -- accessors -----------------------------------------------------------

    def selected_dataset_names(self) -> list[str]:
        """Datasets the sweep should run, in declaration order."""
        if self.experiments.datasets:
            return list(self.experiments.datasets)
        assert self.default is not None  # enforced by _validate_selection_and_refs
        return [self.default]

    def generator_kwargs(self, ds_name: str) -> dict[str, Any]:
        return dict(self.datasets[ds_name].generator or {})

    def resolved_agent(self, ds_name: str) -> dict[str, Any] | None:
        """Per-dataset `agent:` block with `mcp_servers:` resolved against the
        top-level registry into `{name: connection_dict}` form (the shape
        `synalinks.MultiServerMCPClient(connections=...)` expects). Returns
        `None` when the dataset uses a generator instead.
        """
        agent = self.datasets[ds_name].agent
        if agent is None:
            return None
        cfg: dict[str, Any] = {"type": agent.type}
        cfg["mcp_servers"] = {
            n: dict(self.mcp_servers[n]) for n in agent.mcp_servers
        }
        cfg.update(dict(agent.__pydantic_extra__ or {}))
        return cfg

    def reward_spec(self, ds_name: str) -> dict[str, Any]:
        """Reward spec ready for `src.rewards.get(...)` — `direction:` stripped."""
        return self.datasets[ds_name].reward.spec()

    def reward_direction(self, ds_name: str) -> Direction:
        return self.datasets[ds_name].reward.direction

    def dataset_metrics(self, ds_name: str) -> list[MetricEntry]:
        """Per-dataset metric entries (NOT merged with `Config.metrics:`)."""
        return list(self.datasets[ds_name].metrics)
