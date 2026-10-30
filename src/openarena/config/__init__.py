# License Apache 2.0: (c) 2026 Atena-Reply

"""Pydantic validation for the openarena YAML config.

A config has the shape of a workspace's `configs/`: the registries, each
keyed by name, and the one evaluation to run, which names what it needs by
those keys.

    evaluation:                      # what to sweep, by key
      name: math-weekly
      language_models: [ministral, qwen]      # keys of `models:`
      datasets: [gsm8k]                        # keys of `datasets:`
      batch_size: 8                            # optional; metrics, seed and
                                               # default_*_model too
    models:                          # name -> the litellm id it calls, its type, its params
      ministral: {model: openrouter/mistralai/ministral-14b-2512, type: llm, params: {}}
      judge: {model: openai/gpt-4o}
    rewards:                         # name -> a catalog reward, its kwargs, its direction
      graded: {type: lm_as_judge, direction: max, config: {language_model: judge}}
    datasets:                        # name -> provider `type` + `config` (its Jinja templates), one kind
      gsm8k: {type: huggingface, config: {...}, reward: graded}
    mcp_servers: {...}               # name -> connection, what `agent.mcp_servers` names
    a2a_agents: {...}                # name -> address, what an `a2a` sweep's axis names

Every entry is read with its registry's strict model (`entries`), the one
the app's registries and `openarena check` use: a field no model
knows is refused. References resolve here, not in the file: a dataset's
`reward:` is a key of `rewards:`; a model named by the sweep axis, a reward
or a metric (`language_model`, `embedding_model`, `panel_language_models`,
`smart_language_model`, `sub_language_model`, the `default_*_model`s) is a
key of `models:`.

`Config.load(path)` is the single entry point; its accessors hand the sweep
everything resolved (`language_models`, `model_id`, `lm_kwargs`,
`dataset_entry`, `resolved_agent`, `reward_spec`, `reward_direction`,
`dataset_metrics`, `metrics`). Pydantic raises on structural and reference
problems before the first trial starts; instantiating framework objects
from the validated specs stays in `evaluate.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from openarena.config.entries import (
    ARENA_KEYS as ARENA_KEYS,
    ENTRIES as ENTRIES,
    KIND_BLOCKS as KIND_BLOCKS,
    NAME_PATTERN as NAME_PATTERN,
    TRANSPORT_FIELDS as TRANSPORT_FIELDS,
    A2aAgentEntry as A2aAgentEntry,
    A2aAgentSpec as A2aAgentSpec,
    AgentBlock as AgentBlock,
    DatasetEntry as DatasetEntry,
    Direction as Direction,
    HarborBlock as HarborBlock,
    EnvironmentEntry as EnvironmentEntry,
    ClockSpec as ClockSpec,
    TaskClock as TaskClock,
    EnvironmentSpec as EnvironmentSpec,
    HealthcheckSpec as HealthcheckSpec,
    EvaluationEntry as EvaluationEntry,
    EvaluationSpec as EvaluationSpec,
    LeaderboardEntry as LeaderboardEntry,
    McpServerEntry as McpServerEntry,
    McpServerSpec as McpServerSpec,
    McpTransport as McpTransport,
    MetricEntry as MetricEntry,
    metric_direction as metric_direction,
    ModelEntry as ModelEntry,
    ModelSpec as ModelSpec,
    Name as Name,
    RewardEntry as RewardEntry,
)
from openarena.config.errors import Invalid as Invalid
from openarena.config.tags import Tag as Tag, Tags as Tags

ModelType = Literal["llm", "embedding", "decision"]

# Kwargs of a reward or a metric that name one model, and the type it must be.
MODEL_KWARGS: dict[str, ModelType] = {
    "language_model": "llm",
    "smart_language_model": "llm",
    "sub_language_model": "llm",
    "embedding_model": "embedding",
    "decision_model": "decision",
}
# Kwargs that name several.
MODEL_LIST_KWARGS: dict[str, ModelType] = {"panel_language_models": "llm"}
# A model type in words, for the messages.
A_MODEL: dict[str, str] = {"llm": "a language model", "embedding": "an embedding model", "decision": "a decision model"}
MODELS_OF: dict[str, str] = {"llm": "language models", "embedding": "embedding models", "decision": "decision models"}


def dataset_kind(entry: dict[str, Any]) -> str:
    """`agent`, `a2a`, `embedding`, `tools`, `harbor` or `model`, read off a dataset entry's blocks."""
    for kind in ("agent", "a2a", "embedding", "tools", "harbor"):
        if entry.get(kind) is not None:
            return kind
    return "model"


def flatten_dataset(entry: dict[str, Any]) -> dict[str, Any]:
    """A registry dataset (`type`, `config`, a kind block, `reward`, `metrics`) as the flat mapping the providers and the sweep read.

    The provider kwargs, the Jinja templates and the schemas come out of
    `config`; what the app keeps beside an entry (`description`, `tags`) is
    dropped.
    """
    flat: dict[str, Any] = {"type": entry["type"], **(entry.get("config") or {})}
    for key in (*KIND_BLOCKS, "reward"):
        if entry.get(key) is not None:
            flat[key] = entry[key]
    if entry.get("metrics"):
        flat["metrics"] = list(entry["metrics"])
    return flat


class EvaluationBlock(EvaluationEntry):
    """`evaluation:` of a run's config: the saved evaluation, and the name it was saved under."""

    name: str | None = None


class Config(BaseModel):
    """The full validated `config.yaml`.

    `Config.load(path)` is the entry point — it parses YAML, validates,
    cross-checks every reference, and returns this object. The accessors
    below hand the sweep what it needs with the references resolved.
    """

    model_config = ConfigDict(extra="forbid")

    evaluation: EvaluationBlock
    models: dict[Name, ModelEntry] = Field(default_factory=dict)
    rewards: dict[Name, RewardEntry] = Field(default_factory=dict)
    datasets: dict[Name, DatasetEntry]
    # What `agent.mcp_servers` names.
    mcp_servers: dict[Name, McpServerEntry] = Field(default_factory=dict)
    # What an `a2a` sweep's axis names.
    a2a_agents: dict[Name, A2aAgentEntry] = Field(default_factory=dict)
    # What a `harbor` dataset names.
    environments: dict[Name, EnvironmentEntry] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_selection_and_refs(self) -> "Config":
        names = self.evaluation.datasets
        missing = [n for n in names if n not in self.datasets]
        if missing:
            raise ValueError(
                f"`evaluation.datasets:` references unknown dataset(s) "
                f"{missing}; available: {sorted(self.datasets)}."
            )
        for ds_name in names:
            if not self.datasets[ds_name].reward:
                raise ValueError(f"dataset {ds_name!r} has no `reward:`; set one before evaluating on it.")
        for ds_name, ds in self.datasets.items():
            if ds.reward is not None and ds.reward not in self.rewards:
                available = sorted(self.rewards) or ["(empty registry)"]
                raise ValueError(
                    f"dataset {ds_name!r}: `reward:` names unknown reward {ds.reward!r}; declared in `rewards:`: {available}."
                )
            if ds.agent is None:
                continue
            unknown = [s for s in ds.agent.mcp_servers if s not in self.mcp_servers]
            if unknown:
                available = sorted(self.mcp_servers) or ["(empty registry)"]
                raise ValueError(
                    f"dataset {ds_name!r}: `agent.mcp_servers:` references "
                    f"unknown server(s) {unknown}. Declared in top-level "
                    f"`mcp_servers:` registry: {available}."
                )
        # An `a2a` sweep's axis is A2A agents, not models, so it runs on its own.
        kinds = {self.datasets[n].kind for n in names}
        if "a2a" in kinds and len(kinds) > 1:
            raise ValueError("an A2A dataset cannot share a run with other kinds of dataset; run it on its own.")
        if "harbor" in kinds and len(kinds) > 1:
            raise ValueError("a Harbor dataset cannot share a run with other kinds of dataset; run it on its own.")
        for ds_name in names:
            harbor = self.datasets[ds_name].harbor
            if harbor is None:
                continue
            if harbor.environment is None:
                # Task folders: each brings its environment; the agent must still drive the axis's models.
                from openarena.harbor import agent_problem

                for name in self.evaluation.language_models:
                    if name in self.models and (problem := agent_problem(harbor.agent, self.models[name].model, None)):
                        raise ValueError(f"dataset {ds_name!r}, model {name!r}: {problem}; pick another agent or model.")
                continue
            environment = self.environments.get(harbor.environment)
            if environment is None:
                available = sorted(self.environments) or ["(empty registry)"]
                raise ValueError(
                    f"dataset {ds_name!r}: `harbor.environment:` names unknown environment {harbor.environment!r}; declared in `environments:`: {available}."
                )
            unknown = [s for s in environment.mcp_servers if s not in self.mcp_servers]
            if unknown:
                raise ValueError(f"environment {harbor.environment!r} names unknown MCP server(s) {unknown}; declare them in `mcp_servers:`.")
            from openarena.harbor import task_files

            try:
                task_files(environment, {n: self.mcp_servers[n].connection() for n in environment.mcp_servers})
            except Invalid as e:
                raise ValueError(f"environment {harbor.environment!r}: {e}") from None
            # The agent drives every model of the axis: one it cannot call fails each row.
            from openarena.harbor import agent_problem

            for name in self.evaluation.language_models:
                if name in self.models and (problem := agent_problem(harbor.agent, self.models[name].model, environment)):
                    raise ValueError(f"dataset {ds_name!r}, model {name!r}: {problem}; pick another agent or model.")
        if "embedding" in kinds and len(kinds) > 1:
            raise ValueError("an embedding dataset cannot share a run with other kinds of dataset; run it on its own.")
        if kinds == {"a2a"}:
            for name in self.evaluation.language_models:
                if name not in self.a2a_agents:
                    available = sorted(self.a2a_agents) or ["(empty registry)"]
                    raise ValueError(f"`evaluation.language_models:` {name!r} is not in `a2a_agents:`; declared: {available}.")
        else:
            # The axis is swept over one type of model: embedding datasets over
            # embedding models, every other kind over LLMs.
            wanted = "embedding" if kinds == {"embedding"} else "llm"
            for name in self.evaluation.language_models:
                entry = self.models.get(name)
                if entry is None:
                    raise ValueError(f"`evaluation.language_models:` {name!r} is not in `models:`; declared: {sorted(self.models) or ['(empty registry)']}.")
                if entry.type != wanted:
                    raise ValueError(f"`evaluation.language_models:` {name!r} is {A_MODEL[entry.type]}; this sweep needs {MODELS_OF[wanted]}.")
        # Every model a reward or a metric names that is a key must be of the right type.
        for where, block in self._model_blocks():
            self._resolve(block, where)
        return self

    def _model_blocks(self):
        """`(where, kwargs)` of every reward and metric, the places a model is named."""
        for name, reward in self.rewards.items():
            yield f"rewards.{name}.config", reward.config
        for name, ds in self.datasets.items():
            for m in ds.metrics:
                yield f"datasets.{name}.metrics", MetricEntry.model_validate(m).kwargs
        for m in self.evaluation.metrics:
            yield "evaluation.metrics", MetricEntry.model_validate(m).kwargs
        yield "evaluation", {
            "language_model": self.evaluation.default_language_model,
            "embedding_model": self.evaluation.default_embedding_model,
        }

    def _id(self, value: Any, wanted: ModelType, where: str) -> Any:
        """The model a reward, a metric or the evaluation names by key: its id, or `{model, **params}` when its entry has params.

        `rewards.get` (and `rewards.build_models`) builds the mapping with
        those params, as the models under test are built with `lm_kwargs`.
        """
        if not isinstance(value, str) or not value:
            return value  # unset: the default model
        entry = self.models.get(value)
        if entry is None:
            raise ValueError(f"{where}: {value!r} is not in `models:`; declared: {sorted(self.models) or ['(empty registry)']}.")
        if entry.type != wanted:
            raise ValueError(f"{where}: {value!r} is {A_MODEL[entry.type]}; this setting needs {A_MODEL[wanted]}.")
        params = self.lm_kwargs(value)
        return {"model": entry.model, **params} if params else entry.model

    def _resolve(self, kwargs: dict[str, Any], where: str) -> dict[str, Any]:
        """`kwargs` with every model it names by key turned into that entry's id, or `{model, **params}` (`_id`)."""
        out = dict(kwargs)
        for key, wanted in MODEL_KWARGS.items():
            if key in out:
                out[key] = self._id(out[key], wanted, f"{where}.{key}")
        for key, wanted in MODEL_LIST_KWARGS.items():
            if isinstance(out.get(key), list):
                out[key] = [self._id(v, wanted, f"{where}.{key}") for v in out[key]]
        return out

    def _resolved_metric(self, entry: MetricEntry, where: str) -> MetricEntry:
        if not entry.kwargs:
            return entry
        return MetricEntry.model_validate(
            {
                "class": entry.class_name,
                "alias": entry.alias,
                "direction": entry.direction,
                "objective": entry.objective,
                **self._resolve(entry.kwargs, where),
            }
        )

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        """Parse + validate a YAML config file, each `${NAME}` in it replaced by the environment variable (`files.expand`)."""
        from openarena.config.files import expand

        path = Path(path)
        with path.open() as f:
            raw = yaml.safe_load(f) or {}
        try:
            return cls.model_validate(expand(raw))
        except ValidationError as e:
            raise ValueError(f"{path}: invalid config:\n{e}") from e

    # -- accessors -----------------------------------------------------------

    @property
    def language_models(self) -> list[str]:
        """The sweep axis, by name."""
        return list(self.evaluation.language_models)

    @property
    def seed(self) -> int | None:
        return self.evaluation.seed

    @property
    def batch_size(self) -> int | None:
        return self.evaluation.batch_size

    @property
    def default_language_model(self) -> Any:
        """The evaluation's default language model: its id, or the model built with its entry's params."""
        from openarena.rewards import build_models

        return build_models({"language_model": self._id(self.evaluation.default_language_model, "llm", "evaluation.default_language_model")})["language_model"]

    @property
    def default_embedding_model(self) -> Any:
        """The evaluation's default embedding model: its id, or the model built with its entry's params."""
        from openarena.rewards import build_models

        return build_models({"embedding_model": self._id(self.evaluation.default_embedding_model, "embedding", "evaluation.default_embedding_model")})["embedding_model"]

    @property
    def metrics(self) -> list[MetricEntry]:
        """The evaluation's metrics, added to every dataset's, their models resolved."""
        return [self._resolved_metric(MetricEntry.model_validate(m), "evaluation.metrics") for m in self.evaluation.metrics]

    def selected_dataset_names(self) -> list[str]:
        """Datasets the sweep should run, in the evaluation's order."""
        return list(self.evaluation.datasets)

    def model_id(self, name: str) -> str:
        """The litellm id an entry of the sweep axis calls: its `models.<name>.model`; an `a2a` sweep's axis is agent names, kept as they are."""
        entry = self.models.get(name)
        return entry.model if entry is not None else name

    def lm_kwargs(self, name: str) -> dict[str, Any]:
        """Keyword arguments for `LanguageModel(model=self.model_id(name), ...)`, or `EmbeddingModel(...)` for an embedding model.

        Empty for a model with no params, in which case the program builders
        hand the framework the bare id.
        """
        entry = self.models.get(name)
        if entry is None:
            return {}
        return {k: v for k, v in entry.params.items() if v is not None}

    def dataset_entry(self, ds_name: str) -> dict[str, Any]:
        """The flat mapping the dataset providers read (`datasets.load_dataset_from_config`)."""
        return self.datasets[ds_name].to_entry()

    def generator_kwargs(self, ds_name: str) -> dict[str, Any]:
        return dict(self.datasets[ds_name].generator or {})

    def resolved_agent(self, ds_name: str) -> dict[str, Any] | None:
        """Per-dataset `agent:` block with `mcp_servers:` resolved against the
        top-level registry into `{name: connection_dict}` form (the shape
        `MultiServerMCPClient(connections=...)` expects). Returns
        `None` when the dataset uses a generator instead.
        """
        agent = self.datasets[ds_name].agent
        if agent is None:
            return None
        cfg: dict[str, Any] = {"type": agent.type}
        cfg["mcp_servers"] = {n: self.mcp_servers[n].connection() for n in agent.mcp_servers}
        cfg.update(dict(agent.__pydantic_extra__ or {}))
        return cfg

    def dataset_kind(self, ds_name: str) -> str:
        return self.datasets[ds_name].kind

    def harbor_spec(self, ds_name: str) -> dict[str, Any]:
        """A Harbor dataset's run: its block, and, over rows, its environment's entry and the MCP connections that environment names.

        Over task folders (`harbor_tasks`), `tasks` is set and there is no environment: each task brings its own, and its own tests.
        """
        block = self.datasets[ds_name].harbor
        if block.environment is None:
            return {**block.model_dump(), "tasks": True}
        environment = self.environments[block.environment]
        return {
            **block.model_dump(),
            "environment_entry": environment.model_dump(mode="json"),
            "mcp_servers": {n: self.mcp_servers[n].connection() for n in environment.mcp_servers},
        }

    def a2a_connection(self, name: str) -> dict[str, Any]:
        """Where the A2A agent `name` on an `a2a` sweep's axis is: `{url, headers, card_path, timeout_sec}`."""
        return self.a2a_agents[name].connection()

    def embedding_kwargs(self, ds_name: str) -> dict[str, Any]:
        return dict(self.datasets[ds_name].embedding or {})

    def tools_kwargs(self, ds_name: str) -> dict[str, Any]:
        return dict(self.datasets[ds_name].tools or {})

    def reward_spec(self, ds_name: str) -> dict[str, Any]:
        """Reward spec ready for `rewards.get(...)`: the dataset's reward entry, `direction:` stripped, models resolved.

        An MCP agent answers with the schema's fields *and* its `messages`, the
        tool-calling transcript, which a target never carries; so a reward
        that sets no mask of its own gets `out_mask: ["messages"]`, lest a
        reward comparing whole objects (`exact_match`) score every row 0. A
        Harbor trial answers with its whole ATIF trajectory, so such a reward
        reads its `answer` alone (`in_mask: ["answer"]`); the verifier's
        reward (`SELF_SCORED`) reads the trajectory's `rewards`. One that names
        masks is left as written, so a judge can be pointed at the steps on
        purpose.
        """
        ds = self.datasets[ds_name]
        reward = self.rewards[ds.reward]
        spec = self._resolve({"name": reward.type, **reward.config}, f"rewards.{ds.reward}.config")
        masks = ("in_mask", "out_mask", "in_mask_pattern", "out_mask_pattern")
        if not any(spec.get(k) for k in masks):
            from openarena.rewards import SELF_SCORED

            if ds.kind == "agent":
                spec["out_mask"] = ["messages"]
            # A Harbor trial's prediction is its whole trajectory: a reward reads its `answer`, the verifier's its `rewards`.
            elif ds.kind == "harbor" and reward.type not in SELF_SCORED:
                spec["in_mask"] = ["answer"]
        return spec

    def reward_direction(self, ds_name: str) -> Direction:
        return self.rewards[self.datasets[ds_name].reward].direction

    def dataset_metrics(self, ds_name: str) -> list[MetricEntry]:
        """Per-dataset metric entries (NOT merged with the evaluation's), their models resolved."""
        return [self._resolved_metric(MetricEntry.model_validate(m), f"datasets.{ds_name}.metrics") for m in self.datasets[ds_name].metrics]


# The checks read the schema above, so they come after it. `check_registry` is
# the one the backend calls on a registry it holds; `check_file` the CLI's.
from openarena.config.check import (  # noqa: E402
    Problem as Problem,
    Report as Report,
    check_file as check_file,
    check_folder as check_folder,
    check_registry as check_registry,
)
