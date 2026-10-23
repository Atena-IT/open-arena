# License Apache 2.0: (c) 2026 Atena-Reply

"""The strict schema of every registry entry: what a `configs/<registry>.yaml` may hold under a name.

One model per registry, each refusing a field it does not know:
`ModelEntry`, `RewardEntry`, `DatasetEntry`, `McpServerEntry`,
`A2aAgentEntry`, `EnvironmentEntry`, `EvaluationEntry`, `LeaderboardEntry`. The same models
read a run's `config.yaml` (`Config`), back `openarena check`
(`check.py`, which turns their errors into fixes) and are what the app's
API models extend (`data_models`, adding the registry `name` and what the
server works out). A validator that refuses something raises `Invalid`,
carrying how to fix it.

The `*Spec` models are the part of an entry that says how to reach the
thing (a model id and its params, an MCP connection, an A2A agent's address, a
Harbor sandbox, a sweep's grid), which the app also takes alone to check or launch it.
"""

from __future__ import annotations

import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator

from openarena.config.errors import Invalid
from openarena.config.model_types import MODEL_TYPES, Modality, ModelKind, clean_modalities
from openarena.config.tags import Tags

NAME_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.-]*$"
# A registry key: what other entries name this one by.
Name = Annotated[str, Field(pattern=NAME_PATTERN, max_length=128)]

Direction = Literal["max", "min"]

# The blocks that say what answers a dataset's rows; at most one is set.
KIND_BLOCKS = ("generator", "agent", "a2a", "embedding", "tools", "harbor")
# Keys the engine reads at a dataset's top level; they cannot be provider kwargs.
ARENA_KEYS = ("type", *KIND_BLOCKS, "reward", "metrics")

# What each MCP transport reads, per the MCP client's sessions; what a transport does not read is refused.
TRANSPORT_FIELDS: dict[str, tuple[str, ...]] = {
    "stdio": ("command", "args", "env", "cwd"),
    "streamable_http": ("url", "headers"),
    "sse": ("url", "headers"),
    "websocket": ("url",),
}
McpTransport = Literal["stdio", "streamable_http", "sse", "websocket"]

# Keys a reward's block carries itself; they cannot double as constructor kwargs.
RESERVED_REWARD_KEYS = ("name", "direction")


def _blank_is_none(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _clean_args(args: list[str]) -> list[str]:
    return [a for a in (x.strip() for x in args) if a]


def _clean_mapping(mapping: dict[str, str], what: str) -> dict[str, str]:
    cleaned = {k.strip(): v for k, v in mapping.items() if k.strip()}
    for key in cleaned:
        if any(c.isspace() for c in key):
            raise Invalid(f"{what} {key!r} must not contain whitespace", f"rename {key!r} without spaces, e.g. {key.replace(' ', '_')!r}")
    return cleaned


# -- metrics and the agent block ---------------------------------------------

# The measurements where less is better: what the calls spend (tokens, money,
# time) and how often they fail, read off the metric's snake_case name, so the
# reward / embedding / optimizer / per-call variants (`reward_cost`,
# `avg_embedding_latency`, `avg_output_tokens_per_call`) follow. Rates
# (`throughput`, `tokens_per_second`, `cache_hit_rate`) and cached tokens are
# better higher.
_LOWER_IS_BETTER = re.compile(
    r"((?<!cached_)tokens|token_share|cost|latency|time_to_[a-z_]+|elapsed_time|error_rate|failed_calls|fallback_activations)"
    r"(_per_call|_per_invocation)?$"
)


def metric_direction(class_name: str) -> Direction:
    """Which way is better for a metric written without `direction:`: `min` for a cost, tokens or latency measurement, else `max`."""
    snake = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", class_name).lower()
    return "min" if _LOWER_IS_BETTER.search(snake) else "max"


class MetricEntry(BaseModel):
    """One entry of a `metrics:` list (an evaluation's or a dataset's).

    Written as a bare string (`- total_tokens`) or a mapping with `class:`
    plus an optional `alias:` / `direction:` / `objective:` and any
    constructor kwargs; the string is read as `{class: ...}`. Without
    `direction:`, a measurement of what the calls spend (`cost`,
    `total_tokens`, `avg_latency`, ...) is better lower and anything else
    higher (`metric_direction`).
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    class_name: str = Field(alias="class")
    alias: str | None = None
    direction: Direction | None = Field(
        default=None,
        description="Which way is better. Unset, `min` for a cost, tokens or latency measurement, `max` for anything else.",
    )
    objective: bool = False

    @model_validator(mode="before")
    @classmethod
    def _string_shortcut(cls, value: Any) -> Any:
        if isinstance(value, str):
            return {"class": value}
        return value

    @model_validator(mode="after")
    def _default_direction(self) -> MetricEntry:
        if self.direction is None:
            self.direction = metric_direction(self.class_name)
        return self

    @property
    def kwargs(self) -> dict[str, Any]:
        """Extra fields forwarded as constructor kwargs (incl. `name:`)."""
        return dict(self.__pydantic_extra__ or {})


def _metrics(metrics: list[Any]) -> list[Any]:
    """Each metric readable as a `MetricEntry`; kept as written, so a string stays a string."""
    for m in metrics:
        MetricEntry.model_validate(m)
    return metrics


Metrics = list[str | dict[str, Any]]


class AgentBlock(BaseModel):
    """A dataset's `agent:` block: a `FunctionCallingAgent` over the MCP servers it names.

    Every other key (`max_iterations`, `autonomous`, `instructions`, ...)
    flows to `FunctionCallingAgent`.
    """

    model_config = ConfigDict(extra="allow")

    type: Literal["function_calling"] = "function_calling"
    mcp_servers: list[str] = Field(min_length=1, description="Keys of the `mcp_servers` registry the agent calls.")

    @model_validator(mode="after")
    def _no_reserved(self) -> AgentBlock:
        overlap = {"language_model", "tools"} & set(self.__pydantic_extra__ or {})
        if overlap:
            raise Invalid(
                f"`agent:` keys {sorted(overlap)} are set by the run",
                f"remove {', '.join(f'`{k}:`' for k in sorted(overlap))} from `agent:`",
            )
        return self


# Harbor's agents a model under test can drive: those that leave an ATIF trajectory, which is what the dataset's reward scores.
HARBOR_AGENTS = (
    "terminus-2", "claude-code", "codex", "gemini-cli", "goose", "opencode", "openhands",
    "mini-swe-agent", "swe-agent", "cline-cli", "cursor-cli",
)


class HarborBlock(BaseModel):
    """A dataset's `harbor:` block: each row a Harbor task, done by an agent the model under test drives in a sandbox.

    Two sources of tasks. A dataset of `type: harbor_tasks` reads Harbor task
    folders as they are (`task.toml`, `instruction.md`, `environment/`,
    `tests/`, `solution/`), one row each, and runs each with Harbor's verifier:
    its `tests/` score it, and the `harbor_verifier` reward reads that score.
    Any other dataset names an `environment`: each row's `input_template`
    renders its question, which, after `instructions`, is the `instruction.md`
    of a task generated from that environment, whose default `tests/` collects
    the agent's answer for the dataset's reward to score on the arena's side.

    What the agent did is the prediction, Harbor's own ATIF trajectory
    (`HarborTrajectory`), with its `answer`: what it wrote to `answer_file`
    (an agent that works a terminal acts rather than replies), else its last
    message; and, for a task with its own tests, the verifier's `rewards`.
    """

    model_config = ConfigDict(extra="forbid")

    environment: Name | None = Field(
        default=None, description="Key of the environments registry: the sandbox every row's generated task runs in. None for `harbor_tasks`."
    )
    agent: str = Field(default="terminus-2", description=f"The Harbor agent the model drives: {', '.join(HARBOR_AGENTS)}.")
    instructions: str | None = Field(default=None, description="Said to the agent before every row's question.")
    answer_file: str | None = Field(
        default=None,
        description="An absolute path in the sandbox the agent is asked to write its final answer to, read back once it is done; blank scores its last message.",
    )
    timeout_sec: float | None = Field(default=None, gt=0, description="How long the agent may work on one row; blank keeps Harbor's.")

    @field_validator("agent")
    @classmethod
    def _agent(cls, agent: str) -> str:
        if agent not in HARBOR_AGENTS:
            raise Invalid(f"no Harbor agent {agent!r}", f"set `agent:` to one of: {', '.join(HARBOR_AGENTS)}")
        return agent

    @field_validator("instructions")
    @classmethod
    def _blank(cls, value: str | None) -> str | None:
        return _blank_is_none(value)

    @field_validator("answer_file")
    @classmethod
    def _answer_file(cls, value: str | None) -> str | None:
        value = _blank_is_none(value)
        if value is not None and (not value.startswith("/") or ".." in value.split("/") or value.endswith("/")):
            raise Invalid(f"`answer_file` {value!r} is not a file's absolute path in the sandbox", "write it like `/app/answer.txt`")
        return value


# -- models ------------------------------------------------------------------


class ModelSpec(BaseModel):
    """A model id plus how to build it: its type and that type's parameters."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1, description="Model id, e.g. `ollama/llama3.2`.")
    type: ModelKind = Field(default="llm", description="What the model does; `/catalog/model-types` lists the kinds.")
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Settings of this type, forwarded when the model is built; `/catalog/model-types` lists them per type, with their bounds.",
    )

    @model_validator(mode="after")
    def _clean_params(self) -> ModelSpec:
        kind = MODEL_TYPES[self.type]
        try:
            self.params = kind.clean_params(self.params)
        except ValueError as e:
            takes = ", ".join(p.name for p in kind.params) or "no parameters"
            raise Invalid(str(e), f"keep only what a {kind.label.lower()} takes under `params:`: {takes}") from None
        return self


class ModelEntry(ModelSpec):
    """A `models.<name>:` entry."""

    description: str | None = None
    modalities: list[Modality] = Field(
        default_factory=lambda: ["text"],
        description="What the model takes in. `text` alone is a plain text model; add `image` for a vision model.",
    )
    tags: Tags = Field(
        default_factory=list,
        description="Labels for browsing, e.g. `openai`, `local`, `tools`, `128k-context`. Lower-case, de-duplicated; not sent to the model.",
    )

    @field_validator("modalities")
    @classmethod
    def _clean_modalities(cls, modalities: list[str]) -> list[str]:
        return clean_modalities(modalities)


# -- rewards -----------------------------------------------------------------


class RewardEntry(BaseModel):
    """A `rewards.<name>:` entry: a catalog reward (`type`), its constructor kwargs (`config`), which way is better."""

    model_config = ConfigDict(extra="forbid")

    type: str = Field(min_length=1, description="A catalog reward: its snake_case identifier, e.g. `lm_as_judge`.")
    description: str | None = None
    tags: Tags = Field(default_factory=list, description="Labels for browsing, e.g. `math`, `judge`, `strict`.")
    direction: Direction = Field(default="max", description="Which way is better on the leaderboard.")
    config: dict[str, Any] = Field(
        default_factory=dict,
        description="Constructor kwargs for the reward: `language_model`, `instructions`, `in_mask`, ... Anything but `name` and `direction`.",
    )

    @field_validator("config")
    @classmethod
    def _no_reserved(cls, config: dict[str, Any]) -> dict[str, Any]:
        clash = [k for k in RESERVED_REWARD_KEYS if k in config]
        if clash:
            raise Invalid(
                f"config cannot set {', '.join(clash)}: use `type` and `direction`",
                "take those keys out of `config:`; `type:` and `direction:` sit beside `config:`, not in it",
            )
        return config

    @property
    def block(self) -> dict[str, Any]:
        """The spec `rewards.get` builds it from: `{name, direction, ...config}`."""
        return {"name": self.type, "direction": self.direction, **self.config}


# -- datasets ----------------------------------------------------------------

# The provider reading Harbor task folders (`openarena.datasets.harbor_tasks`): its rows are the tasks.
HARBOR_TASKS = "harbor_tasks"

# A tool declaration as a Chat Completions request carries it: what `tools.tool_schemas` and a row's `tools_template` must render.
TOOL_SHAPE = "{type: function, function: {name: <function>, description: ..., parameters: {type: object, properties: {}}}}"


def tool_declaration_problem(tool: Any) -> str | None:
    """Why `tool` is not a Chat Completions tool declaration, or None when it is.

    The declarations go to the provider verbatim, as the request's `tools`,
    so only the wire shape `{"type": "function", "function": {"name", ...}}`
    works: a bare `{name, description, parameters}` is refused by the API,
    and better refused here with the shape to write.
    """
    if not isinstance(tool, dict):
        return f"a tool declaration is a mapping, not {type(tool).__name__}"
    if tool.get("type") != "function" or not isinstance(tool.get("function"), dict):
        return "a tool declaration is `{type: function, function: {...}}`" + (
            ": wrap it as `{type: function, function: <the declaration>}`" if isinstance(tool.get("name"), str) else ""
        )
    if not isinstance(tool["function"].get("name"), str) or not tool["function"]["name"]:
        return "a tool declaration's `function` names the function (`name`)"
    return None


# What a dataset with no question template is shown to fix it.
_TEMPLATES = (
    "config:\n"
    "  input_template: '{\"question\": {{ question | tojson }}}'\n"
    "  output_template: '{\"answer\": {{ answer | string | tojson }}}'"
)


class DatasetEntry(BaseModel):
    """A `datasets.<name>:` entry: a provider (`type`, `config`, with the row's Jinja templates), what answers each row, the reward that scores it."""

    model_config = ConfigDict(extra="forbid")

    type: str = Field(description="Dataset provider, e.g. `local`, `huggingface`.")
    config: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Provider kwargs: `path`, `split`, `limit`, `batch_size`, ..., and the row's Jinja2 templates: "
            "`input_template` (required) and `output_template`, with their `input_schema` / `output_schema`, "
            "and `tools_template`, the row's tool declarations for a `tools:` dataset."
        ),
    )
    description: str | None = None
    tags: Tags = Field(
        default_factory=list,
        description="Labels for browsing, e.g. `math`, `science`, `multiple-choice`. Lower-case, de-duplicated; not seen by the engine.",
    )
    generator: dict[str, Any] | None = Field(default=None, description="Kwargs for `Generator` (single-shot eval).")
    agent: AgentBlock | None = Field(default=None, description="`agent:` block (FunctionCallingAgent + MCP tools).")
    a2a: dict[str, Any] | None = Field(
        default=None,
        description="`a2a: {}`: each row is sent to an A2A agent as one message; the run's axis names the agents (`a2a_agents` keys).",
    )
    embedding: dict[str, Any] | None = Field(
        default=None,
        description="`embedding:` block: retrieval over the row's `chunks` with the embedding model under test; `k` chunks returned.",
    )
    tools: dict[str, Any] | None = Field(
        default=None,
        description=(
            "`tools:` block: one call per row with `tool_schemas` in hand (and the row's own, from `config.tools_template`), "
            "no server behind them — the model's tool calls are the prediction."
        ),
    )
    harbor: HarborBlock | None = Field(
        default=None,
        description="`harbor:` block: each row's question asked of a Harbor agent the model drives in an environment's sandbox.",
    )
    reward: str | None = Field(
        default=None, description="Primary reward: a key of the rewards registry. Required to run an evaluation."
    )
    metrics: Metrics = Field(default_factory=list, description="Extra per-dataset metrics (string or `{class, ...}`).")

    @field_validator("config")
    @classmethod
    def _no_arena_keys(cls, config: dict[str, Any]) -> dict[str, Any]:
        clash = [k for k in ARENA_KEYS if k in config]
        if clash:
            raise Invalid(
                f"`config` cannot contain arena-level keys {clash}",
                f"move {', '.join(f'`{k}:`' for k in clash)} out of `config:` to the dataset's top level",
            )
        return config

    @field_validator("metrics")
    @classmethod
    def _readable_metrics(cls, metrics: list[Any]) -> list[Any]:
        return _metrics(metrics)

    @model_validator(mode="after")
    def _one_kind(self) -> DatasetEntry:
        set_blocks = [k for k in KIND_BLOCKS if getattr(self, k) is not None]
        if len(set_blocks) > 1:
            raise Invalid(
                f"`{'`, `'.join(set_blocks)}` are mutually exclusive",
                "keep one of those blocks: a dataset is answered one way (generator, agent, a2a, embedding, tools or harbor)",
            )
        if self.a2a:
            raise Invalid(
                f"`a2a:` takes no settings, got {sorted(self.a2a)}",
                "write `a2a: {}`; the agents are the evaluation's `language_models`, keys of `a2a_agents`",
            )
        per_row = self.config.get("tools_template") is not None
        if self.tools is not None and not self.tools.get("tool_schemas") and not per_row:
            raise Invalid(
                "`tools.tool_schemas:` lists at least one function declaration, or `config.tools_template` renders each row's",
                f"list the functions under `tools:`:\ntools:\n  tool_schemas:\n    - {TOOL_SHAPE}\n"
                "or render them from the row in `config:`:\n  tools_template: '{{ tools | tojson }}'",
            )
        for tool in (self.tools or {}).get("tool_schemas") or []:
            if (problem := tool_declaration_problem(tool)) is not None:
                raise Invalid(f"`tools.tool_schemas:` {problem}", f"write each entry as {TOOL_SHAPE}")
        if per_row and self.tools is None:
            raise Invalid(
                "`config.tools_template` only offers tools to a `tools:` dataset",
                "add `tools: {}` to hand the model each row's tools, or remove `tools_template:`",
            )
        if self.type == HARBOR_TASKS:
            if self.harbor is None:
                raise Invalid(
                    "a `harbor_tasks` dataset is Harbor task folders: it needs a `harbor:` block",
                    "add `harbor: {agent: terminus-2}`; the tasks' own tests score them, with the `harbor_verifier` reward",
                )
            if self.harbor.environment is not None:
                raise Invalid(
                    "a `harbor_tasks` dataset's tasks bring their own environment",
                    "remove `harbor.environment:`; each task's `environment/` and `task.toml` say how its sandbox is made",
                )
            return self
        if self.harbor is not None and self.harbor.environment is None:
            raise Invalid(
                "a `harbor:` dataset over rows needs the environment its tasks run in",
                "set `harbor.environment:` to a key of environments.yaml, or make it `type: harbor_tasks` over task folders",
            )
        if not isinstance(self.config.get("input_template"), str) or not self.config["input_template"].strip():
            raise Invalid(
                "no `config.input_template`: nothing says what each row asks",
                "write the Jinja2 templates over the row's columns in `config:`:\n" + _TEMPLATES,
            )
        return self

    @computed_field(  # type: ignore[prop-decorator]
        description="What answers each row: `model` (a generator), `agent` (MCP tools), `a2a` (A2A agents), `embedding` (retrieval), `tools` (a model with tool schemas, one call), `harbor` (a task in a sandbox)."
    )
    @property
    def kind(self) -> Literal["model", "agent", "a2a", "embedding", "tools", "harbor"]:
        if self.agent is not None:
            return "agent"
        if self.a2a is not None:
            return "a2a"
        if self.embedding is not None:
            return "embedding"
        if self.tools is not None:
            return "tools"
        if self.harbor is not None:
            return "harbor"
        return "model"

    def to_entry(self) -> dict[str, Any]:
        """The flat mapping the dataset providers read (`config.flatten_dataset`)."""
        from openarena.config import flatten_dataset

        return flatten_dataset(self.model_dump(mode="json", exclude={"kind"}))


# -- MCP servers and A2A agents ----------------------------------------------


class McpServerSpec(BaseModel):
    """One MCP connection: a transport plus what that transport needs."""

    model_config = ConfigDict(extra="forbid")

    transport: McpTransport = Field(default="stdio", description="How to reach the server: spawn a `command`, or connect to a `url`.")
    command: str | None = Field(default=None, description="stdio: the executable to spawn, e.g. `npx` or `python`.")
    args: list[str] = Field(default_factory=list, description="stdio: arguments of `command`, one per item.")
    env: dict[str, str] = Field(default_factory=dict, description="stdio: environment variables set for the process, on top of the backend's.")
    cwd: str | None = Field(default=None, description="stdio: working directory of the process.")
    url: str | None = Field(default=None, description="streamable_http / sse / websocket: the endpoint, e.g. `http://localhost:8080/mcp`.")
    headers: dict[str, str] = Field(
        default_factory=dict, description="streamable_http / sse: HTTP headers sent on every request, e.g. `Authorization`."
    )

    @field_validator("command", "cwd", "url")
    @classmethod
    def _blank(cls, value: str | None) -> str | None:
        return _blank_is_none(value)

    @field_validator("args")
    @classmethod
    def _args(cls, args: list[str]) -> list[str]:
        return _clean_args(args)

    @field_validator("env", "headers")
    @classmethod
    def _clean(cls, mapping: dict[str, str]) -> dict[str, str]:
        return _clean_mapping(mapping, "key")

    @model_validator(mode="after")
    def _fields_match_transport(self) -> McpServerSpec:
        allowed = TRANSPORT_FIELDS[self.transport]
        if self.transport == "stdio":
            if not self.command:
                raise Invalid("a stdio server needs a `command` to spawn", "add `command: <executable>`, e.g. `command: python`")
        elif not self.url:
            raise Invalid(f"a {self.transport} server needs a `url`", "add `url: https://<host>/mcp`")
        elif not self.url.lower().startswith(("http://", "https://", "ws://", "wss://")):
            raise Invalid(f"`url` must start with http://, https://, ws:// or wss://, got {self.url!r}", "write the full address, e.g. `url: http://localhost:8080/mcp`")
        stray = [
            f
            for f, value in (
                ("command", self.command),
                ("args", self.args),
                ("env", self.env),
                ("cwd", self.cwd),
                ("url", self.url),
                ("headers", self.headers),
            )
            if value and f not in allowed
        ]
        if stray:
            readers = sorted({t for t, fields in TRANSPORT_FIELDS.items() for f in stray if f in fields} - {self.transport})
            raise Invalid(
                f"a {self.transport} server does not take {stray}; it reads {list(allowed)}",
                f"remove {', '.join(f'`{f}:`' for f in stray)}, or set `transport:` to one that reads them ({', '.join(readers)})",
            )
        return self

    def connection(self) -> dict[str, Any]:
        """The dict `MultiServerMCPClient(connections={name: ...})` takes: the transport and only the fields it reads."""
        conn: dict[str, Any] = {"transport": self.transport}
        for f in TRANSPORT_FIELDS[self.transport]:
            value = getattr(self, f)
            if value:
                conn[f] = value
        return conn

    @property
    def target(self) -> str:
        """`command arg arg` or the url: the one line that says where the server is."""
        if self.transport == "stdio":
            return " ".join([self.command or "", *self.args]).strip()
        return self.url or ""


class McpServerEntry(McpServerSpec):
    """An `mcp_servers.<name>:` entry: what a dataset's `agent.mcp_servers` names."""

    description: str | None = None
    tags: Tags = Field(default_factory=list, description="Labels for browsing, e.g. `search`, `filesystem`, `local`.")


class A2aAgentSpec(BaseModel):
    """How to reach one A2A agent: its address, the headers it wants, how long a row may take."""

    model_config = ConfigDict(extra="forbid")

    url: str = Field(min_length=1, description="The agent's base URL; its card is read from `<url>/.well-known/agent-card.json`.")
    headers: dict[str, str] = Field(default_factory=dict, description="HTTP headers sent on every request, e.g. `Authorization: Bearer ...`.")
    card_path: str | None = Field(default=None, description="Where the agent card is, relative to `url`, when not at the well-known path.")
    timeout_sec: float = Field(default=300, gt=0, description="How long one row may take, from sending the message to the agent's last word.")

    @field_validator("url", "card_path")
    @classmethod
    def _blank(cls, value: str | None) -> str | None:
        return _blank_is_none(value)

    @field_validator("url")
    @classmethod
    def _http(cls, url: str) -> str:
        if not url.lower().startswith(("http://", "https://")):
            raise Invalid(f"`url` must start with http:// or https://, got {url!r}", "write the full address, e.g. `url: http://localhost:9999`")
        return url.rstrip("/")

    @field_validator("headers")
    @classmethod
    def _clean_headers(cls, headers: dict[str, str]) -> dict[str, str]:
        return _clean_mapping(headers, "header")

    def connection(self) -> dict[str, Any]:
        """What `programs.a2a_program.A2aAgent` and the check reach: `{url, headers, card_path, timeout_sec}`."""
        return {"url": self.url, "headers": dict(self.headers), "card_path": self.card_path, "timeout_sec": self.timeout_sec}

    @property
    def target(self) -> str:
        """The url: the one line that says where the agent is."""
        return self.url


class A2aAgentEntry(A2aAgentSpec):
    """An `a2a_agents.<name>:` entry: what an `a2a` sweep's axis names."""

    description: str | None = None
    tags: Tags = Field(default_factory=list, description="Labels for browsing, e.g. `coding`, `remote`.")


# -- Harbor environments -------------------------------------------------------

NetworkMode = Literal["public", "no-network", "allowlist"]


class HealthcheckSpec(BaseModel):
    """A command Harbor runs in the container until it succeeds: `[environment.healthcheck]`."""

    model_config = ConfigDict(extra="forbid")

    command: str = Field(min_length=1, description="Run inside the container; exit 0 means ready.")
    interval_sec: float = Field(default=5, gt=0)
    timeout_sec: float = Field(default=30, gt=0)
    start_period_sec: float = Field(default=0, ge=0)
    start_interval_sec: float = Field(default=5, gt=0)
    retries: int = Field(default=3, ge=0)

    @field_validator("command")
    @classmethod
    def _blank(cls, value: str) -> str:
        return value.strip()


class EnvironmentSpec(BaseModel):
    """The sandbox a Harbor task's agent works in: its `[environment]` and its `environment/Dockerfile`.

    Either a prebuilt `docker_image` or a `dockerfile` to build, never both.
    `mcp_servers` are keys of the MCP servers registry, written into the
    task as `[[environment.mcp_servers]]`.
    """

    model_config = ConfigDict(extra="forbid")

    docker_image: str | None = Field(default=None, description="A prebuilt image, e.g. `python:3.12-slim`; the task then needs no `environment/` folder.")
    dockerfile: str | None = Field(default=None, description="The `environment/Dockerfile` Harbor builds, when no `docker_image` is given.")
    os: Literal["linux", "windows"] = "linux"
    cpus: int | None = Field(default=None, ge=1, description="CPUs the sandbox gets; blank leaves it to the provider.")
    memory_mb: int | None = Field(default=None, ge=1, description="Memory, in MB.")
    storage_mb: int | None = Field(default=None, ge=1, description="Disk, in MB.")
    gpus: int | None = Field(default=None, ge=0, description="GPUs the sandbox gets.")
    gpu_types: list[str] = Field(default_factory=list, description="Accepted GPU types, e.g. `H100`; empty accepts any.")
    network_mode: NetworkMode = Field(default="public", description="`public`, `no-network`, or `allowlist` (only `allowed_hosts`).")
    allowed_hosts: list[str] = Field(default_factory=list, description="The hosts an `allowlist` sandbox may reach.")
    env: dict[str, str] = Field(default_factory=dict, description="Environment variables set in the container.")
    workdir: str | None = Field(default=None, description="The directory the agent starts in, e.g. `/app`.")
    build_timeout_sec: float = Field(default=600, gt=0, description="How long building the image may take.")
    healthcheck: HealthcheckSpec | None = None
    mcp_servers: list[Name] = Field(default_factory=list, description="MCP servers the agent is given, by their key in the MCP servers registry.")

    @field_validator("docker_image", "dockerfile", "workdir")
    @classmethod
    def _blank(cls, value: str | None) -> str | None:
        return _blank_is_none(value)

    @field_validator("gpu_types", "allowed_hosts")
    @classmethod
    def _items(cls, items: list[str]) -> list[str]:
        return _clean_args(items)

    @field_validator("mcp_servers")
    @classmethod
    def _unique(cls, names: list[str]) -> list[str]:
        return list(dict.fromkeys(names))

    @field_validator("env")
    @classmethod
    def _clean_env(cls, env: dict[str, str]) -> dict[str, str]:
        return _clean_mapping(env, "variable")

    @model_validator(mode="after")
    def _consistent(self) -> EnvironmentSpec:
        if self.docker_image and self.dockerfile:
            raise Invalid("an environment is a `docker_image` or a `dockerfile`, not both", "remove one of `docker_image:` and `dockerfile:`")
        if not self.docker_image and not self.dockerfile:
            raise Invalid(
                "an environment needs an image",
                "add `docker_image: python:3.12-slim` (prebuilt), or `dockerfile: |` with the Dockerfile to build",
            )
        if self.network_mode == "allowlist" and not self.allowed_hosts:
            raise Invalid("an `allowlist` network lets nothing through without `allowed_hosts`", "add `allowed_hosts: [pypi.org]`, or set `network_mode: public`")
        if self.network_mode != "allowlist" and self.allowed_hosts:
            raise Invalid(f"`allowed_hosts` is only read with `network_mode: allowlist`, not {self.network_mode!r}", "set `network_mode: allowlist`, or remove `allowed_hosts:`")
        return self

    @property
    def target(self) -> str:
        """The image, or `Dockerfile (FROM <base>)`: the one line that says what runs."""
        if self.docker_image:
            return self.docker_image
        base = next((line.split()[1] for line in (self.dockerfile or "").splitlines() if line.strip().upper().startswith("FROM ") and len(line.split()) > 1), None)
        return f"Dockerfile (FROM {base})" if base else "Dockerfile"


class EnvironmentEntry(EnvironmentSpec):
    """An `environments.<name>:` entry: a Harbor task environment."""

    description: str | None = None
    tags: Tags = Field(default_factory=list, description="Labels for browsing, e.g. `python`, `gpu`, `offline`.")


# -- evaluations and leaderboards --------------------------------------------


class EvaluationSpec(BaseModel):
    """What an evaluation sweeps: the grid and the knobs every run of it uses."""

    model_config = ConfigDict(extra="forbid")

    language_models: list[str] = Field(
        min_length=1,
        description="Models to compare, by their key in the models registry. An `a2a` sweep compares agents instead: keys of the A2A agents registry.",
    )
    datasets: list[str] = Field(min_length=1, description="Keys of the datasets registry to evaluate on.")
    metrics: Metrics = Field(default_factory=list, description="Metrics added to every dataset (string or `{class, ...}`).")
    batch_size: int | None = Field(
        default=None,
        ge=1,
        le=1024,
        description=(
            "Rows per batch for every dataset in this sweep, overriding each dataset's own `batch_size`. A batch "
            "is evaluated as one step, so this is the concurrency knob. None keeps each dataset's."
        ),
    )
    seed: int | None = None
    default_language_model: str | None = None
    default_embedding_model: str | None = None

    @field_validator("metrics")
    @classmethod
    def _readable_metrics(cls, metrics: list[Any]) -> list[Any]:
        return _metrics(metrics)


class EvaluationEntry(EvaluationSpec):
    """An `evaluations.<name>:` entry: a saved sweep."""

    description: str | None = None
    tags: Tags = Field(default_factory=list, description="Labels for browsing (`math`, `weekly`).")


class Budget(BaseModel):
    """A hard limit on one metric of a leaderboard: a model breaking it is left out of the ranking, and listed as such."""

    model_config = ConfigDict(extra="forbid")

    metric: str = Field(min_length=1, description="A metric of the board, e.g. `cost`, `avg_latency`, `reward`.")
    op: Literal["lte", "gte"] = Field(description="`lte`: at most `value`; `gte`: at least `value`.")
    value: float


def check_shares(shares: dict[str, float]) -> dict[str, float]:
    """A leaderboard's shares as the app takes them: each 0 or more, at least one above 0 (or none, the default)."""
    negative = [m for m, v in shares.items() if v < 0]
    if negative:
        raise Invalid(f"a share is a percent of the score, 0 or more: {negative}", f"set {', '.join(f'`{m}:`' for m in negative)} to 0 or more")
    if shares and not any(v > 0 for v in shares.values()):
        raise Invalid(
            "no metric carries weight",
            "give at least one metric a share above 0 (`shares: {reward: 100}`), or remove `shares:` for the default",
        )
    return shares


def check_weights(weights: dict[str, float]) -> dict[str, float]:
    """Group or dataset weights: each 0 or more. All at 0 is fine, the ones left out still weigh 1."""
    negative = [k for k, v in weights.items() if v < 0]
    if negative:
        raise Invalid(f"a weight is 0 or more: {negative}", f"set {', '.join(f'`{k}:`' for k in negative)} to 0 or more")
    return weights


class LeaderboardEntry(BaseModel):
    """A `leaderboards.<name>:` entry: a scope, and each metric's share of the score.

    Empty `datasets` / `models` mean every one scored; empty `shares` the
    default preset (`leaderboard.default_shares`).
    """

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=200, description="Shown on the card and the page.")
    description: str | None = None
    tags: Tags = Field(default_factory=list, description="Labels for browsing: the domain (`math`), the lens (`reward-first`), the purpose (`weekly`).")
    datasets: list[str] = Field(default_factory=list, description="Dataset names in scope. Empty means every dataset with scores.")
    models: list[str] = Field(default_factory=list, description="Models in scope, by their registry name. Empty means every scored model.")
    shares: dict[str, float] = Field(default_factory=dict, description="Percent of the score per metric. Empty means the default preset.")
    budgets: list[Budget] = Field(default_factory=list, description="Hard limits, e.g. cost at most 0.05: a model breaking one is left out of the ranking.")
    group_shares: dict[str, float] = Field(default_factory=dict, description="Weight per group of datasets (an import's suites); a group left out weighs 1.")
    dataset_shares: dict[str, float] = Field(default_factory=dict, description="Weight per dataset within its group; a dataset left out weighs 1, at 0 it is out of the score.")

    @field_validator("shares")
    @classmethod
    def _shares(cls, shares: dict[str, float]) -> dict[str, float]:
        return check_shares(shares)

    @field_validator("group_shares", "dataset_shares")
    @classmethod
    def _weights(cls, weights: dict[str, float]) -> dict[str, float]:
        return check_weights(weights)


# Each registry's entry model, by section.
ENTRIES: dict[str, type[BaseModel]] = {
    "models": ModelEntry,
    "rewards": RewardEntry,
    "datasets": DatasetEntry,
    "mcp_servers": McpServerEntry,
    "a2a_agents": A2aAgentEntry,
    "environments": EnvironmentEntry,
    "evaluations": EvaluationEntry,
    "leaderboards": LeaderboardEntry,
}
