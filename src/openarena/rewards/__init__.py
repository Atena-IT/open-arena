# License Apache 2.0: (c) 2026 Atena-Reply

"""Reward registry + YAML resolver.

Combines the framework's built-in `Reward` subclasses (auto-discovered via
the framework's reward registry) with the project-local rewards the framework
has no equivalent of (`MultiJudgePanel`, `AnswerAccuracy`, `RetrievalRecall`,
`ToolCallsAccuracy`) into a
single `_REWARD_TYPES` map keyed by the reward's snake_case class name. The
rubrics, agent, recursive and deep judges that used to live here ship with
the framework since 0.10. `get(spec)` instantiates a
reward from either a string name (`"exact_match"`) or a YAML-shaped dict;
string `language_model:` / `embedding_model:` values flow straight into
the reward constructor — the framework resolves them via
`language_models.get` / `embedding_models.get` — and a `{model, ...params}`
mapping (a models.yaml entry with params) is built into that model first.
"""

import inspect

import synalinks
from synalinks.src import ops, tree
from synalinks.src.rewards import ALL_OBJECTS as _SYNALINKS_REWARDS
from synalinks.src.rewards.batch_reward import BatchReward, BatchRewardFunctionWrapper
from synalinks.src.rewards.reward_wrappers import RewardFunctionWrapper, ProgramAsJudge
from synalinks.src.utils.naming import to_snake_case

from openarena.rewards.answer_accuracy import AnswerAccuracy
from openarena.rewards.multi_judge_panel import MultiJudgePanel
from openarena.rewards.retrieval_recall import RetrievalRecall
from openarena.rewards.harbor_verifier import HarborVerifier
from openarena.rewards.tool_calls_accuracy import ToolCallsAccuracy

_BASES = (
    synalinks.rewards.Reward,
    RewardFunctionWrapper,
    BatchReward,
    BatchRewardFunctionWrapper,
    ProgramAsJudge,
)

# Project-local reward classes that should also be selectable from YAML by
# their snake_case class name.
_LOCAL_REWARDS = (
    MultiJudgePanel,
    AnswerAccuracy,
    RetrievalRecall,
    ToolCallsAccuracy,
    HarborVerifier,
)



def _is_yaml_instantiable(cls):
    """Concrete `Reward` subclass whose `__init__` needs no runtime objects.

    Excludes bases (`Reward`, `RewardFunctionWrapper`) and judge/wrapper
    classes like `ProgramAsJudge` whose `__init__` requires a `program`
    instance — those can't be configured purely from YAML.
    """
    if not issubclass(cls, synalinks.rewards.Reward) or cls in _BASES:
        return False
    sig = inspect.signature(cls.__init__)
    for name, param in sig.parameters.items():
        if name == "self" or param.kind == inspect.Parameter.VAR_KEYWORD:
            continue
        if param.default is inspect.Parameter.empty:
            return False
    return True


_REWARD_TYPES = {
    to_snake_case(cls.__name__): cls
    for cls in (*_SYNALINKS_REWARDS, *_LOCAL_REWARDS)
    if _is_yaml_instantiable(cls)
}


# Kwargs of a reward's constructor no entry sets, and no form shows: the
# judge's prompt engineering (its template, examples, final instructions,
# score type), the agent runtime (tools, skills, sandbox, working directory),
# and the fine budgets behind it. A reward built from an entry gets the
# engine's defaults for all of these.
HIDDEN_KWARGS = {
    "name",
    "in_mask",
    "out_mask",
    "in_mask_pattern",
    "out_mask_pattern",
    "program",
    "reduction",
    # prompt engineering
    "prompt_template",
    "examples",
    "final_instructions",
    "score_type",
    # agent runtime
    "tools",
    "native_tools",
    "skills",
    "sandbox",
    "sandbox_type",
    "workdir",
    "reset_sandbox",
    "recursive",
    "max_subagent_depth",
    # fine budgets and metric internals
    "max_llm_calls",
    "max_output_chars",
    "axis",
    "schema",
}
# Kwargs every reward takes that an entry may set although no form lists
# them: which fields to compare, and how per-sample rewards are reduced.
MASK_KWARGS = ("in_mask", "out_mask", "in_mask_pattern", "out_mask_pattern", "reduction")


# Rewards that read a score the program itself produced (a Harbor task's verifier), so need no expected answer.
SELF_SCORED = ("harbor_verifier",)


def accepted_kwargs(name: str) -> set[str]:
    """The `config` keys the reward registered as `name` takes: its named constructor kwargs, less the hidden ones, plus the masks."""
    cls = _REWARD_TYPES[name]
    named = {
        key
        for key, param in inspect.signature(cls.__init__).parameters.items()
        if key != "self" and param.kind not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
    }
    return (named - HIDDEN_KWARGS) | set(MASK_KWARGS)


# The constructor kwargs that make a reward call out to a model, by kind.
_JUDGE_MODEL_KWARGS = {"language_model", "panel_language_models", "smart_language_model"}
# Local rewards the class hierarchy cannot place, by name; none today.
_CATEGORY_BY_NAME: dict[str, str] = {}


def needs(cls) -> str | None:
    """Which kind of model a reward class calls out to: `language_model`, `embedding_model` or None."""
    names = set(inspect.signature(cls.__init__).parameters)
    if names & _JUDGE_MODEL_KWARGS:
        return "language_model"
    if "embedding_model" in names:
        return "embedding_model"
    return None


def category(name: str) -> str:
    """How the reward registered as `name` scores: `rubrics_judge`, `agent_judge`, `lm_judge` or `deterministic`.

    A named override, else its base class, else what it calls out to: a
    deterministic reward compares by rule or by embedding, with no judge.
    """
    if name in _CATEGORY_BY_NAME:
        return _CATEGORY_BY_NAME[name]
    cls = _REWARD_TYPES[name]
    if issubclass(cls, synalinks.rewards.RubricsAsJudge):
        return "rubrics_judge"
    if issubclass(cls, (synalinks.rewards.AgentAsJudge, synalinks.rewards.RLMAsJudge, synalinks.rewards.DeepAgentAsJudge)):
        return "agent_judge"
    return "lm_judge" if needs(cls) == "language_model" else "deterministic"


# The spec key naming the answer's fields a reward is scoped to (`AnswerFields`); not a constructor kwarg.
ANSWER_FIELDS = "answer_fields"

# Constructor kwargs that name one model, or several, and the kind of model each builds.
_MODEL_KWARGS = {
    "language_model": "llm",
    "smart_language_model": "llm",
    "sub_language_model": "llm",
    "embedding_model": "embedding",
    # A rubric judge may grade with a decision model instead of its language model.
    "decision_model": "decision",
}
_MODEL_LIST_KWARGS = {"panel_language_models": "llm"}


class AnswerFields(synalinks.rewards.Reward):
    """`reward` scored on the named top-level fields of both sides, each field's value whole.

    What the program returns beside its answer (returned inputs, a chain of
    thought, a message's `content`) stays out of the score. A mask cannot do
    this: `in_mask` keeps the names it lists at every nesting level, so it
    would empty an answer field that holds objects (each tool call, a nested
    answer) and score every prediction the same.
    """

    def __init__(self, reward, fields, name=None):
        super().__init__(name=name or reward.name, reduction=reward.reduction)
        self.reward = reward
        self.fields = list(fields)
        # Where the program's metrics look for the models a reward calls.
        self.rewards = [reward]

    def _keep(self, data):
        if data is None:
            return None
        return ops.convert_to_json_data_model(data).in_mask(mask=self.fields, recursive=False)

    async def __call__(self, y_true, y_pred):
        return await self.reward(tree.map_structure(self._keep, y_true), tree.map_structure(self._keep, y_pred))

    def get_config(self):
        from synalinks.src.saving import serialization_lib

        return {
            "reward": serialization_lib.serialize_synalinks_object(self.reward),
            "fields": self.fields,
            "name": self.name,
        }

    @classmethod
    def from_config(cls, config):
        from synalinks.src.saving import serialization_lib

        config = dict(config)
        config["reward"] = serialization_lib.deserialize_synalinks_object(config["reward"])
        return cls(**config)


def _model(value, kind: str):
    """A `{model, ...params}` mapping built into the model it describes; anything else as it is."""
    if not isinstance(value, dict) or "model" not in value:
        return value
    if kind == "embedding":
        return synalinks.EmbeddingModel(**value)
    if kind == "decision":
        return synalinks.DecisionModel(**value)
    from openarena.programs.utils import language_model

    return language_model(value["model"], {k: v for k, v in value.items() if k != "model"})


def build_models(kwargs: dict) -> dict:
    """A reward's or a metric's kwargs, each model given as `{model, ...params}` built with its params.

    A model named by its bare id is left to the framework, which builds it
    with no params of its own.
    """
    out = dict(kwargs)
    for key, kind in _MODEL_KWARGS.items():
        if key in out:
            out[key] = _model(out[key], kind)
    for key, kind in _MODEL_LIST_KWARGS.items():
        if isinstance(out.get(key), list):
            out[key] = [_model(v, kind) for v in out[key]]
    return out


def get(spec, **kwargs):
    """Resolve a reward spec into an instantiated `rewards.Reward`.

    `spec` can be:
      - A string snake_case reward name, e.g. `"exact_match"`.
      - A dict shaped like the YAML config:
        ```yaml
        name: lm_as_judge
        language_model: ollama/llama3.2
        instructions: |
          ...
        ```
        Every key besides `name` is forwarded as a kwarg to the reward
        constructor. String `language_model:` / `embedding_model:` values
        pass through verbatim — the framework resolves them into concrete
        `LanguageModel` / `EmbeddingModel` instances on demand. A model
        given as a mapping, `{model: openai/gpt-4o, temperature: 0.0}`
        (what `Config` resolves a models.yaml entry with params to), is
        built here with those params. An `answer_fields:` list scopes the
        reward to those top-level fields (`AnswerFields`).

    Args:
        spec: Snake_case name or config dict (see above).
        **kwargs: Extra constructor kwargs, merged on top of any in `spec`.

    Raises:
        KeyError: If no reward is registered under the resolved name.
    """
    if isinstance(spec, dict):
        spec = dict(spec)
        name = spec.pop("name")
        kwargs = {**spec, **kwargs}
    else:
        name = spec

    cls = _REWARD_TYPES.get(name)
    if cls is None:
        known = ", ".join(sorted(_REWARD_TYPES))
        raise KeyError(f"Unknown reward {name!r}. Known: {known}")
    fields = kwargs.pop(ANSWER_FIELDS, None)
    reward = cls(**build_models(kwargs))
    return AnswerFields(reward, fields) if fields else reward
