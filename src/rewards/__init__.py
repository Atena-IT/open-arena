# License Apache 2.0: (c) 2026 Athena-Reply

"""Reward registry + YAML resolver.

Combines synalinks's built-in `Reward` subclasses (auto-discovered via
`synalinks.src.rewards.ALL_OBJECTS`) with project-local rewards
(`JudgePanel`, `RecursiveLMAsJudge`) into a single `_REWARD_TYPES` map
keyed by the reward's snake_case class name. `get(spec)` instantiates a
reward from either a string name (`"exact_match"`) or a YAML-shaped dict;
string `language_model:` / `embedding_model:` values flow straight into
the reward constructor — synalinks resolves them via
`synalinks.language_models.get` / `synalinks.embedding_models.get`.
"""

import inspect

import synalinks
from synalinks.src.rewards import ALL_OBJECTS as _SYNALINKS_REWARDS
from synalinks.src.rewards.reward_wrappers import RewardFunctionWrapper, ProgramAsJudge
from synalinks.src.utils.naming import to_snake_case

from src.rewards.deep_eval import DeepEval
from src.rewards.judge_panel import JudgePanel
from src.rewards.recursive_language_model_reward import RecursiveLMAsJudge

_BASES = (synalinks.rewards.Reward, RewardFunctionWrapper, ProgramAsJudge)

# Project-local reward classes that should also be selectable from YAML by
# their snake_case class name.
_LOCAL_REWARDS = (JudgePanel, RecursiveLMAsJudge, DeepEval)


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


def get(spec, **kwargs):
    """Resolve a reward spec into an instantiated `synalinks.rewards.Reward`.

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
        pass through verbatim — synalinks resolves them into concrete
        `LanguageModel` / `EmbeddingModel` instances on demand.

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
    return cls(**kwargs)
