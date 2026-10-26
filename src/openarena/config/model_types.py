# License Apache 2.0: (c) 2026 Atena-Reply

"""The kinds of model the registry knows, defined once.

A *model type* says what a model does — `llm` answers, `embedding` encodes,
`decision` grades —
and with it: which rows of litellm's table list its ids (`litellm_mode`),
which parameters the arena forwards when it builds the framework object, and
whether the sweep engine can evaluate it yet. Everything else reads from
here: the registry validation in `openarena.config.entries`, `/catalog/model-types`
(which the add-model form renders from), the per-type provider lists, and the
health check's dispatch. Adding a type is one more `ModelType` below plus a
checker in `services.model_check`.

Modalities are orthogonal to the type — a vision LLM and a CLIP-style
embedding model both take `image` — so they are a separate set on the entry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

MODALITIES = ("text", "image", "audio")
Modality = Literal[*MODALITIES]


@dataclass(frozen=True)
class Param:
    """One parameter the form offers for a type, and how to validate it."""

    name: str
    label: str
    kind: Literal["number", "integer", "choice"]
    hint: str = ""
    choices: tuple[str, ...] = ()
    min: float | None = None
    max: float | None = None
    # How the form names each choice ("low" -> "Light"); a choice left out
    # shows as its value. Only the words change: the stored value is the choice.
    labels: dict[str, str] = field(default_factory=dict)

    def clean(self, value: Any) -> Any:
        """`value` as stored, or `ValueError` saying what is wrong with it."""
        if self.kind == "choice":
            if value not in self.choices:
                raise ValueError(f"{self.name}: one of {', '.join(self.choices)}")
            return value
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise ValueError(f"{self.name}: a number")
        try:
            if self.kind == "integer":
                if isinstance(value, float) and not value.is_integer():
                    raise ValueError
                number: int | float = int(value)
            else:
                number = float(value)
        except ValueError:
            kind = "a whole number" if self.kind == "integer" else "a number"
            raise ValueError(f"{self.name}: {kind}") from None
        if (self.min is not None and number < self.min) or (
            self.max is not None and number > self.max
        ):
            low = "" if self.min is None else f"{self.min:g}"
            high = "" if self.max is None else f"{self.max:g}"
            bounds = (
                f"between {low} and {high}"
                if low and high
                else f"at least {low}"
                if low
                else f"at most {high}"
            )
            raise ValueError(f"{self.name}: {bounds}")
        return number

    def describe(self) -> dict[str, Any]:
        """The field as `/catalog/model-types` serves it."""
        return {
            "name": self.name,
            "label": self.label,
            "kind": self.kind,
            "hint": self.hint,
            "choices": list(self.choices),
            "min": self.min,
            "max": self.max,
            "labels": dict(self.labels),
        }


@dataclass(frozen=True)
class ModelType:
    name: str
    # What the form calls it, a noun that reads alone: "Chat model".
    label: str
    description: str
    # The `mode` of the rows in `litellm.model_cost` that list this type's ids.
    litellm_mode: str
    # Can `EvaluationRequest.language_models` name one? Only chat models today.
    evaluable: bool
    params: tuple[Param, ...] = ()
    # Example names for the local servers, which litellm prices nothing for.
    local_examples: dict[str, str] | None = None

    def clean_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """`params` validated for this type; unknown names are an error, unset ones are dropped."""
        known = {p.name: p for p in self.params}
        unknown = sorted(set(params) - set(known))
        if unknown:
            allowed = ", ".join(known) or "no parameters"
            raise ValueError(
                f"{self.label}s take {allowed}, not {', '.join(unknown)}"
            )
        return {k: known[k].clean(v) for k, v in params.items() if v is not None}

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "description": self.description,
            "evaluable": self.evaluable,
            "params": [p.describe() for p in self.params],
        }


REASONING_EFFORTS = ("disable", "low", "medium", "high")

MODEL_TYPES: dict[str, ModelType] = {
    t.name: t
    for t in (
        ModelType(
            name="llm",
            label="Chat model",
            description="Answers prompts: chat, structured output, agents. What the arena evaluates.",
            litellm_mode="chat",
            evaluable=True,
            params=(
                Param(
                    "reasoning_effort",
                    "Thinking",
                    "choice",
                    # `disable` turns native thinking off (ollama's qwen3,
                    # deepseek-r1); the levels set the effort where the model
                    # supports one.
                    hint=(
                        "How much the model thinks before answering, for models that can. "
                        "Default leaves it to the provider."
                    ),
                    choices=REASONING_EFFORTS,
                    labels={"disable": "Off", "low": "Light", "medium": "Normal", "high": "Deep"},
                ),
                Param(
                    "temperature",
                    "Temperature",
                    "number",
                    hint="Lower gives more consistent answers; blank keeps the provider default.",
                    min=0,
                    max=2,
                ),
                Param(
                    "max_tokens",
                    "Longest answer (tokens)",
                    "integer",
                    hint="Blank keeps the provider default.",
                    min=1,
                ),
            ),
            local_examples={"vllm": "Qwen/Qwen3-4B", "ollama": "llama3.2"},
        ),
        ModelType(
            name="embedding",
            label="Embedding model",
            description=(
                "Turns text into vectors, for similarity rewards and retrieval. "
                "Evaluating embedding models is in the works."
            ),
            litellm_mode="embedding",
            evaluable=False,
            params=(
                Param(
                    "dimensions",
                    "Dimensions",
                    "integer",
                    hint="Vector size, where the model lets you pick one.",
                    min=1,
                ),
            ),
            local_examples={"vllm": "BAAI/bge-m3", "ollama": "nomic-embed-text"},
        ),
        ModelType(
            name="decision",
            label="Decision model",
            description=(
                "Answers typed questions (yes/no, a choice, a score) with calibrated probabilities, "
                "fast and cheap; does not generate text. Grades for the rubric judges."
            ),
            litellm_mode="evaluation",
            evaluable=False,
            local_examples={"ollama": "laya"},
        ),
    )
}

MODEL_TYPE_NAMES = tuple(MODEL_TYPES)
ModelKind = Literal[*MODEL_TYPE_NAMES]


def clean_modalities(values: list[str]) -> list[str]:
    """`values` de-duplicated, in canonical order; at least one is required."""
    ordered = [m for m in MODALITIES if m in values]
    if not ordered:
        raise ValueError(f"at least one modality: {', '.join(MODALITIES)}")
    return ordered
