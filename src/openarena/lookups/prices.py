# License Apache 2.0: (c) 2026 Atena-Reply

"""What a model id costs per token, from the first source that knows.

1. A local server (`ollama/...`, `vllm/...`, `sglang/...`): free, there is no bill.
2. litellm's pricing table: the live one litellm downloads at import, filled
   with the rows only the copy shipped inside the package has
   (`eol.fill_from_bundled`); the source says which of the two it was.
3. OpenRouter's own catalogue (`/api/v1/models`, cached for a day with the
   end-of-life dates). It prices its own ids, and a vendor's direct id too
   when OpenRouter routes the same model: it passes the vendor's price
   through.

A source that prices both sides wins over one that prices one; a side nobody
prices stays None: unknown is not free.

An `openrouter/...` id also has a routed price (`routed_price`): OpenRouter
sends each call to one of the model's hosts, at that host's price, among
those that take every parameter the call sends, so its price is a range over
the hosts the arena's calls can reach, with, per kind of call, the reason
none can when that happens. `model_price` keeps answering one number for it
(the LangSmith price rows read it).

A run is priced by litellm as it goes (the framework reads litellm's cost of each
call). Rows from the shipped copy are in litellm's table already; a price only
OpenRouter has is put there too before the run (`ensure_priced`), except for
`openrouter/...` ids, whose every answer carries its own cost, which litellm
reads.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass
from typing import Literal

from openarena.lookups.endpoints import openrouter_endpoints
from openarena.lookups.eol import (
    _OPENROUTER_VENDORS,
    ARENA_ONLY_PRICE,
    fill_from_bundled,
    openrouter_model,
    replace_table,
    table_entry,
    table_key,
)

log = logging.getLogger("uvicorn.error")

Source = Literal["local", "litellm", "litellm-bundled", "openrouter"]

# Served by a server of your own: no bill to price.
LOCAL_PROVIDERS = frozenset(
    {"ollama", "ollama_chat", "vllm", "hosted_vllm", "sglang", "lm_studio", "llamafile"}
)

# A vendor's own prefix -> how OpenRouter spells that vendor (`gemini/x` is `google/x` there).
_VENDOR_ON_OPENROUTER = {
    direct: vendor for vendor, direct in _OPENROUTER_VENDORS.items()
}

# Marks a row's prices as the arena's, put there from OpenRouter: a lookup passes over them to
# OpenRouter itself, so the source shown stays true and a price that moved is picked up again.
_ARENA_PRICED = "arena_priced"
_register_lock = threading.Lock()


@dataclass(frozen=True)
class Price:
    """USD per token, input and output (None for a side the source does not price), and where it was found."""

    input: float | None
    output: float | None
    source: Source

    @property
    def complete(self) -> bool:
        return self.input is not None and self.output is not None


def _sides(row: dict | None) -> tuple[float | None, float | None] | None:
    """A table row's input and output price per token; None when it prices neither."""
    if not row:
        return None

    def side(key: str) -> float | None:
        value = row.get(key)
        return float(value) if isinstance(value, (int, float)) else None

    sides = side("input_cost_per_token"), side("output_cost_per_token")
    return sides if sides != (None, None) else None


def _openrouter_sides(model_id: str) -> tuple[float, float] | None:
    """OpenRouter's price for its own id, or for the vendor's direct id of the same model."""
    prefix, _, rest = model_id.partition("/")
    if prefix == "openrouter":
        row = openrouter_model(rest)
    elif prefix in _VENDOR_ON_OPENROUTER and rest:
        row = openrouter_model(f"{_VENDOR_ON_OPENROUTER[prefix]}/{rest}")
    else:
        return None
    pricing = (row or {}).get("pricing") or {}
    try:
        cin, cout = float(pricing["prompt"]), float(pricing["completion"])
    except (KeyError, TypeError, ValueError):
        return None
    # A negative price is OpenRouter's "decided per request" (its auto router): no fixed price.
    return (cin, cout) if cin >= 0 and cout >= 0 else None


def model_price(model_id: str) -> Price | None:
    """The price of a model id per token, from the first source that has one (a complete one first)."""
    if model_id.partition("/")[0] in LOCAL_PROVIDERS:
        return Price(0.0, 0.0, "local")
    try:
        import litellm

        from_bundled = fill_from_bundled()
        entry = table_entry(model_id, litellm.model_cost)
    except Exception:  # noqa: BLE001 (litellm is optional at import time)
        from_bundled, entry = frozenset(), None
    found: list[Price] = []
    if entry and not entry[1].get(_ARENA_PRICED):
        if sides := _sides(entry[1]):
            found.append(
                Price(
                    *sides, "litellm-bundled" if entry[0] in from_bundled else "litellm"
                )
            )
    if not (found and found[0].complete):
        if sides := _openrouter_sides(model_id):
            found.append(Price(*sides, "openrouter"))
    return next((p for p in found if p.complete), found[0] if found else None)


def ensure_priced(model_id: str, mode: str = "chat") -> Price | None:
    """Make litellm price a run's calls to `model_id` with a price only OpenRouter has.

    Nothing to do when litellm's table prices it, when the id is local or an
    `openrouter/...` one (litellm counts a local call as free, and reads
    OpenRouter's own cost off each answer), or when nobody knows both sides of
    its price: a half-known price would bill the other side as free. The two
    prices go into the model's row when litellm has one (its other facts kept),
    else into a row of their own (`mode`: `chat`, `embedding`). The table is
    replaced, never changed in place. Never raises: an unpriced run still runs.
    """
    try:
        price = model_price(model_id)
        if (
            price is None
            or price.source != "openrouter"
            or not price.complete
            or model_id.startswith("openrouter/")
        ):
            return price
        import litellm

        with _register_lock:
            live = litellm.model_cost
            entry = table_entry(model_id, live)
            key, row = entry if entry else (table_key(model_id), {})
            priced = {
                **row,
                "input_cost_per_token": price.input,
                "output_cost_per_token": price.output,
                _ARENA_PRICED: True,
            }
            if not entry:
                priced |= {
                    "litellm_provider": key.partition("/")[0],
                    "mode": mode,
                    ARENA_ONLY_PRICE: True,
                }
            replace_table({**live, key: priced})
        return price
    except Exception as e:  # noqa: BLE001 (an unpriced run still runs, its spend reads 0)
        log.info(
            "price of %s not given to litellm (%s: %s)", model_id, type(e).__name__, e
        )
        return None


CallKind = Literal["structured", "tools"]
# Why no host serves a kind of call. A closed set: nothing else is ever given as a reason.
Reason = Literal[
    "structured_output",
    "tools",
    "reasoning",
    "max_tokens",
    "temperature",
    "parameter",
    "platform_exclusion",
    "no_provider",
    "price_unavailable",
]
CALL_KINDS: tuple[CallKind, ...] = ("structured", "tools")


@dataclass(frozen=True)
class RoutedPrice:
    """USD per token over the hosts an `openrouter/...` model's evaluation calls can reach, and per kind of call why none can.

    The four prices are None when no kind of call has a host with a price.
    `kinds` maps each kind to None when at least one host serves it.
    """

    input_min: float | None
    input_max: float | None
    output_min: float | None
    output_max: float | None
    kinds: dict[CallKind, Reason | None]


# The stand-ins a structured and a tool call are built with: only the parameter names they lead to matter.
_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}
_LOOKUP_TOOL = {
    "type": "function",
    "function": {
        "name": "lookup",
        "description": "Look a word up.",
        "parameters": {
            "type": "object",
            "properties": {"word": {"type": "string"}},
            "required": ["word"],
        },
    },
}
# What a request carries that is not a parameter OpenRouter's hosts take or refuse.
_NOT_PARAMETERS = frozenset(
    {"model", "messages", "provider", "extra_body", "api_base", "api_key", "stream"}
)
# A request's parameter -> the names OpenRouter lists it under, and the reason when no host takes it.
_PARAMETERS: dict[str, tuple[tuple[str, ...], Reason]] = {
    "response_format": (("response_format", "structured_outputs"), "structured_output"),
    "tools": (("tools",), "tools"),
    "tool_choice": (("tool_choice",), "tools"),
    # By the name the request sends: litellm passes `reasoning_effort` to OpenRouter as it is, and
    # `require_parameters` is not known to read it as `reasoning`. A host that lists only `reasoning`
    # (Claude's, Gemini's on 25 Nov) is not counted for a level sent this way.
    "reasoning_effort": (("reasoning_effort",), "reasoning"),
    "reasoning": (("reasoning",), "reasoning"),
    "max_tokens": (("max_tokens",), "max_tokens"),
    "temperature": (("temperature",), "temperature"),
}


async def _request(lm, kind: CallKind) -> dict:
    """What a call of this kind sends, built by the framework without sending it."""
    import synalinks

    messages = synalinks.ChatMessages(
        messages=[synalinks.ChatMessage(role="user", content="?")]
    )
    if kind == "structured":
        return await lm.build_request(messages, schema=_ANSWER_SCHEMA)
    # An agent asks for its final answer with `tool_choice="required"`: a host must take both.
    return await lm.build_request(
        messages, tool_schemas=[_LOOKUP_TOOL], tool_choice="required"
    )


def _required(request: dict) -> list[tuple[str, Reason]]:
    """The parameter names a host must list to serve `request`, each with the reason when none does, in filter order."""
    sent = {k: v for k, v in request.items() if k not in _NOT_PARAMETERS}
    extra = request.get("extra_body") or {}
    sent |= {k: v for k, v in extra.items() if k != "provider"}
    required: list[tuple[str, Reason]] = []
    for key, (names, reason) in _PARAMETERS.items():
        if key in sent:
            if key == "response_format" and not (sent[key] or {}).get("json_schema"):
                names = ("response_format",)
            required += [(name, reason) for name in names]
    required += [(key, "parameter") for key in sent if key not in _PARAMETERS]
    return required


def _host(endpoint: dict) -> str:
    """The provider slug of an endpoint (`coreweave` for `coreweave/bf16`), as `provider.ignore` names it."""
    return str(endpoint.get("tag") or "").split("/", 1)[0].lower()


def _admitted(endpoints: list[dict], request: dict) -> tuple[list[dict], Reason | None]:
    """The endpoints OpenRouter's default routing could send `request` to; when none, the filter that left none."""
    left = list(endpoints)
    for name, reason in _required(request):
        left = [e for e in left if name in (e.get("supported_parameters") or ())]
        if not left:
            return [], reason
    max_tokens = request.get("max_tokens")
    if isinstance(max_tokens, int):
        left = [
            e
            for e in left
            if not isinstance(e.get("max_completion_tokens"), int)
            or max_tokens <= e["max_completion_tokens"]
        ]
        if not left:
            return [], "max_tokens"
    provider = request.get("provider") or {}
    ignored = {str(name).lower() for name in provider.get("ignore") or ()}
    only = {str(name).lower() for name in provider.get("only") or ()}
    left = [e for e in left if _host(e) not in ignored and (not only or _host(e) in only)]
    # Default routing skips the flex and priority tiers.
    left = [e for e in left if not e.get("service_tier")]
    return (left, None) if left else ([], "platform_exclusion")


def _per_token(endpoint: dict) -> tuple[float, float] | None:
    """An endpoint's input and output price per token; None when unknown or decided per request (negative)."""
    pricing = endpoint.get("pricing") or {}
    try:
        cin, cout = float(pricing["prompt"]), float(pricing["completion"])
    except (KeyError, TypeError, ValueError):
        return None
    return (cin, cout) if cin >= 0 and cout >= 0 else None


async def routed_price(model_id: str, params: dict | None = None) -> RoutedPrice | None:
    """The price range of an `openrouter/...` model's evaluation calls, and per kind of call why no host serves it.

    Each kind of call the arena makes (`structured`: a model dataset or a
    judge; `tools`: a tools or agent dataset) is built by the framework with
    the model entry's `params`, as a run builds it (`programs.utils.
    language_model`, which adds the routing), and not sent. The hosts are
    OpenRouter's endpoints of the model (`lookups.endpoints`); a call can
    reach those that list every parameter it sends, allow its `max_tokens`,
    are not ruled out by its `provider` block, and are on the default tier.
    The range is over the hosts any kind can reach, a host with no fixed
    price left out. Endpoints that cannot be read give `price_unavailable`
    for both kinds: the price is unknown, not the model unservable.

    None for an id that is not `openrouter/...`.
    """
    if not model_id.startswith("openrouter/"):
        return None
    # Read in a thread: a first read waits on OpenRouter, and must not hold up the caller's event loop.
    endpoints = await asyncio.to_thread(openrouter_endpoints, model_id.removeprefix("openrouter/"))
    if endpoints is None:
        return RoutedPrice(
            None, None, None, None, dict.fromkeys(CALL_KINDS, "price_unavailable")
        )
    if not endpoints:
        # OpenRouter lists the model and no host for it: no call of any kind can be served.
        return RoutedPrice(None, None, None, None, dict.fromkeys(CALL_KINDS, "no_provider"))
    import synalinks

    from openarena.programs.utils import language_model

    lm = language_model(model_id, params)
    if isinstance(lm, str):
        lm = synalinks.LanguageModel(model=lm)
    kinds: dict[CallKind, Reason | None] = {}
    reached: dict[int, dict] = {}
    for kind in CALL_KINDS:
        admitted, reason = _admitted(endpoints, await _request(lm, kind))
        kinds[kind] = reason
        reached |= {id(e): e for e in admitted}
    prices = [p for e in reached.values() if (p := _per_token(e)) is not None]
    if not prices:
        return RoutedPrice(None, None, None, None, kinds)
    ins, outs = [p[0] for p in prices], [p[1] for p in prices]
    return RoutedPrice(min(ins), max(ins), min(outs), max(outs), kinds)
