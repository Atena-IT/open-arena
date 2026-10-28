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

A run is priced by litellm as it goes (the framework reads litellm's cost of each
call). Rows from the shipped copy are in litellm's table already; a price only
OpenRouter has is put there too before the run (`ensure_priced`), except for
`openrouter/...` ids, whose every answer carries its own cost, which litellm
reads.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Literal

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
