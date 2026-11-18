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

A host may charge a model's calls at its list price times a factor
(`ensure_priced(..., factor=)`). litellm charges from its table, so the
factor goes into the model's row: every price field holds the list value
times the factor, and the row keeps the list values beside them. The price
of a model (`model_price`) stays its list price; `is_priced_at` says whether
its calls are charged at a factor.
"""

from __future__ import annotations

import logging
import math
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
# On a row charged at a factor: the list value of each of its price fields, and the factor. A new
# factor starts from the list values, never from the charged ones, so none is applied twice.
_LIST_PRICE = "arena_list_price"
_PRICE_FACTOR = "arena_price_factor"
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
        # A row charged at a factor still says its list price.
        if sides := _sides(entry[1].get(_LIST_PRICE) or entry[1]):
            found.append(
                Price(
                    *sides, "litellm-bundled" if entry[0] in from_bundled else "litellm"
                )
            )
    if not (found and found[0].complete):
        if sides := _openrouter_sides(model_id):
            found.append(Price(*sides, "openrouter"))
    return next((p for p in found if p.complete), found[0] if found else None)


def _price_fields(row: dict) -> dict[str, float]:
    """The price fields of a litellm row, each with the price it holds: every numeric field whose name says `cost`."""
    return {
        key: float(value)
        for key, value in row.items()
        if "cost" in key and isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def _flat_prices(row: dict) -> bool:
    """Whether every price of a row is a number of its own: a table of prices inside it (tiers) takes no factor."""
    return not any(
        isinstance(value, (dict, list)) and ("cost" in key or "pricing" in key)
        for key, value in row.items()
    )


def _is_factor(factor) -> bool:
    return (
        isinstance(factor, (int, float))
        and not isinstance(factor, bool)
        and math.isfinite(factor)
        and factor > 0
    )


def ensure_priced(
    model_id: str, mode: str = "chat", factor: float | None = None
) -> Price | None:
    """Make litellm price a run's calls to `model_id`: with a price only OpenRouter has, and at `factor` when one is given.

    With no factor, nothing to do when litellm's table prices the model, when
    the id is local or an `openrouter/...` one (litellm counts a local call as
    free, and reads OpenRouter's own cost off each answer), or when nobody
    knows both sides of its price: a half-known price would bill the other side
    as free. The two prices go into the model's row when litellm has one (its
    other facts kept), else into a row of their own (`mode`: `chat`,
    `embedding`).

    With a `factor`, every price field of the row is set to its list value
    times the factor, and the row keeps the list values and the factor. The
    row is built once and the table replaced once, on every call, so the table
    never holds the list price of a model given a factor. A call with no
    factor leaves a row charged at one as it is, at the same factor. A model
    takes no factor, and None is answered, when the factor is no positive
    number, when nobody prices the model in full, when its row holds a table
    of prices of its own, or when its cost is not litellm's to compute (a
    local id, an `openrouter/...` one).

    The table is replaced, never changed in place. Never raises: an unpriced
    run still runs. Whether a factor is in force is `is_priced_at`'s to say.
    """
    try:
        if factor is not None and not _is_factor(factor):
            log.warning(
                "price factor %r of %s is no positive number: not applied", factor, model_id
            )
            return None
        price = model_price(model_id)
        from_openrouter = (
            price is not None
            and price.source == "openrouter"
            and price.complete
            and not model_id.startswith("openrouter/")
        )
        if factor is None and not from_openrouter:
            return price
        if factor is not None and (
            price is None
            or not price.complete
            or price.source == "local"
            or model_id.startswith("openrouter/")
        ):
            return None
        import litellm

        with _register_lock:
            live = litellm.model_cost
            entry = table_entry(model_id, live)
            key, row = entry if entry else (table_key(model_id), {})
            if factor is None:
                # A row another run charges at a factor stays charged at it.
                factor = row.get(_PRICE_FACTOR)
            elif not _flat_prices(row):
                return None
            listed = dict(row.get(_LIST_PRICE) or _price_fields(row))
            priced = dict(row)
            if from_openrouter:
                listed |= {
                    "input_cost_per_token": price.input,
                    "output_cost_per_token": price.output,
                }
                priced[_ARENA_PRICED] = True
                if not entry:
                    priced |= {
                        "litellm_provider": key.partition("/")[0],
                        "mode": mode,
                        ARENA_ONLY_PRICE: True,
                    }
            if factor is None:
                priced |= listed
            else:
                priced |= {field: value * factor for field, value in listed.items()}
                priced |= {_LIST_PRICE: listed, _PRICE_FACTOR: factor}
            replace_table({**live, key: priced})
        return price
    except Exception as e:  # noqa: BLE001 (an unpriced run still runs, its spend reads 0)
        log.info(
            "price of %s not given to litellm (%s: %s)", model_id, type(e).__name__, e
        )
        return None


def is_priced_at(model_id: str, factor: float) -> bool:
    """Whether litellm charges a model's calls at its list price times `factor`.

    Read off the prices of the model's row, each against its list value, not
    off the factor the row says: a row whose prices went back to list value is
    not charged at the factor, whatever else it still carries.
    """
    try:
        import litellm

        entry = table_entry(model_id, litellm.model_cost)
    except Exception:  # noqa: BLE001 (litellm is optional at import time)
        return False
    listed = entry[1].get(_LIST_PRICE) if entry else None
    if not listed or not _is_factor(factor):
        return False
    row = entry[1]
    return all(
        isinstance(row.get(field), (int, float))
        and math.isclose(row[field], value * factor, rel_tol=1e-9, abs_tol=0.0)
        for field, value in listed.items()
    )
