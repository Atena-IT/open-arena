# License Apache 2.0: (c) 2026 Atena-Reply

"""End-of-life dates for model ids, from three sources.

1. The provider's own announcement, as recorded in litellm's pricing table
   (`deprecation_date`). OpenAI only announces a shutdown once a model is on
   its way out, so most current models carry nothing here.
2. Azure's OpenAI retirement schedule, which sets a retirement date for every
   OpenAI model at launch (eighteen months for GA models) and publishes it as
   a markdown table in the Azure docs. It is the earliest firm date most
   OpenAI models have, and it is what people mean by a model's EOL before
   OpenAI itself says anything, so it stands in when litellm has no date.

3. OpenRouter's own `expiration_date`, for `openrouter/...` ids: the day
   OpenRouter stops routing the model, which it often sets for models no
   vendor has dated (previews, smaller labs). It answers only for its own
   ids, and wins when it comes before the vendor's date: that is the day
   the id stops working.

The schedule is fetched from the docs' GitHub source and cached for a day; a
snapshot shipped next to this module answers when the network does not.
OpenRouter's list is cached for a day too; with no network it adds nothing.
"""

from __future__ import annotations

import functools
import logging
import re
import json
import threading
import time
import urllib.request
from datetime import date
from pathlib import Path
from typing import Literal

log = logging.getLogger("uvicorn.error")

Source = Literal["provider", "azure", "openrouter"]

# How far ahead the leaderboard looks. Time to end of life is capped here: a
# model whose date is further away, and a model with no date at all, both
# count as this many days. No date means no end of life, not "unknown": an
# open-weight model is served for as long as anyone runs it, and a provider
# that has announced nothing has, for ranking purposes, promised the same.
# Five years, so that the dates providers do announce, one to two years out,
# still spread over the lower part of the scale.
EOL_HORIZON_DAYS = 5 * 365

AZURE_SCHEDULE_URL = (
    "https://raw.githubusercontent.com/MicrosoftDocs/azure-ai-docs/main/"
    "articles/foundry/openai/includes/concepts-model-retirement-schedule-content.md"
)
_SNAPSHOT = Path(__file__).with_name("data") / "azure_openai_retirement_schedule.md"
_TTL_S = 24 * 3600

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
# A failed fetch is asked again sooner than a good one is refreshed.
_RETRY_S = 3600

# Arena prefixes whose ids name OpenAI models, and so fall under the schedule.
_OPENAI_PREFIXES = {"openai", "azure", "azure_ai"}
# `vllm` and `sglang` are synalinks' prefixes for a server litellm calls `hosted_vllm`.
_PROVIDER_ALIASES = {"vllm": "hosted_vllm", "sglang": "hosted_vllm", "cohere": "cohere_chat"}

# OpenRouter's vendor segment -> the prefix that vendor's own models are keyed
# by, where the two differ (`google` is `gemini` in litellm's table). Only
# vendors that publish retirement dates are worth listing; anything absent is
# left alone rather than guessed at.
_OPENROUTER_VENDORS = {
    "anthropic": "anthropic",
    "openai": "openai",
    "google": "gemini",
    "deepseek": "deepseek",
    "mistralai": "mistral",
    "x-ai": "xai",
    "cohere": "cohere",
}

_lock = threading.Lock()
_cache: tuple[float, dict[str, str], dict[tuple[str, str], str]] | None = None
_openrouter_cache: tuple[float, float, dict[str, dict], dict[str, str]] | None = None  # (at, ttl, rows by same_name, dates)
_openrouter_fetch_lock = threading.Lock()
_openrouter_refreshing = False

# litellm's live table as last filled from its shipped copy, and the keys that came from the copy.
_fill_lock = threading.Lock()
_filled_table: dict | None = None
_from_bundled: frozenset[str] = frozenset()

# Marks a row the arena put into litellm's table only so a run is priced (`lookups.prices`): it has
# the two prices and nothing else to say about the model.
ARENA_ONLY_PRICE = "arena_only_price"


def parse_azure_schedule(
    markdown: str,
) -> tuple[dict[str, str], dict[tuple[str, str], str]]:
    """The "Azure OpenAI" table of the schedule, as `{model: date}` and `{(model, version): date}`.

    A model listed under several versions gets the latest date, since an
    unversioned id resolves to the newest snapshot. The fine-tuned table that
    follows is not read: its dates are for training and deployment of
    fine-tunes, not for calling the model.
    """
    by_model: dict[str, str] = {}
    by_version: dict[tuple[str, str], str] = {}
    section = re.search(r"^### Azure OpenAI\s*$(.*?)^#{3,4} ", markdown, re.M | re.S)
    if not section:
        return by_model, by_version
    for line in section.group(1).splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 4 or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", cells[3]):
            continue
        model, version, date = cells[0], cells[1], cells[3]
        by_version[(model, version)] = date
        if date > by_model.get(model, ""):
            by_model[model] = date
    return by_model, by_version


def _fetch_schedule() -> str | None:
    try:
        with urllib.request.urlopen(AZURE_SCHEDULE_URL, timeout=5) as resp:  # noqa: S310 — fixed https URL
            return resp.read().decode("utf-8")
    except Exception as e:  # noqa: BLE001 — the snapshot answers instead
        log.info(
            "azure retirement schedule not fetched (%s: %s); using the shipped snapshot",
            type(e).__name__,
            e,
        )
        return None


def azure_retirements() -> tuple[dict[str, str], dict[tuple[str, str], str]]:
    """The schedule, fetched at most once a day, else the shipped snapshot."""
    global _cache
    with _lock:
        if _cache and time.monotonic() - _cache[0] < _TTL_S:
            return _cache[1], _cache[2]
        text = _fetch_schedule()
        by_model, by_version = parse_azure_schedule(text) if text else ({}, {})
        if not by_model:
            by_model, by_version = parse_azure_schedule(
                _SNAPSHOT.read_text(encoding="utf-8")
            )
        _cache = (time.monotonic(), by_model, by_version)
        return by_model, by_version


def _azure_eol(name: str) -> str | None:
    by_model, by_version = azure_retirements()
    if name in by_model:
        return by_model[name]
    # A dated snapshot id, `gpt-4o-2024-08-06`, is one version's row.
    m = re.fullmatch(r"(.+)-(\d{4}-\d{2}-\d{2})", name)
    return by_version.get((m.group(1), m.group(2))) if m else None


def parse_openrouter_models(payload: dict) -> dict[str, str]:
    """`{openrouter id: YYYY-MM-DD}` for the models of `/api/v1/models` that carry an `expiration_date`."""
    out: dict[str, str] = {}
    for row in payload.get("data") or []:
        date_ = row.get("expiration_date") if isinstance(row, dict) else None
        if isinstance(date_, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_[:10]):
            out[str(row.get("id"))] = date_[:10]
    return out


def _fetch_openrouter() -> dict | None:
    try:
        with urllib.request.urlopen(OPENROUTER_MODELS_URL, timeout=5) as resp:  # noqa: S310 — fixed https URL
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:  # noqa: BLE001 — no OpenRouter dates, the other sources still answer
        log.info("openrouter model list not fetched (%s: %s)", type(e).__name__, e)
        return None


def same_name(model_id: str) -> str:
    """A model id with its punctuation evened out, so two spellings of one model meet.

    `anthropic/claude-opus-5-5` (the vendor's own) and `anthropic/claude-opus-5.5`
    (OpenRouter's) are the same model.
    """
    return re.sub(r"[._:]", "-", model_id.lower())


def _refresh_openrouter() -> tuple[dict[str, dict], dict[str, str]]:
    """Fetch OpenRouter's list (outside `_lock`, one fetch at a time) and keep it indexed."""
    global _openrouter_cache, _openrouter_refreshing
    with _openrouter_fetch_lock:
        try:
            payload = _fetch_openrouter()
            data = payload.get("data") if isinstance(payload, dict) else None
            rows = [r for r in data or [] if isinstance(r, dict)]
            by_name = {same_name(str(r.get("id", ""))): r for r in rows}
            dates = parse_openrouter_models({"data": rows})
            with _lock:
                if rows or _openrouter_cache is None:
                    _openrouter_cache = (
                        time.monotonic(),
                        _TTL_S if rows else _RETRY_S,
                        by_name,
                        dates,
                    )
                else:
                    # Offline this time: keep what was known, ask again sooner.
                    _openrouter_cache = (
                        time.monotonic(),
                        _RETRY_S,
                        *_openrouter_cache[2:],
                    )
                return _openrouter_cache[2], _openrouter_cache[3]
        finally:
            _openrouter_refreshing = False


def _openrouter() -> tuple[dict[str, dict], dict[str, str]]:
    """OpenRouter's rows by `same_name` of their id, and their expiration dates.

    Fetched once and then refreshed at most once a day, behind the caller: a
    stale list answers while a thread fetches the next one, so no request
    waits on OpenRouter after the first.
    """
    global _openrouter_refreshing
    with _lock:
        cache = _openrouter_cache
        if cache is not None:
            if time.monotonic() - cache[0] >= cache[1] and not _openrouter_refreshing:
                _openrouter_refreshing = True
                threading.Thread(target=_refresh_openrouter, daemon=True).start()
            return cache[2], cache[3]
    return _refresh_openrouter()


def openrouter_model(model_id: str) -> dict | None:
    """OpenRouter's row for one of its ids (`vendor/model`, any spelling); None offline or unknown."""
    return _openrouter()[0].get(same_name(model_id))


def openrouter_catalog() -> dict[str, dict]:
    """OpenRouter's rows by `same_name` of their id (see `openrouter_model`)."""
    return _openrouter()[0]


def openrouter_expirations() -> dict[str, str]:
    """OpenRouter's expiration dates by its own id (`vendor/model`)."""
    return _openrouter()[1]


@functools.cache
def bundled_table() -> dict:
    """The pricing table shipped inside the litellm package, read once; empty if it cannot be."""
    try:
        import litellm

        path = Path(litellm.__file__).with_name(
            "model_prices_and_context_window_backup.json"
        )
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001 (the live table and the other sources still answer)
        log.info("litellm's shipped table not read (%s: %s)", type(e).__name__, e)
        return {}


def fill_from_bundled() -> frozenset[str]:
    """Give litellm's live table the rows only its shipped copy has; the keys that came from there.

    litellm downloads a fresh table at import, and that one can drop rows: in
    October 2026 it lost every `openrouter/...` row the shipped copy still had.
    Filling the live table from the copy serves every reader at once: the
    prices, the end-of-life dates, the add form's facts and catalogue, and
    litellm's own price of each call in a run. The live row wins where both
    have one. The table is replaced, never changed in place, so a request
    reading it meanwhile is not disturbed; a table litellm reloads is filled
    again on the next call.
    """
    global _filled_table, _from_bundled
    try:
        import litellm
    except Exception:  # noqa: BLE001 (litellm is optional at import time)
        return _from_bundled
    with _fill_lock:
        if litellm.model_cost is _filled_table:
            return _from_bundled
        live = litellm.model_cost
        missing = {k: v for k, v in bundled_table().items() if k not in live}
        if missing:
            replace_table({**live, **missing})
        _from_bundled = _from_bundled | frozenset(missing)
        _filled_table = litellm.model_cost
        return _from_bundled


def replace_table(table: dict) -> None:
    """Make `table` litellm's live pricing table, at once (see `fill_from_bundled`)."""
    import litellm

    litellm.model_cost = table
    # litellm keeps a lower-cased index of the table, rebuilt on demand.
    invalidate = getattr(litellm.utils, "_invalidate_model_cost_lowercase_map", None)
    if invalidate:
        invalidate()


def table_key(model_id: str) -> str:
    """The key an arena `provider/model` id has in a litellm table (`vllm/x` is `hosted_vllm/x`)."""
    prefix, _, name = model_id.partition("/")
    return f"{_PROVIDER_ALIASES.get(prefix, prefix)}/{name}"


def table_entry(model_id: str, table: dict) -> tuple[str, dict] | None:
    """The key and row of a litellm-shaped table for an arena `provider/model` id, or None."""
    prefix, _, name = model_id.partition("/")
    if not name:
        return None
    provider = _PROVIDER_ALIASES.get(prefix, prefix)
    key = f"{provider}/{name}"
    if key in table:
        return key, table[key]
    bare = table.get(name)
    return (name, bare) if bare and bare.get("litellm_provider") == provider else None


def litellm_model_info(model_id: str) -> dict | None:
    """litellm's row for an arena `provider/model` id, or None when it has none.

    The row carries what litellm knows about the model: token prices, the
    context window, `supports_*` capability flags, and a `deprecation_date`
    when the provider announced one. The live table, filled from the copy
    shipped with litellm (`fill_from_bundled`). A row the arena wrote only to
    price a run (`ARENA_ONLY_PRICE`) knows nothing else, and is not answered.
    """
    try:
        import litellm
    except Exception:  # noqa: BLE001 (litellm is optional at import time)
        return None
    fill_from_bundled()
    entry = table_entry(model_id, litellm.model_cost)
    return entry[1] if entry and not entry[1].get(ARENA_ONLY_PRICE) else None
def provider_eol(model_id: str) -> str | None:
    """The provider's own announced date for an id, from litellm's table."""
    info = litellm_model_info(model_id)
    return info.get("deprecation_date") if info else None


def direct_id(model_id: str) -> str | None:
    """The id as the vendor's own API spells it, for an aggregator's id.

    OpenRouter spells everything `openrouter/<vendor>/<model>` and proxies the
    vendor, so a retirement date announced for `anthropic/claude-sonnet-5`
    belongs to `openrouter/anthropic/claude-sonnet-5` just as much: it is the
    same model, and it stops answering on the same day. Nothing in litellm's
    table or Azure's schedule is keyed that way, though, so the id is unwrapped
    before it is looked up.

    None when the id is not an aggregator's, or names a vendor with no direct
    prefix of its own (`openrouter/mancer/...`).
    """
    prefix, _, rest = model_id.partition("/")
    if prefix != "openrouter":
        return None
    vendor, sep, name = rest.partition("/")
    direct = _OPENROUTER_VENDORS.get(vendor.lower())
    return f"{direct}/{name}" if sep and name and direct else None


def model_eol(model_id: str) -> tuple[str, Source] | None:
    """The end-of-life date (YYYY-MM-DD) of a model id and where it comes from, if any.

    An `openrouter/...` id also has OpenRouter's own date, which wins when it
    is the earlier of the two.
    """
    vendor = _vendor_eol(model_id)
    prefix, _, rest = model_id.partition("/")
    routed = (
        openrouter_expirations().get(rest) if prefix == "openrouter" and rest else None
    )
    if routed and (vendor is None or routed < vendor[0]):
        return routed, "openrouter"
    return vendor


def _vendor_eol(model_id: str) -> tuple[str, Source] | None:
    """The date the vendor announced, or Azure's schedule for an OpenAI model."""
    # An aggregator's id is asked about under the vendor's own spelling too,
    # which is the only one the sources are keyed by.
    for candidate in (model_id, direct_id(model_id)):
        if not candidate:
            continue
        date = provider_eol(candidate)
        if date:
            return date, "provider"
        prefix, _, name = candidate.partition("/")
        if name and prefix in _OPENAI_PREFIXES:
            date = _azure_eol(name)
            if date:
                return date, "azure"
    return None


def eol_days(model_id: str, today: date) -> float:
    """Days from `today` to the model's end of life, the leaderboard's `eol_days`.

    The horizon (`EOL_HORIZON_DAYS`) when no date is known, since that reads
    as no end of life, and never more than the horizon otherwise. A past date
    goes negative: the model may already refuse calls.
    """
    eol = model_eol(model_id)
    if not eol:
        return float(EOL_HORIZON_DAYS)
    return float(min((date.fromisoformat(eol[0]) - today).days, EOL_HORIZON_DAYS))
