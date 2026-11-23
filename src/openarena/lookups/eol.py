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
   the id stops working. An `openrouter/...` id is dated by the vendor's own
   id and by OpenRouter; litellm's row for the OpenRouter id, whose date is
   copied from one of the vendor's endpoints, answers only for a model
   OpenRouter's list does not have (offline, or no longer listed), and only
   when litellm fetched its table rather than loading its shipped copy.
   OpenRouter's ids are matched in any spelling (`same_name`).

A row litellm's live table dropped, filled back from the copy shipped with
litellm, comes without its date (`fill_from_bundled`).

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
_openrouter_cache: tuple[float, float, dict[str, dict], dict[str, dict]] | None = None  # (at, ttl, rows by same_name, rows by id)
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
        if date_ := _expiration(row):
            out[str(row.get("id"))] = date_
    return out


def _expiration(row: object) -> str | None:
    """A row's `expiration_date` as YYYY-MM-DD, when it carries a well-formed one."""
    date_ = row.get("expiration_date") if isinstance(row, dict) else None
    if isinstance(date_, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_[:10]):
        return date_[:10]
    return None


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


def _refresh_openrouter() -> tuple[dict[str, dict], dict[str, dict]]:
    """Fetch OpenRouter's list (outside `_lock`, one fetch at a time) and keep it indexed."""
    global _openrouter_cache, _openrouter_refreshing
    with _openrouter_fetch_lock:
        try:
            payload = _fetch_openrouter()
            data = payload.get("data") if isinstance(payload, dict) else None
            rows = [r for r in data or [] if isinstance(r, dict)]
            by_name = {same_name(str(r.get("id", ""))): r for r in rows}
            by_id = {str(r.get("id", "")): r for r in rows}
            with _lock:
                if rows or _openrouter_cache is None:
                    _openrouter_cache = (
                        time.monotonic(),
                        _TTL_S if rows else _RETRY_S,
                        by_name,
                        by_id,
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


def _openrouter() -> tuple[dict[str, dict], dict[str, dict]]:
    """OpenRouter's rows by `same_name` of their id, and by the id itself.

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
    """OpenRouter's row for one of its ids (`vendor/model`); None offline or unknown.

    The id as OpenRouter spells it first, else any spelling (`same_name`): two
    listed ids that differ only in punctuation (`x:beta`, `x-beta`) share a
    `same_name`, and only the exact one says which is meant.
    """
    by_name, by_id = _openrouter()
    return by_id.get(model_id) or by_name.get(same_name(model_id))


def openrouter_catalog() -> dict[str, dict]:
    """OpenRouter's rows by `same_name` of their id (see `openrouter_model`)."""
    return _openrouter()[0]


def openrouter_expirations() -> dict[str, str]:
    """OpenRouter's expiration dates by its own id (`vendor/model`)."""
    return parse_openrouter_models({"data": list(_openrouter()[1].values())})


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
    have one. A filled row comes without its `deprecation_date`: the live table
    dropped the row, so the date is no longer anyone's announcement (the copy
    kept Vertex's day on `openrouter/google/...` rows OpenRouter still serves).
    The table is replaced, never changed in place, so a request reading it
    meanwhile is not disturbed; a table litellm reloads is filled again on the
    next call.
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
        missing = {
            k: _undated(v) for k, v in bundled_table().items() if k not in live
        }
        if missing:
            replace_table({**live, **missing})
        _from_bundled = _from_bundled | frozenset(missing)
        _filled_table = litellm.model_cost
        return _from_bundled


def _undated(row: object) -> object:
    """A shipped row without its `deprecation_date` (see `fill_from_bundled`)."""
    if isinstance(row, dict) and "deprecation_date" in row:
        return {k: v for k, v in row.items() if k != "deprecation_date"}
    return row


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
    routed = _openrouter_date(rest) if prefix == "openrouter" and rest else None
    if routed and (vendor is None or routed < vendor[0]):
        return routed, "openrouter"
    return vendor


def _openrouter_date(model_id: str) -> str | None:
    """OpenRouter's expiration date (YYYY-MM-DD) for one of its ids: the row `openrouter_model` finds."""
    return _expiration(openrouter_model(model_id))


def _vendor_eol(model_id: str) -> tuple[str, Source] | None:
    """The date the vendor announced, or Azure's schedule for an OpenAI model.

    An `openrouter/...` id is asked about under the vendor's own spelling,
    which is the one the sources are keyed by. litellm's row for the
    OpenRouter id itself is a last resort: its `deprecation_date` is copied
    from one of the vendor's endpoints (Vertex's day for
    `openrouter/google/...`) and stays after OpenRouter moves the model to
    another one. It is read only when OpenRouter's list does not have the
    model (offline, or no longer listed), and only from a table litellm
    fetched: its shipped copy is where the stale dates are.
    """
    prefix, _, rest = model_id.partition("/")
    if prefix != "openrouter":
        return _dated(model_id)
    direct = direct_id(model_id)
    if direct and (found := _dated(direct)):
        return found
    if openrouter_model(rest) is None and _live_table_fetched():
        return _dated(model_id)
    return None


def _dated(model_id: str) -> tuple[str, Source] | None:
    """litellm's date for an id, else Azure's schedule for an OpenAI one."""
    date_ = provider_eol(model_id)
    if date_:
        return date_, "provider"
    prefix, _, name = model_id.partition("/")
    if name and prefix in _OPENAI_PREFIXES:
        date_ = _azure_eol(name)
        if date_:
            return date_, "azure"
    return None


def _live_table_fetched() -> bool:
    """Whether litellm's live table was downloaded, rather than loaded from its shipped copy.

    litellm loads the copy when told to (`LITELLM_LOCAL_MODEL_COST_MAP`) or when
    the download fails. Only an explicit `"remote"` source counts as downloaded.
    `"local"` is the copy. Any other answer (none, an unknown source, a shape
    that cannot be read) is taken as the copy too and logged, so the copy's
    stale rows are not read on a guess. A litellm too old to have the call is
    taken as downloaded, as before the call existed.
    """
    try:
        from litellm.litellm_core_utils.get_model_cost_map import (
            get_model_cost_map_source_info,
        )
    except ImportError:
        return True
    try:
        source = get_model_cost_map_source_info()["source"]
    except Exception as e:  # noqa: BLE001 (litellm changed its answer)
        log.warning(
            "litellm did not say where its pricing table came from (%s: %s); "
            "treating it as its shipped copy",
            type(e).__name__,
            e,
        )
        return False
    if source == "remote":
        return True
    if source != "local":
        log.warning(
            "litellm said its pricing table came from %r; treating it as its shipped copy",
            source,
        )
    return False


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
