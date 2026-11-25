# License Apache 2.0: (c) 2026 Atena-Reply

"""OpenRouter's endpoints of a model: one per provider that serves it, with its price and parameters.

OpenRouter routes each call of an `openrouter/...` id to one of these
endpoints (`/api/v1/models/{author}/{slug}/endpoints`), so a model's price is
a range over them, and which ones a call can reach depends on the parameters
it sends (`lookups.prices.routed_price`). The list is read once a day per
model, like the catalogue in `lookups.eol`: a stale list answers while one
refresh runs behind it, and a refresh that fails keeps the list it had and
asks again within the hour; a first read that fails is not asked again
before then either. Each model is fetched on its own, once at a time, so a
slow model holds up no other; the least recently read models are dropped
past `_MAX_SLUGS`. Only an id in OpenRouter's own form is ever put into the
URL, whose host is fixed.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import urllib.parse
import urllib.request

log = logging.getLogger("uvicorn.error")

ENDPOINTS_URL = "https://openrouter.ai/api/v1/models/{}/endpoints"
# `author/model`, as OpenRouter spells its ids (lowercase, a variant after `:`). Each part starts
# with a letter or a digit, so no `.` or `..` part can walk the URL's path.
OPENROUTER_SLUG = re.compile(r"[a-z0-9][a-z0-9._-]*/[a-z0-9][a-z0-9._:-]*")
_TTL_S = 24 * 3600
_RETRY_S = 3600
_MAX_SLUGS = 4096

_lock = threading.Lock()
_fetch_locks: dict[str, threading.Lock] = {}  # slug -> the lock its fetch holds
# slug -> (fetched at, ttl, endpoints or None), least recently read first.
_cache: dict[str, tuple[float, float, list[dict] | None]] = {}
_refreshing: set[str] = set()


def endpoints_url(slug: str) -> str:
    """The URL of a model's endpoints, `slug` encoded into OpenRouter's fixed host."""
    return ENDPOINTS_URL.format(urllib.parse.quote(slug, safe="/:"))


def _fetch_endpoints(slug: str) -> dict | None:
    try:
        with urllib.request.urlopen(endpoints_url(slug), timeout=5) as resp:  # noqa: S310 — fixed https host
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:  # noqa: BLE001 — no endpoints this time; a stale list or no price answers
        log.info("openrouter endpoints of %s not fetched (%s: %s)", slug, type(e).__name__, e)
        return None


def _rows(payload: dict | None) -> list[dict] | None:
    data = payload.get("data") if isinstance(payload, dict) else None
    rows = data.get("endpoints") if isinstance(data, dict) else None
    return [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else None


def _keep(slug: str, entry: tuple[float, float, list[dict] | None]) -> None:
    """Keep a model's entry as the most recently read; drop the least recently read past `_MAX_SLUGS`. Under `_lock`."""
    _cache.pop(slug, None)
    _cache[slug] = entry
    while len(_cache) > _MAX_SLUGS:
        oldest = next(iter(_cache))
        _cache.pop(oldest)
        _fetch_locks.pop(oldest, None)


def _refresh(slug: str, *, cold: bool = False) -> list[dict] | None:
    """Fetch a model's endpoints (outside `_lock`, one fetch per model at a time) and keep them."""
    with _lock:
        fetch_lock = _fetch_locks.setdefault(slug, threading.Lock())
    with fetch_lock:
        try:
            if cold:
                with _lock:
                    # Another reader fetched it while this one waited for the lock.
                    if (known := _cache.get(slug)) is not None:
                        return known[2]
            rows = _rows(_fetch_endpoints(slug))
            with _lock:
                known = _cache.get(slug)
                if rows is not None:
                    _keep(slug, (time.monotonic(), _TTL_S, rows))
                else:
                    # Offline this time: keep what was known (nothing, on a first read), ask again sooner.
                    _keep(slug, (time.monotonic(), _RETRY_S, known[2] if known else None))
                return _cache[slug][2]
        finally:
            with _lock:
                _refreshing.discard(slug)


def openrouter_endpoints(slug: str) -> list[dict] | None:
    """The endpoints of OpenRouter's model `slug` (`author/model`).

    `[]` when OpenRouter lists none; None for a slug not in OpenRouter's form
    (never fetched), or when no read of it has succeeded yet.
    """
    if not OPENROUTER_SLUG.fullmatch(slug):
        return None
    with _lock:
        known = _cache.get(slug)
        if known is not None:
            _keep(slug, known)
            if time.monotonic() - known[0] >= known[1] and slug not in _refreshing:
                _refreshing.add(slug)
                threading.Thread(target=_refresh, args=(slug,), daemon=True).start()
            return known[2]
        _refreshing.add(slug)
    return _refresh(slug, cold=True)
