# License Apache 2.0: (c) 2026 Atena-Reply

"""LangSmith's price table, kept in step with the arena's prices of the models a run calls.

LangSmith reads no cost over OpenTelemetry: it prices a call's tokens with its own
table, one row per model name (the span's `ls_model_name`, the arena's model id).
Its built-in rows name a provider's own ids (`openai/gpt-oss-120b` for Groq), never
an `openrouter/...` one, so such a run showed no cost. Before a run each of its
models gets a row of the arena's own, `arena: <id>`, at the arena's price
(`lookups.prices.model_price`): written when missing, rewritten when the price
moved. That is the model's list price; OpenRouter bills each call at the provider
it picked, so the exact cost rides on each call too, as metadata (`arena_cost_usd`,
`otel._cost_attributes`).
"""

from __future__ import annotations

import json
import logging
import math
import re
import urllib.parse
import urllib.request
from collections.abc import Iterable

from openarena.hooks.otel import TraceConfig
from openarena.lookups.prices import model_price

log = logging.getLogger("uvicorn.error")

ROW = "arena: "
# What this process last wrote, per price table, account and model, with the row's id: a
# launch whose prices did not move asks LangSmith nothing, and a row the search misses is
# still the one rewritten, never written twice.
_written: dict[tuple[str, str, str], tuple[tuple[float, float], str | None]] = {}


def _call(method: str, url: str, key: str, body: dict | None = None):
    """One request to LangSmith's price table API, its JSON answer."""
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"X-API-Key": key, "Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(request, timeout=10) as answer:  # noqa: S310 — the configured LangSmith endpoint
        text = answer.read()
        return json.loads(text) if text else None


def _same(a, b) -> bool:
    # LangSmith may answer a price as text ("1.5e-07").
    return a is not None and math.isclose(float(a), b, rel_tol=1e-9)


def _table(endpoint: str) -> str:
    # The OTLP endpoint's API: LangSmith's cloud answers at its root, a self-hosted one at `.../api/v1`.
    base = endpoint.removesuffix("/otel/v1/traces").removesuffix("/api/v1")
    return f"{base}/api/v1/model-price-map"


def _price(api: str, key: str, model: str) -> None:
    price = model_price(model)
    # Unknown is not free, and a local server has no bill: no row for either.
    if price is None or not price.complete or price.source == "local":
        return
    cache = (api, key, model)
    known = _written.get(cache)
    if known and known[0] == (price.input, price.output):
        return
    row = {
        "name": ROW + model,
        "match_pattern": f"^{re.escape(model)}$",
        "prompt_cost": price.input,
        "completion_cost": price.output,
    }
    found = _call(
        "GET", f"{api}?{urllib.parse.urlencode({'q': model, 'limit': 100})}", key
    )
    ours = next((r for r in found or [] if r.get("name") == row["name"]), None)
    row_id = ours["id"] if ours else (known[1] if known else None)
    if row_id is None:
        row_id = (_call("POST", api, key, row) or {}).get("id")
    elif not (
        ours
        and _same(ours.get("prompt_cost"), price.input)
        and _same(ours.get("completion_cost"), price.output)
    ):
        _call("PUT", f"{api}/{row_id}", key, row)
    _written[cache] = ((price.input, price.output), row_id)


def price_models(config: TraceConfig, models: Iterable[str]) -> None:
    """Give each model a row of LangSmith's price table, for every LangSmith destination in `config`; never fails."""
    models = sorted({m for m in models if m})
    for destination in config.destinations:
        if destination.name != "langsmith":
            continue
        api = _table(destination.endpoint)
        key = dict(destination.headers).get("x-api-key", "")
        for model in models:
            try:
                _price(api, key, model)
            # A price LangSmith did not take never stops the run: its calls show no cost there.
            # The first failure ends this destination's turn, so a LangSmith that does not
            # answer delays the run by one timeout, not one per model.
            except Exception as e:  # noqa: BLE001
                log.warning(
                    "LangSmith prices: %s not written (%s); the run's cost there stays empty.",
                    model,
                    e,
                )
                break
