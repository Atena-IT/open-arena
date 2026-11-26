# License Apache 2.0: (c) 2026 Atena-Reply

"""What each model paid per token in a run, estimated from what its calls were billed.

OpenRouter bills each call of an `openrouter/...` model at the price of the
host it picked, and litellm reads that cost off the answer. A run's calls of
one model, each with its input tokens, output tokens and cost, then give the
model's price per token as the least-squares fit `cost ≈ a·in + b·out`, with
no intercept: the average of the hosts the run reached, weighted by what each
served. A model with billed calls always gets the two prices: the fit when
it can be trusted (`estimate_prices`); else, marked approximate, the same fit
without its thresholds (typically when the calls reached hosts at different
prices, or were few) when it prices output over input as the hosts its calls
could reach do, give or take a factor of two (`loose_prices`, `host_ratios`);
else the run's cost split by those hosts' median ratio (`split_prices`).

Calls are kept by registry entry, not by model id: two entries may share an
id with different params, which reach different hosts. A judge's calls count
for the judge's own entry, found by building each entry's model as a reward
builds it (`judge_keys`); a judge no entry builds (the framework's default
model) is not counted. Cache hits, failed calls and calls with no cost are
left out; so are Harbor tasks (their agents call models inside the sandbox),
A2A agents and embedding models.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import synalinks

from openarena.callbacks.otel import _judges
from openarena.lookups.endpoints import openrouter_endpoints
from openarena.lookups.prices import CallKind, _admitted, _per_token, _request, model_price

# Below these the fit says nothing reliable: too few calls, input and output
# tokens in a fixed ratio (they cannot be told apart), or a fit that explains
# too little of the costs. Tuned on staging's 17 Nov 2026 runs.
MIN_CALLS = 10
MAX_COSINE = 0.99
MIN_R2 = 0.85

# The run recording in this context: a model the whole process shares (synalinks' default
# judge) is called by every run, and each call counts only for the run that made it.
_RECORDING: contextvars.ContextVar[RunPrices | None] = contextvars.ContextVar(
    "openarena_run_prices", default=None
)

# A dataset's kind -> the kind of call its model makes; the others are not counted.
_CALL_KIND: dict[str, CallKind] = {"model": "structured", "tools": "tools", "agent": "tools"}


@dataclass(frozen=True)
class BilledCall:
    input_tokens: int
    output_tokens: int
    cost: float  # USD
    kind: CallKind


def estimate_prices(calls: Sequence[BilledCall]) -> tuple[float, float] | None:
    """USD per input and per output token, fitted on `calls`; None when the fit cannot be trusted.

    It needs `MIN_CALLS` calls, input and output token counts that are not
    proportional (their cosine below `MAX_COSINE`), two prices of at least 0,
    and an R² of at least `MIN_R2`.
    """
    if len(calls) < MIN_CALLS:
        return None
    return _fit(calls, min_r2=MIN_R2)


def loose_prices(calls: Sequence[BilledCall]) -> tuple[float, float] | None:
    """The fit of `estimate_prices` without its count and R² thresholds: what the calls paid, approximately.

    Calls whose cost is not a number are left out. It still needs two calls, input and output token counts
    that are not proportional, and two prices of at least 0: otherwise None, as the two cannot be told apart.
    """
    kept = [c for c in calls if math.isfinite(c.cost)]
    return _fit(kept, min_r2=None) if len(kept) >= 2 else None


def _fit(calls: Sequence[BilledCall], *, min_r2: float | None) -> tuple[float, float] | None:
    x = np.array([[c.input_tokens, c.output_tokens] for c in calls], dtype=float)
    y = np.array([c.cost for c in calls], dtype=float)
    if not np.isfinite(y).all():
        return None
    norms = np.linalg.norm(x, axis=0)
    if not norms.all() or x[:, 0] @ x[:, 1] / (norms[0] * norms[1]) >= MAX_COSINE:
        return None
    try:
        (a, b), *_ = np.linalg.lstsq(x, y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if a < 0 or b < 0:
        return None
    if min_r2 is not None:
        residual = float(((y - x @ (a, b)) ** 2).sum())
        spread = float(((y - y.mean()) ** 2).sum())
        if spread <= 0 or 1 - residual / spread < min_r2:
            return None
    return float(a), float(b)


def split_prices(calls: Sequence[BilledCall], ratio: float) -> tuple[float, float] | None:
    """The run's cost split into USD per input and per output token, output priced `ratio` times input.

    `a·in + ratio·a·out` gives back what the calls cost. Calls whose cost is not a number are left out;
    None when none is left or they took no token.
    """
    kept = [c for c in calls if math.isfinite(c.cost)]
    weighted = sum(c.input_tokens + ratio * c.output_tokens for c in kept)
    if not kept or weighted <= 0:
        return None
    per_input = sum(c.cost for c in kept) / weighted
    return per_input, ratio * per_input


# How far a loose fit may price output over input from the hosts its calls could reach: past this, it
# put the cost on the wrong side (two calls from two hosts) and gives way to the split.
RATIO_SLACK = 2.0


def host_ratios(model, kinds: Sequence[CallKind]) -> list[float]:
    """What each host `model`'s calls of `kinds` could reach charges per output token over per input token, sorted.

    `model` is a language model, or its id. An `openrouter/...` model: its OpenRouter endpoints with a price
    per token (read as `routed_price` reads them, cached), only those each kind's request is admitted to when
    `model` is a language model and any is; by id, every one. Any other: litellm's table, or OpenRouter's
    catalogue (`model_price`). Empty when no source says. Blocking (a first read waits on OpenRouter): call it
    off an event loop. Never raises.
    """
    model_id = model if isinstance(model, str) else getattr(model, "model", None)
    if not model_id:
        return []
    try:
        if model_id.startswith("openrouter/"):
            endpoints = openrouter_endpoints(model_id.removeprefix("openrouter/")) or []
            if not isinstance(model, str) and kinds:
                endpoints = asyncio.run(_reached(model, endpoints, kinds)) or endpoints
            ratios = sorted(out / inp for e in endpoints if (p := _per_token(e)) for inp, out in [p] if inp > 0)
            if ratios:
                return ratios
        price = model_price(model_id)
        if price is not None and price.input and price.output and price.input > 0:
            return [price.output / price.input]
    except Exception:  # noqa: BLE001 (no ratio: the cost is split evenly)
        return []
    return []


async def _reached(model, endpoints: list[dict], kinds: Sequence[CallKind]) -> list[dict]:
    """The endpoints any of `kinds`' requests is admitted to, as `routed_price` filters them."""
    reached: dict[int, dict] = {}
    for kind in kinds:
        admitted, _ = _admitted(endpoints, await _request(model, kind))
        reached |= {id(e): e for e in admitted}
    return list(reached.values())


def _plausible(prices: tuple[float, float], ratios: Sequence[float]) -> bool:
    """Two prices above 0 whose output over input is within `RATIO_SLACK` of the hosts' ratios (any, without them)."""
    per_input, per_output = prices
    if per_input <= 0 or per_output <= 0:
        return False
    return not ratios or min(ratios) / RATIO_SLACK <= per_output / per_input <= max(ratios) * RATIO_SLACK


def _key(model) -> tuple[str, str]:
    """A language model by what its calls send: its id, its default parameters, and what it is declared to take.

    `supports_reasoning`, `supports_vision` and `supports_audio` are constructor arguments, not
    default parameters, yet they change the request (a level is sent only to a declared reasoner).
    """
    declared = {
        flag: bool(getattr(model, f"_{flag}", False))
        for flag in ("supports_reasoning", "supports_vision", "supports_audio")
    }
    sent = {"default_kwargs": getattr(model, "default_kwargs", {}) or {}, **declared}
    return model.model, json.dumps(sent, sort_keys=True, default=str)


def judge_keys(models: Mapping[str, object]) -> dict[tuple[str, str], str]:
    """The registry entry each judge model is built from, by `_key`.

    `models` maps an entry's name to what a reward or a metric that names it is
    handed (`Config.judge_models`: the bare id, or `{model, **params}`). Each is
    built as a reward builds its judge (`rewards.build_models`, then the
    framework for a bare id); two entries that build the same model cannot be
    told apart and are left out.
    """
    from synalinks.src.modules.language_models import get as get_language_model

    from openarena.rewards import build_models

    found: dict[tuple[str, str], list[str]] = {}
    for name, named in models.items():
        try:
            built = build_models({"language_model": named})["language_model"]
            if isinstance(built, str):
                built = get_language_model(built)
        except Exception:  # noqa: BLE001 — an entry that cannot be built is no judge's
            continue
        if isinstance(built, synalinks.LanguageModel):
            found.setdefault(_key(built), []).append(name)
    return {key: names[0] for key, names in found.items() if len(names) == 1}


def _language_models(program) -> list:
    """The language models a program calls, each with its fallbacks."""
    found: dict[int, object] = {}
    for module in program._flatten_modules():
        model = module
        while isinstance(model, synalinks.LanguageModel):
            found.setdefault(id(model), model)
            model = getattr(model, "fallback", None)
    return list(found.values())


class _BilledHook(synalinks.hooks.Hook):
    """Record each billed call of one model for its registry entry, while its run records."""

    def __init__(self, prices: RunPrices, entry: str, kind: CallKind):
        super().__init__()
        self._prices, self._entry, self._kind = prices, entry, kind

    def on_call_end(self, call_id, parent_call_id=None, outputs=None, exception=None):
        if exception is not None or outputs is None or _RECORDING.get() is not self._prices:
            return
        # Read in the calling task right after the call, as `hooks.otel` reads it.
        from synalinks.src.modules.language_models.language_model import current_call_usage

        usage = current_call_usage() or {}
        if usage.get("error") or usage.get("cache_hit"):
            return
        cost = usage.get("cost")
        tokens_in = int(usage.get("input_tokens") or 0)
        tokens_out = int(usage.get("output_tokens") or 0)
        if (
            not isinstance(cost, (int, float))
            or not math.isfinite(cost)
            or cost <= 0
            or not (tokens_in or tokens_out)
        ):
            return
        self._prices.record(self._entry, BilledCall(tokens_in, tokens_out, float(cost), self._kind))


class _PriceWatch(synalinks.callbacks.Callback):
    """As a cell's evaluation begins, watch its program's models and its judges'."""

    def __init__(self, prices: RunPrices, entry: str, kind: CallKind):
        super().__init__()
        self._prices, self._entry, self._kind = prices, entry, kind

    def on_test_begin(self, logs=None):
        self._hooks = self._prices.watch_program(self.program, self._entry, self._kind)
        for judge in _judges(self.program):
            self._hooks += self._prices.watch_judge(judge)

    def on_test_end(self, logs=None):
        # The cell is over: its models go with its program, nothing of them is kept.
        self._prices.unwatch(getattr(self, "_hooks", []))
        self._hooks = []


class RunPrices:
    """A run's billed calls by registry entry, and what each entry paid per token."""

    def __init__(self, judges: Mapping[tuple[str, str], str]):
        self._judges = dict(judges)
        self.calls: dict[str, list[BilledCall]] = {}
        # Each entry's model, as watched (its id, and the language model itself): what `host_ratios` reads.
        self.models: dict[str, str] = {}
        self._models: dict[str, object] = {}
        self._hooked: list[tuple[object, _BilledHook]] = []
        self._token: contextvars.Token | None = None

    def start(self) -> None:
        """Count, in this context, the calls of the models this run watches."""
        self._token = _RECORDING.set(self)

    def stop(self) -> None:
        """Stop counting and take this run's hooks off its models (a shared model outlives the run)."""
        self.unwatch([h for _, h in self._hooked])
        if self._token is not None:
            _RECORDING.reset(self._token)
            self._token = None

    @contextlib.contextmanager
    def recording(self):
        self.start()
        try:
            yield self
        finally:
            self.stop()

    def record(self, entry: str, call: BilledCall) -> None:
        self.calls.setdefault(entry, []).append(call)

    def watch(self, model, entry: str, kind: CallKind) -> list[_BilledHook]:
        """Record `model`'s billed calls for `entry`; a model this run already watches keeps its hook.

        Returns the hook it added (`[]` for a model already watched), for `unwatch`.
        """
        if any(isinstance(h, _BilledHook) and h._prices is self for h in model._hooks.hooks):
            return []
        if isinstance(model_id := getattr(model, "model", None), str):
            self.models.setdefault(entry, model_id)
            self._models.setdefault(entry, model)
        hook = _BilledHook(self, entry, kind)
        hook.set_module(model)
        model.add_hook(hook)
        self._hooked.append((model, hook))
        return [hook]

    def unwatch(self, hooks: Sequence[_BilledHook]) -> None:
        """Take these hooks off their models, and let the models go."""
        gone = {id(h) for h in hooks}
        for model, hook in [(m, h) for m, h in self._hooked if id(h) in gone]:
            with contextlib.suppress(ValueError):
                model._hooks.hooks.remove(hook)
        self._hooked = [(m, h) for m, h in self._hooked if id(h) not in gone]

    def watch_program(self, program, entry: str, kind: CallKind) -> list[_BilledHook]:
        return [h for model in _language_models(program) for h in self.watch(model, entry, kind)]

    def watch_judge(self, program) -> list[_BilledHook]:
        """A judge program's models, each for the registry entry it is built from; one no entry builds is left alone."""
        return [
            h
            for model in _language_models(program)
            if (entry := self._judges.get(_key(model))) is not None
            for h in self.watch(model, entry, "structured")
        ]

    def callback(self, entry: str, dataset_kind: str) -> synalinks.callbacks.Callback | None:
        """The callback that watches a cell's models; None for a cell whose calls are not counted."""
        kind = _CALL_KIND.get(dataset_kind)
        return _PriceWatch(self, entry, kind) if kind else None

    def estimates(self) -> dict[str, dict]:
        """What each entry with billed calls paid: USD per input and per output token, the kinds of call
        covered, and `approx`. Not approximate only when the fit can be trusted (`estimate_prices`); else
        the fit without its thresholds (`loose_prices`) when it is plausible next to the hosts its calls
        could reach (`host_ratios`); else the cost split by their median ratio, even when none is known
        (`split_prices`). Blocking (`host_ratios`): a sweep calls it in a thread."""
        out = {}
        for entry, calls in self.calls.items():
            kinds = sorted({c.kind for c in calls})
            if (fit := estimate_prices(calls)) is not None:
                out[entry] = {"input": fit[0], "output": fit[1], "kinds": kinds, "approx": False}
                continue
            ratios = host_ratios(self._models.get(entry) or self.models.get(entry), kinds)
            loose = loose_prices(calls)
            paid = loose if loose is not None and _plausible(loose, ratios) else None
            paid = paid or split_prices(calls, float(np.median(ratios)) if ratios else 1.0)
            if paid is not None:
                out[entry] = {"input": paid[0], "output": paid[1], "kinds": kinds, "approx": True}
        return out
