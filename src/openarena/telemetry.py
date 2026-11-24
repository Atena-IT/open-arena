# License Apache 2.0: (c) 2026 Atena-Reply

"""Anonymous usage telemetry: one event per evaluation run, to PostHog's EU Cloud.

What the event holds, and why, is the README's "Telemetry" section; the two
are kept together by `tests/test_telemetry.py`, which also fails a release
whose telemetry could be sent around the opt-out. The event never holds a
prompt, a row, a key, a result, a name the user chose, a path or a host: the
models are named only when they are in litellm's public table, the providers
and rewards only when the engine ships them.

`ARENA_TELEMETRY` decides: `0` (or any value but the ones below) is off, `1`
is on, `print` prints the event on stderr instead of sending it. `DO_NOT_TRACK`
turns it off too, and so does a CI runner unless `ARENA_TELEMETRY=1`. Both are
read at each run, never at import.

The event is sent with the standard library from a background thread, with a
short timeout; a finishing process waits for it at most that long. Any failure
is swallowed: it never fails, slows or blocks a run.
"""

from __future__ import annotations

import atexit
import functools
import json
import logging
import os
import platform
import sys
import threading
import time
import urllib.request
import uuid
from datetime import datetime, timezone
from importlib.metadata import version
from typing import Any

log = logging.getLogger(__name__)

OPT_OUT = "ARENA_TELEMETRY"
EVENT = "evaluation_run"
# PostHog's capture endpoint on the EU Cloud: the one place an event goes.
ENDPOINT = "https://eu.i.posthog.com/i/v0/e/"
# The shared Open Arena project's API key (`phc_...`), public and write-only. The placeholder sends
# nothing: the real key goes here before the release that turns telemetry on.
POSTHOG_KEY = "phc_OPEN_ARENA_PROJECT_KEY_PLACEHOLDER"
# On unless opted out (the ticket's choice); False makes it opt-in (`ARENA_TELEMETRY=1`).
DEFAULT_ON = True
# Seconds a send may take, and the most a finishing process waits for it.
TIMEOUT = 2.0

_ON = ("1", "true", "yes", "on")
# shortcut: the common CI runners' variables; add one when a runner is missed.
_CI = ("CI", "TF_BUILD", "JENKINS_URL", "CODEBUILD_BUILD_ID", "TEAMCITY_VERSION")
_notice = threading.Event()
_threads: list[threading.Thread] = []


@atexit.register
def _flush() -> None:
    """Give the events still in flight at most `TIMEOUT` seconds in all, as the process ends."""
    deadline = time.monotonic() + TIMEOUT
    for thread in list(_threads):
        thread.join(max(0.0, deadline - time.monotonic()))


def mode() -> str:
    """`on`, `off` or `print`, from the environment as it is now."""
    value = os.environ.get(OPT_OUT, "").strip().lower()
    if value == "print":
        return "print"
    if (
        value not in ("", *_ON)
        or _truthy("DO_NOT_TRACK")
        or POSTHOG_KEY.endswith("_PLACEHOLDER")
    ):
        return "off"
    if value in _ON:
        return "on"
    return "on" if DEFAULT_ON and not any(_truthy(name) for name in _CI) else "off"


def _truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() not in (
        "",
        "0",
        "false",
        "no",
        "off",
    )


def capture_run(cfg: Any) -> None:
    """Send a run's event in the background, or print it; never raises."""
    try:
        chosen = mode()
        if chosen == "off":
            return
        payload = event(cfg)
        if chosen == "print":
            print(json.dumps(payload, indent=2), file=sys.stderr)
            return
        if not _notice.is_set():
            _notice.set()
            print(
                f"openarena sends one anonymous usage event per run; {OPT_OUT}=0 turns it off. "
                "What it holds: https://github.com/Atena-IT/open-arena#telemetry",
                file=sys.stderr,
            )
        thread = threading.Thread(
            target=_send, args=(payload,), name="openarena-telemetry", daemon=True
        )
        thread.start()
        _threads[:] = [t for t in _threads if t.is_alive()] + [thread]
    except Exception as e:  # noqa: BLE001 (telemetry never breaks a run)
        log.debug("telemetry skipped (%s: %s)", type(e).__name__, e)


def event(cfg: Any) -> dict[str, Any]:
    """The request body PostHog's capture endpoint takes, exactly as it is sent."""
    return {
        "api_key": POSTHOG_KEY,
        "event": EVENT,
        "distinct_id": _process_id(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "properties": run_properties(cfg),
    }


@functools.cache
def _process_id() -> str:
    # A random id for this process only: nothing is stored on the machine, and runs are not tied together.
    return str(uuid.uuid4())


def run_properties(cfg: Any) -> dict[str, Any]:
    """What a run's event says: the environment, and the kinds of things the run compares."""
    from openarena.datasets import PROVIDERS
    from openarena.lookups.eol import bundled_table, table_entry
    from openarena.rewards import _REWARD_TYPES

    # The table shipped inside litellm, never the live one: a user's register_model or private
    # price map puts their own ids in that.
    table = bundled_table()
    datasets = cfg.selected_dataset_names()
    kinds = {cfg.dataset_kind(n) for n in datasets}
    # An A2A run's axis is the user's agent names, never sent even when one looks like a model id.
    model_ids = (
        [] if "a2a" in kinds else [cfg.model_id(name) for name in cfg.language_models]
    )
    rewards = {
        cfg.rewards[cfg.datasets[n].reward].type
        for n in datasets
        if cfg.datasets[n].reward in cfg.rewards
    }
    return {
        "source": "sdk",
        "$lib": "openarena",
        "$lib_version": version("openarena"),
        # PostHog: no person profile, and no location looked up from the request's address.
        "$process_person_profile": False,
        "$geoip_disable": True,
        "python_version": "{}.{}".format(*sys.version_info[:2]),
        "os": platform.system().lower(),
        "arch": platform.machine().lower(),
        "models_count": len(cfg.language_models),
        "datasets_count": len(datasets),
        "model_providers": sorted({_provider(m, table) for m in model_ids}),
        "models": sorted(
            {m if m in table or table_entry(m, table) else "other" for m in model_ids}
        ),
        "dataset_kinds": sorted(kinds),
        "dataset_providers": sorted(
            {_known(cfg.datasets[n].type, PROVIDERS) for n in datasets}
        ),
        "reward_types": sorted({_known(r, _REWARD_TYPES) for r in rewards}),
    }


def _known(name: str, shipped) -> str:
    """The name when the engine or litellm ships it, else `other`: a name the user chose is never sent."""
    return name if name in shipped else "other"


def _provider(model_id: str, table: dict) -> str:
    """A `provider/model` id's provider, or the one litellm's table gives a bare id (`gpt-4o-mini`)."""
    import litellm

    prefix, slash, _ = model_id.partition("/")
    name = prefix if slash else (table.get(model_id) or {}).get("litellm_provider", "")
    # litellm's enum, not its provider_list, which a user's custom_provider_map appends to.
    return _known(name, {p.value for p in litellm.LlmProviders})


def _send(payload: dict[str, Any]) -> None:
    try:
        request = urllib.request.Request(
            ENDPOINT,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(request, timeout=TIMEOUT).close()
    except Exception as e:  # noqa: BLE001 (an unreachable PostHog costs nothing but the event)
        log.debug("telemetry not sent (%s: %s)", type(e).__name__, e)
