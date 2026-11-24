# License Apache 2.0: (c) 2026 Atena-Reply

"""Usage telemetry: what a run's event holds, when it is sent, and the release guard.

The guard (the last tests) fails the suite, and with it a release
(`.github/workflows/release.yml`), when anything but `openarena.telemetry`
sends to PostHog, when the opt-out does not stop it, or when the README's
"Telemetry" section does not say what is sent and how to turn it off. No test
sends a real event: the transport is faked.
"""

import json
import os
import re
import threading
import urllib.request
from pathlib import Path

import pytest

from openarena import telemetry
from openarena.config import Config

ROOT = Path(__file__).resolve().parents[1]
# A name the user chose, in every place a config holds one: none may reach the event.
PRIVATE = (
    "acme",
    "alice",
    "secret-dataset",
    "my-reward",
    "tuned",
    "local-thing",
    "rows.jsonl",
)


@pytest.fixture
def sent(monkeypatch):
    """The payloads handed to the transport, with telemetry on under a fake key, outside CI."""
    events: list[dict] = []
    monkeypatch.setattr(telemetry, "POSTHOG_KEY", "phc_test")
    monkeypatch.setattr(telemetry, "_send", events.append)
    # A send that went around `_send` lands in the list too.
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda *a, **k: events.append("urlopen")
    )
    monkeypatch.setattr(telemetry, "_notice", threading.Event())
    for name in (telemetry.OPT_OUT, "DO_NOT_TRACK", *telemetry._CI):
        monkeypatch.delenv(name, raising=False)
    return events


def _config():
    return Config.model_validate(
        {
            "evaluation": {
                "language_models": ["mini", "tuned", "local-thing"],
                "datasets": ["secret-dataset"],
            },
            "models": {
                "mini": {"model": "openai/gpt-4o-mini"},
                "tuned": {"model": "openai/ft:gpt-4o-mini:acme-corp::abc123"},
                "local-thing": {"model": "ollama/acme-internal-model"},
            },
            "rewards": {"my-reward": {"type": "exact_match"}},
            "datasets": {
                "secret-dataset": {
                    "type": "local",
                    "config": {
                        "path": "/home/alice/acme/rows.jsonl",
                        "input_template": "{}",
                    },
                    "reward": "my-reward",
                }
            },
        }
    )


def _sent_after_run(events):
    for thread in threading.enumerate():
        if thread.name == "openarena-telemetry":
            thread.join(5)
    return events


# -- what the event holds ----------------------------------------------------------------


def test_a_run_event_names_only_what_the_engine_or_litellm_publish():
    body = telemetry.event(_config())
    properties = body["properties"]
    assert (
        body["event"],
        properties["source"],
        properties["$process_person_profile"],
    ) == (telemetry.EVENT, "sdk", False)
    assert properties["models"] == [
        "openai/gpt-4o-mini",
        "other",
    ]  # the fine-tune and the local model are not named
    assert properties["model_providers"] == ["ollama", "openai"]
    assert properties["dataset_providers"] == ["local"] and properties[
        "dataset_kinds"
    ] == ["model"]
    assert properties["reward_types"] == ["exact_match"]
    assert (properties["models_count"], properties["datasets_count"]) == (3, 1)
    text = json.dumps(body)
    assert not [name for name in PRIVATE if name in text]


def test_an_a2a_agents_name_is_not_sent():
    cfg = Config.model_validate(
        {
            "evaluation": {"language_models": ["acme-support-bot"], "datasets": ["qa"]},
            "rewards": {"r": {"type": "exact_match"}},
            "a2a_agents": {"acme-support-bot": {"url": "http://agent.example/"}},
            "datasets": {
                "qa": {
                    "type": "local",
                    "config": {"path": "x.jsonl", "input_template": "{}"},
                    "a2a": {},
                    "reward": "r",
                }
            },
        }
    )
    properties = telemetry.run_properties(cfg)
    assert properties["models"] == [] and properties["model_providers"] == []
    assert properties["models_count"] == 1
    assert properties["dataset_kinds"] == ["a2a"]


def test_the_id_is_random_and_this_processs_alone():
    import uuid

    first = telemetry.event(_config())["distinct_id"]
    assert (
        uuid.UUID(first).version == 4
        and telemetry.event(_config())["distinct_id"] == first
    )
    telemetry._process_id.cache_clear()  # a new process
    assert telemetry.event(_config())["distinct_id"] != first


@pytest.mark.skipif(not hasattr(os, "fork"), reason="the platform does not fork")
# The child only writes to a pipe and exits: no lock it could deadlock on.
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded")
def test_a_forked_child_has_its_own_id_and_its_own_notice(sent):
    parent = telemetry.event(_config())["distinct_id"]
    telemetry._notice.set()
    read, write = os.pipe()
    pid = os.fork()
    if pid == 0:  # the child: report, then leave without running pytest's teardown
        os.write(
            write, f"{telemetry._process_id()} {telemetry._notice.is_set()}".encode()
        )
        os._exit(0)
    os.close(write)
    child_id, child_noticed = os.read(read, 200).decode().split()
    os.waitpid(pid, 0)
    assert child_id != parent and child_noticed == "False"
    assert telemetry._process_id() == parent  # the parent's own is untouched


def test_a_model_the_user_registers_with_litellm_is_still_not_named(monkeypatch):
    import litellm

    monkeypatch.setattr(
        litellm,
        "model_cost",
        {
            **litellm.model_cost,
            "hosted_vllm/acme-vision-ft": {"litellm_provider": "hosted_vllm"},
        },
    )
    cfg = _config()
    cfg.models["mini"].model = "hosted_vllm/acme-vision-ft"
    assert "acme" not in json.dumps(telemetry.run_properties(cfg))


def test_a_provider_the_user_adds_to_litellm_is_not_named(monkeypatch):
    import litellm

    monkeypatch.setattr(
        litellm, "provider_list", [*litellm.provider_list, "acme-internal"]
    )
    cfg = _config()
    cfg.models["mini"].model = "acme-internal/foo"
    assert "acme" not in json.dumps(telemetry.run_properties(cfg))


def test_a_bare_public_id_is_named_with_its_provider():
    cfg = _config()
    cfg.models["mini"].model = "gpt-4o-mini"
    properties = telemetry.run_properties(cfg)
    assert (
        "gpt-4o-mini" in properties["models"]
        and "openai" in properties["model_providers"]
    )


def test_the_transport_posts_the_body_to_posthog_and_swallows_any_failure(monkeypatch):
    requests = []

    def urlopen(request, timeout):
        requests.append((request.full_url, json.loads(request.data), timeout))
        raise OSError("unreachable")

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    telemetry._send({"event": "x"})  # does not raise
    assert requests == [(telemetry.ENDPOINT, {"event": "x"}, telemetry.TIMEOUT)]


# -- when it is sent -----------------------------------------------------------------------


def test_a_run_sends_one_event_in_the_background_after_a_notice(sent, capsys):
    telemetry.capture_run(_config())
    telemetry.capture_run(_config())
    assert len(_sent_after_run(sent)) == 2 and sent[0]["properties"]["source"] == "sdk"
    assert (
        capsys.readouterr().err.count(f"{telemetry.OPT_OUT}=0 turns it off") == 1
    )  # once per process, before the first


@pytest.mark.parametrize(
    "env",
    [
        {"ARENA_TELEMETRY": "0"},
        {"ARENA_TELEMETRY": "off"},
        {"ARENA_TELEMETRY": "disable"},  # a value it does not know is off, never on
        {"DO_NOT_TRACK": "1"},
        {"DO_NOT_TRACK": "true", "ARENA_TELEMETRY": "1"},
        {"CI": "true"},
        {"TF_BUILD": "True"},
    ],
)
def test_the_opt_out_and_ci_send_nothing(sent, monkeypatch, capsys, env):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    telemetry.capture_run(_config())
    assert _sent_after_run(sent) == [] and capsys.readouterr().err == ""


def test_a_ci_runner_sends_when_asked(sent, monkeypatch):
    monkeypatch.setenv("CI", "true")
    monkeypatch.setenv("ARENA_TELEMETRY", "1")
    telemetry.capture_run(_config())
    assert len(_sent_after_run(sent)) == 1


def test_print_shows_the_exact_body_and_sends_nothing(sent, monkeypatch, capsys):
    monkeypatch.setenv("ARENA_TELEMETRY", "print")
    telemetry.capture_run(_config())
    assert _sent_after_run(sent) == []
    shown = json.loads(capsys.readouterr().err)
    assert set(shown) == {"api_key", "event", "distinct_id", "timestamp", "properties"}
    assert shown["properties"] == telemetry.run_properties(_config())


def test_opt_in_when_the_default_is_off(sent, monkeypatch):
    monkeypatch.setattr(telemetry, "DEFAULT_ON", False)
    telemetry.capture_run(_config())
    assert _sent_after_run(sent) == []
    monkeypatch.setenv("ARENA_TELEMETRY", "1")
    telemetry.capture_run(_config())
    assert len(_sent_after_run(sent)) == 1


def test_a_broken_config_never_breaks_the_run(sent):
    telemetry.capture_run(object())  # building the event raises inside: swallowed
    assert _sent_after_run(sent) == []


# -- the release guard: no telemetry without the opt-out and the notice ------------------


def test_guard_only_the_telemetry_module_reaches_posthog():
    others = [
        p.relative_to(ROOT).as_posix()
        for p in (ROOT / "src").rglob("*.py")
        if p.name != "telemetry.py"
        and re.search(
            r"posthog|from openarena\.telemetry import|\btelemetry\.(?!capture_run\b|py\b)\w+",
            p.read_text(encoding="utf-8"),
            re.I,
        )
    ]
    assert others == [], (
        f"sent from outside openarena/telemetry.py, around its opt-out: {others}"
    )


def test_guard_the_sweep_sends_only_through_capture_run():
    """`run_sweep` reaches telemetry by `capture_run` alone, whose opt-out the tests above check."""
    source = (ROOT / "src/openarena/evaluate.py").read_text(encoding="utf-8")
    assert re.findall(r"telemetry\.\w+", source) == ["telemetry.capture_run"]


def test_guard_the_readme_says_what_is_sent_and_how_to_turn_it_off():
    readme = (ROOT / "README.md").read_text(encoding="utf-8").replace("\r\n", "\n")
    assert "\n## Telemetry\n" in readme, "README.md has no Telemetry section"
    section = readme.split("\n## Telemetry\n", 1)[1].split("\n## ", 1)[0]
    # The endpoint as shipped: conftest points the module's at a closed port.
    source = (ROOT / "src/openarena/telemetry.py").read_text(encoding="utf-8")
    host = re.search(r'ENDPOINT = "https://([^/"]+)', source)[1]
    for needed in (
        f"{telemetry.OPT_OUT}=0",
        "DO_NOT_TRACK",
        f"{telemetry.OPT_OUT}=print",
        f"`{telemetry.EVENT}`",
        host,
        *telemetry._CI,
    ):
        assert needed in section, f"the Telemetry section does not mention {needed}"
    rows = [
        line.split("|")[1] for line in section.splitlines() if line.startswith("| `")
    ]
    documented = set(re.findall(r"`([$\w]+)`", "".join(rows)))
    assert documented == set(telemetry.run_properties(_config())), (
        "the README's property table and the event differ"
    )
