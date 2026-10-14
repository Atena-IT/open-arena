# License Apache 2.0: (c) 2026 Atena-Reply

from openarena.programs.utils import language_model, with_provider_routing


def test_openrouter_requires_parameters():
    assert with_provider_routing("openrouter/deepseek/deepseek-r1", None) == {
        "extra_body": {"provider": {"require_parameters": True}}
    }


def test_openrouter_keeps_options_and_explicit_routing():
    kwargs = with_provider_routing(
        "openrouter/x/y",
        {
            "temperature": 0.0,
            "extra_body": {"provider": {"require_parameters": False, "order": ["a"]}},
        },
    )
    assert kwargs == {
        "temperature": 0.0,
        "extra_body": {"provider": {"require_parameters": False, "order": ["a"]}},
    }


def test_other_providers_untouched():
    assert with_provider_routing("openai/gpt-4.1", None) == {}
    assert language_model("openai/gpt-4.1") == "openai/gpt-4.1"


def test_openrouter_builds_language_model():
    lm = language_model("openrouter/deepseek/deepseek-r1")
    assert lm.default_kwargs["extra_body"] == {"provider": {"require_parameters": True}}
