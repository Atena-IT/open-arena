# License Apache 2.0: (c) 2026 Atena-Reply

import copy

from openarena.programs.utils import language_model, with_provider_routing


def test_openrouter_requires_parameters():
    assert with_provider_routing("openrouter/deepseek/deepseek-r1", None) == {
        "provider": {"require_parameters": True}
    }


def test_openrouter_keeps_options_and_explicit_routing():
    kwargs = {
        "temperature": 0.0,
        "provider": {"require_parameters": False, "order": ["a"]},
    }
    assert with_provider_routing("openrouter/x/y", kwargs) == kwargs


def test_an_extra_body_provider_moves_to_the_top_and_the_top_level_one_wins():
    # litellm merges `extra_body` into the request last: a block left there would replace
    # the framework's own `provider` (its excluded hosts), so it moves to the top level.
    kwargs = {
        "extra_body": {"provider": {"order": ["a"], "only": ["b"]}, "seed": 1},
        "provider": {"order": ["c"]},
    }
    assert with_provider_routing("openrouter/x/y", kwargs) == {
        "provider": {"order": ["c"], "only": ["b"], "require_parameters": True},
        "extra_body": {"seed": 1},
    }


def test_an_extra_body_left_empty_is_dropped():
    kwargs = {"extra_body": {"provider": {"order": ["a"]}}}
    assert with_provider_routing("openrouter/x/y", kwargs) == {
        "provider": {"order": ["a"], "require_parameters": True}
    }


def test_the_callers_params_are_not_changed():
    kwargs = {"extra_body": {"provider": {"order": ["a"]}}, "provider": {"only": ["b"]}}
    snapshot = copy.deepcopy(kwargs)
    with_provider_routing("openrouter/x/y", kwargs)
    assert kwargs == snapshot


def test_other_providers_untouched():
    assert with_provider_routing("openai/gpt-4.1", None) == {}
    kept = {"extra_body": {"provider": {"order": ["a"]}}}
    assert with_provider_routing("openai/gpt-4.1", kept) == kept
    assert language_model("openai/gpt-4.1") == "openai/gpt-4.1"


def test_openrouter_builds_language_model():
    lm = language_model("openrouter/deepseek/deepseek-r1")
    assert lm.default_kwargs == {"provider": {"require_parameters": True}}
