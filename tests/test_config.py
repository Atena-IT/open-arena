# License Apache 2.0: (c) 2026 Atena-Reply

"""`Config`: what the validated run config hands the sweep, and the references it refuses.

The accessors the sweep reads (`reward_spec` with the masks a kind needs,
the metrics with their models resolved, a model's params, the defaults, an
A2A connection) and the cross-checks on load: every name an entry uses is
declared, of the right type, and of a kind that can share the run.
"""

import pytest

from openarena.config import Config, MetricEntry


def _raw(**overrides):
    raw = {
        "evaluation": {"language_models": ["m"], "datasets": ["qa"]},
        "models": {
            "m": {"model": "openai/gpt-4o-mini"},
            "judge": {"model": "openai/gpt-4o", "params": {"temperature": 0.0, "max_tokens": None}},
            "embed": {"model": "openai/text-embedding-3-small", "type": "embedding", "params": {"dimensions": 64}},
            "jev": {"model": "typesafe/jev-latest", "type": "decision"},
        },
        "rewards": {"exact": {"type": "exact_match"}},
        "mcp_servers": {"calc": {"transport": "stdio", "command": "calc-mcp"}},
        "a2a_agents": {"bot": {"url": "http://agent.example/", "headers": {"Authorization": "Bearer t"}, "timeout_sec": 30}},
        "datasets": {"qa": {"type": "local", "config": {"path": "x.jsonl", "input_template": "{}"}, "reward": "exact"}},
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(raw.get(key), dict):
            raw[key] = {**raw[key], **value}
        else:
            raw[key] = value
    return raw


def _config(**overrides):
    return Config.model_validate(_raw(**overrides))


def _dataset(**fields):
    return {"qa": {"type": "local", "config": {"path": "x.jsonl", "input_template": "{}"}, "reward": "exact", **fields}}


# -- what the sweep reads --------------------------------------------------------------


def test_a_reward_spec_carries_the_kinds_mask_unless_the_entry_sets_one():
    assert _config().reward_spec("qa") == {"name": "exact_match"}
    agent = _config(datasets=_dataset(agent={"mcp_servers": ["calc"]}))
    assert agent.reward_spec("qa") == {"name": "exact_match", "out_mask": ["messages"]}  # the trajectory stays out
    own = _config(rewards={"exact": {"type": "exact_match", "config": {"in_mask": ["answer"]}}}, datasets=_dataset(agent={"mcp_servers": ["calc"]}))
    assert own.reward_spec("qa") == {"name": "exact_match", "in_mask": ["answer"]}  # a mask the user set is theirs
    assert _config(rewards={"exact": {"type": "exact_match", "direction": "min"}}).reward_direction("qa") == "min"
    assert _config().reward_direction("qa") == "max"


def test_a_model_a_reward_or_a_metric_names_is_its_id_or_its_entry_with_params():
    cfg = _config(
        rewards={"graded": {"type": "lm_as_judge", "config": {"language_model": "judge", "instructions": "Grade."}}, "close": {"type": "cosine_similarity", "config": {"embedding_model": "embed"}}, "bare": {"type": "lm_as_judge", "config": {"language_model": "m"}}},
        datasets={**_dataset(reward="graded", metrics=[{"class": "lm_as_judge", "alias": "judged", "language_model": "judge"}, "exact_match"]), "sim": {"type": "local", "config": {"path": "y.jsonl", "input_template": "{}"}, "reward": "close"}, "plain": {"type": "local", "config": {"path": "y.jsonl", "input_template": "{}"}, "reward": "bare"}},
        evaluation={"language_models": ["m"], "datasets": ["qa"], "metrics": [{"class": "multi_judge_panel", "panel_language_models": ["m", "judge"], "smart_language_model": "judge"}]},
    )
    assert cfg.reward_spec("qa") == {"name": "lm_as_judge", "language_model": {"model": "openai/gpt-4o", "temperature": 0.0}, "instructions": "Grade."}
    assert cfg.reward_spec("sim") == {"name": "cosine_similarity", "embedding_model": {"model": "openai/text-embedding-3-small", "dimensions": 64}}
    assert cfg.reward_spec("plain") == {"name": "lm_as_judge", "language_model": "openai/gpt-4o-mini"}  # no params: the bare id
    fast = _config(rewards={"fast": {"type": "faithfulness", "config": {"decision_model": "jev"}}}, datasets=_dataset(reward="fast", metrics=[{"class": "toxicity", "decision_model": "jev"}]))
    assert fast.reward_spec("qa") == {"name": "faithfulness", "decision_model": "typesafe/jev-latest"}
    assert fast.dataset_metrics("qa")[0].kwargs == {"decision_model": "typesafe/jev-latest"}
    judged, exact = cfg.dataset_metrics("qa")
    assert judged.alias == "judged" and judged.kwargs == {"language_model": {"model": "openai/gpt-4o", "temperature": 0.0}}
    assert exact.class_name == "exact_match" and exact.kwargs == {}
    [panel] = cfg.metrics
    assert panel.kwargs == {"panel_language_models": ["openai/gpt-4o-mini", {"model": "openai/gpt-4o", "temperature": 0.0}], "smart_language_model": {"model": "openai/gpt-4o", "temperature": 0.0}}
    assert isinstance(panel, MetricEntry) and panel.direction == "max"


def test_a_models_params_defaults_and_an_a2a_connection():
    cfg = _config(evaluation={"language_models": ["m"], "datasets": ["qa"], "default_language_model": "judge", "default_embedding_model": "embed"})
    assert cfg.lm_kwargs("judge") == {"temperature": 0.0} and cfg.lm_kwargs("m") == {} and cfg.lm_kwargs("nope") == {}  # a None param is dropped
    assert cfg.model_id("judge") == "openai/gpt-4o" and cfg.model_id("bot") == "bot"
    assert cfg.default_language_model.model == "openai/gpt-4o" and cfg.default_language_model.default_kwargs == {"temperature": 0.0}
    assert cfg.default_embedding_model.default_kwargs == {"dimensions": 64}
    plain = _config(evaluation={"language_models": ["m"], "datasets": ["qa"], "default_language_model": "m"})
    assert plain.default_language_model == "openai/gpt-4o-mini" and plain.default_embedding_model is None
    assert cfg.a2a_connection("bot") == cfg.a2a_agents["bot"].connection()
    assert cfg.a2a_connection("bot")["url"] == "http://agent.example" and cfg.a2a_connection("bot")["headers"] == {"Authorization": "Bearer t"} and cfg.a2a_connection("bot")["timeout_sec"] == 30
    assert cfg.language_models == ["m"] and cfg.selected_dataset_names() == ["qa"] and cfg.seed is None and cfg.batch_size is None
    assert _config(evaluation={"language_models": ["m"], "datasets": ["qa"], "seed": 7, "batch_size": 4}).batch_size == 4


# -- what is refused on load -------------------------------------------------------------


def _refused(match, **overrides):
    with pytest.raises(ValueError, match=match):
        _config(**overrides)


def test_every_name_an_entry_uses_must_be_declared():
    _refused(r"references unknown dataset\(s\) \['qb'\]; available: \['qa'\]", evaluation={"language_models": ["m"], "datasets": ["qb"]})
    _refused("dataset 'qa' has no `reward:`", datasets={"qa": {"type": "local", "config": {"path": "x", "input_template": "{}"}}})
    _refused("names unknown reward 'exact-mach'; declared in `rewards:`: \\['exact'\\]", datasets=_dataset(reward="exact-mach"))
    _refused(r"`agent.mcp_servers:` references unknown server\(s\) \['ghost'\]", datasets=_dataset(agent={"mcp_servers": ["ghost"]}))
    _refused("'ghost' is not in `models:`; declared: \\['embed', 'jev', 'judge', 'm'\\]", evaluation={"language_models": ["ghost"], "datasets": ["qa"]})
    _refused("rewards.graded.config.language_model: 'ghost' is not in `models:`", rewards={"graded": {"type": "lm_as_judge", "config": {"language_model": "ghost"}}})
    _refused("evaluation.language_model: 'ghost' is not in `models:`", evaluation={"language_models": ["m"], "datasets": ["qa"], "default_language_model": "ghost"})


def test_a_model_must_be_of_the_kind_the_setting_or_the_sweep_needs():
    _refused("'embed' is an embedding model; this sweep needs language models", evaluation={"language_models": ["embed"], "datasets": ["qa"]})
    passages = {"type": "local", "config": {"path": "p.jsonl", "input_template": "{}"}, "embedding": {"k": 1}, "reward": "exact"}
    _refused("'m' is a language model; this sweep needs embedding models", datasets={"qa": passages})
    assert _config(datasets={"qa": passages}, evaluation={"language_models": ["embed"], "datasets": ["qa"]}).dataset_kind("qa") == "embedding"
    _refused("rewards.graded.config.language_model: 'embed' is an embedding model; this setting needs a language model", rewards={"graded": {"type": "lm_as_judge", "config": {"language_model": "embed"}}})
    _refused("datasets.qa.metrics.language_model: 'embed' is an embedding model", datasets=_dataset(metrics=[{"class": "lm_as_judge", "language_model": "embed"}]))
    _refused("rewards.fast.config.decision_model: 'm' is a language model; this setting needs a decision model", rewards={"fast": {"type": "faithfulness", "config": {"decision_model": "m"}}})
    _refused("rewards.graded.config.language_model: 'jev' is a decision model; this setting needs a language model", rewards={"graded": {"type": "lm_as_judge", "config": {"language_model": "jev"}}})
    _refused("'jev' is a decision model; this sweep needs language models", evaluation={"language_models": ["jev"], "datasets": ["qa"]})
    _refused("evaluation.embedding_model: 'm' is a language model; this setting needs an embedding model", evaluation={"language_models": ["m"], "datasets": ["qa"], "default_embedding_model": "m"})


def test_a_run_is_one_kind_of_sweep_and_an_a2a_axis_names_agents():
    a2a = {"type": "local", "config": {"path": "a.jsonl", "input_template": "{}"}, "a2a": {}, "reward": "exact"}
    both = {**_dataset(), "bots": a2a}
    _refused("an A2A dataset cannot share a run", datasets=both, evaluation={"language_models": ["bot"], "datasets": ["qa", "bots"]})
    _refused("'ghost' is not in `a2a_agents:`; declared: \\['bot'\\]", datasets={"bots": a2a}, evaluation={"language_models": ["ghost"], "datasets": ["bots"]})
    cfg = _config(datasets={"bots": a2a}, evaluation={"language_models": ["bot"], "datasets": ["bots"]})
    assert cfg.dataset_kind("bots") == "a2a" and cfg.model_id("bot") == "bot"
    passages = {"type": "local", "config": {"path": "p.jsonl", "input_template": "{}"}, "embedding": {"k": 1}, "reward": "exact"}
    _refused("an embedding dataset cannot share a run", datasets={**_dataset(), "p": passages}, evaluation={"language_models": ["m"], "datasets": ["qa", "p"]})
