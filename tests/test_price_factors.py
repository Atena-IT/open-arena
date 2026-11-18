# License Apache 2.0: (c) 2026 Atena-Reply

"""A run given price factors: its calls are charged at them, and it says what the same calls cost at list price.

The provider is replaced and litellm is kept: each call is answered by litellm's
own mock, so litellm counts the tokens and prices them from its table, the
framework adds the cost to its counters, and the sweep reads both figures off
those. No network, no key.
"""

import asyncio
import json
import os

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import litellm
import pytest
import synalinks
import tenacity
import yaml
from synalinks.src.modules.language_models import (
    language_model as language_model_module,
)

from openarena.evaluate import PriceFactorLost, run_sweep
from openarena.lookups import eol
from openarena.lookups.prices import is_priced_at

MODEL = "mistral/A-Model-Of-This-Test"
# Its row in litellm's table for these tests, USD per token: two prices and nothing a factor cannot scale.
LIST_INPUT, LIST_OUTPUT = 1e-06, 2e-06
# Other models litellm prices, each at a price of its own.
OTHER, JUDGE, EMBEDDER = "mistral/Another-Model-Of-This-Test", "mistral/A-Judge-Of-This-Test", "mistral/An-Embedder-Of-This-Test"
Q_SCHEMA = {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]}
A_SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}
ROWS = [{"question": "2 + 2?", "answer": "4"}, {"question": "3 + 3?", "answer": "6"}, {"question": "5 + 5?", "answer": "10"}]


@pytest.fixture(autouse=True)
def litellms_table():
    """litellm's table with the test's model in it at its list price; the table found is put back afterwards."""
    table = litellm.model_cost

    def row(input_price, output_price, mode="chat"):
        return {"input_cost_per_token": input_price, "output_cost_per_token": output_price, "litellm_provider": "mistral", "mode": mode}

    rows = {MODEL: row(LIST_INPUT, LIST_OUTPUT), OTHER: row(5e-06, 5e-06), JUDGE: row(3e-06, 3e-06), EMBEDDER: row(1e-07, 0.0, "embedding")}
    eol.replace_table({**table, **rows})
    yield
    eol.replace_table(table)


def _config(tmp_path, model=MODEL, others=None):
    rows = tmp_path / "rows.jsonl"
    rows.write_text("\n".join(json.dumps(r) for r in ROWS) + "\n")
    config = {
        "models": {"tested": {"model": model}, **(others or {})},
        "rewards": {"exact": {"type": "exact_match"}},
        "datasets": {
            "qa": {
                "type": "local",
                "config": {
                    "path": str(rows),
                    "input_template": '{"question": {{ question | tojson }}}',
                    "output_template": '{"answer": {{ answer | tojson }}}',
                    "input_schema": Q_SCHEMA,
                    "output_schema": A_SCHEMA,
                },
                "reward": "exact",
                "metrics": ["cost"],
            }
        },
        "evaluation": {"name": "e", "language_models": ["tested"], "datasets": ["qa"], "batch_size": 2},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _answered_by_litellm(monkeypatch, *, garbled_first=False, on_call=None, named=None):
    """Replace the provider, not litellm. Returns the answers litellm gave, each with the tokens and the cost it counted.

    `garbled_first` makes the first answer to each question one the framework cannot read, which it retries.
    `on_call` runs before each answer. `named` is the model name the provider puts in its answers,
    when that is not the one it was asked for.
    """
    answers, asked = [], set()

    async def acompletion(self, formatted_messages, **kwargs):
        if on_call:
            on_call()
        question = next(r for r in ROWS if r["question"] in json.dumps(formatted_messages))
        text = json.dumps({"answer": question["answer"]})
        if garbled_first and question["question"] not in asked:
            asked.add(question["question"])
            text = "no JSON here"
        if named:
            text = litellm.ModelResponse(
                model=named,
                choices=[{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
                usage=litellm.Usage(prompt_tokens=40, completion_tokens=8, total_tokens=48),
            )
        # Everything the framework sends with a call reaches litellm, as it does with a real provider.
        response = await litellm.acompletion(model=self.model, messages=formatted_messages, mock_response=text, **kwargs)
        answers.append(
            {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "cost": response._hidden_params["response_cost"],
            }
        )
        return response

    monkeypatch.setattr(synalinks.LanguageModel, "_acompletion", acompletion)
    # A retry is not made to wait.
    monkeypatch.setattr(language_model_module, "rate_limit_aware_wait", lambda max_wait: tenacity.wait_none())
    return answers


def _run(tmp_path, events, **kwargs):
    return asyncio.run(run_sweep(str(_config(tmp_path, **kwargs.pop("config", {}))), state_dir=tmp_path / "state", on_event=events.append, **kwargs))


def _of(events, kind):
    return [e for e in events if e["event"] == kind]


def test_a_run_given_a_factor_is_charged_at_it_and_says_what_its_calls_cost_at_list_price(tmp_path, monkeypatch):
    answers, events = _answered_by_litellm(monkeypatch), []
    _run(tmp_path, events, price_factors={MODEL: 2.0})
    assert len(answers) == 3
    # litellm charged each call at twice the list price of its tokens.
    for answer in answers:
        assert answer["cost"] == pytest.approx(2.0 * (answer["input_tokens"] * LIST_INPUT + answer["output_tokens"] * LIST_OUTPUT))
    charged = sum(a["cost"] for a in answers)
    (end,) = _of(events, "cell_end")
    assert end["metrics"]["cost"] == pytest.approx(charged)
    assert end["list_usage"] == {MODEL: {"cost": pytest.approx(charged / 2.0)}}
    # Each batch says what the cell has spent so far, both ways.
    batches = _of(events, "batch")
    assert len(batches) == 2
    for batch in batches:
        assert batch["list_usage"] == {MODEL: {"cost": pytest.approx(batch["usage"]["cost"] / 2.0)}}


def test_a_run_given_no_factor_tells_nothing_new_and_leaves_the_prices_alone(tmp_path, monkeypatch):
    answers, events = _answered_by_litellm(monkeypatch), []
    _run(tmp_path, events)
    for answer in answers:
        assert answer["cost"] == pytest.approx(answer["input_tokens"] * LIST_INPUT + answer["output_tokens"] * LIST_OUTPUT)
    assert len(_of(events, "batch")) == 2 and len(_of(events, "cell_end")) == 1
    # The events are the ones a run had before it could be given factors, field for field.
    assert all(set(batch) == {"event", "dataset", "model", "batch", "batches_total", "reward", "usage"} for batch in _of(events, "batch"))
    assert set(_of(events, "cell_end")[0]) == {"event", "dataset", "model", "metrics"}
    assert not any("list_usage" in event for event in events)


def test_every_attempt_of_a_retried_call_is_in_both_figures(tmp_path, monkeypatch):
    answers, events = _answered_by_litellm(monkeypatch, garbled_first=True), []
    _run(tmp_path, events, price_factors={MODEL: 2.0})
    # Each question was answered twice, and the provider bills both answers.
    assert len(answers) == 6
    charged = sum(a["cost"] for a in answers)
    (end,) = _of(events, "cell_end")
    assert end["metrics"]["cost"] == pytest.approx(charged)
    assert end["list_usage"] == {MODEL: {"cost": pytest.approx(charged / 2.0)}}


def test_an_answer_that_names_another_model_is_charged_at_the_factor_of_the_model_asked_for(tmp_path, monkeypatch):
    # litellm prices an answer by the model the answer names, when its table knows that name:
    # left alone, these calls would be charged at the other model's list price without a word.
    answers, events = _answered_by_litellm(monkeypatch, named=OTHER.partition("/")[2]), []
    _run(tmp_path, events, price_factors={MODEL: 2.0})
    assert len(answers) == 3
    for answer in answers:
        assert answer["cost"] == pytest.approx(2.0 * (answer["input_tokens"] * LIST_INPUT + answer["output_tokens"] * LIST_OUTPUT))
    charged = sum(a["cost"] for a in answers)
    (end,) = _of(events, "cell_end")
    assert end["list_usage"] == {MODEL: {"cost": pytest.approx(charged / 2.0)}}


def test_a_factor_given_for_a_judge_and_for_an_embedding_model_is_applied(tmp_path, monkeypatch):
    answers, events = _answered_by_litellm(monkeypatch), []
    others = {"judge": {"model": JUDGE}, "embedder": {"model": EMBEDDER, "type": "embedding"}}
    _run(tmp_path, events, price_factors={JUDGE: 1.5, EMBEDDER: 3.0}, config={"others": others})
    assert is_priced_at(JUDGE, 1.5) and is_priced_at(EMBEDDER, 3.0)
    # The model under test was given no factor: charged at list price, and counted as charged.
    charged = sum(a["cost"] for a in answers)
    assert charged == pytest.approx(sum(a["input_tokens"] * LIST_INPUT + a["output_tokens"] * LIST_OUTPUT for a in answers))
    (end,) = _of(events, "cell_end")
    assert end["list_usage"] == {MODEL: {"cost": pytest.approx(charged)}}


def test_a_factor_for_a_model_the_run_does_not_call_stops_it_before_its_first_cell(tmp_path, monkeypatch):
    answers, events = _answered_by_litellm(monkeypatch), []
    # The id spelt another way: left alone, the run would be charged at list price without a word.
    with pytest.raises(ValueError, match="mistral/a-model-of-this-test"):
        _run(tmp_path, events, price_factors={MODEL.lower(): 2.0})
    assert answers == [] and _of(events, "cell_start") == []


def test_a_factor_that_cannot_be_applied_stops_the_run_before_its_first_cell(tmp_path, monkeypatch):
    answers, events = _answered_by_litellm(monkeypatch), []
    # An `openrouter/...` answer carries its own cost: litellm's table is not what charges it.
    model = "openrouter/meta-llama/llama-3.3-70b-instruct"
    with pytest.raises(ValueError, match=model):
        _run(tmp_path, events, price_factors={model: 2.0}, config={"model": model})
    assert answers == [] and _of(events, "cell_start") == []


def test_a_model_that_stops_being_charged_at_its_factor_stops_the_run_and_the_batch_is_not_reported(tmp_path, monkeypatch):
    table = litellm.model_cost

    def litellm_replaces_its_table():
        # As litellm does when a fresh table arrives: the row is back at its list price.
        eol.replace_table(table)

    events = []
    answers = _answered_by_litellm(monkeypatch, on_call=litellm_replaces_its_table)
    with pytest.raises(PriceFactorLost, match=MODEL):
        _run(tmp_path, events, price_factors={MODEL: 2.0})
    assert not is_priced_at(MODEL, 2.0)
    # The calls of that batch were charged at list price: no figure of the run counts them.
    assert answers and _of(events, "batch") == [] and _of(events, "cell_end") == []
