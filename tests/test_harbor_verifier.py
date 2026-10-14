# License Apache 2.0: (c) 2026 Atena-Reply

"""`harbor_verifier` reads the score a task's own tests wrote, held to [0, 1]."""

import asyncio
import logging

import pytest
from synalinks.src.backend import JsonDataModel

from openarena.rewards.harbor_verifier import HarborVerifier

SCHEMA = {"type": "object", "properties": {"rewards": {"type": "object"}}}


def _score(rewards, key="reward"):
    prediction = JsonDataModel(json={"rewards": rewards}, schema=SCHEMA)
    return asyncio.run(HarborVerifier(key=key).call(None, prediction))


@pytest.mark.parametrize(("written", "scored"), [(1, 1.0), (0, 0.0), (0.25, 0.25), ("0.5", 0.5)])
def test_a_score_in_range_is_read_as_written(written, scored):
    assert _score({"reward": written}) == scored


@pytest.mark.parametrize(("written", "scored"), [(11, 1.0), (-3, 0.0), (1.0000001, 1.0)])
def test_a_score_out_of_range_is_clamped_and_said(written, scored, caplog):
    with caplog.at_level(logging.WARNING):
        assert _score({"reward": written}) == scored
    assert "outside [0, 1]" in caplog.text


@pytest.mark.parametrize("written", [float("nan"), float("inf"), float("-inf"), "nan"])
def test_a_score_that_is_not_a_number_scores_zero_and_is_said(written, caplog):
    with caplog.at_level(logging.WARNING):
        assert _score({"reward": written}) == 0.0
    assert "not a number" in caplog.text


def test_nothing_written_scores_zero():
    assert _score({}) == 0.0
    assert _score({"reward": "passed"}) == 0.0
    assert asyncio.run(HarborVerifier().call(None, None)) == 0.0


def test_another_key_of_reward_json():
    assert _score({"reward": 1, "style": 0.5}, key="style") == 0.5
