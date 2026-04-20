"""Unit tests for the LLM-as-a-Judge evaluator.

Covers score normalisation (1-5 → [0, 1]), reference vs no-reference prompt
branching, trajectory injection, and error propagation. LangChain/ChatLiteLLM
are sidestepped by reassigning `evaluator._judge` to a stub exposing an
async `ainvoke` method.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from src.evaluation.evaluators.llm_as_judge import LLMAsJudgeEvaluator
from src.evaluation.types import JudgeResponse
from src.llms import AgentStep, ToolInvocation


_SYS_REF = "judge against expected_output"
_SYS_NO_REF = "judge from input alone"


def _make_evaluator(**overrides) -> LLMAsJudgeEvaluator:
    """Build an evaluator with no results; we drive `_score` directly."""
    return LLMAsJudgeEvaluator(
        results=[],
        llm_config={"model": "openai/gpt-x"},
        system_prompt=_SYS_REF,
        system_prompt_no_reference=_SYS_NO_REF,
        **overrides,
    )


class _Stub:
    """Captures the messages passed to `ainvoke` and returns a pre-set
    `JudgeResponse` (or raises if configured to)."""

    def __init__(self, response: JudgeResponse | None = None, error: Exception | None = None):
        self._response = response
        self._error = error
        self.captured_messages: list | None = None
        self.call_count = 0

    async def ainvoke(self, messages, config=None):
        self.call_count += 1
        self.captured_messages = messages
        if self._error is not None:
            raise self._error
        return self._response


# ---------------------------------------------------------------------------
# Score normalisation — 1-5 → [0, 1]
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [(1, 0.0), (2, 0.25), (3, 0.5), (4, 0.75), (5, 1.0)],
)
def test_score_normalises_1_to_5_to_unit_interval(raw, expected):
    evaluator = _make_evaluator()
    evaluator._judge = _Stub(JudgeResponse(thinking="r", score=raw))  # type: ignore[arg-type]

    score, explanation, error = asyncio.run(
        evaluator._score(input="q", output="a", expected_output="a")
    )
    assert score == pytest.approx(expected)
    assert explanation == "r"
    assert error is None


# ---------------------------------------------------------------------------
# Prompt branching + payload shape
# ---------------------------------------------------------------------------

def _system_text(messages) -> str:
    return str(messages[0].content)


def _user_payload(messages) -> dict:
    return json.loads(str(messages[1].content))


def test_score_uses_reference_prompt_when_expected_output_given():
    evaluator = _make_evaluator()
    stub = _Stub(JudgeResponse(thinking="r", score=4))
    evaluator._judge = stub  # type: ignore[assignment]

    asyncio.run(evaluator._score(input="q", output="a", expected_output="gt"))
    assert _system_text(stub.captured_messages) == _SYS_REF
    payload = _user_payload(stub.captured_messages)
    assert payload["input"] == "q"
    assert payload["output"] == "a"
    assert payload["expected_output"] == "gt"


def test_score_uses_no_reference_prompt_when_expected_output_none():
    evaluator = _make_evaluator()
    stub = _Stub(JudgeResponse(thinking="r", score=3))
    evaluator._judge = stub  # type: ignore[assignment]

    asyncio.run(evaluator._score(input="q", output="a", expected_output=None))
    assert _system_text(stub.captured_messages) == _SYS_NO_REF
    payload = _user_payload(stub.captured_messages)
    assert "expected_output" not in payload


def test_score_includes_trajectory_in_payload_when_provided():
    evaluator = _make_evaluator()
    stub = _Stub(JudgeResponse(thinking="r", score=5))
    evaluator._judge = stub  # type: ignore[assignment]

    trajectory = [
        AgentStep(
            thought="thinking...",
            reasoning=None,
            tool_calls=[ToolInvocation(name="search", args={"q": "x"}, output="result")],
        ),
    ]
    asyncio.run(evaluator._score(
        input="q", output="a", expected_output="gt", trajectory=trajectory,
    ))
    payload = _user_payload(stub.captured_messages)
    assert "trajectory" in payload
    assert payload["trajectory"][0]["thought"] == "thinking..."
    assert payload["trajectory"][0]["tool_calls"][0]["name"] == "search"


def test_score_omits_trajectory_when_none():
    evaluator = _make_evaluator()
    stub = _Stub(JudgeResponse(thinking="r", score=2))
    evaluator._judge = stub  # type: ignore[assignment]

    asyncio.run(evaluator._score(input="q", output="a", expected_output="gt"))
    payload = _user_payload(stub.captured_messages)
    assert "trajectory" not in payload


# ---------------------------------------------------------------------------
# Error propagation
# ---------------------------------------------------------------------------

def test_score_returns_error_when_ainvoke_raises():
    evaluator = _make_evaluator()
    evaluator._judge = _Stub(error=RuntimeError("rate limited"))  # type: ignore[assignment]

    score, explanation, error = asyncio.run(
        evaluator._score(input="q", output="a", expected_output="gt")
    )
    assert score is None
    assert explanation is None
    assert "rate limited" in error


def test_score_returns_thinking_as_explanation():
    evaluator = _make_evaluator()
    evaluator._judge = _Stub(JudgeResponse(thinking="chain of thought", score=4))  # type: ignore[assignment]

    _, explanation, _ = asyncio.run(
        evaluator._score(input="q", output="a", expected_output="gt")
    )
    assert explanation == "chain of thought"


# ---------------------------------------------------------------------------
# EvaluationConfig.max_retries — schema wiring
# ---------------------------------------------------------------------------

def test_evaluation_config_accepts_max_retries():
    from src.config.types import EvaluationConfig

    cfg = EvaluationConfig(
        method="llm_as_judge",
        litellm={"model": "x"},
        system_prompt="sys",
        system_prompt_no_reference="sys-nr",
        max_retries=5,
    )
    assert cfg.max_retries == 5


def test_evaluation_config_rejects_non_positive_max_retries():
    from pydantic import ValidationError
    from src.config.types import EvaluationConfig

    with pytest.raises(ValidationError):
        EvaluationConfig(
            method="llm_as_judge",
            litellm={"model": "x"},
            system_prompt="sys",
            system_prompt_no_reference="sys-nr",
            max_retries=0,
        )
