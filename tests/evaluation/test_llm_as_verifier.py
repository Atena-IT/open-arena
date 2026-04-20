"""Unit tests for the LLM-as-a-Verifier evaluator.

Covers the prompt rendering (name + description placeholders), tag-based
logprob extraction (max aggregation, case-fold, text fallback), the
Ollama → OpenAI-compat routing shim, and the criteria decomposition
path on the evaluator class.

All LLM I/O is mocked at the `litellm.acompletion` module attribute.
`ChatLiteLLM` calls that attribute via its own `self.client = litellm`
reference, so the same patch intercepts both the evaluator's code path
and any stray direct litellm calls.
"""

from __future__ import annotations

import asyncio
import math
from unittest.mock import patch

import pytest

from src.evaluation.evaluators.llm_as_verifier import (
    LLMAsVerifierEvaluator,
    _as_criterion,
    _build_score_tokens,
    _coerce_ollama_to_openai_compat,
    _extract_score_at_tag,
    _pairwise_reward,
    _render_prompt,
    _scale_description,
    _score_letter,
)
from src.llms.base import build_chat_model


# ---------------------------------------------------------------------------
# _as_criterion / _build_score_tokens / _scale_description
# ---------------------------------------------------------------------------

def test_as_criterion_accepts_dict():
    out = _as_criterion({"name": "x", "description": "y"})
    assert out == {"name": "x", "description": "y"}


def test_as_criterion_rejects_string():
    with pytest.raises(TypeError, match="name"):
        _as_criterion("just a name")


def test_as_criterion_accepts_pydantic_model():
    from src.config.types import CriterionConfig
    out = _as_criterion(CriterionConfig(name="x", description="y"))
    assert out == {"name": "x", "description": "y"}


def test_build_score_tokens_basic():
    assert _build_score_tokens(2) == ["A", "B"]
    assert _build_score_tokens(8) == list("ABCDEFGH")
    assert _build_score_tokens(20) == list("ABCDEFGHIJKLMNOPQRST")
    assert _build_score_tokens(26)[-1] == "Z"


@pytest.mark.parametrize("g", [1, 0, -1, 27, 100])
def test_build_score_tokens_rejects_out_of_range(g):
    with pytest.raises(ValueError, match="granularity"):
        _build_score_tokens(g)


def test_scale_description_mentions_endpoints_and_size():
    text = _scale_description(list("ABCDEFGH"))
    assert "8-point" in text
    assert "A through H" in text
    assert "best" in text
    assert "worst" in text


# ---------------------------------------------------------------------------
# _render_prompt
# ---------------------------------------------------------------------------

def test_render_prompt_substitutes_all_placeholders():
    prompt = (
        "G={granularity} letters={score_letters} best={best_letter} "
        "worst={worst_letter} crit={criterion} desc={criterion_description} "
        "scale={scale_description}"
    )
    rendered = _render_prompt(
        prompt,
        ["A", "B", "C"],
        {"name": "clarity", "description": "explain things clearly"},
    )
    assert "G=3" in rendered
    assert "letters=A, B, C" in rendered
    assert "best=A" in rendered
    assert "worst=C" in rendered
    assert "crit=clarity" in rendered
    assert "desc=explain things clearly" in rendered
    assert "scale=Rate each output" in rendered  # _scale_description prefix


def test_render_prompt_tolerates_stray_braces():
    """User prompts may contain literal JSON braces — replace() must not
    choke on them the way str.format would."""
    prompt = '{{"output": "{}"}} G={granularity}'
    out = _render_prompt(prompt, ["A", "B"], {"name": "x", "description": "y"})
    assert "G=2" in out
    assert '{{"output": "{}"}}' in out


# ---------------------------------------------------------------------------
# _score_letter
# ---------------------------------------------------------------------------

SCORE_TOKENS: set[str] = set("ABCDEFGH")


@pytest.mark.parametrize(
    "token,expected",
    [
        ("A", "A"),
        ("H", "H"),
        # Lower-case (reference treats A and a as equivalent).
        ("a", "A"),
        ("h", "H"),
        # Trailing whitespace / newline.
        ("A ", "A"),
        ("B\n", "B"),
        ("a\t", "A"),
        # Leading punctuation (`<score_A>` + `>A` split).
        (">A", "A"),
        (":A", "A"),
        ("\tC", "C"),
        # Trailing punctuation (`<score_A>A</` split).
        ("A>", "A"),
        ("A:", "A"),
        ('A"', "A"),
        ("A<", "A"),
    ],
)
def test_score_letter_accepts(token, expected):
    assert _score_letter(token, SCORE_TOKENS) == expected


@pytest.mark.parametrize(
    "token",
    [
        "",
        "   ",
        # Adjacent letters → ambiguous, must reject.
        "Alpha",
        "AB",
        "aA",
        "Aa",
        # Out-of-range letters.
        "Z",
        "1",
        "score",
    ],
)
def test_score_letter_rejects(token):
    assert _score_letter(token, SCORE_TOKENS) is None


def test_score_letter_handles_none_token():
    assert _score_letter(None, SCORE_TOKENS) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _extract_score_at_tag — fed (content, text) directly
# ---------------------------------------------------------------------------

def _logprob_entry(token: str, top: list[tuple[str, float]] | None = None) -> dict:
    return {
        "token": token,
        "top_logprobs": [{"token": t, "logprob": lp} for t, lp in (top or [])],
    }


def _tag_tokens(tag: str) -> list[dict]:
    """Tokenize a tag string char-by-char as plain entries with no logprobs;
    enough to drive `_extract_score_at_tag`'s text-accumulation logic."""
    return [_logprob_entry(ch) for ch in tag]


def test_extract_score_at_tag_normalises_to_one():
    """80% A, 20% B at the position right after `<score_A>`."""
    content = _tag_tokens("<score_A>") + [
        _logprob_entry("A", [("A", math.log(0.8)), ("B", math.log(0.2))]),
    ]
    dist = _extract_score_at_tag(content, None, SCORE_TOKENS, "<score_A>")
    assert dist is not None
    assert dist["A"] == pytest.approx(0.8)
    assert dist["B"] == pytest.approx(0.2)
    assert sum(dist.values()) == pytest.approx(1.0)


def test_extract_score_at_tag_uses_max_for_duplicate_letter_tokens():
    """`A`, ` A`, `a` all map to letter A — reference takes the max so
    the same letter isn't double-counted across alternative spellings."""
    content = _tag_tokens("<score_A>") + [
        _logprob_entry(
            "A",
            [
                ("A", math.log(0.5)),
                (" A", math.log(0.3)),
                ("a", math.log(0.2)),
                ("B", math.log(0.4)),
            ],
        ),
    ]
    dist = _extract_score_at_tag(content, None, SCORE_TOKENS, "<score_A>")
    assert dist is not None
    # Pre-normalisation: max(A) = 0.5, B = 0.4; total = 0.9. After
    # normalisation: A = 5/9, B = 4/9.
    assert dist["A"] == pytest.approx(5 / 9)
    assert dist["B"] == pytest.approx(4 / 9)


def test_extract_score_at_tag_finds_score_b_after_score_a():
    """The tag-finding walk must locate `<score_B>` even though
    `<score_A>` appears earlier in the same response."""
    content = (
        _tag_tokens("<score_A>")
        + [_logprob_entry("A", [("A", math.log(0.99))])]
        + _tag_tokens("</score_A><score_B>")
        + [_logprob_entry("D", [("D", math.log(0.6)), ("E", math.log(0.4))])]
    )
    dist_a = _extract_score_at_tag(content, None, SCORE_TOKENS, "<score_A>")
    dist_b = _extract_score_at_tag(content, None, SCORE_TOKENS, "<score_B>")
    assert dist_a is not None and dist_a["A"] == pytest.approx(1.0)
    assert dist_b is not None
    assert dist_b["D"] == pytest.approx(0.6)
    assert dist_b["E"] == pytest.approx(0.4)


def test_extract_score_at_tag_falls_back_to_text_when_no_logprobs():
    """When logprobs content is empty, the reference parses the score
    letter from the assistant's text."""
    text = "some reasoning <score_A>C</score_A><score_B>F</score_B>"
    dist_a = _extract_score_at_tag([], text, SCORE_TOKENS, "<score_A>")
    dist_b = _extract_score_at_tag([], text, SCORE_TOKENS, "<score_B>")
    assert dist_a == {"C": 1.0}
    assert dist_b == {"F": 1.0}


def test_extract_score_at_tag_returns_none_when_tag_not_found_anywhere():
    content = [_logprob_entry("hello", [("hello", math.log(0.99))])]
    text = "no score tag here"
    assert _extract_score_at_tag(content, text, SCORE_TOKENS, "<score_A>") is None


def test_extract_score_at_tag_returns_none_when_content_empty_and_no_text():
    assert _extract_score_at_tag([], None, SCORE_TOKENS, "<score_A>") is None
    assert _extract_score_at_tag(None, None, SCORE_TOKENS, "<score_A>") is None


# ---------------------------------------------------------------------------
# _coerce_ollama_to_openai_compat
# ---------------------------------------------------------------------------

def test_coerce_rewrites_ollama_chat_to_openai_compat():
    cfg = {"model": "ollama_chat/qwen2.5:7b", "temperature": 0.0}
    out = _coerce_ollama_to_openai_compat(cfg)
    assert out["model"] == "openai/qwen2.5:7b"
    assert out["api_base"].endswith("/v1")
    assert out["api_key"] == "ollama"
    assert cfg["model"] == "ollama_chat/qwen2.5:7b"


def test_coerce_rewrites_bare_ollama_prefix():
    out = _coerce_ollama_to_openai_compat({"model": "ollama/llama3:8b"})
    assert out["model"] == "openai/llama3:8b"


def test_coerce_preserves_user_supplied_api_base_and_key():
    cfg = {
        "model": "ollama/llama3",
        "api_base": "http://custom:9000/v1",
        "api_key": "secret",
    }
    out = _coerce_ollama_to_openai_compat(cfg)
    assert out["api_base"] == "http://custom:9000/v1"
    assert out["api_key"] == "secret"


def test_coerce_appends_v1_when_user_supplies_native_ollama_base():
    """User-supplied `api_base: http://host:11434` (the native Ollama
    endpoint) must be rewritten to `.../v1` so OpenAI-compat dispatch
    hits the right path."""
    cfg = {"model": "ollama_chat/qwen3:8b", "api_base": "http://localhost:11434"}
    out = _coerce_ollama_to_openai_compat(cfg)
    assert out["api_base"] == "http://localhost:11434/v1"


def test_coerce_tolerates_trailing_slash_on_user_supplied_base():
    cfg = {"model": "ollama/llama3", "api_base": "http://localhost:11434/"}
    out = _coerce_ollama_to_openai_compat(cfg)
    assert out["api_base"] == "http://localhost:11434/v1"


def test_coerce_passes_through_non_ollama_models():
    cfg = {"model": "openai/gpt-4o-mini", "temperature": 0.2}
    assert _coerce_ollama_to_openai_compat(cfg) == cfg


# ---------------------------------------------------------------------------
# _pairwise_reward + evaluator end-to-end — mocked at litellm.acompletion
# ---------------------------------------------------------------------------
#
# `ChatLiteLLM` calls `self.client.acompletion(...)` where `self.client` is
# the litellm module. Patching `llm_as_verifier.litellm.acompletion`
# therefore intercepts the call made by LangChain without any extra
# plumbing. The response must be dict-shaped so `_create_chat_result` can
# walk `response["choices"]`, `res["message"]`, `res.get("logprobs")`, etc.


def _litellm_response(content_entries: list[dict], text: str = "") -> dict:
    """Build a dict-shaped response that mimics enough of litellm's
    ModelResponse for `ChatLiteLLM._create_chat_result` to consume."""
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": text},
                "logprobs": {"content": content_entries} if content_entries else None,
                "finish_reason": "stop",
            }
        ],
        "usage": {},
    }


def _pairwise_response(top_a, top_b):
    """Build a litellm-shaped response with logprobs at `<score_A>` and
    `<score_B>`."""
    content = (
        _tag_tokens("<score_A>")
        + [_logprob_entry("A", top_a)]
        + _tag_tokens("</score_A><score_B>")
        + [_logprob_entry("A", top_b)]
    )
    return _litellm_response(content)


def _async_return(value):
    async def _fn(**_kwargs):
        return value
    return _fn


def _model(model_id: str = "openai/gpt-x"):
    """Build a ChatLiteLLM for tests. litellm calls go through the patched
    `litellm.acompletion`."""
    return build_chat_model({"model": model_id})


def test_pairwise_reward_normalises_to_zero_one_range():
    """Pure A on score_A (best → 1.0); pure worst-letter on score_B (→ 0.0)."""
    response = _pairwise_response(
        top_a=[("A", math.log(1.0))],
        top_b=[("H", math.log(1.0))],
    )
    with patch(
        "src.evaluation.evaluators.llm_as_verifier.litellm.acompletion",
        _async_return(response),
    ):
        rewards, error = asyncio.run(
            _pairwise_reward(
                model=_model(),
                callbacks=[],
                system_prompt="sys",
                user_payload="payload",
                granularity=8,
                repeats=1,
            )
        )
    assert error is None
    assert rewards == pytest.approx((1.0, 0.0))


def test_pairwise_reward_uses_expected_value_under_distribution():
    # 50/50 over A (1.0) and H (0.0) on an 8-letter scale → reward 0.5.
    same = [("A", math.log(0.5)), ("H", math.log(0.5))]
    response = _pairwise_response(top_a=same, top_b=same)
    with patch(
        "src.evaluation.evaluators.llm_as_verifier.litellm.acompletion",
        _async_return(response),
    ):
        rewards, error = asyncio.run(
            _pairwise_reward(
                model=_model(),
                callbacks=[],
                system_prompt="sys",
                user_payload="payload",
                granularity=8,
                repeats=1,
            )
        )
    assert error is None
    assert rewards == pytest.approx((0.5, 0.5))


def test_pairwise_reward_averages_over_k_samples():
    """K samples are fired in parallel; the reward is their mean."""
    letters = [("A", math.log(1.0)), ("B", math.log(1.0)),
               ("C", math.log(1.0)), ("D", math.log(1.0))]
    # value_map for G=4: A=1.0, B=2/3, C=1/3, D=0.0 → mean = 0.5.
    responses = iter([
        _pairwise_response(top_a=[top], top_b=[("A", math.log(1.0))])
        for top in letters
    ])
    call_count = 0

    async def _serve(**_kwargs):
        nonlocal call_count
        call_count += 1
        return next(responses)

    with patch(
        "src.evaluation.evaluators.llm_as_verifier.litellm.acompletion",
        _serve,
    ):
        rewards, error = asyncio.run(
            _pairwise_reward(
                model=_model(),
                callbacks=[],
                system_prompt="sys",
                user_payload="payload",
                granularity=4,
                repeats=4,
            )
        )
    assert error is None
    assert call_count == 4
    assert rewards is not None
    # Mean of (1.0, 2/3, 1/3, 0.0) = 0.5 on side A; pure A on side B = 1.0.
    assert rewards[0] == pytest.approx(0.5)
    assert rewards[1] == pytest.approx(1.0)


def test_pairwise_reward_returns_error_when_completion_raises():
    async def _boom(**_kwargs):
        raise RuntimeError("rate limited")

    with patch(
        "src.evaluation.evaluators.llm_as_verifier.litellm.acompletion",
        _boom,
    ):
        rewards, error = asyncio.run(
            _pairwise_reward(
                model=_model(),
                callbacks=[],
                system_prompt="sys",
                user_payload="payload",
                granularity=4,
                repeats=2,
            )
        )
    assert rewards is None
    assert "rate limited" in error


def test_pairwise_reward_reports_missing_second_score():
    """Only score_A logprobs present → missing score_B is reported."""
    content = _tag_tokens("<score_A>") + [_logprob_entry("A", [("A", math.log(1.0))])]
    response = _litellm_response(content)
    with patch(
        "src.evaluation.evaluators.llm_as_verifier.litellm.acompletion",
        _async_return(response),
    ):
        rewards, error = asyncio.run(
            _pairwise_reward(
                model=_model(),
                callbacks=[],
                system_prompt="sys",
                user_payload="payload",
                granularity=4,
                repeats=1,
            )
        )
    assert rewards is None
    assert "score_A" in error or "score_B" in error


def test_pairwise_reward_forwards_callbacks_to_langchain():
    """`callbacks` must be threaded through `model.agenerate` so Langfuse's
    CallbackHandler (or any other LangChain callback) fires on every
    grader call."""
    from langchain_core.callbacks import BaseCallbackHandler

    class _RecordingCallback(BaseCallbackHandler):
        def __init__(self):
            self.starts = 0
            self.ends = 0

        def on_llm_start(self, *_args, **_kwargs):
            self.starts += 1

        def on_llm_end(self, *_args, **_kwargs):
            self.ends += 1

    cb = _RecordingCallback()
    response = _pairwise_response(
        top_a=[("A", math.log(1.0))],
        top_b=[("D", math.log(1.0))],
    )
    with patch(
        "src.evaluation.evaluators.llm_as_verifier.litellm.acompletion",
        _async_return(response),
    ):
        asyncio.run(
            _pairwise_reward(
                model=_model(),
                callbacks=[cb],
                system_prompt="sys",
                user_payload="payload",
                granularity=4,
                repeats=1,
            )
        )
    # The LangChain run manager fires `on_llm_start` and `on_llm_end` for
    # every completion; if callbacks weren't threaded through, both stay 0.
    assert cb.starts == 1
    assert cb.ends == 1


# ---------------------------------------------------------------------------
# LLMAsVerifierEvaluator — end-to-end (mocked litellm)
# ---------------------------------------------------------------------------

CRIT_CORRECTNESS = {"name": "correctness", "description": "Is the answer right?"}
CRIT_CLARITY = {"name": "clarity", "description": "Is the answer clear?"}

# Minimal test prompt — the evaluator just needs a non-empty string that
# renders through `_render_prompt`. The actual prompt text is irrelevant
# when `litellm.acompletion` is mocked.
_TEST_PROMPT = "Judge on {criterion}: {criterion_description}"


def _make_evaluator(criteria=None, granularity=4) -> LLMAsVerifierEvaluator:
    """Build an evaluator with no groups; we drive `_verify_pair` directly
    so we don't need real ExecutionResults or a Langfuse client."""
    return LLMAsVerifierEvaluator(
        groups=[],
        llm_config={"model": "openai/gpt-x"},
        system_prompt=_TEST_PROMPT,
        system_prompt_no_reference=_TEST_PROMPT,
        granularity=granularity,
        repeats=1,
        criteria=criteria,
    )


def test_verify_pair_single_holistic_call_when_no_criteria():
    response = _pairwise_response(
        top_a=[("A", math.log(1.0))],
        top_b=[("D", math.log(1.0))],
    )
    call_count = 0

    async def _spy(**_kwargs):
        nonlocal call_count
        call_count += 1
        return response

    evaluator = _make_evaluator(granularity=4)
    with patch(
        "src.evaluation.evaluators.llm_as_verifier.litellm.acompletion", _spy
    ):
        rewards, error = asyncio.run(
            evaluator._verify_pair(
                "in", "a", "out_a", "b", "out_b",
                expected_output="ref", trajectory_a=None, trajectory_b=None,
            )
        )
    assert error is None
    assert rewards == pytest.approx((1.0, 0.0))
    assert call_count == 1


def test_verify_pair_averages_rewards_across_criteria():
    """Two criteria where (R_A, R_B) is (1.0, 0.0) and (0.0, 1.0)
    → averaged per-pair reward is (0.5, 0.5)."""
    high_a = _pairwise_response(
        top_a=[("A", math.log(1.0))],
        top_b=[("D", math.log(1.0))],
    )
    high_b = _pairwise_response(
        top_a=[("D", math.log(1.0))],
        top_b=[("A", math.log(1.0))],
    )
    responses = iter([high_a, high_b])
    call_count = 0

    async def _serve(**_kwargs):
        nonlocal call_count
        call_count += 1
        return next(responses)

    evaluator = _make_evaluator(criteria=[CRIT_CORRECTNESS, CRIT_CLARITY], granularity=4)
    with patch(
        "src.evaluation.evaluators.llm_as_verifier.litellm.acompletion", _serve
    ):
        rewards, error = asyncio.run(
            evaluator._verify_pair(
                "in", "a", "out_a", "b", "out_b",
                expected_output="ref", trajectory_a=None, trajectory_b=None,
            )
        )
    assert error is None
    assert rewards == pytest.approx((0.5, 0.5))
    assert call_count == 2


def test_verify_pair_tolerates_partial_criterion_failure():
    """If one criterion fails the others still produce a reward; the
    remaining criteria are averaged into the per-pair reward."""
    good = _pairwise_response(
        top_a=[("A", math.log(1.0))],
        top_b=[("D", math.log(1.0))],
    )
    bad = _litellm_response([])  # no logprobs at all → triggers error path
    responses = iter([good, bad])

    async def _serve(**_kwargs):
        return next(responses)

    evaluator = _make_evaluator(criteria=[CRIT_CORRECTNESS, CRIT_CLARITY], granularity=4)
    with patch(
        "src.evaluation.evaluators.llm_as_verifier.litellm.acompletion", _serve
    ):
        rewards, error = asyncio.run(
            evaluator._verify_pair(
                "in", "a", "out_a", "b", "out_b",
                expected_output="ref", trajectory_a=None, trajectory_b=None,
            )
        )
    assert rewards == pytest.approx((1.0, 0.0))
    # `_verify_pair` only reports an error when *every* criterion fails.
    assert error is None


def test_verify_pair_returns_error_when_all_criteria_fail():
    bad = _litellm_response([])

    async def _serve(**_kwargs):
        return bad

    evaluator = _make_evaluator(
        criteria=[
            {"name": "a", "description": "x"},
            {"name": "b", "description": "y"},
        ],
        granularity=4,
    )
    with patch(
        "src.evaluation.evaluators.llm_as_verifier.litellm.acompletion", _serve
    ):
        rewards, error = asyncio.run(
            evaluator._verify_pair(
                "in", "a", "out_a", "b", "out_b",
                expected_output=None, trajectory_a=None, trajectory_b=None,
            )
        )
    assert rewards is None
    assert "all criteria failed" in error


# ---------------------------------------------------------------------------
# EvaluationConfig.criteria — schema-level guardrails
# ---------------------------------------------------------------------------

def test_evaluation_config_rejects_string_criteria_with_clear_error():
    """The pre-decomposition format `criteria: ["foo", "bar"]` must error
    with a message that points the user at the new shape."""
    from pydantic import ValidationError

    from src.config.types import EvaluationConfig

    with pytest.raises(ValidationError) as excinfo:
        EvaluationConfig(
            method="llm_as_verifier",
            litellm={"model": "x"},
            system_prompt="sys",
            system_prompt_no_reference="sys-nr",
            criteria=["correctness", "clarity"],
        )
    assert "name" in str(excinfo.value) and "description" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Score bounds — match the [0, 1] contract used by other evaluators
# ---------------------------------------------------------------------------

def _eval_with_two_models(response, *, criteria=None, granularity=4):
    """Drive `_score_group` on a 2-model group and return the score dict."""
    evaluator = LLMAsVerifierEvaluator(
        groups=[],
        llm_config={"model": "openai/gpt-x"},
        system_prompt=_TEST_PROMPT,
        system_prompt_no_reference=_TEST_PROMPT,
        granularity=granularity,
        repeats=1,
        criteria=criteria,
    )
    with patch(
        "src.evaluation.evaluators.llm_as_verifier.litellm.acompletion",
        _async_return(response),
    ):
        scores, _explanation, _error = asyncio.run(
            evaluator._score_group(
                input="in", outputs={"a": "out_a", "b": "out_b"},
                expected_output="ref",
            )
        )
    return scores


@pytest.mark.parametrize(
    "top_a,top_b,expected",
    [
        # Pure best on A, pure worst on B → (1.0, 0.0).
        ([("A", math.log(1.0))], [("D", math.log(1.0))], (1.0, 0.0)),
        # Mirror.
        ([("D", math.log(1.0))], [("A", math.log(1.0))], (0.0, 1.0)),
        # Uniform → (1/3, 1/3) for granularity=4 with letters A..D collapsing
        # to value_map {A:1.0, B:2/3, C:1/3, D:0}; uniform expectation = 0.5.
        (
            [("A", math.log(0.25)), ("B", math.log(0.25)),
             ("C", math.log(0.25)), ("D", math.log(0.25))],
            [("A", math.log(0.25)), ("B", math.log(0.25)),
             ("C", math.log(0.25)), ("D", math.log(0.25))],
            (0.5, 0.5),
        ),
        # 50/50 best-vs-worst on both sides.
        (
            [("A", math.log(0.5)), ("D", math.log(0.5))],
            [("A", math.log(0.5)), ("D", math.log(0.5))],
            (0.5, 0.5),
        ),
    ],
)
def test_score_group_returns_values_in_unit_interval(top_a, top_b, expected):
    """Per-model scores must lie in [0, 1] regardless of the distribution
    the verifier emits — same range as the LLM judge ((score-1)/4)."""
    response = _pairwise_response(top_a=top_a, top_b=top_b)
    scores = _eval_with_two_models(response, granularity=4)
    assert scores is not None
    assert set(scores) == {"a", "b"}
    for m, s in scores.items():
        assert 0.0 <= s <= 1.0, f"{m} score {s} out of [0, 1]"
    assert (scores["a"], scores["b"]) == pytest.approx(expected)


def test_score_group_stays_in_unit_interval_under_criteria_decomposition():
    """With multiple criteria, the per-pair reward is the mean of per-criterion
    rewards — so the per-model score stays in [0, 1]."""
    high_a = _pairwise_response(
        top_a=[("A", math.log(1.0))], top_b=[("D", math.log(1.0))],
    )
    high_b = _pairwise_response(
        top_a=[("D", math.log(1.0))], top_b=[("A", math.log(1.0))],
    )
    responses = iter([high_a, high_b])

    async def _serve(**_kwargs):
        return next(responses)

    evaluator = LLMAsVerifierEvaluator(
        groups=[],
        llm_config={"model": "openai/gpt-x"},
        system_prompt=_TEST_PROMPT,
        system_prompt_no_reference=_TEST_PROMPT,
        granularity=4,
        repeats=1,
        criteria=[CRIT_CORRECTNESS, CRIT_CLARITY],
    )
    with patch(
        "src.evaluation.evaluators.llm_as_verifier.litellm.acompletion", _serve
    ):
        scores, _explanation, _error = asyncio.run(
            evaluator._score_group(
                input="in", outputs={"a": "out_a", "b": "out_b"},
                expected_output="ref",
            )
        )
    assert scores is not None
    for m, s in scores.items():
        assert 0.0 <= s <= 1.0
    # Two opposite criteria → per-pair reward (0.5, 0.5).
    assert scores["a"] == pytest.approx(0.5)
    assert scores["b"] == pytest.approx(0.5)


def test_score_group_text_fallback_score_is_in_unit_interval():
    """Text fallback returns a delta distribution → reward is exactly
    `value_map[letter]`, which is also in [0, 1]."""
    response = _litellm_response(
        content_entries=[],
        text="<score_A>A</score_A><score_B>D</score_B>",
    )
    scores = _eval_with_two_models(response, granularity=4)
    assert scores == {"a": pytest.approx(1.0), "b": pytest.approx(0.0)}


def test_evaluation_config_accepts_dict_criteria():
    from src.config.types import EvaluationConfig

    cfg = EvaluationConfig(
        method="llm_as_verifier",
        litellm={"model": "x"},
        system_prompt="sys",
        system_prompt_no_reference="sys-nr",
        criteria=[
            {"name": "correctness", "description": "Is it right?"},
            {"name": "clarity", "description": "Is it clear?"},
        ],
    )
    assert len(cfg.criteria) == 2
    assert cfg.criteria[0].name == "correctness"
    assert cfg.criteria[0].description == "Is it right?"
