"""Logprob-based reward helpers for the LLM-as-a-Verifier evaluators.

Implements the reward R(t, τ) = (1/CK) Σ_c Σ_k Σ_g p(v_g|t,c,τ)·φ(v_g)
from Kwok et al. 2026.

* `verifier_reward` — pointwise: one output, one score tag, one reward.
* `verifier_pairwise_reward` — pairwise: two outputs in a single prompt,
  extract logprob distributions at both `<score_A>` and `<score_B>`.

Both helpers reuse the same score-letter extraction and the Ollama →
OpenAI-compat routing so `logprobs`/`top_logprobs` flow through.
"""

import logging
import math
import os
from typing import Any

import litellm

_logger = logging.getLogger(__name__)

_DEFAULT_OLLAMA_OPENAI_BASE = "http://localhost:11434/v1"


def _coerce_ollama_to_openai_compat(cfg: dict[str, Any]) -> dict[str, Any]:
    """Route Ollama calls through the OpenAI-compatible endpoint.

    LiteLLM's dedicated `ollama`/`ollama_chat` providers currently reject
    `logprobs`/`top_logprobs` (see BerriAI/litellm#17165), but Ollama itself
    (>= 0.12.11) returns them on `/v1/chat/completions`. Rewriting the
    provider prefix to `openai/` with `api_base=<ollama>/v1` lets the
    params flow through untouched.
    """
    model = cfg.get("model")
    if not isinstance(model, str):
        return cfg
    for prefix in ("ollama_chat/", "ollama/"):
        if model.startswith(prefix):
            bare = model[len(prefix) :]
            rewritten = dict(cfg)
            rewritten["model"] = f"openai/{bare}"
            rewritten.setdefault(
                "api_base", os.getenv("OLLAMA_API_BASE", _DEFAULT_OLLAMA_OPENAI_BASE)
            )
            rewritten.setdefault("api_key", "ollama")
            _logger.info(
                "verifier: routing %s via OpenAI-compat endpoint %s (logprobs support)",
                model,
                rewritten["api_base"],
            )
            return rewritten
    return cfg


def build_score_tokens(granularity: int) -> list[str]:
    """Score letters ordered best → worst (A is best)."""
    if granularity < 2 or granularity > 26:
        raise ValueError(f"granularity must be in [2, 26]; got {granularity}")
    return [chr(ord("A") + i) for i in range(granularity)]


def render_verifier_prompt(
    prompt: str, score_tokens: list[str], criterion: str = "overall quality"
) -> str:
    """Substitute the standard verifier placeholders without requiring full
    `str.format` semantics — user-overridden prompts may contain stray
    braces (e.g. JSON snippets) that would break `str.format`.

    `criterion` fills the `{criterion}` placeholder used by the pointwise
    verifier for criteria decomposition (Phase 3). Prompts that do not
    contain `{criterion}` are unaffected.
    """
    return (
        prompt.replace("{granularity}", str(len(score_tokens)))
        .replace("{score_letters}", ", ".join(score_tokens))
        .replace("{best_letter}", score_tokens[0])
        .replace("{worst_letter}", score_tokens[-1])
        .replace("{criterion}", criterion)
    )


async def verifier_reward(
    llm_config: dict[str, Any],
    system_prompt: str,
    user_payload: str,
    granularity: int = 8,
    repeats: int = 1,
) -> tuple[float | None, str | None]:
    """Sample the verifier `repeats` times and return the average expected score.

    Returns `(reward, error)`. If every sample fails (no logprobs, API
    error, no score token found), `reward` is None and `error` holds the
    most useful message.
    """
    score_tokens = build_score_tokens(granularity)
    n = len(score_tokens)
    # Normalise score letters to [0, 1] so the verifier's Langfuse score is
    # directly comparable with the judge evaluators (which also emit [0, 1]).
    # A (best) → 1.0, worst → 0.0. EvaluationConfig enforces granularity >= 2
    # so n - 1 >= 1 and there is no divide-by-zero risk.
    value_map = {tok: (n - 1 - i) / (n - 1) for i, tok in enumerate(score_tokens)}

    completion_kwargs = {k: v for k, v in llm_config.items() if v is not None}
    completion_kwargs = _coerce_ollama_to_openai_compat(completion_kwargs)
    completion_kwargs.pop("tools", None)
    completion_kwargs.pop("tool_choice", None)
    completion_kwargs["logprobs"] = True
    completion_kwargs["top_logprobs"] = granularity
    completion_kwargs["messages"] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_payload},
    ]

    rewards: list[float] = []
    errors: list[str] = []
    for _ in range(max(1, repeats)):
        try:
            response = await litellm.acompletion(**completion_kwargs)
        except Exception as e:
            errors.append(f"completion failed: {e}")
            continue
        dists = _extract_score_distributions(response, set(score_tokens), n=1)
        if not dists:
            errors.append("no score-token logprobs in response")
            continue
        expected = sum(prob * value_map[tok] for tok, prob in dists[0].items())
        rewards.append(expected)

    if not rewards:
        return None, "; ".join(dict.fromkeys(errors)) or "no usable samples"
    return sum(rewards) / len(rewards), None


async def verifier_pairwise_reward(
    llm_config: dict[str, Any],
    system_prompt: str,
    user_payload: str,
    granularity: int = 8,
    repeats: int = 1,
) -> tuple[tuple[float, float] | None, str | None]:
    """Pairwise variant: one prompt with both trajectories, extract two
    score-letter distributions (`<score_A>`, `<score_B>`), return
    `(R_A, R_B)` averaged over `repeats` samples.

    Matches the paper's headline methodology (Kwok et al. 2026, §3):
    the grader sees A and B together, which mitigates absolute-scoring
    drift relative to separate pointwise calls.
    """
    score_tokens = build_score_tokens(granularity)
    n = len(score_tokens)
    value_map = {tok: (n - 1 - i) / (n - 1) for i, tok in enumerate(score_tokens)}

    completion_kwargs = {k: v for k, v in llm_config.items() if v is not None}
    completion_kwargs = _coerce_ollama_to_openai_compat(completion_kwargs)
    completion_kwargs.pop("tools", None)
    completion_kwargs.pop("tool_choice", None)
    completion_kwargs["logprobs"] = True
    completion_kwargs["top_logprobs"] = granularity
    completion_kwargs["messages"] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_payload},
    ]

    rewards_a: list[float] = []
    rewards_b: list[float] = []
    errors: list[str] = []
    for _ in range(max(1, repeats)):
        try:
            response = await litellm.acompletion(**completion_kwargs)
        except Exception as e:
            errors.append(f"completion failed: {e}")
            continue
        dists = _extract_score_distributions(response, set(score_tokens), n=2)
        if not dists or len(dists) < 2:
            errors.append("missing <score_A>/<score_B> logprobs in response")
            continue
        rewards_a.append(sum(p * value_map[t] for t, p in dists[0].items()))
        rewards_b.append(sum(p * value_map[t] for t, p in dists[1].items()))

    if not rewards_a:
        return None, "; ".join(dict.fromkeys(errors)) or "no usable samples"
    return (sum(rewards_a) / len(rewards_a), sum(rewards_b) / len(rewards_b)), None


def _score_letter(token: str, score_tokens: set[str]) -> str | None:
    """Return the score letter carried by a token, or None.

    Handles three tokenizer patterns we see in practice:
    * a bare letter (`"A"`),
    * a letter fused with surrounding punctuation (`">A"`, `":A"`, `"\tA"`),
    * a letter with trailing whitespace (`"A "`, `"A\n"`).
    Anything where the letter is adjacent to another letter (e.g. `"Alpha"`)
    is rejected to avoid false positives.
    """
    stripped = (token or "").strip()
    if not stripped:
        return None
    if stripped in score_tokens:
        return stripped
    last = stripped[-1]
    if last in score_tokens and (len(stripped) == 1 or not stripped[-2].isalpha()):
        return last
    return None


def _extract_score_distributions(
    response: Any, score_tokens: set[str], n: int = 1
) -> list[dict[str, float]] | None:
    """Find the first `n` response tokens that carry a score letter and
    return their normalised probability distributions, in order.

    Because tokenizers often fuse the score letter with adjacent
    punctuation (e.g. `">A"`), we group `top_logprobs` entries by the
    score letter they carry and sum their probabilities per letter.
    """

    choices = _get(response, "choices", default=[]) or []
    if not choices:
        return None
    logprobs = _get(choices[0], "logprobs", default=None)
    if logprobs is None:
        return None
    content = _get(logprobs, "content", default=None) or []

    dists: list[dict[str, float]] = []
    for token_entry in content:
        tok_text = _get(token_entry, "token", default="") or ""
        if _score_letter(tok_text, score_tokens) is None:
            continue
        top = _get(token_entry, "top_logprobs", default=[]) or []
        probs: dict[str, float] = {}
        for entry in top:
            t = _get(entry, "token", default="") or ""
            lp = _get(entry, "logprob", default=None)
            letter = _score_letter(t, score_tokens)
            if letter is None or lp is None:
                continue
            probs[letter] = probs.get(letter, 0.0) + math.exp(float(lp))
        if not probs:
            continue
        total = sum(probs.values())
        if total <= 0:
            continue
        dists.append({k: v / total for k, v in probs.items()})
        if len(dists) >= n:
            break
    return dists or None


def _get(obj: Any, attr: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)
