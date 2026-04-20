"""LLM-as-a-Verifier evaluator (pairwise, logprob-based).

Faithful port of Kwok et al. 2026 — see
https://github.com/llm-as-a-verifier/llm-as-a-verifier
(scripts/verifier_core.py is the canonical reference). The reward is

  R(t,τ) = (1/CK) Σ_c Σ_k Σ_g p_θ(v_g | t, c, τ) · φ(v_g)

where:
  G — score-token granularity (number of discrete score letters A..)
  K — repeated verification samples per (pair, criterion)
  C — number of evaluation criteria (decomposition; if 1, holistic)

Each (pair × criterion × repeat) is a single verifier call. Both
trajectories sit in one prompt so the grader emits two score tags
(`<score_A>X</score_A>` / `<score_B>Y</score_B>`); we recover the score
distributions by locating the position right after each tag in the
returned token stream and reading its `top_logprobs`.

Per-row provider calls = num_pairs · max(C, 1) · K. Pairs are unordered
(`itertools.combinations`) — the reference does NOT swap A/B; variance
reduction comes from K and from criterion decomposition.

The grader model is wrapped in `ChatLiteLLM` so LangChain callbacks
(Langfuse's `CallbackHandler`, retries, streaming, ...) fire on every
grader call exactly as they do for `LLMAsJudgeEvaluator`. `logprobs` and
`top_logprobs` are forwarded through `agenerate`'s kwargs to litellm's
completion API and surfaced back via `ChatGeneration.generation_info`.
"""

import asyncio
import itertools
import json
import logging
import math
import os
import re
from dataclasses import asdict
from typing import Any

# `litellm` is imported (unused at call time) so tests can patch
# `llm_as_verifier.litellm.acompletion` — ChatLiteLLM internally calls
# `litellm.acompletion` via its own `self.client` reference to the same
# module, so the patch intercepts both code paths.
import litellm  # noqa: F401
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel

from src.evaluation.base import GroupEvaluator
from src.execution import ExecutionResult
from src.llms import AgentStep
from src.llms.base import build_chat_model, to_langchain_messages

_logger = logging.getLogger(__name__)
_DEFAULT_OLLAMA_OPENAI_BASE = "http://localhost:11434/v1"

# Default holistic criterion when the user passes no `criteria` list.
# Mirrors the pre-decomposition pointwise behaviour: reference mode →
# correctness; no-reference → overall quality.
_DEFAULT_CRITERION_REFERENCE = {
    "name": "correctness",
    "description": (
        "Compare the output to the expected output and judge how closely "
        "the output matches it (semantically and factually). Account for "
        "valid variations in phrasing but penalise factual errors and "
        "omissions."
    ),
}
_DEFAULT_CRITERION_NO_REFERENCE = {
    "name": "overall quality",
    "description": (
        "Judge the output's correctness, relevance, and clarity as a "
        "response to the input. Consider whether it directly answers the "
        "question, whether the information is accurate, and whether the "
        "presentation is clear and well structured."
    ),
}


class LLMAsVerifierEvaluator(GroupEvaluator):
    """Round-robin pairwise verifier with continuous per-match rewards
    derived from the verifier's `top_logprobs` over score letters,
    optionally decomposed across multiple criteria.

    Per-model score is in [0, 1] (best→1.0, worst→0.0) — same range as
    the LLM judge, so the two are directly comparable in Langfuse. The
    bound holds because per-pair rewards are convex combinations
    (Σ p·v with Σp=1) of `value_map` ∈ [0, 1], and the per-model score
    is the mean of those.
    """

    def __init__(
        self,
        groups: list[dict[str, ExecutionResult]],
        llm_config: dict[str, Any],
        system_prompt: str,
        system_prompt_no_reference: str,
        granularity: int = 8,
        repeats: int = 1,
        criteria: list[dict[str, str]] | None = None,
        score_name: str = "verifier_score",
        max_concurrency: int = 10,
        callbacks: list[BaseCallbackHandler] | None = None,
        timeout_s: float | None = None,
    ):
        super().__init__(
            groups=groups,
            score_name=score_name,
            max_concurrency=max_concurrency,
            timeout_s=timeout_s,
        )
        # Tools would derail the `<score_A>X</score_A><score_B>Y</score_B>`
        # output contract the extractor depends on, so strip them up front.
        coerced = _coerce_ollama_to_openai_compat(
            {k: v for k, v in llm_config.items() if v is not None}
        )
        coerced.pop("tools", None)
        coerced.pop("tool_choice", None)
        self._model: BaseChatModel = build_chat_model(coerced)
        self._callbacks = list(callbacks or [])
        self.granularity = granularity
        self.repeats = max(1, repeats)
        self.criteria = [_as_criterion(c) for c in criteria] if criteria else None
        self._score_tokens = _build_score_tokens(granularity)
        self._base_prompt = system_prompt
        self._base_prompt_no_reference = system_prompt_no_reference

    async def _score_group(
        self,
        input: str,
        outputs: dict[str, str],
        expected_output: str | None = None,
        trajectories: dict[str, list[AgentStep]] | None = None,
    ) -> tuple[dict[str, float] | None, str | None, str | None]:
        models = list(outputs)
        if len(models) < 2:
            return None, None, "LLM-as-Verifier requires at least 2 models"

        # Faithful to reference: unordered pairs, no position swap. Use
        # `repeats >= 2` for variance reduction.
        match_specs = list(itertools.combinations(models, 2))

        results = await asyncio.gather(*[
            self._verify_pair(input, a, outputs[a], b, outputs[b], expected_output,
                              (trajectories or {}).get(a), (trajectories or {}).get(b))
            for a, b in match_specs
        ])

        per_model_rewards: dict[str, list[float]] = {m: [] for m in models}
        wins: dict[str, float] = {m: 0.0 for m in models}
        lines: list[str] = []
        pair_errors: list[str] = []
        for (a, b), (pair_result, pair_error) in zip(match_specs, results):
            if pair_result is None:
                pair_errors.append(f"{a} vs {b}: {pair_error or 'unknown'}")
                continue
            r_a, r_b = pair_result
            per_model_rewards[a].append(r_a)
            per_model_rewards[b].append(r_b)
            if r_a > r_b:
                wins[a] += 1.0
            elif r_b > r_a:
                wins[b] += 1.0
            else:
                wins[a] += 0.5
                wins[b] += 0.5
            lines.append(f"{a} vs {b}: R_A={r_a:.3f} R_B={r_b:.3f}")

        if not any(per_model_rewards.values()):
            return None, None, "all pair verifications failed: " + "; ".join(pair_errors)

        scores = {
            m: (sum(rs) / len(rs)) if rs else None
            for m, rs in per_model_rewards.items()
        }
        tournament = ", ".join(f"{m}:{wins[m]:.1f}W" for m in models)
        criteria_note = f" C={len(self.criteria)}" if self.criteria else ""
        explanation = (
            f"G={self.granularity} K={self.repeats}{criteria_note}  wins: {tournament}\n"
            + "\n".join(lines)
        )
        error = (
            f"{len(pair_errors)}/{len(match_specs)} pair verifications failed: "
            + "; ".join(pair_errors)
            if pair_errors
            else None
        )
        return {m: s for m, s in scores.items() if s is not None}, explanation, error

    async def _verify_pair(
        self,
        input: str,
        name_a: str,
        output_a: str,
        name_b: str,
        output_b: str,
        expected_output: str | None,
        trajectory_a: list[AgentStep] | None,
        trajectory_b: list[AgentStep] | None,
    ) -> tuple[tuple[float, float] | None, str | None]:
        if expected_output is not None:
            base, default_criterion = self._base_prompt, _DEFAULT_CRITERION_REFERENCE
        else:
            base, default_criterion = self._base_prompt_no_reference, _DEFAULT_CRITERION_NO_REFERENCE
        criteria = self.criteria or [default_criterion]

        payload_obj: dict[str, Any] = {
            "input": input,
            "output_A": output_a,
            "output_B": output_b,
        }
        if expected_output is not None:
            payload_obj["expected_output"] = expected_output
        if trajectory_a:
            payload_obj["trajectory_A"] = [asdict(step) for step in trajectory_a]
        if trajectory_b:
            payload_obj["trajectory_B"] = [asdict(step) for step in trajectory_b]
        payload = json.dumps(payload_obj, indent=2, default=str)

        # One verifier call per criterion, in parallel. Each call internally
        # averages over `repeats` samples (the K in R = (1/CK) Σ Σ Σ ...).
        rendered = [_render_prompt(base, self._score_tokens, c) for c in criteria]
        outcomes = await asyncio.gather(*[
            _pairwise_reward(
                model=self._model,
                callbacks=self._callbacks,
                system_prompt=system,
                user_payload=payload,
                granularity=self.granularity,
                repeats=self.repeats,
            )
            for system in rendered
        ])

        rewards_a: list[float] = []
        rewards_b: list[float] = []
        errors: list[str] = []
        for criterion, (rewards, error) in zip(criteria, outcomes):
            if rewards is None:
                errors.append(f"'{criterion['name']}': {error}")
            else:
                rewards_a.append(rewards[0])
                rewards_b.append(rewards[1])

        if not rewards_a:
            return None, "all criteria failed: " + "; ".join(errors)
        return (sum(rewards_a) / len(rewards_a), sum(rewards_b) / len(rewards_b)), None


# ---------------------------------------------------------------------------
# Logprob math + Ollama compat. Private to this module.
# ---------------------------------------------------------------------------

def _as_criterion(c: Any) -> dict[str, str]:
    """Normalise a criterion entry to a {name, description} dict.

    Accepts either a Pydantic CriterionConfig (from EvaluationConfig) or a
    raw dict (e.g. from tests / programmatic use).
    """
    if hasattr(c, "model_dump"):
        c = c.model_dump()
    if not isinstance(c, dict) or "name" not in c or "description" not in c:
        raise TypeError(
            f"criterion must be {{'name': str, 'description': str}}; got {c!r}"
        )
    return {"name": str(c["name"]), "description": str(c["description"])}


def _build_score_tokens(granularity: int) -> list[str]:
    """Score letters ordered best → worst (A is best)."""
    if granularity < 2 or granularity > 26:
        raise ValueError(f"granularity must be in [2, 26]; got {granularity}")
    return [chr(ord("A") + i) for i in range(granularity)]


def _scale_description(score_tokens: list[str]) -> str:
    """Render the rating-scale block embedded in the prompt — same shape
    as the reference's `SCALE['scale_description']` but generalised over
    arbitrary granularities (the reference hardcodes G=20)."""
    n = len(score_tokens)
    return (
        f"Rate each output on a {n}-point scale using letters "
        f"{score_tokens[0]} through {score_tokens[-1]}:\n"
        f"  {score_tokens[0]} = clearly satisfies this criterion (best)\n"
        f"  {score_tokens[-1]} = clearly fails this criterion (worst)\n"
        f"  intermediate letters reflect varying degrees of quality."
    )


def _render_prompt(
    prompt: str,
    score_tokens: list[str],
    criterion: dict[str, str],
) -> str:
    """Substitute the standard verifier placeholders without requiring full
    `str.format` semantics — user-overridden prompts may contain stray
    braces (e.g. JSON snippets) that would break `str.format`.

    Recognised placeholders:
      {granularity}            — number of score letters
      {score_letters}          — comma-separated letters (e.g. "A, B, C")
      {best_letter}            — first letter (e.g. "A")
      {worst_letter}           — last letter
      {scale_description}      — the multi-line rating-scale rubric
      {criterion}              — short criterion name
      {criterion_description}  — long criterion rubric (paper-faithful)
    """
    return (
        prompt.replace("{granularity}", str(len(score_tokens)))
        .replace("{score_letters}", ", ".join(score_tokens))
        .replace("{best_letter}", score_tokens[0])
        .replace("{worst_letter}", score_tokens[-1])
        .replace("{scale_description}", _scale_description(score_tokens))
        .replace("{criterion}", criterion["name"])
        .replace("{criterion_description}", criterion["description"])
    )


def _coerce_ollama_to_openai_compat(cfg: dict[str, Any]) -> dict[str, Any]:
    """Route Ollama calls through the OpenAI-compatible endpoint.

    LiteLLM's dedicated `ollama`/`ollama_chat` providers currently reject
    `logprobs`/`top_logprobs` (see BerriAI/litellm#17165), but Ollama itself
    (>= 0.12.11) returns them on `/v1/chat/completions`. Rewriting the
    provider prefix to `openai/` with `api_base=<ollama>/v1` lets the
    params flow through untouched.

    The user may have supplied the native Ollama `api_base`
    (`http://host:11434`) for their experiments; in that case we append
    `/v1` so the OpenAI-compat dispatcher hits the right path. If the
    user already supplied a `/v1` base we leave it alone.
    """
    model = cfg.get("model")
    if not isinstance(model, str):
        return cfg
    for prefix in ("ollama_chat/", "ollama/"):
        if model.startswith(prefix):
            bare = model[len(prefix) :]
            rewritten = dict(cfg)
            rewritten["model"] = f"openai/{bare}"
            supplied = rewritten.get("api_base")
            if isinstance(supplied, str) and supplied:
                rewritten["api_base"] = _ensure_openai_compat_base(supplied)
            else:
                rewritten["api_base"] = os.getenv(
                    "OLLAMA_API_BASE", _DEFAULT_OLLAMA_OPENAI_BASE
                )
            rewritten.setdefault("api_key", "ollama")
            _logger.info(
                "verifier: routing %s via OpenAI-compat endpoint %s (logprobs support)",
                model,
                rewritten["api_base"],
            )
            return rewritten
    return cfg


def _ensure_openai_compat_base(base: str) -> str:
    """Append `/v1` to a bare Ollama host URL; leave `/v1`-suffixed URLs
    alone. Trailing slashes are tolerated on either side."""
    trimmed = base.rstrip("/")
    if trimmed.endswith("/v1") or trimmed.endswith("/v1/"):
        return base
    return f"{trimmed}/v1"


async def _pairwise_reward(
    model: BaseChatModel,
    callbacks: list[BaseCallbackHandler],
    system_prompt: str,
    user_payload: str,
    granularity: int,
    repeats: int,
) -> tuple[tuple[float, float] | None, str | None]:
    """Sample the verifier `repeats` times and return averaged (R_A, R_B)
    for the two score tags. Returns `(None, error)` if every sample fails.

    Grader calls go through `ChatLiteLLM.agenerate` so LangChain callbacks
    (Langfuse tracing, retries) fire on every sample. `logprobs` /
    `top_logprobs` are forwarded to litellm as per-call kwargs; the
    returned `ChatGeneration.generation_info["logprobs"]` carries the raw
    OpenAI-shaped content list used by the tag extractor.
    """
    score_tokens = _build_score_tokens(granularity)
    score_set = set(score_tokens)
    n = len(score_tokens)
    # Normalise score letters to [0, 1] so the verifier's Langfuse score is
    # directly comparable with the judge evaluators (which also emit [0, 1]).
    # A (best) → 1.0, worst → 0.0. The reference computes raw values 1..G
    # then normalises by (max - min); the index-based map below is
    # mathematically equivalent.
    value_map = {tok: (n - 1 - i) / (n - 1) for i, tok in enumerate(score_tokens)}

    messages = to_langchain_messages([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_payload},
    ])

    # Samples are i.i.d. by design (same prompt, nondeterministic grader) —
    # fire all K in parallel and average. Serial latency = one grader call
    # instead of K. Per-pair and per-criterion parallelism sit one level up.
    async def _one_sample() -> tuple[float | None, float | None, str | None]:
        try:
            result = await model.agenerate(
                [messages],
                callbacks=callbacks,
                logprobs=True,
                top_logprobs=granularity,
            )
        except Exception as e:
            return None, None, f"completion failed: {e}"
        content, text = _unpack_generation(result)
        dist_a = _extract_score_at_tag(content, text, score_set, "<score_A>")
        dist_b = _extract_score_at_tag(content, text, score_set, "<score_B>")
        if dist_a is None or dist_b is None:
            return None, None, "missing <score_A>/<score_B> logprobs in response"
        r_a = sum(p * value_map[t] for t, p in dist_a.items())
        r_b = sum(p * value_map[t] for t, p in dist_b.items())
        return r_a, r_b, None

    outcomes = await asyncio.gather(*[_one_sample() for _ in range(max(1, repeats))])

    rewards_a: list[float] = []
    rewards_b: list[float] = []
    errors: list[str] = []
    for r_a, r_b, err in outcomes:
        if err is not None:
            errors.append(err)
            continue
        rewards_a.append(r_a)
        rewards_b.append(r_b)

    if not rewards_a:
        return None, "; ".join(dict.fromkeys(errors)) or "no usable samples"
    return (sum(rewards_a) / len(rewards_a), sum(rewards_b) / len(rewards_b)), None


def _unpack_generation(result: Any) -> tuple[list[Any], str | None]:
    """Pull (logprobs content, assistant text) out of a LangChain LLMResult.

    `ChatLiteLLM._create_chat_result` sticks the raw OpenAI-shaped
    `{"content": [...]}` logprobs block into `generation_info["logprobs"]`.
    Missing or unusual shapes degrade to an empty content list so the tag
    extractor can fall back to the assistant text.
    """
    try:
        gen = result.generations[0][0]
    except (AttributeError, IndexError):
        return [], None
    info = getattr(gen, "generation_info", None) or {}
    logprobs = info.get("logprobs") if isinstance(info, dict) else None
    content = _get(logprobs, "content", default=None) if logprobs is not None else None
    return list(content or []), getattr(gen, "text", None)


def _score_letter(token: str, score_tokens: set[str]) -> str | None:
    """Return the score letter (uppercase) carried by a token, or None.

    Case-insensitive (matches the reference's valid_tokens dict that
    includes both A-T and a-t). Handles tokenizer patterns we see in
    practice:
    * a bare letter (`"A"`, `"a"`),
    * a letter fused with leading punctuation (`">A"`, `":A"`, `"\tA"`),
    * a letter fused with trailing punctuation (`"A>"`, `"A:"`, `"A\""`),
    * a letter with trailing whitespace (`"A "`, `"A\n"`).
    Anything where the letter is adjacent to another letter (e.g. `"Alpha"`)
    is rejected to avoid false positives.
    """
    stripped = (token or "").strip().upper()
    if not stripped:
        return None
    if stripped in score_tokens:
        return stripped
    last = stripped[-1]
    if last in score_tokens and (len(stripped) == 1 or not stripped[-2].isalpha()):
        return last
    first = stripped[0]
    if first in score_tokens and (len(stripped) == 1 or not stripped[1].isalpha()):
        return first
    return None


def _extract_score_at_tag(
    logprobs_content: list[Any] | None,
    assistant_text: str | None,
    score_tokens: set[str],
    tag: str,
) -> dict[str, float] | None:
    """Find the token immediately after `tag` (e.g. `"<score_A>"`) in the
    logprobs content stream and return its normalised distribution over
    score letters.

    Faithful to the reference (`_find_tag_logprobs` + `extract_score`):
    walk the chosen tokens, accumulate text, locate the position whose
    accumulated text ends with the tag, then read `top_logprobs` at the
    NEXT position. Per-letter probabilities are aggregated with `max`
    rather than `sum` to collapse `"A"` / `" A"` / `"a"` duplicates
    without double-counting.

    Falls back to a regex over `assistant_text` when logprobs are not
    present at the tag — the reference does the same.
    """
    content = list(logprobs_content or [])

    text_so_far = ""
    for i, entry in enumerate(content):
        tok_text = _get(entry, "token", default="") or ""
        text_so_far += tok_text
        if not text_so_far.rstrip().endswith(tag):
            continue
        if i + 1 >= len(content):
            break
        top = _get(content[i + 1], "top_logprobs", default=[]) or []
        probs: dict[str, float] = {}
        for alt in top:
            t = _get(alt, "token", default="") or ""
            lp = _get(alt, "logprob", default=None)
            letter = _score_letter(t, score_tokens)
            if letter is None or lp is None:
                continue
            p = math.exp(float(lp))
            # `max` not `sum`: same letter may appear under several token
            # spellings (`"A"`, `" A"`, `"a"`); the reference treats them
            # as alternative encodings of the same score, not independent
            # samples. Summing would double-count.
            probs[letter] = max(probs.get(letter, 0.0), p)
        if not probs:
            break
        total = sum(probs.values())
        if total <= 0:
            break
        return {k: v / total for k, v in probs.items()}

    # Fallback: regex-parse `<tag>X</tag>` from the assistant's text. Used
    # when logprobs aren't returned at the tag (some providers, some
    # truncations) — gives a single-letter delta distribution.
    return _extract_from_text(assistant_text, score_tokens, tag)


def _extract_from_text(
    text: str | None, score_tokens: set[str], tag: str
) -> dict[str, float] | None:
    if not text:
        return None
    tag_name = tag.strip("<>")
    pattern = rf"<{re.escape(tag_name)}>\s*(.+?)\s*</{re.escape(tag_name)}>"
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return None
    letter = _score_letter(match.group(1), score_tokens)
    if letter is None:
        return None
    return {letter: 1.0}


def _get(obj: Any, attr: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)
