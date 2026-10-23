---
name: openarena-add-reward
description: Use when an openarena dataset needs something to score its answers (rewards.yaml) — choosing between an exact comparison and a judge model, writing the reward entry with its catalog type, its kwargs and its direction, and naming it from a dataset. Lists every catalog reward with the kwargs it accepts, and what `openarena check` refuses.
---

# openarena-add-reward

A reward scores one answer against its row, from 0 to 1. `configs/rewards.yaml`
maps a **name** to a catalog reward (`type`) with its kwargs (`config`); a
dataset names one with `reward: <name>`, and a dataset needs one to run.
Depth: `docs/guides/rewards.md`.

## The entry

```yaml
rewards:
  exact-match:
    type: exact_match                # required: a catalog reward, below
    direction: max                   # max (higher is better, default) or min
    description: One when the answer is exactly the reference.
    tags: [strict]
  judge:
    type: lm_as_judge
    config:                          # the type's kwargs, and only those
      language_model: judge-4o       # a models.yaml KEY, never a litellm id: add it first (openarena-add-model)
      instructions: |
        You grade the answer against the reference (the gold_ fields).
        Score 1.0 when it is correct and complete, 0.5 when it is correct but
        misses a step, 0.0 when it is wrong. Length and style do not count.
  rubrics:
    type: rubrics_as_judge
    config:
      language_model: judge-4o
      rubrics:                       # required
        - The final answer is correct
        - name: units
          description: The answer states its unit.
          weight: 0.5
  tool-calls:
    type: tool_calls_accuracy        # no kwargs, no mask needed
```

`type`, `direction`, `description`, `tags` sit beside `config`; `name` or
`direction` inside `config` is an error, and so is a kwarg the type does not
take (`reduction` included, on a type whose constructor has none), or a model
setting naming an id instead of a `models` key (or a model of the wrong
`type`: `embedding_model` needs `type: embedding`, the others `llm`).

Every type also takes the masks `in_mask`, `out_mask`, `in_mask_pattern`,
`out_mask_pattern` (field names or regexes, applied to both sides at every
nesting level). Setting any mask turns off the automatic field choice: a
deterministic reward otherwise sees only the top-level fields the
`output_template` renders, each value compared whole (nested objects and
lists included), and a judge sees everything the program returns.

## Choosing a reward

Deterministic (the dataset's `output_template` must give the expected answer):

| type | kwargs | scores |
|---|---|---|
| `exact_match` | — | 1 when the answer's JSON equals the reference's, else 0 (`"42"` ≠ `42`; case and spaces count) |
| `answer_accuracy` | — | each reference field compared by its value's type, then averaged: bool and number by equality, text/list/object by token overlap (partial credit); a missing field scores 0 |
| `cosine_similarity` | `embedding_model` | embedding similarity rescaled `(cos+1)/2` (unrelated texts ≈ 0.5) |
| `retrieval_recall` | — | share of gold chunks in the top `k` (`embedding:` datasets) |
| `tool_calls_accuracy` | — | matched calls (same name, same arguments) over the longer list (`tools:` datasets) |
| `harbor_verifier` | `key` (default `reward`) | what a Harbor task's own tests wrote; needs no reference (`harbor_tasks` datasets) |

`answer_accuracy` picks the comparison from the **expected value**: a number
rendered as text (`{{ x | string | tojson }}`) is compared by tokens; render it
as a number (`{{ x | tojson }}`) to compare it as one.

`tool_calls_accuracy` needs no mask: scoped to `tool_calls`, it compares each
call's name and arguments, and the call ids (which differ per run) do not
count. The gold calls are `{name, arguments}` each; the wire shape
`{id, type, function: {name, arguments}}` (arguments a dict or a JSON string)
is read the same way, so a benchmark's gold column goes in as published.

A model judges (reference optional; the judge reads it as `gold_` fields):

| type | kwargs |
|---|---|
| `lm_as_judge` | `language_model`, `instructions` |
| `rubrics_as_judge` | `language_model`, `rubrics` (required: strings, `{name, description, weight}` mappings, or a preset name), `instructions` |
| `multi_judge_panel` | `panel_language_models` (required, non-empty), `smart_language_model` (required), `agreement_threshold` (0.2), `instructions` |
| `rlm_as_judge`, `deep_agent_as_judge` | `language_model`, `sub_language_model`, `instructions`, `max_iterations`, `timeout`, `use_chain_of_thought` |

`agent_as_judge` is listed by the engine but cannot be built from YAML (its
`tools` cannot be set): `check` refuses it. Use `rlm_as_judge` or
`deep_agent_as_judge` instead.

Ready-made rubric judges, each taking only `language_model` and `instructions`:
`answer_relevancy`, `faithfulness`, `hallucination`, `citation_faithfulness`,
`contextual_precision`, `contextual_recall`, `contextual_relevancy`,
`summarization`, `task_completion`, `goal_accuracy`, `plan_quality`,
`plan_adherence`, `step_efficiency`, `tool_correctness`, `tool_use`,
`tool_permission`, `argument_correctness`, `agent_loop_detection`,
`topic_adherence`, `role_adherence`, `role_violation`, `prompt_alignment`,
`knowledge_retention`, `conversation_completeness`, `turn_faithfulness`,
`turn_relevancy`, `bias`, `toxicity`, `pii_leakage`, `misuse`, `non_advice`.

The engine's own list, if in doubt (run from the openarena project):

```bash
uv run python -c "from openarena import rewards as r; [print(n, sorted(r.accepted_kwargs(n) - set(r.MASK_KWARGS))) for n in sorted(r._REWARD_TYPES)]"
```

Rule of thumb: a short, exact answer (a number, a letter) → `exact_match`;
several fields with partial credit → `answer_accuracy`; free text →
`lm_as_judge` or `rubrics_as_judge`; tool calls → `tool_calls_accuracy`;
Harbor task folders → `harbor_verifier`; code that must run →
`rlm_as_judge`. Prefer a rule whenever one can decide: it is free and
reproducible.

## Judges

- Name `language_model` explicitly; unset, it falls back to the evaluation's
  `default_language_model`.
- Give the judge its **own** `models.yaml` entry (openarena-add-model), never
  one of the evaluation's `language_models`, and preferably from another
  provider than the candidates (self-preference). A judge is called with its
  entry's `params`: give it `temperature: 0` so its scores are as repeatable
  as the provider allows.
- The judge does not see the question unless the program returns it: set
  `generator: {return_inputs: true}` in the dataset when grading needs it
  (relevancy, instruction following).
- Agent presets (`step_efficiency`, ...) on an MCP agent dataset need the
  `messages` the default hides: set a mask such as `out_mask: [thinking]`.
- Safety and hallucination presets do not say whether high means clean or
  bad: run ~5 rows whose answer you know, read the scores, then set
  `direction`. Or use `rubrics_as_judge` with a criterion that states the good
  side ("The answer contains no insult").
- A judge costs a model call per row (`reward_cost`) and needs its provider key.

`direction: min` is for a reward where lower is better; leaderboards flip it.

## What `openarena check` refuses

Beyond an unknown type or kwarg, these are errors, each with its fix:

- `rubrics_as_judge` without `rubrics`.
- `multi_judge_panel` without `smart_language_model`, or with an empty
  `panel_language_models`.
- Any `agent_as_judge`.
- `reduction` on a reward that does not take it (the deterministic ones);
  a judge takes it under `config:`.

It does not call the judge: a wrong `instructions` or an inverted
`direction` shows only in the scores.

## Steps

1. `uv run openarena rewards list --configs my-arena/configs`.
2. Add the entry; for a judge, make sure its `language_model` is a
   `models.yaml` key (openarena-add-model).
3. Point the dataset at it: `reward: judge` (openarena-add-dataset). Extra
   rewards go under the dataset's `metrics:` (`class:`, `alias:`).
4. `uv run openarena check my-arena/configs` until it exits 0.
