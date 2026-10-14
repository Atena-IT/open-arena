---
name: openarena-add-reward
description: Use when an openarena dataset needs something to score its answers (rewards.yaml) — choosing between an exact comparison and a judge model, writing the reward entry with its catalog type, its kwargs and its direction, and naming it from a dataset. Lists every catalog reward with the kwargs it accepts.
---

# openarena-add-reward

A reward scores one answer against its row, from 0 to 1. `configs/rewards.yaml`
maps a **name** to a catalog reward (`type`) with its kwargs (`config`); a
dataset names one with `reward: <name>`, and a dataset needs one to run.

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
      language_model: gpt-4o-mini    # a models.yaml KEY, never a litellm id
      instructions: |
        Score 1.0 when the answer is correct and complete, 0.0 when it is wrong.
  rubrics:
    type: rubrics_as_judge
    config:
      language_model: gpt-4o-mini
      rubrics:
        - The final answer is correct
        - The reasoning shows each step
```

A kwarg the type does not take is an error, and so is a model setting naming
an id instead of a `models` key.

## Choosing a reward

Deterministic, with a reference (the dataset's `output_template` must give one):

| type | kwargs | scores |
|---|---|---|
| `exact_match` | — | 1 when the answer equals the reference |
| `answer_accuracy` | — | the answer's field equals the reference's, normalized |
| `cosine_similarity` | `embedding_model` | embedding similarity to the reference |
| `retrieval_recall` | — | gold chunks among those retrieved (`embedding:` datasets) |
| `tool_calls_accuracy` | — | right functions with right arguments (`tools:` datasets) |
| `harbor_verifier` | `key` | what a Harbor task's own tests wrote (`harbor_tasks` datasets) |

A model judges (reference optional):

| type | kwargs |
|---|---|
| `lm_as_judge` | `language_model`, `instructions` |
| `rubrics_as_judge` | `language_model`, `rubrics` (list of criteria), `instructions` |
| `multi_judge_panel` | `panel_language_models` (list), `smart_language_model`, `agreement_threshold`, `instructions` |
| `agent_as_judge` | `language_model`, `instructions`, `max_iterations`, `use_chain_of_thought` |
| `deep_agent_as_judge`, `rlm_as_judge` | the above, plus `sub_language_model`, `timeout` |

Ready-made rubric judges, each taking `language_model` and `instructions`:
`answer_relevancy`, `faithfulness`, `hallucination`, `citation_faithfulness`,
`contextual_precision`, `contextual_recall`, `contextual_relevancy`,
`summarization`, `task_completion`, `goal_accuracy`, `plan_quality`,
`plan_adherence`, `step_efficiency`, `tool_correctness`, `tool_use`,
`tool_permission`, `argument_correctness`, `agent_loop_detection`,
`topic_adherence`, `role_adherence`, `role_violation`, `prompt_alignment`,
`knowledge_retention`, `conversation_completeness`, `turn_faithfulness`,
`turn_relevancy`, `bias`, `toxicity`, `pii_leakage`, `misuse`, `non_advice`.

The engine's own list, if in doubt:

```bash
uv run python -c "from openarena import rewards as r; [print(n, sorted(r.accepted_kwargs(n) - set(r.MASK_KWARGS))) for n in sorted(r._REWARD_TYPES)]"
```

Rule of thumb: a short, exact answer (a number, a letter) → `exact_match`;
free text → `lm_as_judge` or `rubrics_as_judge`; several criteria to report
apart → `rubrics_as_judge`; Harbor task folders → `harbor_verifier`. A judge
costs a model call per row: its `language_model` needs its key too.

`direction: min` is for a reward where lower is better (a toxicity or
hallucination score read as "how much"); the leaderboard flips it.

## Steps

1. `uv run openarena rewards list --configs my-arena/configs`.
2. Add the entry; for a judge, make sure its `language_model` is in
   `models.yaml` (openarena-add-model).
3. Point the dataset at it: `reward: judge` (openarena-add-dataset).
4. `uv run openarena check my-arena/configs`.
