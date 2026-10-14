# Rewards

A reward scores one answer against its row, from 0 to 1. `configs/rewards.yaml`
gives each reward a **name**: a catalog reward (`type`) with its settings
(`config`). A dataset names the reward that scores it with `reward:`, and a
dataset needs one to run.

```yaml
rewards:
  exact-match:
    type: exact_match                # required: a catalog reward
    direction: max                   # max (higher is better, the default) or min
    description: One when the answer is exactly the reference.
    tags: [strict]
  judge:
    type: lm_as_judge
    config:
      language_model: gpt-4o-mini    # a models.yaml name, never a litellm id
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

`config` takes the type's settings and only those: a setting the type does not
take is an error, and so is a model setting naming an id instead of a model.

## Choosing a reward

**A short, exact answer** (a number, a letter): compare it with the expected
answer. **Free text**: let a model judge it. **Several criteria to report
apart**: rubrics. **Harbor task folders**: the task's own tests.

### Deterministic

They need the expected answer the dataset's `output_template` renders.

| type | settings | scores |
|---|---|---|
| `exact_match` | | 1 when the answer equals the expected answer |
| `answer_accuracy` | | each field of the answer with the comparison that fits its type, averaged, with partial credit |
| `cosine_similarity` | `embedding_model` | the embedding similarity to the expected answer |
| `retrieval_recall` | | how many gold passages were retrieved ([retrieval datasets](datasets.md#retrieval)) |
| `tool_calls_accuracy` | | the right functions called with the right arguments ([tool calls](datasets.md#tool-calls)) |
| `harbor_verifier` | `key` | what a Harbor task's own tests wrote ([Harbor tasks](harbor-tasks.md)) |

### A model judges

The expected answer is optional; when there is one, the judge reads it as a reference.

| type | settings |
|---|---|
| `lm_as_judge` | `language_model`, `instructions` |
| `rubrics_as_judge` | `language_model`, `rubrics` (a list of criteria), `instructions` |
| `multi_judge_panel` | `panel_language_models` (a list), `smart_language_model`, `agreement_threshold`, `instructions` |
| `agent_as_judge` | `language_model`, `instructions`, `max_iterations`, `use_chain_of_thought` |
| `deep_agent_as_judge`, `rlm_as_judge` | as `agent_as_judge`, plus `sub_language_model`, `timeout` |

Ready-made rubric judges, each taking `language_model` and `instructions`:
`answer_relevancy`, `faithfulness`, `hallucination`, `citation_faithfulness`,
`contextual_precision`, `contextual_recall`, `contextual_relevancy`,
`summarization`, `task_completion`, `goal_accuracy`, `plan_quality`,
`plan_adherence`, `step_efficiency`, `tool_correctness`, `tool_use`,
`tool_permission`, `argument_correctness`, `agent_loop_detection`,
`topic_adherence`, `role_adherence`, `role_violation`, `prompt_alignment`,
`knowledge_retention`, `conversation_completeness`, `turn_faithfulness`,
`turn_relevancy`, `bias`, `toxicity`, `pii_leakage`, `misuse`, `non_advice`.

A judge makes a model call per row: its model needs its key, and its cost is
counted apart from the model under test's (`reward_cost`).

## Direction

`direction: min` is for a reward where lower is better, read as "how much": a
toxicity or hallucination score. Leaderboards take the direction into account;
you never invert a score yourself.
