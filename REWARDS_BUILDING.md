# Building rewards

How to add a new project-local reward to `src/rewards/` so it can be
selected from `config.yaml` (per-dataset primary or experiment-level
candidate).

## Background: how rewards are wired

`src/rewards/__init__.py` builds a single `_REWARD_TYPES` registry by
union-ing two sources:

- **synalinks built-ins** — every concrete subclass of
  `synalinks.rewards.Reward` whose `__init__` has all-default args
  (auto-discovered from `synalinks.src.rewards.ALL_OBJECTS`).
  Examples: `exact_match`, `cosine_similarity`, `lm_as_judge`.
- **project-local** — classes listed in `_LOCAL_REWARDS`. Today:
  `JudgePanel`, `RecursiveLMAsJudge`, `DeepEval`. This is what you
  extend.

Lookup is by `to_snake_case(cls.__name__)`, so `MyClassName` becomes
`my_class_name` in YAML.

`get(spec)` instantiates the chosen class. String-typed
`language_model:` / `sub_language_model:` /
`smart_language_model:` / `panel_language_models:` /
`embedding_model:` values in YAML are auto-coerced into
`synalinks.LanguageModel` / `synalinks.EmbeddingModel` instances —
which is why your reward's `__init__` should accept those types
directly.

## What a reward looks like

A reward is a callable that takes `(y_true, y_pred)` and returns a
scalar `reward` in `[0, 1]` (higher = better). The base classes you
pick from depend on what you need:

- **`synalinks.rewards.Reward`** — bare-bones. Subclass and override
  `call(self, y_true, y_pred)`. Use this for purely-computational
  rewards (string equality, regex, BLEU, custom math).
- **`ProgramAsJudge`** (`synalinks.src.rewards.reward_wrappers`) — wraps
  an inner `Program` that judges `[y_true, y_pred]` and returns
  something with a `reward` field. Use this when the reward needs an LM
  in the loop. `JudgePanel` and `RecursiveLMAsJudge` both follow this
  pattern.
- **`RewardFunctionWrapper`** — wraps a plain async function. Lighter
  than a `Program` when you don't need parameters or training.

Every reward should also accept and forward the `in_mask` / `out_mask`
constructor args so users can restrict the comparison to specific
fields of the chat message (e.g. `in_mask=[content]` to ignore
`role`).

## What `y_pred` actually contains

The harness builds the eval program with `synalinks.Generator(
return_inputs=True, …)` (see `evaluate.py:build_program`). That means
`y_pred` is the **input prompt concatenated with the prediction**, not
just the prediction.

For a schema-based dataset (e.g. `input_schema: {question: str}`,
`output_schema: {answer: "A"|"B"|"C"|"D"}`):

```jsonc
// y_pred (what the reward sees)
{ "question": "<original prompt>", "answer": "A" }

// y_true (what the dataset's output_template produced)
{ "answer": "<gold letter>" }
```

For a default chat-message dataset (no `input_schema` /
`output_schema` set):

```jsonc
// y_pred
{
  "messages": [{"role": "user", "content": "<original prompt>"}],
  "role": "assistant",
  "content": "<the model's answer>"
}

// y_true
{ "role": "assistant", "content": "<gold answer>" }
```

This is deliberate: judge-style rewards (`lm_as_judge`,
`recursive_lm_as_judge`, `judge_panel`, `deep_eval`, custom LM judges)
need to see the original prompt to score whether the response actually
addresses the task. Without it the judge is grading an answer against a
gold reference in a vacuum.

### Implication for `in_mask`

- **Judge-style rewards: do not pass `in_mask: [content]`.** Stripping
  everything but `content` would discard the input `messages` field
  and defeat the point of `return_inputs=True`. Let the judge see the
  full row.
- **Comparison-style rewards (`exact_match`, `cosine_similarity`,
  length checks, regex): mask the comparison down to whatever fields
  appear in `y_true`.** Schema-based dataset → `out_mask` should drop
  the input fields the harness re-attached via `return_inputs=True`
  (e.g. `out_mask: [question]`). Chat-message default → keep
  `in_mask: [content]` so the extra `messages` / `role` fields on
  `y_pred` don't drive the score to 0.

### Opting out per-dataset

If a specific dataset's generator should not include the input in the
output (e.g. you're scoring a transformation where the input would
dominate the prediction signal), override it in the dataset's
`generator:` block:

```yaml
generator:
  return_inputs: false
  instructions: "..."
```

The CLI merges `return_inputs=True` underneath any
`generator_kwargs`, so any explicit setting in YAML wins.

## Wrapping a plain function

`RewardFunctionWrapper` (in `synalinks.src.rewards.reward_wrappers`)
takes an `async fn(y_true, y_pred, **kwargs)` returning a float in
`[0, 1]`. `y_true` and `y_pred` are `JsonDataModel` instances — call
`.get_json()` to get a dict, or index by key.

You can't register `RewardFunctionWrapper` itself in `_LOCAL_REWARDS`:
its `fn` arg has no default, so `_is_yaml_instantiable` rejects it
(and there's no class name to map to a YAML key anyway). Instead,
subclass it and default `fn` to your function:

```python
# src/rewards/length_match.py
from synalinks.src.rewards.reward_wrappers import RewardFunctionWrapper


async def _length_match(y_true, y_pred, tolerance=0.2):
    if not y_true or not y_pred:
        return 0.0
    gold = (y_true.get("content") or "").strip()
    pred = (y_pred.get("content") or "").strip()
    if not gold:
        return 0.0
    ratio = min(len(pred), len(gold)) / max(len(pred), len(gold))
    return 1.0 if ratio >= 1 - tolerance else 0.0


class LengthMatch(RewardFunctionWrapper):
    def __init__(self, tolerance=0.2, name="length_match",
                 in_mask=None, out_mask=None):
        super().__init__(
            fn=_length_match,
            name=name,
            in_mask=in_mask,
            out_mask=out_mask,
            tolerance=tolerance,   # forwarded as **kwargs to fn
        )
```

Register it the same way as a `ProgramAsJudge` reward — add the class
to `_LOCAL_REWARDS` in `src/rewards/__init__.py`. It then resolves
under `name: length_match` in YAML, with `tolerance:` and the usual
`in_mask:` / `out_mask:` as constructor keys.

Use this instead of a full `Program` when there's no LM call and no
trainable state — pure-Python rewards (string ops, regex, BLEU, math,
calls to a deterministic library) belong here.

## The `LMAsJudgeProgram` pattern

`JudgePanel` and `RecursiveLMAsJudge` both follow the same shape
(`DeepEval` is a thinner wrapper around an external library and is
not a `LMAsJudgeProgram`):

1. An inner `Program` subclass that takes `[y_true, y_pred]` as input.
2. Inside `call`: prefix the gold side, concatenate, delegate to one or
   more inner judge modules (`SelfCritique`, generators, RLM, …),
   aggregate, return a dict-like with a `reward` field.
3. A thin `ProgramAsJudge` subclass that constructs the program and
   exposes the user-facing constructor signature.

Reference implementation:
[`src/rewards/judge_panel.py`](src/rewards/judge_panel.py) —
`JudgePanelProgram` (the program) and `JudgePanel` (the
`ProgramAsJudge` wrapper).

The skeleton:

```python
from synalinks.src import ops
from synalinks.src.modules import SelfCritique
from synalinks.src.programs import Program
from synalinks.src.rewards.reward_wrappers import ProgramAsJudge


class MyJudgeProgram(Program):
    def __init__(self, language_model=None, instructions=None,
                 name=None, description=None, trainable=True):
        super().__init__(name=name, description=description,
                         trainable=trainable)
        self.judge = SelfCritique(
            language_model=language_model,
            instructions=instructions,
        )
        self.language_model = language_model
        self.instructions = instructions

    async def call(self, inputs):
        y_true, y_pred = inputs
        if not y_pred:
            return 0.0
        if y_true:
            y_true = await ops.prefix(y_true, prefix="gold")
            judge_input = await ops.concat(y_true, y_pred)
        else:
            judge_input = y_pred
        return await self.judge(judge_input)


class MyJudge(ProgramAsJudge):
    def __init__(self, language_model=None, instructions=None,
                 name="my_judge", in_mask=None, out_mask=None):
        super().__init__(
            program=MyJudgeProgram(
                language_model=language_model,
                instructions=instructions,
            ),
            name=name,
            in_mask=in_mask,
            out_mask=out_mask,
        )
```

If your reward needs serialization (so trained variables round-trip),
also implement `get_config` / `from_config` on the inner `Program` —
again, see `JudgePanelProgram` for an example using
`serialization_lib`.

## Step-by-step: adding a new reward

### 1. Create the file

`src/rewards/<name>.py`. Naming: pick a class name whose snake_case form
is what you want users to type in YAML. `MyJudge` → `my_judge`.

### 2. Implement it

Pick a base class (above). Constructor args become YAML keys. Keep them
named like the existing rewards so the auto-coercion in
`src/rewards/__init__.py:get` finds your `language_model: "..."` /
`embedding_model: "..."` strings and turns them into instances. If you
need a new type that isn't already auto-coerced, add a coercion line
in `get`.

Constructor must have **all-default args** for auto-registration to
pick it up. (See `_is_yaml_instantiable`. If your reward fundamentally
needs a runtime object you can't default — e.g. a pre-built `Program`
— you can register it manually, but the simpler path is to take the
strings/dicts in the constructor and build the object inside.)

### 3. Register it

Edit `src/rewards/__init__.py:_LOCAL_REWARDS`:

```python
from src.rewards.my_judge import MyJudge

_LOCAL_REWARDS = (JudgePanel, RecursiveLMAsJudge, MyJudge)
```

That's it — the registry rebuilds on import.

### 4. Wire it into `config.yaml`

As an experiment-level candidate (the typical autoresearch case):

```yaml
experiments:
  rewards:
    - name: my_judge
      alias: my_judge          # column header in matrices and TSV
      language_model: ollama/llama3.2
      # No `in_mask` for judges — they need to see the input messages
      # the CoT concatenated onto the prediction. See
      # "What `y_pred` actually contains" above.
      instructions: |
        Score 0.0–1.0 on whether the prediction matches the gold answer.
```

Or as a per-dataset primary reward:

```yaml
datasets:
  my_dataset:
    # ... type, path, templates, generator ...
    reward:
      name: my_judge
      language_model: ollama/llama3.2
      instructions: |
        Score 0.0–1.0 on factual correctness.
```

### 5. Clear the cache and run

The HP space changed (a new metric alias was added):

```bash
rm -rf .kt/open_arena
uv run python -u evaluate.py > .kt/run.log 2>&1
```

### 6. Score it

```bash
uv run analyze.py
```

Output is `<alias>\t<agreement>\t<spearman>` per candidate, where
agreement is the fraction of datasets on which the candidate's
argmax-model matches the primary's argmax-model.

### 7. Iterate

Tweak instructions / judge LMs / aggregation logic, commit, re-run,
re-score. The autoresearch loop in `AUTORESEARCH.md` automates this.

## Pitfalls

- **`__init__` with non-default args** — won't be auto-registered.
  Either default them and validate inside `__init__`, or add explicit
  registration in `_LOCAL_REWARDS` plus custom YAML coercion in `get`.
- **Forgetting `in_mask` / `out_mask` on comparison rewards** — with
  `return_inputs=True`, `y_pred` carries the input fields alongside
  the prediction. Equality / similarity rewards must mask those out
  (`out_mask: [<input field names>]` for schema datasets, or
  `in_mask: [content]` for chat-message defaults) so the comparison
  only spans fields that exist in `y_true`. Judge rewards must NOT
  mask away the input — that hides the prompt.
- **Reward not in `[0, 1]`** — break the agreement / Spearman analysis,
  which assumes higher is better and bounded. Clamp or normalize inside
  the reward.
- **Heavy reward × big sweep** — every experiment-level reward triggers
  another full `program.evaluate()` pass per trial (synalinks
  `compile(reward=...)` only takes one). K extras = K× the model calls.
  Cap dataset `limit:` accordingly.
- **Alias collision with `reward`** — the primary metric's alias is
  hardcoded as `reward`; setting `alias: reward` raises in
  `_parse_experiment_rewards`.
- **Same alias twice** — keras-tuner registers each alias once. If two
  candidates share an alias, the second silently overwrites the first.
- **Cache staleness** — adding / renaming / removing an alias changes
  the HP space. `rm -rf .kt/open_arena` (or pass `--no-cache`) before
  the next run, otherwise resumed trials won't have the new metric.
