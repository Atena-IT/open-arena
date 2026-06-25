---
name: arena-build-reward
description: Add or iterate on a custom reward in src/rewards/ and wire it into the sweep.
---

Read `REWARDS_BUILDING.md` end-to-end before writing any reward — it covers base classes, `y_pred` structure, masking rules, and the full 7-step walkthrough. Read `README.md` (Rewards table) for the built-in menu and `config.example.yaml` (metrics block) for YAML wiring examples.

## Built-in rewards (select from YAML, no code needed)

| `name:` | Use case |
|---|---|
| `exact_match` | String equality on masked fields |
| `cosine_similarity` | Semantic similarity via embedding model |
| `lm_as_judge` | Single-LM judge |
| `recursive_lm_as_judge` | RLM agent inspects (gold, prediction) pair with code |
| `multi_judge_panel` | M small LMs vote; smart LM breaks ties on disagreement |
| `deep_eval` | Wraps any `deepeval.metrics` class (`GEval`, `FaithfulnessMetric`, …) |

For built-ins, just wire them in `config.yaml` (see `arena-configure` skill). For custom rewards, follow the steps below.

## What `y_pred` contains

The harness builds programs with `Generator(return_inputs=True, …)`, so `y_pred` is the **input prompt concatenated with the prediction**. This means:
- **Judge rewards**: do NOT pass `in_mask: [content]`. The judge needs the full input.
- **Comparison rewards** (`exact_match`, `cosine_similarity`): pass `in_mask: [content]` (chat-message datasets) or `out_mask: [<input_fields>]` (schema datasets) so the comparison only spans fields present in `y_true`.

## Step 1: Create `src/rewards/<name>.py`

Pick a class name whose `snake_case` form is what you want users to type in YAML. `MyJudge` → `my_judge`.

### Thin functional reward (no LM call)

```python
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
        super().__init__(fn=_length_match, name=name,
                         in_mask=in_mask, out_mask=out_mask,
                         tolerance=tolerance)
```

### LM judge reward (`ProgramAsJudge` pattern)

```python
from synalinks.src import ops
from synalinks.src.modules import SelfCritique
from synalinks.src.programs import Program
from synalinks.src.rewards.reward_wrappers import ProgramAsJudge

class MyJudgeProgram(Program):
    def __init__(self, language_model=None, instructions=None,
                 name=None, description=None, trainable=True):
        super().__init__(name=name, description=description, trainable=trainable)
        self.judge = SelfCritique(
            language_model=language_model,
            instructions=instructions,
        )

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
            name=name, in_mask=in_mask, out_mask=out_mask,
        )
```

See `src/rewards/multi_judge_panel.py` for a full reference implementation with serialization (`get_config` / `from_config`).

**Important**: the constructor must have **all-default args** for auto-registration. If a runtime object is required, accept strings/dicts and build it inside `__init__`.

## Step 2: Register in `src/rewards/__init__.py`

```python
from src.rewards.my_judge import MyJudge

_LOCAL_REWARDS = (MultiJudgePanel, RecursiveLMAsJudge, MyJudge)
```

The registry rebuilds on import. Lookup is by `to_snake_case(cls.__name__)`.

## Step 3: Wire into `config.yaml`

As a candidate in the top-level `metrics:` block (auto-wrapped in `MeanMetricWrapper` — no extra LM calls per trial):

```yaml
metrics:
  - class: my_judge
    alias: my_judge
    language_model: ollama/llama3.2
    # No in_mask for judge rewards — they need the full prompt context
    instructions: |
      Score 0.0–1.0 on whether the prediction matches the gold answer.
```

As a per-dataset primary reward:

```yaml
datasets:
  my_dataset:
    reward:
      name: my_judge
      language_model: ollama/llama3.2
      instructions: "Score 0.0–1.0 on factual correctness."
```

## Step 4: Clear the cache and run

When a new metric alias is added:

```bash
rm -rf .open-arena/*/
uv run arena > .open-arena/run.log 2>&1
```

## Step 5: Score and iterate

```bash
cat .open-arena/last_run.tsv
```

Long-format: `model<TAB>dataset<TAB>metric<TAB>value<TAB>direction`. Check whether the new metric alias column agrees with the `reward` column across datasets. See `arena-autoresearch` skill for the full iteration loop.

## Pitfalls

- **Non-default args in `__init__`** — reward won't auto-register. Default all args or add explicit registration.
- **Forgetting `in_mask` / `out_mask` on comparison rewards** — `y_pred` carries input fields; equality rewards must mask them.
- **Reward not in `[0, 1]`** — breaks agreement/Spearman analysis. Clamp or normalize.
- **Alias `reward`** — collides with the primary metric; raises on startup.
- **Duplicate aliases** — within a list, one silently shadows the other. Keep aliases unique.
- **Cache staleness** — adding/renaming/removing a `metrics:` alias changes the HP space. Always `rm -rf .open-arena/*/` or pass `--no-cache` before the next run.
