# Open Arena

<img src="open-arena.png" width="28%" align="right" alt="Open Arena logo">

Autoresearch loop for reward-R&D and model selection in the modern
LM ops stack — between post-training evals and the next round of
fine-tuning / RL. An agent iterates on candidate rewards built-ins 
(`exact_match`, `cosine_similarity`, `lm_as_judge`), custom rewards
under `src/rewards/` (`multi_judge_panel`, `recursive_lm_as_judge`),
and any `deepeval.metrics` class via the `deep_eval` wrapper.
Validates them by running an evaluation sweep across a (model ×
dataset) grid, and keeps the rewards whose rankings best agree with
each dataset's task-specific primary reward. **Configured entirely
from YAML**; experiment management on top of `keras-tuner`.

Plugs into every major LM provider — OpenAI, Anthropic, Gemini,
Mistral, Cohere, Groq, Together, DeepSeek, xAI, OpenRouter, Azure,
AWS Bedrock, Doubleword, plus self-hosted Ollama / vLLM — and pulls
eval datasets from Langfuse, LangSmith, Opik, Arize Phoenix, and
Braintrust (or any HF / local / folder source). Each dataset can run
as a single-shot `Generator` eval or as a multi-step
`FunctionCallingAgent` driven by MCP tools.

## Install

Requires Python 3.11+ and [`uv`](https://github.com/astral-sh/uv).

```bash
git clone <this repo> && cd open-arena
uv sync                # installs deps + the `arena` console script
cp .env.example .env   # fill in only the providers you actually use
cp config.example.yaml config.yaml
```

For local LLMs, point `OLLAMA_API_BASE` at your Ollama server (defaults to
`http://localhost:11434`). Cloud providers are routed through litellm — any
`*_API_KEY` variable from `.env.example` works (OpenAI, Anthropic, Gemini,
Mistral, Cohere, Groq, Together, DeepSeek, xAI, OpenRouter, Azure, AWS).

## Run

```bash
arena                          # uses ./config.yaml
arena -c configs/eval.yaml     # different config
arena --no-cache               # discard the .open-arena/ trial cache and start over
```

`arena --help` for the full option list.

## Launch the autoresearch agent

A coding agent (Claude Code, Codex, Cursor, etc.) reads `AGENTS.md` /
`CLAUDE.md` on startup. Both files instruct the agent: when the user
asks to **start the research loop**, the agent should open
`AUTORESEARCH.md`, walk through Setup with you (run tag, branch,
smoke-tests, baseline `results.tsv`), wait for confirmation, then enter
the autonomous experiment loop and not stop until you interrupt it.

Trigger phrases (any of these, or an obvious paraphrase, kicks it off):

- "start the research loop"
- "begin autoresearch"
- "run autoresearch"
- "kick off the autoresearch loop"

After confirmation, the agent edits `src/rewards/`, commits, runs
`uv run arena > .open-arena/run.log 2>&1`, reads `.open-arena/last_run.tsv` directly
to score the sweep (computing whatever agreement / correlation /
trade-off statistics fit the iteration), and either advances or
reverts the branch — repeating indefinitely. See `AUTORESEARCH.md`
for the full protocol.

## Configure

A minimal `config.yaml`:

```yaml
datasets:
  mmlu_test:
    type: huggingface
    path: cais/mmlu
    name: all
    split: test
    streaming: true
    limit: 50
    batch_size: 1
    input_template: |
      {"messages": [{"role": "user", "content": {{ ("Q: " ~ question ~ "\nA) " ~ choices[0] ~ " B) " ~ choices[1] ~ " C) " ~ choices[2] ~ " D) " ~ choices[3]) | tojson }}}]}
    output_template: |
      {"role": "assistant", "content": {{ ["A","B","C","D"][answer] | tojson }}}
    generator:
      temperature: 0.0
      instructions: "Reply with one letter: A, B, C, or D."
    reward:
      name: exact_match
      in_mask: [content]

default: mmlu_test

experiments:
  language_models:
    - ollama/mistral
    - ollama/llama3.2
  datasets:
    - mmlu_test
```

Each `datasets:` entry carries its own `generator:` (instructions,
temperature, etc.) and `reward:` because both are task-dependent. The
sweep iterates the cross product `experiments.language_models ×
experiments.datasets`.

`config.example.yaml` is the full menu — every dataset provider and
reward type with annotated examples.

### Dataset providers

| `type:` | Source |
|---|---|
| `huggingface` | HuggingFace `datasets` library |
| `local` | JSONL / CSV / Parquet on disk |
| `folder` | Folder of files, one record per file (JSON / YAML / text / markdown) |
| `langfuse` | [Langfuse](https://langfuse.com) datasets |
| `langsmith` | [LangSmith](https://smith.langchain.com) datasets |
| `opik` | [Comet Opik](https://www.comet.com/docs/opik/) datasets |
| `phoenix` | [Arize Phoenix](https://arize.com/docs/phoenix) datasets |
| `braintrust` | [Braintrust](https://www.braintrust.dev) datasets |

All providers stream rows through Jinja2 templates that render to JSON
matching the input/output data models (the defaults are chat-message
shapes — a list of `{role, content}` for inputs, a single message for
outputs).

### Rewards

| `name:` | What it does |
|---|---|
| `exact_match` | String-equality on the masked fields |
| `cosine_similarity` | Cosine over `embedding_model` outputs |
| `lm_as_judge` | Single-LM judge |
| `recursive_lm_as_judge` | RLM agent inside a `ProgramAsJudge` — inspects the (gold, prediction) pair with code, recursively delegates semantic comparisons to a sub-LM |
| `multi_judge_panel` | M small LMs vote in parallel; on disagreement (max-min spread > `agreement_threshold`) a smart LM breaks the tie |
| `deep_eval` | Wraps any `deepeval.metrics` class (`GEval`, `FaithfulnessMetric`, `ToolCorrectnessMetric`, `PIILeakageMetric`, …). Per-slot mask routing maps `LLMTestCase` slots (input / actual_output / expected_output / context / retrieval_context / tools_called / expected_tools) to data-model fields independently. Install with `uv sync --extra deepeval`. |

**Masking — different rule for comparison vs judge rewards:**

- **Comparison rewards** (`exact_match`, `cosine_similarity`, regex / length
  checks): pass `in_mask: [content]` so the comparison only sees the answer
  field — `role` and friends would otherwise drive the score to 0.
- **Judge rewards** (`lm_as_judge`, `recursive_lm_as_judge`, `multi_judge_panel`,
  `deep_eval`): leave `in_mask` *unset*. The harness builds the eval
  program with `Generator(return_inputs=True, …)`, so `y_pred` carries the
  original input `messages` alongside the answer; the judge needs that
  prompt to score whether the answer addresses the task. `in_mask: [content]`
  would discard `messages` and leave the judge grading in a vacuum. See
  `REWARDS_BUILDING.md` for the full rationale.

### Agent mode (function-calling + MCP)

A dataset that declares an `agent:` block runs as a multi-step
`FunctionCallingAgent` instead of a single Generator call.
MCP servers are declared once in a top-level `mcp_servers:` registry
and referenced by name from each agentic dataset:

```yaml
mcp_servers:
  math:
    transport: stdio
    command: python
    args: ["/abs/path/to/math_server.py"]
  weather:
    transport: streamable_http
    url: http://localhost:8000/mcp

datasets:
  agentic_math_eval:
    type: folder
    path: data/agent_cases
    pattern: "*.json"
    batch_size: 1
    input_template: |
      {"messages":[{"role":"user","content":{{ question | tojson }}}]}
    agent:
      type: function_calling          # only supported value
      mcp_servers: [math]             # references the registry above
      max_iterations: 5
      autonomous: true
      use_chain_of_thought: true
      instructions: "Solve step by step using the available tools."
    reward:
      name: deep_eval
      metric: ToolCorrectnessMetric
```

`agent:` and `generator:` are mutually exclusive on a dataset. Tools
are loaded from the listed MCP servers at trial-build time via
`MultiServerMCPClient.get_tools()` so a misconfigured server fails the
trial fast rather than producing a tool-less agent. Any
`FunctionCallingAgent` constructor kwarg can be set under `agent:`
except `language_model`/`tools`/`data_model`/`schema`, which are wired
from the model and the dataset.

### Extra scoring functions via top-level `metrics:`

To score every `(model, dataset)` trial with additional scoring functions
beyond the dataset's primary `reward:`, list them under the top-level
`metrics:` block. Any reward identifier (e.g. `lm_as_judge`,
`cosine_similarity`, `multi_judge_panel`, `recursive_lm_as_judge`,
`deep_eval`) is auto-wrapped in `synalinks.metrics.MeanMetricWrapper`
so it rides the primary `evaluate()` pass — no extra K× model-call cost:

```yaml
metrics:
  - class: lm_as_judge
    alias: lm_judge
    objective: true              # promote to oracle objective for ranking
    language_model: ollama/llama3.2
    instructions: "Score 0.0–1.0 on factual correctness."
  - class: recursive_lm_as_judge
    alias: rlm_judge
    objective: true
    # `language_model` drives code generation + structured submit, so it
    # must be a capable model — small Ollama models can't do this reliably.
    # `sub_language_model` (used for `llm_query`) can be cheap.
    language_model: openai/gpt-4o
    sub_language_model: openai/gpt-4o-mini
    max_iterations: 8
    max_llm_calls: 10
    instructions: "Score 0.0–1.0 on factual correctness."
```

Each one becomes a column in the result matrix. With `objective: true`,
the entry joins the dataset's tuner objective list (Pareto-ranked
alongside the primary `reward:` and any other tagged entries).

## Layout

```
src/
  evaluate.py                         `arena` entrypoint — sweep runner, writes .open-arena/last_run.tsv (+ optional --json)
  program.py                          Editable program graphs: build_program / build_agent
  config.py                           Pydantic schema + validators for config.yaml
  datasets/                           One file per provider + the registry
  rewards/
    deep_eval.py                      Wraps any DeepEval metric (per-slot mask routing)
    multi_judge_panel.py
    recursive_language_model_reward.py
config.yaml                           Active config
config.example.yaml                   Reference config (every provider / reward / agent example)
.env.example                          Provider env vars
AGENTS.md / CLAUDE.md                 Notes for AI coding agents
AUTORESEARCH.md                       Autonomous experiment-loop protocol
```

## License

Apache 2.0 — see file headers.
