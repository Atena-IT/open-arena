---
name: arena-configure
description: Configure an Open Arena sweep: author/edit config.yaml datasets, rewards, metrics, and experiments blocks.
---

Read `config.example.yaml` for the full menu of every provider, reward type, and knob. Read `README.md` (Configure section) for the narrative. The steps below give the essential workflow.

## Starting point

```bash
cp config.example.yaml config.yaml
cp .env.example .env   # fill in only the providers you actually use
```

## Minimal config skeleton

```yaml
datasets:
  <name>:
    type: <provider>          # huggingface | local | folder | langfuse | langsmith | opik | phoenix | braintrust
    # provider-specific keys (path, split, name, …)
    limit: 50                 # rows cap — keep sweeps fast during development
    batch_size: 1
    input_template: |
      {"messages": [{"role": "user", "content": {{ question | tojson }}}]}
    output_template: |
      {"role": "assistant", "content": {{ answer | tojson }}}
    generator:
      temperature: 0.0
      instructions: "One-sentence task instruction."
    reward:
      name: exact_match       # or lm_as_judge, cosine_similarity, deep_eval, …
      in_mask: [content]      # mask for comparison rewards; omit for judge rewards

default: <name>               # dataset used when none is listed in experiments

experiments:
  language_models:
    - ollama/mistral
    - openai/gpt-4o-mini
  datasets:
    - <name>
```

## Dataset providers

| `type:` | Source |
|---|---|
| `huggingface` | HuggingFace datasets library (add `path`, `name`, `split`, `streaming`) |
| `local` | Single file: `.jsonl` / `.csv` / `.parquet` on disk |
| `folder` | One file per record under a directory (json/yaml/text) |
| `langfuse` | Langfuse-managed dataset |
| `langsmith` | LangSmith dataset |
| `opik` | Comet Opik dataset |
| `phoenix` | Arize Phoenix dataset |
| `braintrust` | Braintrust dataset |

All providers render rows through Jinja2 templates. Always use the `tojson` filter for string values: `{{ field | tojson }}`. Templates with bare `{{ field }}` break on quotes and newlines.

## Reward selection and masking rule

- **Comparison rewards** (`exact_match`, `cosine_similarity`): add `in_mask: [content]` on chat-message datasets, or `out_mask: [<input_field>]` on schema datasets. The eval harness attaches the input back onto `y_pred`; without masking the comparison includes the prompt and scores zero.
- **Judge rewards** (`lm_as_judge`, `recursive_lm_as_judge`, `multi_judge_panel`, `deep_eval`): omit `in_mask`. The judge needs the prompt context to score the answer.

## Extra scoring metrics (top-level `metrics:` block)

Add secondary scoring functions that run alongside the primary reward at no extra LM-call cost (auto-wrapped in `MeanMetricWrapper`):

```yaml
metrics:
  - class: lm_as_judge
    alias: lm_judge           # column header in last_run.tsv
    objective: true           # include in Pareto ranking
    language_model: ollama/llama3.2
    instructions: "Score 0.0–1.0 on factual correctness."
```

With `objective: true` the entry joins the dataset's tuner objective list (Pareto-ranked alongside the primary reward).

## Agent/MCP mode

A dataset that declares `agent:` runs as a `FunctionCallingAgent` instead of a single Generator call. Declare MCP servers once at the top level and reference by name:

```yaml
mcp_servers:
  math:
    transport: stdio
    command: python
    args: ["/abs/path/to/math_server.py"]

datasets:
  agentic_eval:
    type: folder
    path: data/agent_cases
    pattern: "*.json"
    batch_size: 1
    input_template: |
      {"messages":[{"role":"user","content":{{ question | tojson }}}]}
    agent:
      type: function_calling
      mcp_servers: [math]
      max_iterations: 5
      autonomous: true
      use_chain_of_thought: true
      instructions: "Solve step by step using available tools."
    reward:
      name: deep_eval
      metric: ToolCorrectnessMetric
```

`agent:` and `generator:` are mutually exclusive per dataset.

## Top-level defaults

```yaml
default_language_model: ollama/llama3.2   # fallback when reward omits language_model
seed: 42                                   # reproducibility seed for numpy/random
```

## State directory

Use `--state-dir` to isolate trial caches between experiments (see `arena-run-sweep` skill). The config itself does not control `--state-dir`; it is a CLI flag.

## Validation

The Pydantic schema in `src/config.py` validates the YAML on load. Run `uv run python -c "from src.config import Config; Config.load('config.yaml')"` to catch errors before a full sweep.
