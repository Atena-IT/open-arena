# Open Arena 🚀

<img src="open-arena.png" width="28%" align="right" alt="Open Arena logo">

Open Arena is a lightweight evaluation framework for benchmarking LLMs and tool-enabled workflows against curated datasets. It combines LiteLLM, Langfuse, LangChain, and optional MCP integrations so experiments can be executed, traced, and scored from a single Python project.

## ✨ Key Features

- **Experiment orchestration**: run multiple model configurations against the same dataset and compare their outputs consistently.
- **Tool-enabled evaluation**: support MCP-backed tool calling alongside standard LLM completions.
- **Langfuse observability**: capture dataset uploads, execution traces, experiment runs, and evaluation scores in one place.
- **Pluggable dataset sources**: fetch evaluation items from local CSV/Excel, HuggingFace, Langfuse, LangSmith, Arize Phoenix, Opik (Comet), Braintrust, Weave, or MLflow — shaped into `(input, expected_output, metadata)` via Jinja2 templates.
- **Config-driven workflows**: a single `config.yaml` describes the dataset source, templates, experiments, and evaluation in one document.
- **Extensible Python architecture**: add a new dataset source by dropping a module in `src/datasets/dataset_adapters/`; evaluators and LLM clients are similarly modular.

## ⚡ Quickstart

### Prerequisites

- Python 3.12+
- Access to the LLM providers and Langfuse instance you want to use
- `uv` recommended for environment management

### Install

```sh
git clone https://github.com/Atena-IT/open-arena.git
cd open-arena
uv sync
```

For local development, install the project in editable mode so module and entrypoint changes are picked up immediately:

```sh
uv pip install -e .
```

To verify the Open Arena CLI entry point is available:

```sh
arena --help
```

### Configure secrets

Copy the example environment file and fill in the required keys:

```sh
cp .env.example .env
```

At minimum, configure the Langfuse values plus any provider credentials required by the models defined in `config.yaml` or `config.example.yaml`.

### Configure experiments

The runtime configuration lives in a single YAML file (see `config.example.yaml` for a fully annotated example). It defines:

- the dataset source and Jinja2 templates that shape each row
- the system prompt applied to every experiment
- the list of experiments (LiteLLM model config + optional MCP tool servers)
- the evaluator (`llm_as_judge` or `llm_as_verifier`) and grader model

The YAML schema is validated against the Pydantic models in `src/config/types.py`.

### Run the pipeline

```sh
arena --config config.yaml
```

To reuse an existing Langfuse dataset (skip the upload step):

```sh
arena --config config.yaml --skip-upload
```

When `dataset.source.provider` is `"langfuse"`, the upload is skipped automatically — items already live in Langfuse.

## 📚 Dataset Sources

The `dataset.source.provider` field selects which adapter fetches raw rows. Every adapter produces a stream of dicts; two Jinja2 templates (`input` and `expected_output`) turn each dict into a Record. Columns not referenced by a template flow through as metadata.

| Provider | Fetches from | Notes |
|----------|--------------|-------|
| `local` | CSV / Excel file on disk | `path`, `format` (`csv` \| `excel`) |
| `huggingface` | `datasets.load_dataset(repo, config, split)` | Streaming by default; supports `revision` pinning |
| `langfuse` | `get_client().get_dataset(name).items` | Upload is skipped automatically (items already in Langfuse) |
| `langsmith` | `Client().list_examples(dataset_name=...)` | Server-side `limit` push-down |
| `phoenix` | Arize Phoenix `datasets.get_dataset(...)` | Works against self-hosted or cloud; optional `version_id` pin |
| `opik` | `Opik().get_dataset(name).get_items()` | Comet Opik; server-side `nb_samples` push-down |
| `braintrust` | `init_dataset(project, name).fetch()` | Requires `project`; optional `version` pin |
| `weave` | `weave.ref("<name>:<version>").get()` | W&B Weave; requires `project="entity/project"` |
| `mlflow` | `mlflow.genai.datasets.get_dataset(name=...)` | Requires MLflow 3 with GenAI datasets |

Each adapter supports `limit: N` to cap the number of rows processed — useful for smoke tests. A source config looks like:

```yaml
dataset:
  name: "MMLU Anatomy"
  source:
    provider: "huggingface"
    repo: "cais/mmlu"
    config: "anatomy"
    split: "test"
  input: "{{ question }}\nA) {{ choices[0] }}\nB) {{ choices[1] }}..."
  expected_output: "{{ ['A','B','C','D'][answer] }}"
  limit: 100
```

See `config.example.yaml` for per-provider snippets.

## 👁️ Observability

Langfuse is used to capture experiment execution and evaluation metadata so model runs can be inspected and compared more easily. Depending on the workflow you use, Open Arena can track:

- uploaded dataset items
- experiment traces and model outputs
- evaluation results and judge scores
- metadata for MCP-enabled executions

## ⚠️ Limitations

- **Tracing is Langfuse-only**: experiment traces and evaluation scores are written to Langfuse regardless of which dataset source you use. A Langfuse instance (self-hosted or cloud) is therefore required.
- **Single dataset per run**: each `config.yaml` describes one dataset. Running multiple datasets means multiple invocations.

## 🧱 Project Layout

```text
open-arena/
├── config.yaml
├── config.example.yaml
├── pyproject.toml
├── resources/
├── src/
│   ├── config/
│   ├── datasets/
│   │   ├── base.py
│   │   ├── dataset_adapters/        # one file per provider
│   │   └── langfuse_upload.py
│   ├── evaluation/                  # evaluators (llm_as_judge, llm_as_verifier)
│   ├── execution/                   # experiment executor
│   ├── llms/                        # SimpleCaller + AgentCaller
│   └── main_cli.py                  # `arena` entrypoint
└── tests/
    └── datasets/                    # offline adapter tests (mocked clients)
```

## 🧪 Validation

Run the offline test suite (no credentials required — all provider clients are mocked):

```sh
uv run --with pytest pytest
```

For a quick syntax check:

```sh
uv run python -m compileall src
```

The YAML configuration is validated against `src/config/types.py` at load time, so bad shapes fail fast before any model call is made.

## 🤝 Contributing

Open issues and pull requests are welcome. Please keep documentation and configuration examples aligned with the current runtime behavior when changing the evaluation pipeline.

Suggested workflow:

1. Fork the repository and create a focused branch from `develop`.
2. Install dependencies with `uv sync` and configure `.env` from `.env.example`.
3. Make the smallest possible change that solves the issue.
4. Run the relevant validation for the area you touched.
5. Update documentation or examples when behavior changes.
6. Open a pull request with a clear description of the change.

### Commit Convention

This repository follows **Conventional Commits**. Please use commit messages such as:

- `feat: add structured dataset validation`
- `fix: handle missing Langfuse dataset items`
- `docs: expand README contribution guidance`

Using the convention keeps the history easier to scan and makes release or changelog automation more reliable.
