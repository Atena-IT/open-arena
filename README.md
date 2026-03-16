# Open Arena 🚀

<img src="open-arena.png" width="28%" align="right" alt="Open Arena logo">

Open Arena is a lightweight evaluation framework for benchmarking LLMs and tool-enabled workflows against curated datasets. It combines LiteLLM, Langfuse, LangChain, and optional MCP integrations so experiments can be executed, traced, and scored from a single Python project.

## ✨ Key Features

- **Experiment orchestration**: run multiple model configurations against the same dataset and compare their outputs consistently.
- **Tool-enabled evaluation**: support MCP-backed tool calling alongside standard LLM completions.
- **Langfuse observability**: capture dataset uploads, execution traces, experiment runs, and evaluation scores in one place.
- **Config-driven workflows**: use `config.yaml` for the current pipeline or `config.example.yaml` for the newer structured CLI flow.
- **Extensible Python architecture**: swap readers, item models, executors, evaluators, and LLM clients without rewriting the whole pipeline.
- **Practical runtime defaults**: the branch keeps a direct `python -m src.main` entrypoint while also carrying the packaged and module-based structured CLI introduced from `main`.

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

The default runtime configuration lives in `config.yaml`. It defines:

- dataset creation settings
- dataset-specific system prompts
- the list of models to evaluate
- the judge model used for evaluation

The repository also includes `config.example.yaml`, which documents the structured configuration model added from `main` for the newer CLI workflow. The YAML schema for the structured flow is defined in `src/config/types.py`.

### Run the pipeline

Current branch runtime:

```sh
python -m src.main
```

Packaged Open Arena CLI flow:

```sh
arena --config config.example.yaml
```

Module-based structured CLI flow:

```sh
uv run -m src.main_cli --config config.example.yaml
```

If you want to reuse an existing Langfuse dataset with the structured CLI, you can skip the upload step:

```sh
arena --config config.example.yaml --skip-upload
```

## 👁️ Observability

Langfuse is used to capture experiment execution and evaluation metadata so model runs can be inspected and compared more easily. Depending on the workflow you use, Open Arena can track:

- uploaded dataset items
- experiment traces and model outputs
- evaluation results and judge scores
- metadata for MCP-enabled executions

## ⚠️ Limitations

- **CLI is currently Langfuse-backed only**: `src/main_cli.py` runs the end-to-end workflow using Langfuse datasets and experiment traces. If you want to run without Langfuse, you currently need a small custom runner that wires together the in-memory components such as `DatasetLoader`, `GenericExecutor`, and `GenericEvaluator`.

## 🧱 Project Layout

```text
open-arena/
├── config.yaml
├── config.example.yaml
├── open-arena.png
├── pyproject.toml
├── resources/
└── src/
    ├── datasets/
    ├── evaluator/
    ├── evaluation/
    ├── execution/
    ├── llms/
    ├── mcp_server/
    ├── main.py
    └── main_cli.py
```

## 🔍 Notes

- The current entrypoint loads `config.yaml` by default.
- `config.example.yaml` documents the newer structured configuration introduced from `main`.
- Test utilities live under `src/test/`.
- A lightweight syntax validation can be run with `python -m compileall src`.

## 🧪 Validation

For a quick local validation pass, use:

```sh
python -m compileall src
```

If you are working on the structured CLI path, validate your YAML configuration before running long experiments by checking it against the Pydantic-backed schema in `src/config/types.py`.

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
