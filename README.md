# Open Arena 🚀

<img src="open-arena.png" width="28%" align="right" alt="Open Arena logo">

Open Arena is a lightweight evaluation framework for benchmarking LLMs and tool-enabled workflows against curated datasets. It combines LiteLLM, Langfuse, LangChain, and optional MCP integrations so experiments can be executed, traced, and scored from a single Python project.

## ✨ Key Features

- **Generality**: compare multiple model and tool configurations across curated datasets.
- **Observability**: trace experiments and evaluation results through Langfuse.
- **Extensibility**: compose dataset loading, execution, evaluation, and MCP integrations in Python.
- **Practical runtime defaults**: the current pipeline runs directly from `config.yaml` via `src.main`.

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

### Configure secrets

Copy the example environment file and fill in the required keys:

```sh
cp .env.example .env
```

At minimum, configure the Langfuse values plus any provider credentials required by the models defined in `config.yaml`.

### Configure experiments

The default runtime configuration lives in `config.yaml`. It defines:

- dataset creation settings
- dataset-specific system prompts
- the list of models to evaluate
- the judge model used for evaluation

### Run the pipeline

```sh
python -m src.main
```

## 👁️ Observability

Langfuse is used to capture experiment execution and evaluation metadata so model runs can be inspected and compared more easily.

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

## 🤝 Contributing

Open issues and pull requests are welcome. Please keep documentation and configuration examples aligned with the current runtime behavior when changing the evaluation pipeline.
