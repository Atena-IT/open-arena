
# AGENTS.md

## Dev environment tips

- Install repo dependencies with:
  ```sh
  uv sync
  ```
- For development, install the project in **editable mode** (so CLI/entry points reflect your local changes immediately):
  ```sh
  uv pip install -e .
  ```
- Before running anything, copy `.env.example` to `.env` and set provider keys (e.g., `OPENAI_API_KEY` or any other LiteLLM provider variables you plan to use):
  ```sh
  cp .env.example .env
  ```
- **Langfuse is currently required to run the CLI** (no non-Langfuse execution mode is supported yet). Set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_HOST` before running any command.

## Running instructions

- Preferred way (validates packaging + entry point):
  ```sh
  mlmef --config <config.yaml>
  ```
- Example:
  ```sh
  mlmef --config config.example.yaml
  ```
- Alternative (module execution, useful for debugging):
  ```sh
  uv run -m src.main_cli --config <config.yaml>
  ```
- Dataset paths in YAML are relative to the repo root (see `dataset.source` in `config.example.yaml`).
- To avoid uploading the dataset to Langfuse again, use `--skip-upload`.
- MCP servers are optional; when enabled they must be reachable from your machine and support SSE.

## Project structure

- `src/main_cli.py`: CLI runner (loads config, uploads the dataset to Langfuse unless `--skip-upload`, runs experiments, runs evaluation).
- `src/config/types.py`: Pydantic schema for the YAML file (`ExperimentsFile`, dataset types/formats, MCP config).
- `src/datasets/`: readers + loaders + item models.
  - `readers/`: dataset readers for different formats (e.g. csv, excel).
  - `item_models/`: dataset item types (implement `input()` and `expected_output()`).
  - `loaders/`: load from files and (optionally) upload to Langfuse.
- `src/llms/`: LiteLLM/LangChain/LangGraph client, optional MCP tool integration.
- `src/execution/`: executors (generic/in-memory vs Langfuse-backed).
- `src/evaluation/`: evaluators and scoring methods (e.g., `llm_as_judge`).
- `src/prompts.default.yaml`: default prompt.