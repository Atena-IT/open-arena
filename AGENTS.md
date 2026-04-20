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

- Preferred invocation (packaged CLI entry point):
  ```sh
  arena --config <config.yaml>
  ```
- Example:
  ```sh
  arena --config config.example.yaml
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
- `src/config/types.py`: Pydantic schema for the YAML file (`ExperimentsFile`, dataset/experiment/evaluation config, MCP server config).
- `src/datasets/`:
  - `base.py`: `Dataset` base class + Jinja2 template rendering into `(input, expected_output, metadata)` rows.
  - `dataset_adapters/`: one module per provider (`local`, `huggingface`, `langfuse`, `langsmith`, `phoenix`, `opik`, `braintrust`, `weave`, `mlflow`).
  - `langfuse_upload.py`: upload rows to Langfuse (or attach to an existing dataset via `--skip-upload`).
- `src/llms/`: `LLMCaller` base plus `SimpleCaller` (ChatLiteLLM) and `AgentCaller` (LangGraph ReAct + MCP tools).
- `src/execution/`: `Executor` that iterates the dataset, calls the configured LLM per row, and wraps each call in a Langfuse span linked to the uploaded dataset item.
- `src/evaluation/`: `PointwiseEvaluator` / `GroupEvaluator` base classes plus `llm_as_judge` (pointwise, structured JSON) and `llm_as_verifier` (group, pairwise logprob-based).
- Evaluator prompts live in the user's `config.yaml` (`evaluation.system_prompt` / `system_prompt_no_reference`); see `config.example.yaml` for ready-to-copy templates.
