# Multi-Language Model Evaluation Framework 🚀

A framework to evaluate and compare LLMs and tool stacks against a dataset.

Define experiments (model + optional MCP-backed tools) and a dataset; run experiments, collect model I/O and metadata, and score outputs with pluggable evaluation methods.

# ✨ Key Features

- **Generality**: compare arbitrary model + tool combinations; tools are treated as pluggable services and models are provider-agnostic.
- **Observability**: capture and inspect dataset handling, model I/O, and evaluation results to make experiments auditable and reproducible through Langfuse.
- **Extensibility**: clear extension points allow adding new dataset formats, evaluation methods, executors, and integrations.
- **Declarative configuration**: define experiments via YAML configuration files, enabling reproducible runs and easy CI/CD automation.

# ⚡ Quickstart

## 🧰 Prerequisites

- Python 3.10+
- Access to any MCP servers, LLM provider endpoints, or other services you plan to use

## 📦 Install

This project uses `uv` for dependency sync:

```sh
uv sync
```

## 🔒 Environment / Secrets
- Copy `.env.example` to `.env`.
- Add all LiteLLM-related variables required by the providers you plan to call (see each provider's docs for exact names and keys).
- Fill in all the Langfuse related variables for observability (optional).

## ⚙️ Configuration

The YAML schema for experiments is defined by the `ExperimentsFile` model in `src/config/types.py`.

Important top-level fields:

- `dataset`: Global dataset configuration (`name`, `source`, `format`, `type`).
- `system_prompt`: Global system prompt applied to experiments.
- `experiments`: List of experiment blocks with per-experiment LiteLLM config and optional `mcp` server list.
- `evaluation`: Evaluation method and judge model config.

<details>
<summary>Example minimal config</summary>

See full version at [config.example.yaml](config.example.yaml)

```yaml
dataset:
  name: "Example QA Dataset"
  source: "resources/data/my_dataset.xlsx"
  format: "excel"
  type: "QA"

system_prompt: >
  You are a helpful AI assistant designed to answer questions accurately.

experiments:
  - name: "experiment_baseline"
    litellm:
      model: "gpt-4o"

evaluation:
  method: "llm_as_judge"
  litellm:
    model: "gpt-4o"
    temperature: 0.0
```

</details>

### 🤖 Supported Providers and Models

This framework uses LiteLLM. See the public model [index](https://models.litellm.ai/) for supported providers and model IDs.

Refer to each provider’s docs for required environment variables and model configuration options.

## ▶️ Run

Run the CLI with `uv` and the example config:

```sh
uv run -m src.main_cli --config config.example.yaml
```

# 👁️ Observability

## Langfuse Integration ⭐

**Full Langfuse integration is built-in**, providing enterprise-grade observability for your experiments:

- **Datasets**: Automatically uploaded and versioned in Langfuse for reproducibility
- **Traces**: Each experiment execution creates a trace with complete I/O capture
- **Generations**: LLM calls are logged with latency, token usage, and cost tracking
- **Scores**: Evaluation results are automatically attached to traces for easy comparison

# 🤝 Contributing

This framework is designed with extensibilty in mind. We welcome contributions that expand capabilities:

- **New dataset formats**: JSON, Parquet, databases, APIs
- **Evaluation metrics**: Custom scoring methods, domain-specific evaluators
- **Observability integrations**: Alternative to Langfuse (Weights & Biases, MLflow, etc.)

And much more!

## How to Contribute

1. **Report bugs**: Open an issue with reproduction steps
2. **Suggest features**: Describe your use case and proposed solution
3. **Submit PRs**: Include tests and update documentation
4. **Improve docs**: Fix typos, add examples, clarify instructions

# 📃 License

License to be determined. Add the chosen SPDX identifier and include the full license text.

# Code of Conduct

A project code of conduct will be added. Please keep interactions respectful and constructive.

