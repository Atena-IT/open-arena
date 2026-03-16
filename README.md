# Multi-Language Model Evaluation Framework 🚀

A framework to evaluate and compare LLMs and tool stacks against a dataset.

Define experiments (model + optional MCP-backed tools) and a dataset; run experiments, collect model I/O and metadata, and score outputs with pluggable evaluation methods.

## ✨ Key Features

- **Generality**: compare arbitrary model + tool combinations; tools are treated as pluggable services and models are provider-agnostic.
- **Observability**: capture and inspect dataset handling, model I/O, and evaluation results to make experiments auditable and reproducible through Langfuse.
- **Extensibility**: clear extension points allow adding new dataset formats, evaluation methods, executors, and integrations.
- **Declarative configuration**: define experiments via YAML configuration files, enabling reproducible runs and easy CI/CD automation.

## ⚡ Quickstart

### 🧰 Prerequisites
- **Python**: 3.11+
- Access to any **MCP servers**, **LLM provider endpoints**, and other services you plan to use
- (Recommended) **uv** installed and available in your `PATH`

### 🧑‍💻 Development Setup

#### 📦 Install (from source)
Clone the repository and set up the environment:

```sh
cd multi-language-model-evaluation-framework
```

Sync dependencies:

```sh
uv sync
```

Install the project in **editable mode** (recommended for development):

```sh
uv pip install -e .
```

Verify that the CLI entry point is available:

```sh
mlmef --help
```

### 🔒 Environment / Secrets
Copy `.env.example` to `.env` and fill in the required values:

```sh
cp .env.example .env
```

- Add all **LiteLLM-related** variables required by the providers you plan to call (refer to each provider’s documentation for the exact names/keys).
- Fill in the **Langfuse** variables for observability (optional).

### ⚙️ Configuration
The YAML schema for experiments is defined by the `ExperimentsFile` model in `src/config/types.py`.

**Important top-level fields:**
- `dataset`: global dataset configuration (`name`, `source`, `format`, `type`)
- `system_prompt`: global system prompt applied to experiments
- `experiments`: list of experiment blocks with per-experiment LiteLLM config and optional MCP server list
- `evaluation`: evaluation method and judge model config

<details>
<summary>Example minimal config</summary>

See the full version in `config.example.yaml`.

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

#### 🤖 Supported Providers and Models
This framework uses **LiteLLM**. Refer to the LiteLLM documentation/model index for supported providers and model IDs.

Provider credentials and configuration are supplied via environment variables (see each provider’s documentation for the required keys).

### ▶️ Run
Run the CLI using the installed entry point:

```sh
mlmef --config config.example.yaml
```

Alternatively, run the module directly (useful for debugging):

```sh
uv run -m src.main_cli --config config.example.yaml
```

## 👁️ Observability

### Langfuse Integration ⭐

**Full Langfuse integration is built-in**, providing enterprise-grade observability for your experiments:

- **Datasets**: Automatically uploaded and versioned in Langfuse for reproducibility
- **Traces**: Each experiment execution creates a trace with complete I/O capture
- **Generations**: LLM calls are logged with latency, token usage, and cost tracking
- **Scores**: Evaluation results are automatically attached to traces for easy comparison

## ⚠️ Limitations

- **CLI is currently Langfuse-backed only**: `src/main_cli.py` runs the end-to-end workflow using Langfuse datasets/experiments (dataset upload, execution traces, and score writing). If you want to run without Langfuse (fully in-memory execution + evaluation), you currently need to write a small custom runner that wires together the in-memory components (e.g., `DatasetLoader` + `GenericExecutor` + `GenericEvaluator`).

## 🤝 Contributing

This framework is designed with extensibility in mind. We welcome contributions that expand capabilities:

- **New dataset formats**: JSON, Parquet, databases, APIs
- **Evaluation metrics**: Custom scoring methods, domain-specific evaluators
- **Observability integrations**: Alternative to Langfuse (Weights & Biases, MLflow, etc.)

And much more!

### ❓ How to Contribute

1. **Report bugs**: Open an issue with reproduction steps
2. **Suggest features**: Describe your use case and proposed solution
3. **Submit PRs**: Include tests and update documentation
4. **Improve docs**: Fix typos, add examples, clarify instructions

## 📃 License

License to be determined.
