import asyncio
import json
import logging
import sys
import warnings
from collections import defaultdict, deque
from typing import Any, Protocol

import click
from dotenv import load_dotenv
from langfuse import get_client
from pydantic import ValidationError

from src.config.types import ExperimentsFile, DatasetType
from src.datasets.loaders import LangfuseLoader
from src.datasets.readers import ExcelReader, CsvReader
from src.datasets.item_models import QAItem, ToolScaleItem, ToolsExample, DatasetItem
from src.execution import LangfuseExecutor
from src.execution.types import ExecutionResult
from src.evaluation import LangfuseEvaluator, LLMAsJudge
from src.evaluation.types import EvaluationResult
from src.llms import LangfuseLLMClient
from src.llms.types import MCPServerConfig

warnings.filterwarnings('ignore', category=UserWarning, module='pydantic') # TODO: remove when bug fixed

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
_logger = logging.getLogger(__name__)
MISSING_ITEM_PREVIEW_LENGTH = 50

load_dotenv()


class LangfuseDatasetItemLike(Protocol):
    id: str
    input: Any
    expected_output: Any
    metadata: dict[str, Any] | None


def get_item_model(dataset_type: DatasetType) -> type[DatasetItem]:
    """Map dataset type to item model class."""
    mapping = {
        DatasetType.QA: QAItem,
        DatasetType.ToolScale: ToolScaleItem,
        DatasetType.ToolsExample: ToolsExample,
    }

    model = mapping.get(dataset_type)
    if model is None:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    return model


def get_reader(format: str):
    """Get appropriate reader based on format."""
    if format == "excel":
        return ExcelReader()
    elif format == "csv":
        return CsvReader()
    else:
        raise ValueError(f"Unsupported format: {format}")


async def get_evaluation_method(config: ExperimentsFile):
    """Get appropriate evaluation method based on config."""
    if config.evaluation.method == "llm_as_judge":
        judge_config = config.evaluation.litellm.model_dump()
        judge_client = LangfuseLLMClient(judge_config)

        await judge_client.setup()

        return LLMAsJudge(
            llm_client=judge_client,
        )
    else:
        raise ValueError(f"Unsupported evaluation method: {config.evaluation.method}")


async def load_and_upload_dataset(config: ExperimentsFile) -> list[DatasetItem]:
    """Load dataset and upload to Langfuse."""
    _logger.info(f"Loading dataset: {config.dataset.name} from {config.dataset.source}")

    item_model = get_item_model(config.dataset.type)
    reader = get_reader(config.dataset.format)

    loader = LangfuseLoader(
        item_model=item_model,
        reader=reader,
        config={
            "dataset_name": config.dataset.name,
            "source_file": config.dataset.source
        },
        input_path="."
    )

    dataset = loader.load()
    _logger.info(f"Dataset uploaded to Langfuse: {len(dataset)} items in '{config.dataset.name}'")

    return dataset


async def load_dataset_only(config: ExperimentsFile) -> list[DatasetItem]:
    """Load and validate dataset from file without uploading to Langfuse."""
    from src.datasets.loaders import DatasetLoader

    _logger.info(f"Loading dataset (no upload): {config.dataset.name} from {config.dataset.source}")

    item_model = get_item_model(config.dataset.type)
    reader = get_reader(config.dataset.format)

    loader = DatasetLoader(
        item_model=item_model,
        reader=reader,
        config={
            "dataset_name": config.dataset.name,
            "source_file": config.dataset.source
        },
        input_path="."
    )

    dataset = loader.load()
    _logger.info(f"Dataset loaded locally: {len(dataset)} items (not uploaded to Langfuse)")

    return dataset


def _dataset_item_key(
    input_value: Any,
    expected_output: Any,
    metadata: dict[str, Any] | None,
) -> str:
    """Build a stable key used to match local items to Langfuse dataset items."""
    def normalize_for_json_key(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            string_key_items = [(str(key), val) for key, val in value.items()]
            return {
                str(key): normalize_for_json_key(val)
                for key, val in sorted(string_key_items)
            }
        if isinstance(value, (list, tuple)):
            return [normalize_for_json_key(item) for item in value]
        if isinstance(value, set):
            sortable_items = [
                (json.dumps(normalized_item, sort_keys=True), normalized_item)
                for normalized_item in (normalize_for_json_key(item) for item in value)
            ]
            return [item for _, item in sorted(sortable_items, key=lambda sortable_item: sortable_item[0])]
        return str(value)

    return json.dumps(
        {
            "input": normalize_for_json_key(input_value),
            "expected_output": normalize_for_json_key(expected_output),
            "metadata": normalize_for_json_key(metadata or {}),
        },
        sort_keys=True,
    )


async def load_existing_langfuse_dataset(config: ExperimentsFile) -> list[DatasetItem]:
    """
    Load and validate dataset from file, then attach metadata from an existing Langfuse dataset.

    This powers the --skip-upload flow by reusing already-uploaded Langfuse items instead of
    uploading duplicates, while still validating the local source file.
    """
    dataset = await load_dataset_only(config)

    if not dataset:
        _logger.warning("Dataset is empty, skipping Langfuse dataset lookup")
        return dataset

    _logger.info(f"Loading existing Langfuse dataset: {config.dataset.name}")
    langfuse_dataset = get_client().get_dataset(config.dataset.name)

    langfuse_items_by_key: dict[str, deque[LangfuseDatasetItemLike]] = defaultdict(deque)
    for remote_item in langfuse_dataset.items:
        langfuse_items_by_key[
            _dataset_item_key(
                remote_item.input,
                remote_item.expected_output,
                remote_item.metadata,
            )
        ].append(remote_item)

    missing_items: list[str] = []
    for index, item in enumerate(dataset, start=1):
        matching_items = langfuse_items_by_key.get(
            _dataset_item_key(item.input(), item.expected_output(), item.meta())
        )

        if not matching_items:
            missing_items.append(
                f"row {index} ({str(item.input())[:MISSING_ITEM_PREVIEW_LENGTH]!r})"
            )
            continue

        if len(matching_items) > 1:
            _logger.warning(
                "Multiple existing Langfuse items matched row %s in dataset '%s'; reusing one of them",
                index,
                config.dataset.name,
            )

        remote_item = matching_items.popleft()
        item.metadata["lf_item_id"] = remote_item.id
        item.metadata["lf_dataset_name"] = langfuse_dataset.name
        item.metadata["lf_dataset_id"] = langfuse_dataset.id

    if missing_items:
        missing_preview = ", ".join(missing_items[:5])
        if len(missing_items) > 5:
            missing_preview += ", ..."

        raise ValueError(
            "Dataset validation succeeded, but some items were not found in the existing "
            f"Langfuse dataset '{config.dataset.name}'. Missing matches: {missing_preview}. "
            "Re-run without --skip-upload to upload the dataset again."
        )

    extra_remote_items_count = sum(len(items) for items in langfuse_items_by_key.values())
    if extra_remote_items_count:
        _logger.warning(
            "Existing Langfuse dataset '%s' contains %s extra items that were not matched "
            "to the local source file",
            config.dataset.name,
            extra_remote_items_count,
        )

    _logger.info(
        "Reused %s existing Langfuse dataset items from '%s'",
        len(dataset),
        config.dataset.name,
    )
    return dataset


async def run_experiments(config: ExperimentsFile, dataset: list[DatasetItem]) -> list[list[ExecutionResult]]:
    """Run all experiments sequentially."""
    _logger.info(f"Preparing {len(config.experiments)} experiments for execution")

    all_results = []

    for exp_config in config.experiments:
        _logger.info(f"Configuring experiment: {exp_config.name} with model {exp_config.litellm.model}")

        llm_config = exp_config.litellm.model_dump()

        mcp_servers: list[MCPServerConfig] | None = None
        if exp_config.mcp:
            mcp_servers = [
                {"server_name": mcp.name, "url": str(mcp.url)}
                for mcp in exp_config.mcp
            ]
            _logger.info(f"  MCP servers configured: {len(mcp_servers)}")

        lf_client = LangfuseLLMClient(
            llm_config=llm_config,
            mcp_servers=mcp_servers or []
        )

        await lf_client.setup()

        executor = LangfuseExecutor(
            dataset=dataset,
            llm_client=lf_client,
            system_prompt=config.system_prompt,
            experiment_name=exp_config.name,
            experiment_description=f"Experiment: {exp_config.name} with model {exp_config.litellm.model}"
        )

        _logger.info(f"Executing experiment: {exp_config.name}")
        results = await executor.execute()
        all_results.append(results)

        errors = sum(1 for r in results if r.error)
        if errors > 0:
            _logger.warning(f"Experiment '{exp_config.name}' completed: {len(results)} items, {errors} errors")
        else:
            _logger.info(f"Experiment '{exp_config.name}' completed successfully: {len(results)} items")

    _logger.info("All experiments completed")

    return all_results


async def run_evaluations(config: ExperimentsFile, all_results: list[list[ExecutionResult]]) -> list[list[EvaluationResult]]:
    """Run evaluations on all experiment results."""
    _logger.info(f"Preparing evaluation for {len(all_results)} experiments")

    evaluation_method = await get_evaluation_method(config)
    _logger.info(f"Configuring {config.evaluation.method} with model: {config.evaluation.litellm.model}")

    all_eval_results = []

    # Evaluate each experiment's results
    for exp_config, results in zip(config.experiments, all_results):
        _logger.info(f"Evaluating experiment: {exp_config.name}")

        evaluator = LangfuseEvaluator(
            results=results,
            method=evaluation_method,
            score_name=config.evaluation.score_name or "evaluation_score",
            max_concurrency=config.evaluation.max_concurrency or 10
        )

        eval_results = await evaluator.evaluate()
        all_eval_results.append(eval_results)

        scored = sum(1 for r in eval_results if r.score is not None)
        errors = sum(1 for r in eval_results if r.error is not None)
        avg_score = sum(r.score for r in eval_results if r.score is not None) / scored if scored > 0 else 0

        if errors > 0:
            _logger.warning(f"Evaluation '{exp_config.name}' completed: {scored} scored (avg: {avg_score:.2f}), {errors} errors")
        else:
            _logger.info(f"Evaluation '{exp_config.name}' completed: {scored} scored (avg: {avg_score:.2f})")

    _logger.info("All evaluations completed")

    return all_eval_results


@click.command()
@click.option(
    '--config', '-c',
    required=True,
    type=click.Path(exists=True),
    help='Path to YAML configuration file'
)
@click.option(
    '--skip-upload',
    is_flag=True,
    default=False,
    help='Skip dataset upload (assumes dataset already exists in Langfuse)'
)
def main(config: str, skip_upload: bool):
    """
    Run the Open Arena CLI workflow from a YAML configuration file.

    This script:
    1. Loads and validates the configuration
    2. Uploads the dataset to Langfuse (unless --skip-upload)
    3. Runs all experiments sequentially
    4. Evaluates all experiment results
    5. Results and scores are automatically tracked in Langfuse

    Example:
        arena --config experiments.yaml
        arena -c config.yaml --skip-upload
        python -m src.main_cli -c config.yaml --skip-upload
    """
    try:
        _logger.info("Starting Open Arena CLI pipeline")

        _logger.info(f"Loading configuration from: {config}")
        try:
            experiments_config = ExperimentsFile.from_yaml(config)
            _logger.info(f"Configuration validated: {len(experiments_config.experiments)} experiments found")
        except ValidationError as e:
            _logger.error(f"Configuration validation failed:")
            for error in e.errors():
                _logger.error(f"  {error['loc']}: {error['msg']}")
            sys.exit(1)

        async def workflow():
            if not skip_upload:
                dataset = await load_and_upload_dataset(experiments_config)
            else:
                _logger.info("Skipping dataset upload (--skip-upload flag set)")
                dataset = await load_existing_langfuse_dataset(experiments_config)

            results = await run_experiments(experiments_config, dataset)

            await run_evaluations(experiments_config, results)

        asyncio.run(workflow())

        _logger.info("Execution completed successfully")
        _logger.info("View results in Langfuse dashboard")
        sys.exit(0)

    except FileNotFoundError as e:
        _logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        _logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
