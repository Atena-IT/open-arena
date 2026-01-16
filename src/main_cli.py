import asyncio
import logging
import sys
from typing import List
import warnings

import click
from dotenv import load_dotenv
from pydantic import ValidationError

from src.config.types import ExperimentsFile, DatasetType
from src.datasets.loaders import LangfuseLoader
from src.datasets.readers import ExcelReader, CsvReader
from src.datasets.item_models import QAItem, ToolScaleItem, DatasetItem
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

load_dotenv()


def get_item_model(dataset_type: DatasetType) -> type[DatasetItem]:
    """Map dataset type to item model class."""
    mapping = {
        DatasetType.QA: QAItem,
        DatasetType.ToolScale: ToolScaleItem,
        # Add more mappings as needed
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


async def load_and_upload_dataset(config: ExperimentsFile) -> List[DatasetItem]:
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
        input_path=".",
        max_items=2
    )
    
    dataset = loader.load()
    _logger.info(f"Dataset uploaded to Langfuse: {len(dataset)} items in '{config.dataset.name}'")

    return dataset


async def run_experiments(config: ExperimentsFile, dataset: List[DatasetItem]) -> List[List[ExecutionResult]]:
    """Run all experiments sequentially."""
    _logger.info(f"Preparing {len(config.experiments)} experiments for execution")
    
    all_results = []
    
    for exp_config in config.experiments:
        _logger.info(f"Configuring experiment: {exp_config.name} with model {exp_config.litellm.model}")
        
        llm_config = exp_config.litellm.model_dump()
        
        mcp_servers: List[MCPServerConfig] | None = None
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


async def run_evaluations(config: ExperimentsFile, all_results: List[List[ExecutionResult]]) -> List[List[EvaluationResult]]:
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
    Run experiments from a YAML configuration file.
    
    This script:
    1. Loads and validates the configuration
    2. Uploads the dataset to Langfuse (unless --skip-upload)
    3. Runs all experiments sequentially
    4. Evaluates all experiment results
    5. Results and scores are automatically tracked in Langfuse
    
    Example:
        python -m src.main_cli --config experiments.yaml
        python -m src.main_cli -c config.yaml --skip-upload
    """
    try:
        _logger.info("Multi-Language Model Evaluation Framework")
        
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
