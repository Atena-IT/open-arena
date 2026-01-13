"""
CLI script to run experiments from a YAML configuration file.

Usage:
    python -m src.main_cli --config path/to/config.yaml
    python -m src.main_cli -c experiments.yaml
"""

import asyncio
import logging
import sys
from typing import List

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
from src.llms import LangfuseLLMClient, LLMClient
from src.llms.types import MCPServerConfig

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


def get_evaluation_method(config: ExperimentsFile):
    """Get appropriate evaluation method based on config."""
    if config.evaluation.method == "llm_as_judge":
        judge_client = LLMClient()
        judge_config = config.evaluation.litellm.model_dump()
        
        return LLMAsJudge(
            llm_client=judge_client,
            system_prompt="You are an expert judge evaluating the quality of responses. "
                         "Provide a score between 0 and 5, where 0 is completely incorrect and 5 is perfect. "
                         "Return your response as JSON with 'score' and 'explanation' keys.",
            model_config=judge_config
        )
    else:
        raise ValueError(f"Unsupported evaluation method: {config.evaluation.method}")


async def load_and_upload_dataset(config: ExperimentsFile) -> None:
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
        max_items=1 # TODO: remove
    )
    
    dataset = loader.load()
    _logger.info(f"Dataset uploaded to Langfuse: {len(dataset)} items in '{config.dataset.name}'")


async def run_experiments(config: ExperimentsFile) -> List[List[ExecutionResult]]:
    """Run all experiments sequentially."""
    _logger.info(f"Preparing {len(config.experiments)} experiments for execution")
    
    lf_client = LangfuseLLMClient()
    
    item_model = get_item_model(config.dataset.type)
    
    all_results = []
    
    # Run experiments sequentially to avoid event loop conflicts
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
        
        executor = LangfuseExecutor(
            dataset_name=config.dataset.name,
            llm_client=lf_client,
            system_prompt=config.system_prompt,
            llm_config=llm_config,
            from_langfuse_fn=item_model.from_langfuse_item,
            mcp_servers=mcp_servers,
            experiment_name=exp_config.name,
            experiment_description=f"Experiment: {exp_config.name} with model {exp_config.litellm.model}"
        )
        
        # Execute experiment (blocks until complete)
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
    
    evaluation_method = get_evaluation_method(config)
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
        _logger.info("=" * 80)
        _logger.info("Multi-Language Model Evaluation Framework")
        _logger.info("=" * 80)
        
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
                await load_and_upload_dataset(experiments_config)
            else:
                _logger.info("Skipping dataset upload (--skip-upload flag set)")
            
            all_results = await run_experiments(experiments_config)
            
            await run_evaluations(experiments_config, all_results)
        
        asyncio.run(workflow())
        
        _logger.info("=" * 80)
        _logger.info("Execution completed successfully")
        _logger.info("View results in Langfuse dashboard")
        _logger.info("=" * 80)
        sys.exit(0)
        
    except FileNotFoundError as e:
        _logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        _logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
