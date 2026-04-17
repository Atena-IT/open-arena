import asyncio
import logging
import sys
import warnings

import click
from dotenv import load_dotenv
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from pydantic import ValidationError

from src.config.types import ExperimentsFile
from src.datasets import Row, build_dataset
from src.datasets.langfuse_upload import attach_existing_dataset, upload_rows
from src.evaluation import EvaluationResult, build_evaluator
from src.evaluation.evaluators import evaluator_mode
from src.execution import ExecutionResult, Executor
from src.llms import AgentCaller, SimpleCaller
from src.llms.types import MCPServerConfig

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
_logger = logging.getLogger(__name__)

load_dotenv()

def _load_rows(config: ExperimentsFile) -> list[Row]:
    _logger.info(f"Loading dataset: {config.dataset.name} (provider={config.dataset.source.get('provider')})")
    dataset = build_dataset(
        name=config.dataset.name,
        source=config.dataset.source,
        input_template=config.dataset.input,
        expected_output_template=config.dataset.expected_output,
        limit=config.dataset.limit,
    )
    rows = list(dataset)
    _logger.info(f"Fetched {len(rows)} rows")
    return rows




async def _run_experiments(config: ExperimentsFile, rows: list[Row]) -> list[list[ExecutionResult]]:
    _logger.info(f"Running {len(config.experiments)} experiments")
    all_results: list[list[ExecutionResult]] = []

    for exp_config in config.experiments:
        _logger.info(f"Experiment '{exp_config.name}' with model {exp_config.litellm.model}")
        mcp_servers: list[MCPServerConfig] = (
            [{"server_name": m.name, "url": str(m.url)} for m in exp_config.mcp]
            if exp_config.mcp
            else []
        )
        callbacks = [CallbackHandler()]
        caller_cls = AgentCaller if mcp_servers else SimpleCaller
        caller_kwargs: dict = {
            "llm_config": exp_config.litellm.model_dump(),
            "callbacks": callbacks,
        }
        if mcp_servers:
            caller_kwargs["mcp_servers"] = mcp_servers

        async with caller_cls(**caller_kwargs) as client:
            executor = Executor(
                dataset=rows,
                llm_client=client,
                system_prompt=config.system_prompt,
                experiment_name=exp_config.name,
                experiment_description=f"Experiment: {exp_config.name} with model {exp_config.litellm.model}",
            )
            results = await executor.execute()

        errors = sum(1 for r in results if r.error)
        if errors:
            _logger.warning(f"Experiment '{exp_config.name}' completed: {len(results)} items, {errors} errors")
        else:
            _logger.info(f"Experiment '{exp_config.name}' completed: {len(results)} items")
        all_results.append(results)

    return all_results


async def _run_evaluations(
    config: ExperimentsFile, all_results: list[list[ExecutionResult]]
) -> list[list[EvaluationResult]]:
    _logger.info(f"Evaluating {len(all_results)} experiments with {config.evaluation.method}")
    mode = evaluator_mode(config.evaluation.method)
    common_kwargs: dict = dict(
        method=config.evaluation.method,
        llm_config=config.evaluation.litellm.model_dump(),
        score_name=config.evaluation.score_name or "evaluation_score",
        max_concurrency=config.evaluation.max_concurrency or 10,
        callbacks=[CallbackHandler()],
    )
    if config.evaluation.system_prompt:
        common_kwargs["system_prompt"] = config.evaluation.system_prompt
    if config.evaluation.system_prompt_no_reference:
        common_kwargs["system_prompt_no_reference"] = config.evaluation.system_prompt_no_reference
    if config.evaluation.method in ("llm_as_verifier", "pairwise_verifier"):
        if config.evaluation.granularity is not None:
            common_kwargs["granularity"] = config.evaluation.granularity
        if config.evaluation.repeats is not None:
            common_kwargs["repeats"] = config.evaluation.repeats
    if config.evaluation.method == "llm_as_verifier" and config.evaluation.criteria:
        common_kwargs["criteria"] = config.evaluation.criteria
    all_evals: list[list[EvaluationResult]] = []

    if mode == "pointwise":
        for exp_config, results in zip(config.experiments, all_results):
            _logger.info(f"Evaluating experiment: {exp_config.name}")
            evaluator = build_evaluator(results=results, **common_kwargs)
            eval_results = await evaluator.evaluate()
            _log_summary(exp_config.name, eval_results)
            all_evals.append(eval_results)
        return all_evals

    # group mode: bundle results across experiments by lf_item_id
    _logger.info(f"Group evaluation across {len(config.experiments)} experiments")
    groups = _group_by_item(config.experiments, all_results)
    evaluator = build_evaluator(groups=groups, **common_kwargs)
    flat_results = await evaluator.evaluate()

    by_exp: dict[str, list[EvaluationResult]] = {exp.name: [] for exp in config.experiments}
    for r in flat_results:
        by_exp.setdefault(r.experiment_name or r.model_name, []).append(r)
    for exp_config in config.experiments:
        _log_summary(exp_config.name, by_exp.get(exp_config.name, []))
        all_evals.append(by_exp.get(exp_config.name, []))
    return all_evals


def _group_by_item(
    experiments, all_results: list[list[ExecutionResult]]
) -> list[dict[str, ExecutionResult]]:
    groups: dict[str, dict[str, ExecutionResult]] = {}
    for exp_config, results in zip(experiments, all_results):
        for r in results:
            key = r.metadata.get("lf_item_id") or r.input
            groups.setdefault(key, {})[exp_config.name] = r
    return list(groups.values())


def _log_summary(name: str, eval_results: list[EvaluationResult]) -> None:
    scored = sum(1 for r in eval_results if r.score is not None)
    errors = sum(1 for r in eval_results if r.error is not None)
    avg = sum(r.score for r in eval_results if r.score is not None) / scored if scored else 0
    if errors:
        _logger.warning(f"'{name}': {scored} scored (avg: {avg:.2f}), {errors} errors")
    else:
        _logger.info(f"'{name}': {scored} scored (avg: {avg:.2f})")


@click.command()
@click.option("--config", "-c", "config_path", required=True, type=click.Path(exists=True), help="Path to YAML configuration file")
@click.option("--skip-upload", is_flag=True, default=False, help="Skip dataset upload (reuse existing Langfuse dataset)")
def main(config_path: str, skip_upload: bool):
    """Run the Open Arena CLI workflow from a YAML configuration file."""
    try:
        _logger.info(f"Loading configuration from: {config_path}")
        try:
            config = ExperimentsFile.from_yaml(config_path)
        except ValidationError as e:
            _logger.error("Configuration validation failed:")
            for error in e.errors():
                _logger.error(f"  {error['loc']}: {error['msg']}")
            sys.exit(1)
        _logger.info(f"Validated: {len(config.experiments)} experiments")

        async def workflow():
            rows = _load_rows(config)
            if config.dataset.source.get("provider") == "langfuse":
                _logger.info("Source is Langfuse; skipping upload (items already exist remotely)")
            elif skip_upload:
                rows = attach_existing_dataset(rows, config.dataset.name)
            else:
                rows = await upload_rows(rows, dataset_name=config.dataset.name, description=config.dataset.description or "")
            results = await _run_experiments(config, rows)
            await _run_evaluations(config, results)
            get_client().flush()

        asyncio.run(workflow())
        _logger.info("Execution completed. View results in Langfuse.")
        sys.exit(0)

    except FileNotFoundError as e:
        _logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        _logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
