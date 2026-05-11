"""Notebook-aligned runtime helpers for GUI evaluation runs."""

from __future__ import annotations

import logging
from typing import Any

from langfuse.langchain import CallbackHandler

from demo.gui.backend.config_loader import RUNNABLE_CONFIG_PATH, _slugify
from demo.gui.backend.models import MetricSpec, RunState
from src.config.types import ExperimentConfig, ExperimentsFile
from src.datasets import Row, build_dataset
from src.evaluation.evaluators import evaluator_init_params
from src.llms import AgentCaller, ReplayCaller, SimpleCaller
from src.llms.types import MCPServerConfig

_logger = logging.getLogger(__name__)


def build_runtime_config(run: RunState) -> ExperimentsFile:
    base_config = ExperimentsFile.from_yaml(RUNNABLE_CONFIG_PATH)
    experiment_by_key = {
        _slugify(experiment.name): experiment.model_copy(deep=True)
        for experiment in base_config.experiments
    }

    selected_experiments: list[ExperimentConfig] = []
    for model in run.selectedModels:
        experiment = experiment_by_key.get(model.experimentKey)
        if experiment is None:
            raise ValueError(f"Runnable config missing experiment for key: {model.experimentKey!r}")
        selected_experiments.append(experiment)

    dataset = base_config.dataset.model_copy(deep=True)
    dataset.limit = run.sampleLimit
    dataset.name = run.datasetName

    return ExperimentsFile(
        dataset=dataset,
        system_prompt=base_config.system_prompt,
        experiments=selected_experiments,
        evaluation=base_config.evaluation.model_copy(deep=True),
    )


def load_runtime_rows(config: ExperimentsFile) -> list[Row]:
    _logger.info(
        "Loading dataset: %s (provider=%s)",
        config.dataset.name,
        config.dataset.source.get("provider"),
    )
    dataset = build_dataset(
        name=config.dataset.name,
        source=config.dataset.source,
        input_template=config.dataset.input,
        expected_output_template=config.dataset.expected_output,
        limit=config.dataset.limit,
    )
    rows = list(dataset)
    _logger.info("Fetched %s rows", len(rows))
    return rows


def _build_replay_lookup(rows: list[Row], trial_index: int) -> dict[str, tuple[str, list[dict[str, Any]]]]:
    trial_number = trial_index + 1
    output_key = f"trial_{trial_number}_output"
    trajectory_key = f"trial_{trial_number}_trajectory"
    lookup: dict[str, tuple[str, list[dict[str, Any]]]] = {}

    for input_text, _expected, metadata in rows:
        if input_text in lookup:
            raise ValueError(
                "Replay lookup requires unique rendered inputs, but found a duplicate: "
                f"{input_text[:120]!r}"
            )
        if output_key not in metadata or trajectory_key not in metadata:
            raise ValueError(
                f"Replay trial index {trial_index} missing expected metadata keys "
                f"{output_key!r} / {trajectory_key!r}"
            )

        trajectory = metadata[trajectory_key]
        if not isinstance(trajectory, list):
            raise ValueError(f"Expected {trajectory_key!r} to be a list, got {type(trajectory).__name__}")

        lookup[input_text] = (str(metadata[output_key] or ""), trajectory)

    return lookup


def build_execution_caller(
    experiment: ExperimentConfig,
    rows,
) -> tuple[type, dict[str, Any]]:
    callbacks = [CallbackHandler()]
    if experiment.replay_trial_index is not None:
        if experiment.mcp:
            raise ValueError("Replay mode does not support MCP servers")
        return ReplayCaller, {
            "llm_config": experiment.litellm.model_dump(),
            "lookup": _build_replay_lookup(rows, experiment.replay_trial_index),
            "callbacks": callbacks,
        }

    mcp_servers: list[MCPServerConfig] = (
        [{"server_name": server.name, "url": str(server.url)} for server in experiment.mcp]
        if experiment.mcp
        else []
    )
    caller_cls = AgentCaller if mcp_servers else SimpleCaller
    caller_kwargs: dict[str, Any] = {
        "llm_config": experiment.litellm.model_dump(),
        "callbacks": callbacks,
    }
    if mcp_servers:
        caller_kwargs["mcp_servers"] = mcp_servers
    return caller_cls, caller_kwargs


def build_metric_evaluator_kwargs(metric: MetricSpec, config: ExperimentsFile) -> dict[str, Any]:
    evaluation = config.evaluation.model_copy(deep=True)
    evaluation.method = metric.method
    evaluation.score_name = metric.key
    if metric.systemPrompt is not None:
        evaluation.system_prompt = metric.systemPrompt
    if metric.systemPromptNoReference is not None:
        evaluation.system_prompt_no_reference = metric.systemPromptNoReference

    kwargs: dict[str, Any] = {
        "method": metric.method,
        "llm_config": evaluation.litellm.model_dump(),
        "callbacks": [CallbackHandler()],
    }
    accepted_params = evaluator_init_params(metric.method)
    for field_name in evaluation.__class__.model_fields:
        if field_name in {"method", "litellm"}:
            continue
        if field_name not in accepted_params:
            continue
        value = getattr(evaluation, field_name)
        if value is None:
            continue
        kwargs[field_name] = value
    return kwargs
