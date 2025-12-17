from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastmcp import FastMCP
import logging
from pydantic import BaseModel
from src import DATA_LOCATION, load_config
from src.llms import LLMClient
from src.datasets.loaders.general_dataset_loader import GenericDatasetLoader
from src.execution.general_executor import GenericExecutor
from src.evaluator.general_evaluator import GenericEvaluator
from src.datasets.models import QAItem, ToolScaleItem
from typing import Any, Dict


""" CONFIG """
CONFIG = load_config()
CLIENT = LLMClient()
DATASETS = {
    "QADataset": {
        "excel": "QA.xlsx",
        "model_class": QAItem,
        "experiment_prefix": "QA Test",
        "experiment_prompt": CONFIG["datasets_system_prompts"]["qa_system_prompt"],
        "evaluation_prefix": "QA Evaluation",
        "evaluation_prompt": CONFIG["judge_system_prompt"],
    },
    "ToolScaleDataset": {
        "excel": "ToolScale.xlsx",
        "model_class": ToolScaleItem,
        "experiment_prefix": "ToolScale Test",
        "experiment_prompt": CONFIG["datasets_system_prompts"]["tool_scale_system_prompt"],
        "evaluation_prefix": "ToolScale Evaluation",
        "evaluation_prompt": CONFIG["judge_system_prompt"],
    },
}
load_dotenv()
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)


""" FUNCTIONS """
def _run_pipeline_for_dataset(dataset_name: str, dataset_config: dict, config: dict, client: LLMClient) -> Dict[str, Any]:

    # DATA PREPARATION
    LOGGER.info(f"\tDATA PREPARATION for '{dataset_name}':")
    _ = GenericDatasetLoader(
        input_path=DATA_LOCATION,
        excel_file=dataset_config["excel"],
        dataset_config=config["dataset_configuration"],
        dataset_name=dataset_name,
        model_class=dataset_config["model_class"],
    )

    # EXECUTION
    LOGGER.info(f"\tEXECUTION for '{dataset_name}':")
    executor = GenericExecutor(
        client=client,
        model_class=dataset_config["model_class"],
        models_config=config["models_configuration"],
        dataset_prompt=dataset_config["experiment_prompt"],
    )
    experiment_results = executor.langfuse_experiment(
        dataset_name=dataset_name,
        experiment_name_prefix=dataset_config["experiment_prefix"],
    )

    # EVALUATION
    LOGGER.info(f"\tEVALUATION for '{dataset_name}':")
    evaluator = GenericEvaluator(
        client=client,
        judge_model_config=config["judge_model"],
        judge_prompt=dataset_config["evaluation_prompt"],
    )
    evaluator.langfuse_evaluation(
        results_to_evaluate=experiment_results,
        dataset_name=dataset_name,
        evaluation_name_prefix=dataset_config["evaluation_prefix"],
    )

    LOGGER.info(f"\tCOMPLETED processing for dataset '{dataset_name}'.\n")

    return {
        "dataset_name": dataset_name,
        "models": list(experiment_results.keys()),
        "experiment_prefix": dataset_config["experiment_prefix"],
        "evaluation_prefix": dataset_config["evaluation_prefix"],
    }


def _assert_dataset_exists(dataset_name: str) -> None:
    if dataset_name not in DATASETS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown dataset '{dataset_name}'. Allowed: {list(DATASETS.keys())}",
        )


# =========================
# FASTAPI APP (HTTP)
# =========================
app = FastAPI(title="MultiModelEvalServer", version="1.0.0")

class PipelineResponse(BaseModel):
    dataset_name: str
    models: list[str]
    experiment_prefix: str
    evaluation_prefix: str

class AsyncAcceptedResponse(BaseModel):
    status: str
    dataset_name: str

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}

@app.get("/models")
def http_list_models_under_test() -> Dict[str, Any]:
    models = CONFIG.get("models_configuration", [])
    return {"models": [m["name"] if isinstance(m, dict) else m for m in models]}

@app.post("/pipelines/qa", response_model=PipelineResponse)
def http_run_qa_pipeline() -> Dict[str, Any]:
    return _run_pipeline_for_dataset("QADataset", DATASETS["QADataset"], CONFIG, CLIENT)

@app.post("/pipelines/toolscale", response_model=PipelineResponse)
def http_run_toolscale_pipeline() -> Dict[str, Any]:
    return _run_pipeline_for_dataset("ToolScaleDataset", DATASETS["ToolScaleDataset"], CONFIG, CLIENT)

@app.post("/pipelines/{dataset_name}", response_model=PipelineResponse)
def http_run_pipeline(dataset_name: str) -> Dict[str, Any]:
    _assert_dataset_exists(dataset_name)
    return _run_pipeline_for_dataset(dataset_name, DATASETS[dataset_name], CONFIG, CLIENT)

@app.post("/pipelines/{dataset_name}/async", response_model=AsyncAcceptedResponse)
def http_run_pipeline_async(dataset_name: str, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    _assert_dataset_exists(dataset_name)
    background_tasks.add_task(_run_pipeline_for_dataset, dataset_name, DATASETS[dataset_name], CONFIG, CLIENT)
    return {"status": "accepted", "dataset_name": dataset_name}


# =========================
# MCP SERVER (stdio tools)
# =========================
mcp = FastMCP("MultiModelEvalServer")

@mcp.tool
def run_qa_pipeline() -> Dict[str, Any]:
    return _run_pipeline_for_dataset("QADataset", DATASETS["QADataset"], CONFIG, CLIENT)

@mcp.tool
def run_toolscale_pipeline() -> Dict[str, Any]:
    return _run_pipeline_for_dataset("ToolScaleDataset", DATASETS["ToolScaleDataset"], CONFIG, CLIENT)

@mcp.tool
def list_models_under_test() -> Dict[str, Any]:
    models = CONFIG.get("models_configuration", [])
    return {"models": [m["name"] if isinstance(m, dict) else m for m in models]}


# Entry-point MCP (stdio):
def run_mcp() -> None:
    mcp.run()


if __name__ == "__main__":
    # Se lo esegui con `python -m src.mcp.server_hybrid` parte MCP (stdio)
    run_mcp()