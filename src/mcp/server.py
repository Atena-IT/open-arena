from dotenv import load_dotenv
import logging
from typing import Any, Dict, Optional
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel
from src import DATA_LOCATION, load_config
from src.llms import LLMClient
from src.datasets.loaders.general_dataset_loader import GenericDatasetLoader
from src.execution.general_executor import GenericExecutor
from src.evaluator.general_evaluator import GenericEvaluator
from src.datasets.models import QAItem, ToolScaleItem


# =========================
# CONFIG / GLOBALS
# =========================
load_dotenv()

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)

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

app = FastAPI(title="MultiModelEvalServer", version="1.0.0")


# =========================
# CORE LOGIC
# =========================
def _run_pipeline_for_dataset(
    dataset_name: str,
    dataset_config: dict,
    config: dict,
    client: LLMClient,
) -> Dict[str, Any]:
    """
    Run:
      1) DATA PREPARATION (create/upload Langfuse dataset from Excel)
      2) EXECUTION (run all configured models on the dataset)
      3) EVALUATION (LLM-as-a-judge on experiment results)
    """
    excel_file = dataset_config["excel"]
    model_class = dataset_config["model_class"]
    experiment_prefix = dataset_config["experiment_prefix"]
    evaluation_prefix = dataset_config["evaluation_prefix"]
    experiment_prompt = dataset_config["experiment_prompt"]
    evaluation_prompt = dataset_config["evaluation_prompt"]

    # === DATA PREPARATION ===
    LOGGER.info(f"\tDATA PREPARATION for '{dataset_name}':")
    _ = GenericDatasetLoader(
        input_path=DATA_LOCATION,
        excel_files=[excel_file],
        dataset_config=config["dataset_configuration"],
        dataset_name=dataset_name,
        model_class=model_class,
    )

    # === EXECUTION ===
    LOGGER.info(f"\tEXECUTION for '{dataset_name}':")
    executor = GenericExecutor(
        client=client,
        model_class=model_class,
        models_config=config["models_configuration"],
        dataset_prompt=experiment_prompt,
    )
    experiment_results = executor.langfuse_experiment(
        dataset_name=dataset_name,
        experiment_name_prefix=experiment_prefix,
    )

    # === EVALUATION ===
    LOGGER.info(f"\tEVALUATION for '{dataset_name}':")
    evaluator = GenericEvaluator(
        client=client,
        judge_model_config=config["judge_model"],
        judge_prompt=evaluation_prompt,
    )
    evaluator.langfuse_evaluation(
        results_to_evaluate=experiment_results,
        dataset_name=dataset_name,
        evaluation_name_prefix=evaluation_prefix,
    )

    LOGGER.info(f"\tCOMPLETED processing for dataset '{dataset_name}'.\n")

    return {
        "dataset_name": dataset_name,
        "models": list(experiment_results.keys()),
        "experiment_prefix": experiment_prefix,
        "evaluation_prefix": evaluation_prefix,
    }


def _assert_dataset_exists(dataset_name: str) -> None:
    if dataset_name not in DATASETS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown dataset '{dataset_name}'. Allowed: {list(DATASETS.keys())}",
        )


# =========================
# API MODELS
# =========================
class PipelineResponse(BaseModel):
    dataset_name: str
    models: list[str]
    experiment_prefix: str
    evaluation_prefix: str


class AsyncAcceptedResponse(BaseModel):
    status: str
    dataset_name: str


# =========================
# ENDPOINTS (equivalenti ai tool MCP)
# =========================
@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/models")
def list_models_under_test() -> Dict[str, Any]:
    """
    Equivalent of list_models_under_test MCP tool.
    Supports both list-of-dicts and list-of-strings shapes.
    """
    models = CONFIG.get("models_configuration", [])
    return {
        "models": [m["name"] if isinstance(m, dict) else m for m in models],
    }


@app.post("/pipelines/qa", response_model=PipelineResponse)
def run_qa_pipeline() -> Dict[str, Any]:
    """
    Equivalent of run_qa_pipeline MCP tool.
    Synchronous: request waits until the pipeline finishes.
    """
    return _run_pipeline_for_dataset(
        dataset_name="QADataset",
        dataset_config=DATASETS["QADataset"],
        config=CONFIG,
        client=CLIENT,
    )


@app.post("/pipelines/toolscale", response_model=PipelineResponse)
def run_toolscale_pipeline() -> Dict[str, Any]:
    """
    Equivalent of run_toolscale_pipeline MCP tool.
    Synchronous: request waits until the pipeline finishes.
    """
    return _run_pipeline_for_dataset(
        dataset_name="ToolScaleDataset",
        dataset_config=DATASETS["ToolScaleDataset"],
        config=CONFIG,
        client=CLIENT,
    )


# =========================
# OPTIONAL: generic endpoint + async execution
# =========================
@app.post("/pipelines/{dataset_name}", response_model=PipelineResponse)
def run_pipeline(dataset_name: str) -> Dict[str, Any]:
    """
    Generic synchronous endpoint for any dataset in DATASETS.
    """
    _assert_dataset_exists(dataset_name)
    return _run_pipeline_for_dataset(
        dataset_name=dataset_name,
        dataset_config=DATASETS[dataset_name],
        config=CONFIG,
        client=CLIENT,
    )


@app.post("/pipelines/{dataset_name}/async", response_model=AsyncAcceptedResponse)
def run_pipeline_async(dataset_name: str, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """
    Fire-and-forget: returns immediately (202-like semantics) and runs in background.
    Note: without a job store, you won't have status/progress via API.
    """
    _assert_dataset_exists(dataset_name)

    background_tasks.add_task(
        _run_pipeline_for_dataset,
        dataset_name,
        DATASETS[dataset_name],
        CONFIG,
        CLIENT,
    )
    return {"status": "accepted", "dataset_name": dataset_name}


# =========================
# LOCAL RUN (optional)
# =========================
# Run with:
#   uvicorn server_fastapi:app --host 0.0.0.0 --port 8000