import logging
from dotenv import load_dotenv
from fastmcp import FastMCP
from src import DATA_LOCATION, load_config
from src.llms import LLMClient
from src.datasets.loaders.general_dataset_loader import GenericDatasetLoader
from src.execution.general_executor import GenericExecutor
from src.evaluator.general_evaluator import GenericEvaluator
from src.datasets.models import QAItem
from src.datasets.models import ToolScaleItem
from typing import Dict, Any


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
MCP = FastMCP("MultiModelEvalServer")


""" FUNCTIONS """
def _run_pipeline_for_dataset(dataset_name: str, dataset_config: dict, config: dict, client: LLMClient) -> dict:
    """
    Run the full pipeline for a single dataset:
      1. DATA PREPARATION (create/upload Langfuse dataset from Excel)
      2. EXECUTION (run all configured models on the dataset)
      3. EVALUATION (run LLM-as-a-judge on the experiment results)

    Parameters
    ----------
    dataset_name : str
        Langfuse dataset name (e.g. "QADataset", "ToolScaleDataset").
    dataset_config : dict
        Per-dataset configuration from DATASETS, e.g.:
        {
          "excel": "QA.xlsx",
          "model_class": QAItem,
          "experiment_prefix": "QA Test",
          "experiment_prompt": "...",
          "evaluation_prefix": "QA Evaluation",
          "evaluation_prompt": "..."
        }
    config : dict
        Global CONFIG loaded via load_config().
    client : LLMClient
        Shared LLM client instance.
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

    # You can return whatever is useful for callers (e.g. results + some metadata)
    return {
        "dataset_name": dataset_name,
        "models": list(experiment_results.keys()),
        "experiment_prefix": experiment_prefix,
        "evaluation_prefix": evaluation_prefix,
    }


@MCP.tool
def run_qa_pipeline() -> Dict[str, Any]:
    """
    Run the full pipeline (prepare + execute + evaluate) for the QA dataset.
    """
    return _run_pipeline_for_dataset(
        dataset_name="QADataset",
        dataset_config=DATASETS["QADataset"],
        config=CONFIG,
        client=CLIENT,
    )


@MCP.tool
def run_toolscale_pipeline() -> Dict[str, Any]:
    """
    Run the full pipeline (prepare + execute + evaluate) for the ToolScale dataset.
    """
    return _run_pipeline_for_dataset(
        dataset_name="ToolScaleDataset",
        dataset_config=DATASETS["ToolScaleDataset"],
        config=CONFIG,
        client=CLIENT,
    )


@MCP.tool
def list_models_under_test() -> Dict[str, Any]:
    """
    Return the list of models configured in models_configuration.
    Supports both list-of-dicts and list-of-strings shapes.
    """
    models = CONFIG.get("models_configuration", [])
    return {
        "models": [
            m["name"] if isinstance(m, dict) else m
            for m in models
        ]
    }


""" MAIN """
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    MCP.run()
