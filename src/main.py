"""Open Arena pipeline runner."""

import logging
from dotenv import load_dotenv
from src import DATA_LOCATION, load_config
from src.datasets.loaders import GenericDatasetLoader
from src.datasets.models import QAItem, ToolScaleItem, ToolsExample
from src.llms import LLMClient
from src.evaluator import GenericEvaluator
from src.execution import GenericExecutor


CONFIG = load_config()
CLIENT = LLMClient()
DATASETS = {
    "ToolsExample": {
        "excel": "ToolsExample.xlsx",
        "model_class": ToolsExample,
        "experiment_prefix": "ToolsExample Test",
        "experiment_prompt": CONFIG["datasets_system_prompts"]["tools_example_system_prompt"],
        "evaluation_prefix": "ToolsExample Evaluation",
        "evaluation_prompt": CONFIG["judge_system_prompt"],
    },
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


if __name__ == "__main__":

    # PIPELINE START
    LOGGER.info("\tPIPELINE START\n")
    for dataset_name, dataset_config in DATASETS.items():

        # === DATA PREPARATION ===
        LOGGER.info(f"\tDATA PREPARATION for '{dataset_name}':")
        loader = GenericDatasetLoader(
            input_path=DATA_LOCATION,
            excel_file=dataset_config["excel"],
            dataset_config=CONFIG["dataset_configuration"],
            dataset_name=dataset_name,
            model_class=dataset_config["model_class"],
        )

        # === EXECUTION ===
        LOGGER.info(f"\tEXECUTION for '{dataset_name}':")
        executor = GenericExecutor(
            client=CLIENT,
            model_class=dataset_config["model_class"],
            models_config=CONFIG["models_configuration"],
            dataset_prompt=dataset_config["experiment_prompt"],
        )
        experiment_results = executor.langfuse_experiment(
            dataset_name=dataset_name,
            experiment_name_prefix=dataset_config["experiment_prefix"],
        )

        # === EVALUATION ===
        LOGGER.info(f"\tEVALUATION for '{dataset_name}':")
        evaluator = GenericEvaluator(
            client=CLIENT,
            judge_model_config=CONFIG["judge_model"],
            judge_prompt=dataset_config["evaluation_prompt"],
        )
        evaluator.langfuse_evaluation(
            results_to_evaluate=experiment_results,
            dataset_name=dataset_name,
            evaluation_name_prefix=dataset_config["evaluation_prefix"],
        )
        LOGGER.info(f"\tCOMPLETED processing for dataset '{dataset_name}'.\n")

    # PIPELINE END
    LOGGER.info("\tPIPELINE END")
