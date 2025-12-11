import logging
from dotenv import load_dotenv
from src import DATA_LOCATION, EXECUTION_RESULTS_LOCATION, EVALUATION_RESULTS_LOCATION, load_config, PROMPT_LOCATION
from src.datasets.loaders import GenericDatasetLoader
from src.datasets.models import QAItem, ToolScaleItem
from src.llms import LLMClient
from src.execution import GenericExecutor


""" CONFIG """
CONFIG = load_config()
load_dotenv()
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


""" MAIN """
if __name__ == "__main__":

    # Dataset creation and upload
    logger.info(f"\tDATA PREPARATION:")
    # QA Dataset
    qa_loader = GenericDatasetLoader(
        input_path=DATA_LOCATION,
        excel_files=["QA.xlsx"],
        create_langfuse_dataset_bool=CONFIG['dataset_creation'],
        dataset_name="QADataset",
        model_class=QAItem
    )
    qa_dataset = qa_loader.prepare_data()
    # ToolScale Dataset
    tool_scale_loader = GenericDatasetLoader(
        input_path=DATA_LOCATION,
        excel_files=["ToolScale.xlsx"],
        create_langfuse_dataset_bool=CONFIG['dataset_creation'],
        dataset_name="ToolScaleDataset",
        model_class=ToolScaleItem
    )
    tool_scale_dataset = tool_scale_loader.prepare_data()

    # Execution
    logger.info(f"\tEXECUTION:")
    client = LLMClient()
    executor = GenericExecutor(client=client, dataset=qa_dataset, model_class=QAItem, models_list=CONFIG['models'], prompt_path=PROMPT_LOCATION, results_path=EXECUTION_RESULTS_LOCATION)
    executor.langfuse_experiment(dataset_name="QADataset", experiment_name_prefix="QA Test")
    # executor = GenericExecutor(client=client, dataset=tool_scale_dataset, model_class=ToolScaleItem, models_list=CONFIG['models'], prompt_path=PROMPT_LOCATION, results_path=EXECUTION_RESULTS_LOCATION)
    # executor.langfuse_experiment(dataset_name="ToolScaleDataset", experiment_name_prefix="ToolScale Test")
