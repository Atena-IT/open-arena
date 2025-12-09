import logging
from dotenv import load_dotenv
from src import DATA_LOCATION, load_config
from src.datasets.loaders import GenericDatasetLoader
from src.datasets.models import QAItem, ToolScaleItem
from src.llms import LLMClient


""" CONFIG """
CONFIG = load_config()
load_dotenv()
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


""" MAIN """
if __name__ == "__main__":

    # Dataset creation and upload
    logger.info(f"\tStarting Data Preparation...")

    # QA Dataset
    qa_dataset = GenericDatasetLoader(
        input_path=DATA_LOCATION,
        excel_files=["QA.xlsx"],
        create_langfuse_dataset_bool=CONFIG['dataset_creation'],
        dataset_name="QADataset",
        model_class=QAItem
    )
    qa_items = qa_dataset.prepare_data()

    # ToolScale Dataset
    tool_scale_dataset = GenericDatasetLoader(
        input_path=DATA_LOCATION,
        excel_files=["ToolScale.xlsx"],
        create_langfuse_dataset_bool=CONFIG['dataset_creation'],
        dataset_name="ToolScaleDataset",
        model_class=ToolScaleItem
    )
    tool_scale_items = tool_scale_dataset.prepare_data()

    # Model Client
    client = LLMClient()