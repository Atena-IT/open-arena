import logging
from dotenv import load_dotenv
from src import DATA_LOCATION
from src.dataset.loaders import GenericDatasetLoader
from src.dataset.models import QAItem, ToolScaleItem


""" CONFIG """
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
        create_langfuse_dataset_bool=True,
        dataset_name="QADataset",
        model_class=QAItem
    )
    qa_items = qa_dataset.prepare_data()

    # ToolScale Dataset
    tool_scale_dataset = GenericDatasetLoader(
        input_path=DATA_LOCATION,
        excel_files=["ToolScale.xlsx"],
        create_langfuse_dataset_bool=True,
        dataset_name="ToolScaleDataset",
        model_class=ToolScaleItem
    )
    tool_scale_items = tool_scale_dataset.prepare_data()

