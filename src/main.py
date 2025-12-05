import logging, os
from dotenv import load_dotenv
from src import DATA_LOCATION
from src.dataset.loaders import QADatasetLoader
from src.dataset.loaders.dataset_m import GenericDatasetLoader


""" CONFIG """
load_dotenv()
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


""" MAIN """
if __name__ == "__main__":

    # Dataset creation and upload
    logger.info(f"\tStarting Data Preparation...")

    """
    dataset = QADatasetLoader(input_path=DATA_LOCATION,
                              create_langfuse_dataset_bool=True,
                              dataset_name="QADataset")
    dataset.load()
    qa_items = dataset.prepare_data()
    logger.info(f"\tPrepared {len(qa_items)} QA items.")
    """

    dataset = GenericDatasetLoader(
        input_path=DATA_LOCATION,
        create_langfuse_dataset_bool=True,
        dataset_name="QADataset",
    )
    dataset.load()
    qa_items = dataset.prepare_data()

    print(qa_items[0])