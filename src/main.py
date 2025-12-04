import logging, os
from dotenv import load_dotenv
from src.__init__ import (
    DATA_LOCATION,
    EXECUTION_RESULTS_LOCATION,
    EVALUATION_RESULTS_LOCATION,
)


""" CONFIG """
load_dotenv()
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


""" MAIN """
if __name__ == "__main__":

    # Dataset creation and upload
    logger.info(f"Starting Financial Advisor QA Data Preparation...")

    data_prep = FinancialAdvisorDataPrep(DATA_LOCATION)
    qa_items = data_prep.prepare_data()
    logger.info(f"Prepared {len(qa_items)} QA items.")
