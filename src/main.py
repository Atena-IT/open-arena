import logging
from dotenv import load_dotenv
from src import DATA_LOCATION, EXECUTION_RESULTS_LOCATION, EVALUATION_RESULTS_LOCATION, load_config, PROMPT_LOCATION
from src.datasets.loaders import GenericDatasetLoader
from src.datasets.models import QAItem, ToolScaleItem
from src.llms import LLMClient
from src.evaluator import GenericEvaluator
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
    qa_loader = GenericDatasetLoader(input_path=DATA_LOCATION, excel_files=["QA.xlsx"], create_langfuse_dataset_bool=CONFIG['dataset_creation'], dataset_name="QADataset", max_length_langfuse_dataset=CONFIG['max_length_langfuse_dataset'], model_class=QAItem)
    tool_scale_loader = GenericDatasetLoader(input_path=DATA_LOCATION, excel_files=["ToolScale.xlsx"], create_langfuse_dataset_bool=CONFIG['dataset_creation'], dataset_name="ToolScaleDataset", max_length_langfuse_dataset=CONFIG['max_length_langfuse_dataset'], model_class=ToolScaleItem)

    # Execution
    logger.info(f"\tEXECUTION:")
    client = LLMClient()
    qa_executor = GenericExecutor(client=client, model_class=QAItem, models_list=CONFIG['models'], prompt_path=PROMPT_LOCATION)
    qa_results = qa_executor.langfuse_experiment(dataset_name="QADataset", experiment_name_prefix="QA Test")
    tool_scale_executor = GenericExecutor(client=client, model_class=ToolScaleItem, models_list=CONFIG['models'], prompt_path=PROMPT_LOCATION)
    tool_scale_results = tool_scale_executor.langfuse_experiment(dataset_name="ToolScaleDataset", experiment_name_prefix="ToolScale Test")

    # Evaluation
    logger.info(f"\tEVALUATION:")
    qa_evaluator = GenericEvaluator(client=client, judge_model=CONFIG['judge_model'], prompt_path=PROMPT_LOCATION)
    qa_evaluator.langfuse_evaluation(results_to_evaluate=qa_results, dataset_name="QADataset", evaluation_name_prefix="QA Evaluation")
    tool_scale_evaluator = GenericEvaluator(client=client, judge_model=CONFIG['judge_model'], prompt_path=PROMPT_LOCATION)
    tool_scale_evaluator.langfuse_evaluation(results_to_evaluate=tool_scale_results, dataset_name="ToolScaleDataset", evaluation_name_prefix="ToolScale Evaluation")
