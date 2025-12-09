
import os
from dotenv import load_dotenv

load_dotenv()

import logging


from src.language_models_core.language_models import LMClient
from src.execution_core.fin_adv_executor import FinAdvExecutor
from src.eval_core.fin_adv_evaluator import FinAdvEvaluator

from src.__init__ import (
    DATA_LOCATION,
    EXECUTION_RESULTS_LOCATION,
    EVALUATION_RESULTS_LOCATION,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    models_list = ["gpt-4.1"]

    complete = False
    if complete:
        logger.info("Starting Financial Advisor QA Data Preparation...")

        data_prep = FinancialAdvisorDataPrep(DATA_LOCATION)
        qa_items = data_prep.prepare_data()
        logger.info(f"Prepared {len(qa_items)} QA items.")


        client = LMClient()

        for model_name in models_list:
            # Create model-specific folders
            exec_model_folder = os.path.join(EXECUTION_RESULTS_LOCATION, model_name)
            eval_model_folder = os.path.join(EVALUATION_RESULTS_LOCATION, model_name)
            os.makedirs(exec_model_folder, exist_ok=True)
            os.makedirs(eval_model_folder, exist_ok=True)

            # Step 2: Executor
            executor = FinAdvExecutor(client, qa_items, [model_name], exec_model_folder)
            logger.info(f"Running Executor for model {model_name}...")
            executor.run_on_dataset()
            logger.info(f"Executor finished for {model_name}. Results saved.")

            # executor.run_langfuse_experiments_for_models_with_lmclient(
            #     models_list=models_list
            # )
            # logger.info(f"Langfuse experiments completed for model {model_name}.")

            evaluate = False
            if evaluate:
                # Step 3: Evaluator
                evaluator = FinAdvEvaluator(client, qa_items, [model_name])
                logger.info(f"Running Evaluator for model {model_name}...")
                evaluator.run_evaluation_from_executor_results(
                    model=model_name,
                    multiple_choice_path=os.path.join(
                        exec_model_folder, "multiple_choice_results.json"
                    ),
                    open_ended_path=os.path.join(
                        exec_model_folder, "open_ended_results.json"
                    ),
                    evaluation_results_location=eval_model_folder,
                )
                evaluator.save_overall_statistics(eval_model_folder)
                logger.info(f"Evaluation finished for {model_name}. Results saved.")

                # Only run grouped correctness statistics
                evaluator.save_grouped_correctness_statistics(eval_model_folder)

    # Generate and evaluate synthetic exam for each model
    generate_exams = True
    if generate_exams:
        for model_name in models_list:
            eval_model_folder = os.path.join(EVALUATION_RESULTS_LOCATION, model_name)
            evaluator = FinAdvEvaluator(None, [], [])
            evaluator.generate_and_evaluate_exams(
                evaluation_results_location=eval_model_folder,
                input_filename=f"qa_items_evaluation.json",
                exams_folder=f"synthetic_exams",
            )
