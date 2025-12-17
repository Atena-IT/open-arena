import logging, json
from concurrent.futures import ThreadPoolExecutor, as_completed
from langfuse import get_client
from src.llms import LLMClient
from typing import Any


""" CONFIG """
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)


""" CLASS """
class GenericEvaluator:
    """
    Generic LLM-as-a-judge evaluator for previous task results.
    Parameters:
        :param client (LLMClient): The language model client to use for execution.
        :param judge_model (str): Model to use as a judge.
        :param model_class (Type[BaseModel]): The Pydantic model class represents the structure of the dataset items.
        :param prompt_path (str): Path to pick the right prompt for the completion.
    """
    def __init__(self, client: LLMClient, judge_model_config: dict, judge_prompt: str):
        self.client = client
        self.judge_model_config = judge_model_config
        self.judge_prompt = judge_prompt


    def completion(self, model_config: dict, system_prompt: str, user_prompt: str) -> str:
        """
        Executes a user message using the LMClient and returns the response for open-ended questions.
        Parameters:
            :param model_config: Judge Model configuration to use for this completion.
            :param system_prompt: The system prompt to use on this interaction.
            :param user_prompt: The user message to evaluate.
        Return:
            :return: LLM output message
        """
        messages = self.client.format_messages(system=system_prompt, user=user_prompt)
        return self.client.chat(messages=messages, model_config=model_config)


    @staticmethod
    def judge_prompt_payload(user_input: str, model_output: str, expected_output: str | None, metadata: dict | None = None) -> dict:
        """
        Build the JSON payload to be passed as the user message to the judge model.
        Parameters:
            :param user_input: Original user input / question from the dataset item.
            :param model_output: Prediction produced by the evaluated model.
            :param expected_output: Ground truth or expected answer, if available.
            :param metadata: Optional additional context (e.g., task type, difficulty).
        Return:
            :return: LLM-as-a-judge payload
        """
        payload = {
            "input": user_input,
            "output": model_output,
            "expected_output": expected_output,
        }
        if metadata:
            payload["metadata"] = metadata
        return payload


    @staticmethod
    def parse_judge_response(raw_response: str | dict):
        """
        Parse the judge model response as JSON and extract evaluation fields.
        The judge is expected to return a JSON object with at least:
        - "score": an integer (e.g., 1–5) representing the evaluation score
        - "explanation": a short string explaining the score
        Parameters:
            :param raw_response: Raw response returned by the LLM client.
        Returns:
            :return: tuple[Any, str]:
                A tuple (score, explanation) where:
                - score: the parsed value of the "score" field, or None on error
                - explanation: the parsed "explanation" field, or an error message
                  (e.g. "Parsing error. Raw: ...") if parsing fails.
        """
        try:
            if isinstance(raw_response, dict) and "content" in raw_response:
                content = raw_response["content"]
            else:
                content = raw_response
            parsed = json.loads(content)
            return parsed.get("score"), parsed.get("explanation")
        except Exception:
            return None, f"Parsing error. Raw: {raw_response}"


    def judge_single_experiment(self, experiment_result):
        """
        Apply the LLM judge to ALL item_results of a single ExperimentResult.
        Create a llm_judge_score score for each trace.
        Parameters:
            :param experiment_result: Previous task results
        """
        langfuse = get_client()
        for result in experiment_result.item_results:
            user_input = result.item.input
            expected_output = result.item.expected_output
            metadata = result.item.metadata
            model_output = result.output
            trace_id = result.trace_id

            # JSON payload
            user_payload = self.judge_prompt_payload(
                user_input=user_input,
                model_output=model_output,
                expected_output=expected_output,
                metadata=metadata)
            user_prompt = json.dumps(user_payload, ensure_ascii=False)

            # LLM-as-a-judge
            raw_response = self.completion(
                model_config=self.judge_model_config,
                system_prompt=self.judge_prompt,
                user_prompt=user_prompt)

            # Parsing judge answer
            judge_score, judge_explanation = self.parse_judge_response(raw_response)
            if judge_score is not None:
                langfuse.create_score(
                    trace_id=trace_id,
                    name="llm_judge_score",
                    value=judge_score,
                    comment=judge_explanation)


    def langfuse_evaluation(self, results_to_evaluate: dict[str, Any], dataset_name: str, evaluation_name_prefix: str = "Model Evaluation"):
        """
        Apply LLM-as-a-judge to the results of a previous experiment.
        Parameters:
            :param results_to_evaluate: dict[model_name] -> ExperimentResult (output of the GenericExecutor)
            :param dataset_name: Name of the dataset (used only for naming/logging)
            :param evaluation_name_prefix: Prefix for the evaluation name.
        """
        evaluation_name: dict[str, str] = {}
        evaluation_description: dict[str, str] = {}
        for model_name in results_to_evaluate.keys():
            evaluation_name[model_name] = f"{evaluation_name_prefix} - {model_name}"
            evaluation_description[model_name] = f"LLM-as-a-judge evaluation of model {model_name} on {dataset_name}"

        # Main process
        with ThreadPoolExecutor() as executor:
            # Binding each to future to the relative model
            future_to_model = {
                executor.submit(self.judge_single_experiment, exp_result): model_name
                for model_name, exp_result in results_to_evaluate.items()
            }
            for future in as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    future.result()
                    LOGGER.info(f"Finished judge '{self.judge_model_config['name']}' on experiment results of model '{model_name}'")
                except Exception as e:
                    LOGGER.exception(f"Error while running judge '{self.judge_model_config['name']}' on model '{model_name}': {e}")
