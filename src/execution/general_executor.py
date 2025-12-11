import logging, os, tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from langfuse import get_client
from pydantic import BaseModel
from src.llms import LLMClient
from typing import List, Any, Type
from urllib.parse import quote


""" CONFIG """
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


""" CLASSES """
class GenericExecutor:
    """
    Generic executor for tasks, adattabile a qualunque modello Pydantic
    che annoti i campi con json_schema_extra["role"].
    Parameters:
        :param client (LLMClient): The language model client to use for execution.
        :param dataset (List[BaseModel]): The dataset to run the executor on.
        :param model_class (Type[BaseModel]): The Pydantic model class representing the structure of the dataset items.
        :param models_list (List[str]): List of model names to evaluate.
        :param results_path (str): Path to save the execution results.
    """
    def __init__(self, client: LLMClient, dataset: List[BaseModel], model_class: Type[BaseModel], models_list: str, prompt_path: str, results_path: str):
        self.client = client
        self.dataset = dataset
        self.model_class = model_class
        self.models_list = models_list
        self.prompt_path = prompt_path
        self.results_path = results_path
        self.multiple_choice_results_dict: dict[str, dict] = {}
        self.open_ended_results_dict: dict[str, dict] = {}


    def completion(self, system_prompt: str, user_prompt: str) -> dict:
        """
        Executes a user message using the LMClient and returns the response for open-ended questions.
        Parameters:
            :param user_prompt: The user message to evaluate.
        """
        messages = self.client.format_messages(
            system = system_prompt,
            user = user_prompt,
        )
        results = {}
        for model in self.models_list:
            response = self.client.chat(messages=messages, model=model)
            results[model] = response
        return results


    def langfuse_experiment_for_each_dataset(self, experiment_name: str, experiment_description: str, model_name: str, dataset_name: str) -> Any:
        """
        Runs a Langfuse experiment using LMClient for both multiple choice and open-ended responses.
        Parameters:
            :param experiment_name: Name of the experiment.
            :param experiment_description: Description of the experiment.
            :param model_name: Name of the model to evaluate.
            :param dataset_name: Name of the dataset in Langfuse.
        """
        langfuse = get_client()
        encoded_dataset_name = quote(dataset_name, safe="")
        dataset = langfuse.get_dataset(encoded_dataset_name)

        # Reading the right execution prompt for the current dataset
        with open(os.path.join(self.prompt_path, f"{self.model_class.__name__.removesuffix("Item")}.txt"), "r", encoding="utf-8") as f:
            system_prompt = f.read()

        def task(item, **kwargs):
            # Unpacking dataset
            dataset_item = self.model_class.from_langfuse_item(item)

            # Sending message to the model
            open_ended_result = self.completion(system_prompt=system_prompt, user_prompt=dataset_item.user_prompt()).get(model_name, "")

            # Getting the result
            return str(open_ended_result)

        result = dataset.run_experiment(name=experiment_name, description=experiment_description, task=task, max_concurrency=12)
        print(result.format())
        return result


    def langfuse_experiment(self, dataset_name: str, experiment_name_prefix: str = "Model Evaluation"):
        """
        Runs Langfuse experiment for the passed model using LMClient.
        Parameters:
            :param dataset_name: Name of the dataset in Langfuse.
            :param models_list: List of model names to evaluate.
            :param experiment_name_prefix: Prefix for the experiment name.
        """
        experiment_name = {}
        experiment_description = {}
        for model_name in self.models_list:
            experiment_name[model_name] = f"{experiment_name_prefix} - {model_name}"
            experiment_description[model_name] = f"Evaluation of model {model_name} on {dataset_name}"

        # Main process
        results = {}
        with ThreadPoolExecutor() as executor:
            # Binding each to future to the relative model
            future_to_model = {
                executor.submit(
                    self.langfuse_experiment_for_each_dataset,
                    experiment_name[model_name],
                    experiment_description[model_name],
                    model_name,
                    dataset_name
                ): model_name
                for model_name in self.models_list
            }
            for future in tqdm.tqdm(as_completed(future_to_model), total=len(future_to_model), desc="Evaluating QA items"):
                model_name = future_to_model[future]
                result = future.result()
                results[model_name] = result

        print(results['gpt-4.1-mini'].item_results[0].trace_id)
        for result in results['gpt-4.1-mini'].item_results[:1]:
            print(f"Model Output: {result.output}")
            print(f"TOTAL: {dir(result.item)}")
            print(f"Langfuse: {dir(result.item.langfuse)}")

            print(f"Input: {result.item.input}")
            print(f"Expected Output: {result.item.expected_output}")
            print(f"Metadata: {result.item.metadata}")
            print(f"Output: {dir(result.item)}")
            print(f"Expected: {dir(result.trace_id)}")
            print("-----")