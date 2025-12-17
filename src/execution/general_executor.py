import logging, tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from langfuse import get_client
from pydantic import BaseModel
from src.llms import LLMClient
from typing import Any, Type
from urllib.parse import quote


""" CONFIG """
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)


""" CLASSES """
class GenericExecutor:
    """
    Generic executor for tasks, for any Pydantic model that annotates its fields with json_schema_extra["langfuse_dataset"].
    Parameters:
        :param client (LLMClient): The language model client to use for execution.
        :param model_class (Type[BaseModel]): The Pydantic model class represents the structure of the dataset items.
        :param models_config (List[dict]): List of model name and configuration to evaluate.
        :param prompt_path (str): Path to pick the right prompt for the completion.
    """
    def __init__(self, client: LLMClient, model_class: Type[BaseModel], models_config: list, dataset_prompt: str):
        self.client = client
        self.model_class = model_class
        self.models_config = models_config
        self.dataset_prompt = dataset_prompt


    def completion(self, model_config: dict, system_prompt: str, user_prompt: str) -> str:
        """
        Executes a user message using the LMClient and returns the response for open-ended questions.
        Parameters:
            :param model_config: Model configuration to use for this completion.
            :param system_prompt: The system prompt to use on this interaction.
            :param user_prompt: The user message to evaluate.
        Return:
            :return: LLM output message
        """
        messages = self.client.format_messages(system=system_prompt, user=user_prompt)
        return self.client.chat(messages=messages, model_config=model_config)


    def langfuse_experiment_per_model(self, dataset_name: str, experiment_name: str, experiment_description: str, model_config: dict) -> Any:
        """
        Runs a Langfuse experiment using LMClient for both multiple choice and open-ended responses.
        Parameters:
            :param dataset_name: Name of the dataset in Langfuse.
            :param experiment_name: Name of the experiment.
            :param experiment_description: Description of the experiment.
            :param model_config: Configuration of the model to evaluate.
        Return:
            :return: content result of the experiment on Langfuse
        """
        langfuse = get_client()
        encoded_dataset_name = quote(dataset_name, safe="")
        dataset = langfuse.get_dataset(encoded_dataset_name)

        # Processing single sample
        def task(item):
            dataset_item = self.model_class.from_langfuse_item(item)    # Unpacking dataset
            open_ended_result = self.completion(                        # Sending the message to the model
                model_config=model_config,
                system_prompt=self.dataset_prompt,
                user_prompt=dataset_item.user_prompt()
            )
            return str(open_ended_result)                               # Getting the result

        result = dataset.run_experiment(
            name=experiment_name,
            description=experiment_description,
            task=task,
            max_concurrency=12)
        LOGGER.info(result.format())
        return result


    def langfuse_experiment(self, dataset_name: str, experiment_name_prefix: str = "Model Experiment"):
        """
        Runs Langfuse experiment for the passed model using LMClient.
        Parameters:
            :param dataset_name: Name of the dataset in Langfuse.
            :param experiment_name_prefix: Prefix for the experiment name.
        Return:
            :return: content result of all experiments on Langfuse
        """
        experiment_name = {}
        experiment_description = {}
        for model_config in self.models_config:
            experiment_name[model_config["name"]] = f"{experiment_name_prefix} - {model_config['name']}"
            experiment_description[model_config["name"]] = f"Test of model {model_config['name']} on {model_config['name']}"

        # Main process
        results = {}
        with ThreadPoolExecutor() as executor:
            # Binding each to future to the relative model
            future_to_model = {
                executor.submit(
                    self.langfuse_experiment_per_model,
                    dataset_name,
                    experiment_name[model_config["name"]],
                    experiment_description[model_config["name"]],
                    model_config
                ): model_config
                for model_config in self.models_config
            }
            for future in tqdm.tqdm(as_completed(future_to_model), total=len(future_to_model), desc="Dataset test per model"):
                model_config = future_to_model[future]
                result = future.result()
                results[model_config["name"]] = result

        return results
