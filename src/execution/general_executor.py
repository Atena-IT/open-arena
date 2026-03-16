import os, logging, tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from langfuse import get_client
from pydantic import BaseModel
from src.llms import LLMClient
from src.mcp_server import MCPWorker, MCPWorkerPool
from typing import Any, Type
from urllib.parse import quote


""" CONFIG """
load_dotenv()
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
        :param dataset_prompt (str): Right prompt for the completion.
        :param mcp_workers (int): Number of threads to use for accessing the MCP server.
    """
    def __init__(self, client: LLMClient, model_class: Type[BaseModel], models_config: list, dataset_prompt: str, mcp_workers: int = 4):
        self.client = client
        self.model_class = model_class
        self.models_config = models_config
        self.dataset_prompt = dataset_prompt
        self.workers = mcp_workers
        mcp_url = f'{os.getenv("FAST_API_URL", "")}:{os.getenv("FAST_API_PORT", "")}/{os.getenv("MCP_PATH", "")}'
        token = os.getenv("MCP_TOKEN", "")
        self.mcp_pool = MCPWorkerPool(mcp_url=mcp_url, size=self.workers, token=token)


    async def acompletion_with_tools(self, worker: MCPWorker, model_config: dict, system_prompt: str, user_prompt: str) -> str:
        """
        Executes a user message using the LMClient with MCP tools and returns the response for open-ended questions.
        Parameters:
            :param worker:
            :param model_config: Model configuration to use for this completion.
            :param system_prompt: The system prompt to use on this interaction.
            :param user_prompt: The user message to evaluate.
        Return:
            return LLM output message
        """
        messages = self.client.format_messages(system=system_prompt, user=user_prompt)
        return await self.client.chat_with_mcp_tools(
            messages=messages,
            model_config=model_config,
            mcp_session=worker.mcp_session,
            mcp_tools_openai=worker.mcp_tools,
        )


    def completion_with_tools(self, model_config: dict, system_prompt: str, user_prompt: str) -> str:
        """
        Async completion wrapper which syncs the API used by Langfuse task(). It borrows one MCP worker (one persistent
        SSE connection), runs the coroutine on that worker loop, then releases the worker back to the pool.
        Parameters:
            :param model_config: Model configuration to use for this completion.
            :param system_prompt: The system prompt to use on this interaction.
            :param user_prompt: The user message to evaluate.
        Return:
            :return: LLM output message
        """
        worker = self.mcp_pool.acquire()
        try:
            return worker.submit(self.acompletion_with_tools(worker, model_config, system_prompt, user_prompt))
        finally:
            self.mcp_pool.release(worker)


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

        # Processing each item
        def task(item):
            dataset_item = self.model_class.from_langfuse_item(item)    # Unpacking dataset
            out = self.completion_with_tools(                           # Sending the message to the model
                model_config=model_config,
                system_prompt=self.dataset_prompt,
                user_prompt=dataset_item.user_prompt(),
            )
            return str(out)                                             # Getting the result

        result = dataset.run_experiment(
            name=experiment_name,
            description=experiment_description,
            task=task,
            max_concurrency=self.workers,
        )
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
            future_to_model = {executor.submit(
                self.langfuse_experiment_per_model,
                dataset_name,
                experiment_name[model_config["name"]],
                experiment_description[model_config["name"]],
                model_config): model_config for model_config in self.models_config
            }
            for future in tqdm.tqdm(as_completed(future_to_model), total=len(future_to_model), desc="Dataset test per model"):
                model_config = future_to_model[future]
                results[model_config["name"]] = future.result()
        self.mcp_pool.close()
        return results
