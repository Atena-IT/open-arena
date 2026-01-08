import logging
from dotenv import load_dotenv

from src.datasets.loaders import DatasetLoader, LangfuseLoader
from src.datasets.readers import ExcelReader, CsvReader
from src.datasets.types import DatasetConfig
from src.datasets.item_models import QAItem
from src.execution import GenericExecutor
from src.llms import LLMClient, MCPServerConfig

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

_logger = logging.getLogger(__name__)

load_dotenv()

def main():
    
    ds_config: DatasetConfig = {
        "dataset_name": "QA Dataset",
        "source_file": "QA.xlsx"
    }
    
    _logger.info(f"Loading dataset: {ds_config["dataset_name"]}")
    
    loader = DatasetLoader(
        item_model=QAItem,
        reader=ExcelReader(),
        config=ds_config,
        input_path="./resources/data"
    )
    
    dataset = loader.load()
    _logger.info(f"✓ Loaded {len(dataset)} items from {loader.source_file}")

    # === Alternative 1: load from CSV file ===
    #
    # csv_ds_config: DatasetConfig = {
    #     "dataset_name": "QA Dataset",
    #     "source_file": "QA.csv"
    # }
    #
    # csv_loader = DatasetLoader(
    #     item_model=QAItem,
    #     reader=CsvReader(),
    #     config=ds_config,
    #     input_path="./resources/data"
    # )
    
    # csv_dataset = loader.load()
    # _logger.info(f"✓ Loaded {len(dataset)} items from {loader.source_file}")

    # === Alternative 2: load and upload to Langfuse ===
    #
    # lf_loader = LangfuseLoader(
    #     item_model=QAItem,
    #     reader=ExcelReader(),
    #     config=ds_config,
    #     input_path="./resources/data"
    # )
    
    # lf_dataset = lf_loader.load()
    # lf_dataset_meta = lf_loader.upload()

    # _logger.info(f"✓ Loaded {len(dataset)} items from {loader.source_file} to Langfuse")

    client = LLMClient()
    
    model_config = {
        "name": "gpt-4",
        "max_tokens": 500,
        "temperature": 0.7,
        "stream": False,
    }

    mcp_servers: list[MCPServerConfig] = [
        {
            "server_name": "demo",
            "url": "https://demo-day.mcp.cloudflare.com/sse",
        }
    ]
    
    executor = GenericExecutor(
        dataset=dataset,
        llm_client=client,
        system_prompt="Always call the demo-day MCP tools.",
        model_config=model_config,
        mcp_servers=mcp_servers
    )
    
    results = executor.execute()

    _logger.info(f"✓ Executed {len(results)} items with LLM and MCP tools")
    
    return results


if __name__ == "__main__":
    results = main()
    print(f"\n✓ Complete: {len(results)} items processed")
