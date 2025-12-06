import logging, os, re
from concurrent.futures import ThreadPoolExecutor, as_completed
from langfuse import Langfuse
import pandas as pd
from pydantic import BaseModel, ValidationError
from src.dataset.loaders import DatasetLoader
from typing import Type, List
from tqdm import tqdm


""" CONFIG """
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
MAX_WORKERS = 12


""" FUNCTIONS """
def to_snake_case(name: str) -> str:
    """
    Convert a string to snake_case.
    :param name: Input string.
    """
    name = name.strip()
    name = re.sub(r"[^\w]+", "_", name)
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name)
    return name.lower()


""" CLASS """
class GenericDatasetLoader(DatasetLoader):
    """
    Generic loader that dynamically reads tabular datasets and maps them
    into instances of any Pydantic model.
    Parameters:
        :param input_path: The path to the dataset files.
        :param create_langfuse_dataset_bool: Boolean indicating whether to create a Langfuse dataset.
        :param dataset_name: The name of the Langfuse dataset.
    """


    def __init__(self,
                 input_path: str,
                 excel_files: list,
                 model_class: Type[BaseModel],
                 create_langfuse_dataset_bool: bool = False,
                 dataset_name: str = ""):
        super().__init__(input_path=input_path, create_langfuse_dataset_bool=create_langfuse_dataset_bool, dataset_name=dataset_name)
        self.langfuse = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
            host=os.getenv("LANGFUSE_HOST", ""),
        )
        self.dataframes: List[pd.DataFrame] = []
        self.excel_files = excel_files
        self.model_class = model_class
        self.load()


    def load(self):
        """
        Load all .xlsx files, convert column names to snake_case,
        and keep only model fields.
        """
        for file in self.excel_files:
            dataframe = pd.read_excel(os.path.join(self.input_path, file))

            # Converting all column names to snake_case automatically
            dataframe.columns = [to_snake_case(c) for c in dataframe.columns]

            # Keeping only fields that exist in the Pydantic model
            model_fields = set(self.model_class.model_fields.keys())
            dataframe = dataframe[[col for col in dataframe.columns if col in model_fields]]
            logger.info(f"\tLoaded file {file}, columns: {list(dataframe.columns)}")
            self.dataframes.append(dataframe)


    def prepare_data(self) -> List[BaseModel]:
        """
        Convert each row of each DataFrame into a Pydantic model instance.
        Return:
            :return List[BaseModel]: List of Pydantic model instances.
        """
        items: List[BaseModel] = []
        model_fields = set(self.model_class.model_fields.keys())
        for dataframe in self.dataframes:
            for _, row in dataframe.iterrows():

                # Selecting only fields that exist in model
                row_dict = {field: None if pd.isna(row[field]) else str(row[field])
                    for field in model_fields
                    if field in row
                }
                try:
                    item = self.model_class(**row_dict)
                    items.append(item)
                except ValidationError as e:
                    logger.error(f"\tRow validation error: {e}")
        if self.create_langfuse_dataset_bool:
            self.create_langfuse_dataset(items)

        return items


    def create_langfuse_dataset(self, items: List[BaseModel]):
        """
        Create a Langfuse dataset by sending each Pydantic model instance.
        Uses only fields from the Pydantic model_class.
        Automatically excludes fields that are None, empty strings,
        empty dicts or empty lists.
        Parameters:
            :param items: List of Pydantic model instances to upload.
        """
        logger.info(f"\tCreating Langfuse dataset '{self.dataset_name}'...")

        # Ensuring dataset existence
        try:
            self.langfuse.get_dataset(self.dataset_name)
            logger.info(f"\tDataset '{self.dataset_name}' already exists.")
        except Exception:
            logger.info(f"\tDataset '{self.dataset_name}' not found. Creating new one...")
            self.langfuse.create_dataset(name=self.dataset_name)

        # Processing one item
        def process_item(item: BaseModel):
            raw = item.model_dump()

            # Removing None, empty strings, empty lists, empty dicts
            cleaned = {key: value for key, value in raw.items() if value not in (None, "", [], {})}

            # Splitting fields by role
            input_data = {}
            expected_output_data = {}
            metadata_data = {}
            for field_name, field_def in item.model_fields.items():
                value = cleaned.get(field_name)
                if value is None:
                    continue
                field = "input"
                if field_def.json_schema_extra and "langfuse_dataset" in field_def.json_schema_extra:
                    field = field_def.json_schema_extra["langfuse_dataset"]
                if field == "input":
                    input_data[field_name] = value
                elif field == "expected_output":
                    expected_output_data[field_name] = value
                elif field == "metadata":
                    metadata_data[field_name] = value

            # Creating dataset item in Langfuse
            self.langfuse.create_dataset_item(
                dataset_name=self.dataset_name,
                input=input_data,
                expected_output=expected_output_data,
                metadata=metadata_data,
            )

        # Main process
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(process_item, item) for item in items]
            for _ in tqdm(as_completed(futures), total=len(futures), desc="\tUploading Langfuse dataset items"):
                pass  # just progress bar
        logger.info("\tDataset upload completed successfully.")
