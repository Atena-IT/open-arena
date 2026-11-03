# License Apache 2.0: (c) 2026 Atena-Reply

import os

import synalinks

from openarena.datasets.dataset import Dataset
from openarena.datasets.platform import listing

# Where the arena reads datasets from: not the server evaluation runs are traced to (`MLFLOW_TRACKING_URI`).
SOURCE_URI = "MLFLOW_SOURCE_TRACKING_URI"


class MLflowDataset(Dataset, synalinks.datasets.MLflowDataset):
    """An MLflow evaluation dataset.

    The library's own loader, behind the engine's door, read from the
    arena's MLflow *source* server: `MLFLOW_SOURCE_TRACKING_URI` unless
    `tracking_uri` says. Fields, templates and records are the library's.

    Args:
        name (str): The dataset's name; or give `dataset_id`.
        dataset_id (str): Optional. The dataset's id, instead of its name.
        tracking_uri (str): Optional. The MLflow server to read from.
        input_data_model (DataModel): See `Dataset`.
        input_schema (dict | str): See `Dataset`.
        input_template (str): See `Dataset`.
        output_data_model (DataModel): See `Dataset`.
        output_schema (dict | str): See `Dataset`.
        output_template (str): See `Dataset`.
        tools_template (str): See `Dataset`.
        batch_size (int): Examples per yielded batch. Defaults to 1.
        limit (int): Optional. See `Dataset`.
        repeat (int): See `Dataset`.
    """

    def __init__(
        self,
        name=None,
        *,
        dataset_id=None,
        tracking_uri=None,
        input_data_model=None,
        input_schema=None,
        input_template=None,
        output_data_model=None,
        output_schema=None,
        output_template=None,
        tools_template=None,
        batch_size=1,
        limit=None,
        repeat=1,
    ):
        super().__init__(
            input_data_model=input_data_model,
            input_schema=input_schema,
            input_template=input_template,
            output_data_model=output_data_model,
            output_schema=output_schema,
            output_template=output_template,
            tools_template=tools_template,
            batch_size=batch_size,
            limit=limit,
            repeat=repeat,
            name=name,
            dataset_id=dataset_id,
            tracking_uri=tracking_uri or os.environ.get(SOURCE_URI) or None,
        )

    @classmethod
    def list_datasets(cls, query="", limit=100):
        """The datasets on the source server whose name holds `query`; see `PlatformDataset.list_datasets`."""
        from mlflow import MlflowClient

        # A client of its own: `mlflow.set_tracking_uri` would also move where the runs are traced. Same fallback as the reader.
        client = MlflowClient(os.environ.get(SOURCE_URI) or synalinks.config.mlflow_tracking_uri())
        # MLflow's LIKE is case-sensitive and, with no filter, keeps the last week only: every name, filtered here.
        def found():
            token = None
            while True:
                page = client.search_datasets(filter_string="name LIKE '%'", max_results=100, page_token=token)
                yield from page
                token = page.token
                if not token:
                    return

        return listing(((d.name, None, None) for d in found()), query, limit, setting="name")
