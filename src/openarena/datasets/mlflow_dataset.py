# License Apache 2.0: (c) 2026 Atena-Reply

import os

import synalinks


class MLflowDataset(synalinks.datasets.MLflowDataset):
    """An MLflow evaluation dataset.

    Read from the arena's MLflow *source* server.

    The arena keeps two MLflow servers apart: `MLFLOW_TRACKING_URI`, where
    evaluation runs are traced, and `MLFLOW_SOURCE_TRACKING_URI`, where
    datasets (and traces to judge) are read from. The engine's loader only
    knows the former, so this subclass defaults `tracking_uri` to the latter;
    an explicit `tracking_uri` on the dataset still wins. Everything else —
    fields, templates, records — is the library's.
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
        batch_size=1,
        limit=None,
        repeat=1,
    ):
        super().__init__(
            name=name,
            dataset_id=dataset_id,
            tracking_uri=tracking_uri or os.environ.get("MLFLOW_SOURCE_TRACKING_URI") or None,
            input_data_model=input_data_model,
            input_schema=input_schema,
            input_template=input_template,
            output_data_model=output_data_model,
            output_schema=output_schema,
            output_template=output_template,
            batch_size=batch_size,
            limit=limit,
            repeat=repeat,
        )
