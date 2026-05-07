# License Apache 2.0: (c) 2026 Athena-Reply

import csv
import json
from pathlib import Path

from src.datasets.dataset import Dataset


class LocalDataset(Dataset):
    """Streaming dataset backed by a local file (JSONL, CSV, or Parquet).

    Format is auto-detected from `path`'s extension unless `format` is set
    explicitly. Each row is a `dict` keyed by column name and is exposed as
    Jinja2 variables — same convention as `HuggingFaceDataset`.

    Example:

    ```python
    ds = LocalDataset(
        path="data/eval.jsonl",
        input_template='{"messages":[{"role":"user","content": {{ question | tojson }}}]}',
        output_template='{"role":"assistant","content": {{ answer | tojson }}}',
        batch_size=8,
    )
    program.evaluate(x=ds)
    ```

    Args:
        path (str): The filesystem path to the data file.
        format (str): Optional. Override the auto-detected format. One of
            `"jsonl"`, `"csv"`, `"parquet"`. Defaults to `None` (detect
            from extension).
        input_data_model (DataModel): See `Dataset`. Defaults to
            `synalinks.ChatMessages`.
        input_template (str): See `Dataset`.
        output_data_model (DataModel): See `Dataset`. Defaults to
            `synalinks.ChatMessage`.
        output_template (str): See `Dataset`.
        batch_size (int): Examples per yielded batch. Defaults to `1`.
        limit (int): Optional. See `Dataset`. Caps how many rows are
            consumed.
    """

    _EXT_FORMATS = {
        ".jsonl": "jsonl",
        ".ndjson": "jsonl",
        ".csv": "csv",
        ".tsv": "csv",
        ".parquet": "parquet",
    }

    def __init__(
        self,
        path,
        *,
        format=None,
        input_data_model=None,
        input_schema=None,
        input_template=None,
        output_data_model=None,
        output_schema=None,
        output_template=None,
        batch_size=1,
        limit: int = None,
        repeat: int = 1,
    ):
        super().__init__(
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
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)

        self.format = format or self._EXT_FORMATS.get(self.path.suffix.lower())
        if self.format not in {"jsonl", "csv", "parquet"}:
            raise ValueError(
                f"Unsupported format for {self.path.name!r}. "
                f"Pass `format=` explicitly (jsonl / csv / parquet)."
            )

    def _iter_rows(self):
        if self.format == "jsonl":
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
        elif self.format == "csv":
            delimiter = "\t" if self.path.suffix.lower() == ".tsv" else ","
            with self.path.open("r", encoding="utf-8", newline="") as f:
                yield from csv.DictReader(f, delimiter=delimiter)
        elif self.format == "parquet":
            # Stream in row groups so the whole file doesn't have to fit in RAM.
            import pyarrow.parquet as pq

            pf = pq.ParquetFile(self.path)
            for batch in pf.iter_batches():
                for row in batch.to_pylist():
                    yield row

    def __len__(self):
        if self.limit is not None:
            n = self.limit
        elif self.format == "parquet":
            # Parquet metadata stores the row count without reading data.
            import pyarrow.parquet as pq

            n = pq.ParquetFile(self.path).metadata.num_rows
        else:
            # jsonl / csv would need a full file scan to count rows; treat
            # length as unknown unless the caller pins it via `limit`.
            raise NotImplementedError(
                f"Length of {self.format!r} files is unknown without a full "
                "scan; set `limit` if you need a bounded epoch."
            )
        return self._total_batches(n)
