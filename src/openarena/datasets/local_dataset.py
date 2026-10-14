# License Apache 2.0: (c) 2026 Atena-Reply

import csv
import json
from pathlib import Path

from openarena.datasets.dataset import Dataset


class LocalDataset(Dataset):
    """Streaming dataset backed by a local file (JSON list, JSONL, CSV, or Parquet).

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
            `"json"` (a file holding a list of records, or an object with one
            list in it), `"jsonl"`, `"csv"`, `"parquet"`. Defaults to `None`
            (detect from extension).
        input_data_model (DataModel): See `Dataset`. Defaults to
            `ChatMessages`.
        input_template (str): See `Dataset`.
        output_data_model (DataModel): See `Dataset`. Defaults to
            `ChatMessage`.
        output_template (str): See `Dataset`.
        batch_size (int): Examples per yielded batch. Defaults to `1`.
        limit (int): Optional. See `Dataset`. Caps how many rows are
            consumed.
    """

    _EXT_FORMATS = {
        ".json": "json",
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
        if self.format not in {"json", "jsonl", "csv", "parquet"}:
            raise ValueError(
                f"Unsupported format for {self.path.name!r}. "
                f"Pass `format=` explicitly (json / jsonl / csv / parquet)."
            )

    def _json_records(self):
        """The records of a JSON file: a top-level list, or the one list an object holds (`{"data": [...]}`)."""
        with self.path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            lists = [v for v in data.values() if isinstance(v, list)]
            if len(lists) != 1:
                raise ValueError(
                    f"{self.path.name!r} must hold a list of records, or an object with exactly one list in it."
                )
            data = lists[0]
        if not isinstance(data, list) or not all(isinstance(r, dict) for r in data):
            raise ValueError(f"{self.path.name!r} must hold a list of objects, one per record.")
        return data

    def check(self) -> int:
        """Read the whole file and say what is wrong with it, before a run does; returns its row count.

        Raises `ValueError` naming the file and the first bad line (jsonl) or
        row (csv): a line that is not JSON, a line that is not an object, a
        row with more values than the header, no rows at all.
        """
        name = self.path.name
        rows = 0
        if self.format == "jsonl":
            with self.path.open("r", encoding="utf-8") as f:
                for number, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as e:
                        raise ValueError(f"{name}, line {number}: not valid JSON ({e.msg}).") from None
                    if not isinstance(row, dict):
                        raise ValueError(f"{name}, line {number}: each line must be one JSON object, with a field per column.")
                    rows += 1
        elif self.format == "csv":
            delimiter = "\t" if self.path.suffix.lower() == ".tsv" else ","
            with self.path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                if not reader.fieldnames:
                    raise ValueError(f"{name}: the first line must name the columns.")
                for row in reader:
                    if None in row:
                        raise ValueError(f"{name}, line {reader.line_num}: more values than the header has columns.")
                    rows += 1
        elif self.format == "json":
            rows = len(self._json_records())
        else:
            import pyarrow.parquet as pq

            rows = pq.ParquetFile(self.path).metadata.num_rows
        if rows == 0:
            raise ValueError(f"{name} has no rows.")
        return rows

    def _row_count(self) -> int:
        """Rows in a jsonl or csv file, counted once per version of the file (its size and time)."""
        stat = self.path.stat()
        stamp = (stat.st_mtime_ns, stat.st_size)
        if getattr(self, "_counted", (None, 0))[0] != stamp:
            if self.format == "jsonl":
                with self.path.open("r", encoding="utf-8") as f:
                    n = sum(1 for line in f if line.strip())
            else:
                delimiter = "\t" if self.path.suffix.lower() == ".tsv" else ","
                with self.path.open("r", encoding="utf-8", newline="") as f:
                    n = sum(1 for _ in csv.DictReader(f, delimiter=delimiter))
            self._counted = (stamp, n)
        return self._counted[1]

    def _iter_rows(self):
        if self.format == "json":
            yield from self._json_records()
        elif self.format == "jsonl":
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
        elif self.format == "json":
            n = len(self._json_records())
        else:
            # jsonl / csv hold no count: a scan, cheap for a local file and
            # needed, since the trainer asks every dataset its length.
            n = self._row_count()
        return self._total_batches(n)
