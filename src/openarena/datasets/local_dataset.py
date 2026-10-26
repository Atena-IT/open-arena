# License Apache 2.0: (c) 2026 Atena-Reply

"""A local file as a dataset: a JSON list, JSONL, CSV or Parquet, one reader per format.

Each format is a `Reader`: how its rows are streamed, how they are counted
without reading them (or by a scan, for the line formats), and how the file
is checked before a run, naming the first bad line. The dataset picks the
reader off the file's extension, or the `format` given.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import NamedTuple

from openarena.datasets.dataset import Dataset


class Reader(NamedTuple):
    rows: Callable[[Path], Iterator[dict]]
    count: Callable[[Path], int]
    check: Callable[[Path], int]  # the row count, or `ValueError` naming the file and the first bad line


# -- JSON: one list of records, at the top level or alone inside an object -------------


def _json_records(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        lists = [value for value in data.values() if isinstance(value, list)]
        if len(lists) != 1:
            raise ValueError(f"{path.name!r} must hold a list of records, or an object with exactly one list in it.")
        data = lists[0]
    if not isinstance(data, list) or not all(isinstance(record, dict) for record in data):
        raise ValueError(f"{path.name!r} must hold a list of objects, one per record.")
    return data


# -- JSONL: one object per line, blank lines skipped ----------------------------------


def _jsonl_lines(path: Path) -> Iterator[tuple[int, str]]:
    with path.open("r", encoding="utf-8") as lines:
        for number, line in enumerate(lines, start=1):
            if line.strip():
                yield number, line


def _jsonl_rows(path: Path) -> Iterator[dict]:
    for _, line in _jsonl_lines(path):
        yield json.loads(line)


def _jsonl_check(path: Path) -> int:
    rows = 0
    for number, line in _jsonl_lines(path):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path.name}, line {number}: not valid JSON ({e.msg}).") from None
        if not isinstance(row, dict):
            raise ValueError(f"{path.name}, line {number}: each line must be one JSON object, with a field per column.")
        rows += 1
    return rows


# -- CSV, TSV by extension: the header names the columns, every value a string ---------


def _csv_reader(path: Path):
    handle = path.open("r", encoding="utf-8", newline="")
    return handle, csv.DictReader(handle, delimiter="\t" if path.suffix.lower() == ".tsv" else ",")


def _csv_rows(path: Path) -> Iterator[dict]:
    handle, reader = _csv_reader(path)
    with handle:
        yield from reader


def _csv_check(path: Path) -> int:
    handle, reader = _csv_reader(path)
    with handle:
        if not reader.fieldnames:
            raise ValueError(f"{path.name}: the first line must name the columns.")
        rows = 0
        for row in reader:
            if None in row:
                raise ValueError(f"{path.name}, line {reader.line_num}: more values than the header has columns.")
            rows += 1
    return rows


# -- Parquet: streamed by row group, counted from the footer ----------------------------


def _parquet_rows(path: Path) -> Iterator[dict]:
    from pyarrow import parquet

    for batch in parquet.ParquetFile(path).iter_batches():
        yield from batch.to_pylist()


def _parquet_count(path: Path) -> int:
    from pyarrow import parquet

    return parquet.ParquetFile(path).metadata.num_rows


def _counted(rows: Callable[[Path], Iterator[dict]]) -> Callable[[Path], int]:
    return lambda path: sum(1 for _ in rows(path))


READERS = {
    "json": Reader(rows=_json_records, count=lambda path: len(_json_records(path)), check=lambda path: len(_json_records(path))),
    "jsonl": Reader(rows=_jsonl_rows, count=_counted(_jsonl_rows), check=_jsonl_check),
    "csv": Reader(rows=_csv_rows, count=_counted(_csv_rows), check=_csv_check),
    "parquet": Reader(rows=_parquet_rows, count=_parquet_count, check=_parquet_count),
}
FORMAT_OF_SUFFIX = {".json": "json", ".jsonl": "jsonl", ".ndjson": "jsonl", ".csv": "csv", ".tsv": "csv", ".parquet": "parquet"}


class LocalDataset(Dataset):
    """A local file: a JSON list, JSONL, CSV or Parquet.

    The format is read off the file's extension (`.json`, `.jsonl`,
    `.ndjson`, `.csv`, `.tsv`, `.parquet`) unless `format` says. Each row
    is handed to the templates as a mapping of its columns; a CSV's values
    are all strings.

    Args:
        path (str): The file.
        format (str): Optional. `json` (a list of records, or an object
            holding one list), `jsonl`, `csv` or `parquet`, for a file
            whose extension says otherwise.
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
        path,
        *,
        format=None,
        input_data_model=None,
        input_schema=None,
        input_template=None,
        output_data_model=None,
        output_schema=None,
        output_template=None,
        tools_template=None,
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
            tools_template=tools_template,
            batch_size=batch_size,
            limit=limit,
            repeat=repeat,
        )
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.format = format or FORMAT_OF_SUFFIX.get(self.path.suffix.lower())
        if self.format not in READERS:
            raise ValueError(f"Unsupported format for {self.path.name!r}. Pass `format=` explicitly (json / jsonl / csv / parquet).")
        self.reader = READERS[self.format]
        self._counted: tuple[tuple[int, int], int] | None = None  # the row count, by the file's (mtime, size)

    def _iter_rows(self):
        yield from self.reader.rows(self.path)

    def check(self) -> int:
        """Read the whole file and say what is wrong with it, before a run does; returns its row count."""
        rows = self.reader.check(self.path)
        if rows == 0:
            raise ValueError(f"{self.path.name} has no rows.")
        return rows

    def _row_count(self) -> int:
        """The file's rows, counted once per version of the file: the trainer asks every dataset its length."""
        stat = self.path.stat()
        stamp = (stat.st_mtime_ns, stat.st_size)
        if self._counted is None or self._counted[0] != stamp:
            self._counted = (stamp, self.reader.count(self.path))
        return self._counted[1]

    def __len__(self):
        return self._total_batches(self.limit if self.limit is not None else self._row_count())
