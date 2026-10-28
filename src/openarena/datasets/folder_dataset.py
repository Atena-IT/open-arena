# License Apache 2.0: (c) 2026 Atena-Reply

"""A folder as a dataset: one record per file, each file read by its format."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from openarena.datasets.dataset import Dataset

# How a file's text becomes its record: parsed, or kept as text.
PARSERS = {"json": json.loads, "yaml": yaml.safe_load, "text": None}
FORMAT_OF_SUFFIX = {".json": "json", ".yaml": "yaml", ".yml": "yaml", ".txt": "text", ".md": "text"}


class FolderDataset(Dataset):
    """A folder where each file is one record.

    Files are read in name order. A `.json`, `.yaml` or `.yml` file is
    parsed: a mapping's keys are the record's own fields, any other value
    is handed over whole as `record`. A `.txt` or `.md` file, and any file
    when `format` is `text`, is handed over as `text`. Every record also
    carries `_filename`, `_stem` and `_path` (relative to the folder,
    posix-style).

    Args:
        path (str): The folder.
        pattern (str): The glob the files must match. Defaults to `*`, every
            file at the top level.
        recursive (bool): Walk the subfolders too. Defaults to False.
        format (str): Optional. `json`, `yaml` or `text`, for every file
            whatever its extension.
        encoding (str): The files' text encoding. Defaults to `utf-8`.
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
        pattern="*",
        recursive=False,
        format=None,
        encoding="utf-8",
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
        if not self.path.is_dir():
            raise NotADirectoryError(self.path)
        if format is not None and format not in PARSERS:
            raise ValueError(f"Unsupported `format`={format!r}. Use 'json', 'yaml', or 'text'.")
        self.pattern = pattern
        self.recursive = recursive
        self.format = format
        self.encoding = encoding
        self._media_dir(self.path)

    def files(self) -> list[Path]:
        """The files that are records, in name order."""
        found = self.path.rglob(self.pattern) if self.recursive else self.path.glob(self.pattern)
        return sorted(file for file in found if file.is_file())

    def record(self, file: Path) -> dict:
        """The record a file holds, with where it came from."""
        format = self.format or FORMAT_OF_SUFFIX.get(file.suffix.lower())
        if format is None:
            raise ValueError(f"Cannot infer format for {file.name!r}. Pass `format=` explicitly (json / yaml / text).")
        origin = {"_filename": file.name, "_stem": file.stem, "_path": file.relative_to(self.path).as_posix()}
        text = file.read_text(encoding=self.encoding)
        if PARSERS[format] is None:
            return {**origin, "text": text}
        value = PARSERS[format](text)
        return {**origin, **value} if isinstance(value, dict) else {**origin, "record": value}

    def _iter_rows(self):
        for file in self.files():
            yield self.record(file)

    def __len__(self):
        return self._total_batches(self.limit if self.limit is not None else len(self.files()))
