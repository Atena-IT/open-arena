# License Apache 2.0: (c) 2026 Athena-Reply

import json
from pathlib import Path

import yaml

from src.datasets.dataset import Dataset


class FolderDataset(Dataset):
    """Streaming dataset where each file in a folder is one record.

    Files are read in deterministic alphabetical order and exposed to the
    Jinja2 templates as a flat dict. The schema depends on the file's
    format:

      - `.json` / `.yaml` / `.yml` files are parsed: if the parsed value
        is a `dict`, its keys are spread at the top level; otherwise it is
        exposed as `record`.
      - `.txt` / `.md` (and any other extension when `format="text"`) are
        read as UTF-8 strings and exposed as `text`.

    Every record additionally exposes filesystem metadata as `_filename`
    (full filename), `_stem` (filename without extension), and `_path`
    (path relative to the dataset root, posix-style).

    Example:

    ```python
    ds = FolderDataset(
        path="data/cases",
        pattern="*.json",
        input_template='{"messages":[{"role":"user","content":{{ question | tojson }}}]}',
        output_template='{"role":"assistant","content":{{ answer | tojson }}}',
        batch_size=4,
    )
    program.evaluate(x=ds)
    ```

    Args:
        path (str): The folder containing one file per record.
        pattern (str): Glob pattern selecting which files inside `path`
            count as records. Defaults to `"*"` (everything at the top
            level). Combine with `recursive=True` for nested folders.
        recursive (bool): If `True`, walk subfolders too (uses `rglob`
            with `pattern`). Defaults to `False`.
        format (str): Optional. Override the auto-detected per-file
            format. One of `"json"`, `"yaml"`, `"text"`. Defaults to
            `None` (detect from each file's extension).
        encoding (str): Text encoding for file reads. Defaults to
            `"utf-8"`.
        input_data_model (DataModel): See `Dataset`. Defaults to
            `synalinks.ChatMessages`.
        input_template (str): See `Dataset`.
        output_data_model (DataModel): See `Dataset`. Defaults to
            `synalinks.ChatMessage`.
        output_template (str): See `Dataset`.
        batch_size (int): Examples per yielded batch. Defaults to `1`.
        limit (int): Optional. See `Dataset`. Caps how many files are
            consumed.
    """

    _EXT_FORMATS = {
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".txt": "text",
        ".md": "text",
    }

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
        if not self.path.is_dir():
            raise NotADirectoryError(self.path)

        self.pattern = pattern
        self.recursive = recursive
        self.format = format
        self.encoding = encoding

        if format is not None and format not in {"json", "yaml", "text"}:
            raise ValueError(
                f"Unsupported `format`={format!r}. Use 'json', 'yaml', or 'text'."
            )

    def _list_files(self):
        glob = self.path.rglob if self.recursive else self.path.glob
        files = sorted(p for p in glob(self.pattern) if p.is_file())
        return files

    def _load_file(self, file: Path):
        fmt = self.format or self._EXT_FORMATS.get(file.suffix.lower())
        if fmt is None:
            raise ValueError(
                f"Cannot infer format for {file.name!r}. Pass `format=` "
                f"explicitly (json / yaml / text)."
            )

        text = file.read_text(encoding=self.encoding)
        if fmt == "json":
            data = json.loads(text)
        elif fmt == "yaml":
            data = yaml.safe_load(text)
        else:  # text
            data = text

        row = {
            "_filename": file.name,
            "_stem": file.stem,
            "_path": file.relative_to(self.path).as_posix(),
        }
        if isinstance(data, dict):
            row.update(data)
        elif fmt == "text":
            row["text"] = data
        else:
            row["record"] = data
        return row

    def _iter_rows(self):
        for file in self._list_files():
            yield self._load_file(file)

    def __len__(self):
        n = self.limit if self.limit is not None else len(self._list_files())
        return self._total_batches(n)
