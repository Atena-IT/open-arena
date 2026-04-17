from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

from jinja2 import Environment, StrictUndefined, meta
from jinja2 import nodes as jinja_nodes

_env = Environment(undefined=StrictUndefined, autoescape=False)

# Canonical dataset row produced by adapters + rendered by Dataset.__iter__.
# Used by the upload path, executor, and evaluator.
Row = tuple[str, str, dict[str, Any]]  # (input, expected_output, metadata)


def _referenced(template_source: str) -> set[str]:
    """Return all column names consumed by a template.

    Handles three access patterns:
      - {{ column }}          → direct variable reference
      - {{ row.column }}      → attribute access on row
      - {{ row["column"] }}   → subscript access on row
    """
    ast = _env.parse(template_source)
    refs = meta.find_undeclared_variables(ast)

    for node in ast.find_all((jinja_nodes.Getattr, jinja_nodes.Getitem)):
        if isinstance(node.node, jinja_nodes.Name) and node.node.name == "row":
            if isinstance(node, jinja_nodes.Getattr):
                refs.add(node.attr)
            elif isinstance(node, jinja_nodes.Getitem) and isinstance(node.arg, jinja_nodes.Const):
                refs.add(node.arg.value)

    refs.discard("row")
    return refs


class Dataset(ABC):
    """
    Base class for dataset adapters.

    Subclasses fetch raw rows from a provider (local, huggingface, ...) via
    `iter_raw_rows()`. This base class renders each row into
    `(input, expected_output, metadata)` using the Jinja2 templates from the
    config. Column names are never normalized — they flow through exactly as
    provided by the source adapter.

    Templates can reference columns in three ways:
      - ``{{ column }}``         — direct variable access
      - ``{{ row.column }}``     — attribute access on the row dict
      - ``{{ row["column"] }}``  — subscript access (required for headers
        with spaces or non-identifier characters)

    All three patterns are detected by the metadata filter: columns consumed
    by either template are excluded from metadata, unreferenced columns flow
    through.
    """

    def __init__(
        self,
        name: str,
        input_template: str,
        expected_output_template: str,
        limit: int | None = None,
    ):
        self.name = name
        self._input_tpl = _env.from_string(input_template)
        self._expected_output_tpl = _env.from_string(expected_output_template)
        self._referenced = _referenced(input_template) | _referenced(expected_output_template)
        self.limit = limit

    @abstractmethod
    def iter_raw_rows(self) -> Iterator[dict[str, Any]]:
        raise NotImplementedError

    def __iter__(self) -> Iterator[tuple[str, str, dict[str, Any]]]:
        for i, row in enumerate(self.iter_raw_rows()):
            if self.limit is not None and i >= self.limit:
                break
            metadata = {k: v for k, v in row.items() if k not in self._referenced}
            yield (
                self._input_tpl.render(row=row, **row),
                self._expected_output_tpl.render(row=row, **row),
                metadata,
            )
