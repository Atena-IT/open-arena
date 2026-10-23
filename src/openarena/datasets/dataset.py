# License Apache 2.0: (c) 2026 Atena-Reply

import json as _json

import jinja2
import numpy as np
import synalinks
from synalinks import JsonDataModel

from openarena.config.entries import TOOL_SHAPE, tool_declaration_problem

# The input field a row's tool declarations ride in, from `tools_template` to
# the tools program: `tools`, as a Chat Completions request carries them
# beside its `messages`. The program sends them as the request's tools, not
# as prompt text.
TOOLS_FIELD = "tools"


class Dataset:
    """Base class for streaming datasets compatible with the framework's trainers.

    The framework's `Trainer.fit/evaluate/predict(x=...)` accepts a Python
    generator that yields `(inputs,)` or `(inputs, targets)` tuples — one
    tuple per batch. See
    the framework's generator data adapter and
    its data-adapter dispatch.

    Subclasses implement `_iter_rows()` as a generator yielding raw row
    dicts (one per source example). The base class' `__iter__` then
    renders each row through the Jinja2 templates, validates the shape,
    and yields batched `(x, y)` numpy object arrays — including the
    `repeat` expansion. Calling the dataset returns a fresh
    `types.GeneratorType` suitable for the framework:

    ```python
    program.evaluate(x=my_dataset())
    ```

    The shape of the per-row input/target objects is controlled by either
    a Python `DataModel` class (`input_data_model` / `output_data_model`)
    OR a raw JSON Schema (`input_schema` / `output_schema`). The two are
    mutually exclusive on each side. With a class, rows are validated via
    `cls.model_validate_json(rendered)`. With a schema, rows are wrapped
    as `JsonDataModel(schema=..., json=json.loads(rendered))` — the schema
    flows directly into the LM as a structured-output constraint, so any
    JSON Schema feature (enum, const, oneOf, ...) is supported.

    Args:
        input_data_model (DataModel): Python class describing batch inputs.
            Defaults to `ChatMessages` in subclasses when neither
            this nor `input_schema` is provided.
        input_schema (dict | str): Raw JSON Schema for batch inputs. May be
            given as a dict or as a JSON-encoded string. Mutually exclusive
            with `input_data_model`.
        input_template (str): Jinja2 template string used to render raw
            rows into the input shape. Required.
        output_data_model (DataModel): Python class describing batch targets.
            Defaults to `ChatMessage` when `output_template` is
            given but neither this nor `output_schema` is. Must be omitted
            when `output_template` is omitted.
        output_schema (dict | str): Raw JSON Schema for batch targets.
            Mutually exclusive with `output_data_model`. Must be omitted
            when `output_template` is omitted.
        output_template (str): Jinja2 template string used to render raw
            rows into the target shape. Optional — when omitted, the dataset
            is inputs-only and yields single-element `(x,)` batches (no
            targets). Rewards that need `y_true` will see it missing.
        tools_template (str): Jinja2 template string rendering the row's
            tool declarations: a JSON list of `{"type": "function",
            "function": {...}}`, the shape a Chat Completions request
            carries. Optional — when given, each input is the request's
            shape, the row's fields (`messages` by default) and a `tools`
            field beside them, which a `tools:` dataset sends the model as
            the tools it may call on that row.
        batch_size (int): Number of examples per yielded batch. `None` lets
            the subclass decide (or yield the whole dataset as one batch).
        limit (int): Optional. Maximum number of *raw* (pre-repeat) examples
            to iterate over. `None` (default) means no cap. Useful for
            capping huge or streaming sources for quick experiments / smoke
            tests.
        repeat (int): Number of consecutive copies to emit per raw example.
            Defaults to 1 (no expansion). Setting `repeat == batch_size`
            makes every batch a group of N rollouts of the same prompt —
            the expected layout for GRPO-style RL where reward statistics
            are computed across rollouts of one input.
        **kwargs: Provider-specific fields forwarded by subclasses (e.g. HF
            dataset name, split, revision, API key, file path, ...).
    """

    def __init__(
        self,
        input_data_model=None,
        input_schema=None,
        input_template=None,
        output_data_model=None,
        output_schema=None,
        output_template=None,
        tools_template=None,
        batch_size=None,
        limit: int = None,
        repeat: int = 1,
        **kwargs,
    ):
        if input_template is None:
            raise ValueError("`input_template` is required (Jinja2 template).")
        if input_data_model is not None and input_schema is not None:
            raise ValueError(
                "Pass either `input_data_model` or `input_schema`, not both."
            )
        if output_data_model is not None and output_schema is not None:
            raise ValueError(
                "Pass either `output_data_model` or `output_schema`, not both."
            )
        if output_template is None and (
            output_data_model is not None or output_schema is not None
        ):
            raise ValueError(
                "`output_data_model` / `output_schema` require `output_template` "
                "(omit all three for an inputs-only dataset)."
            )
        if not isinstance(repeat, int) or repeat < 1:
            raise ValueError(f"`repeat` must be a positive int; got {repeat!r}.")
        # Default to ChatMessages / ChatMessage when neither a data_model nor
        # a schema is given — matches the historical per-subclass behavior.
        if input_data_model is None and input_schema is None:
            input_data_model = synalinks.ChatMessages
        if (
            output_template is not None
            and output_data_model is None
            and output_schema is None
        ):
            output_data_model = synalinks.ChatMessage
        self.input_data_model = input_data_model
        self.input_schema = _coerce_schema(input_schema)
        self.input_template = input_template
        self.output_data_model = output_data_model
        self.output_schema = _coerce_schema(output_schema)
        self.output_template = output_template
        self.tools_template = tools_template
        self.batch_size = batch_size
        self.limit = limit
        self.repeat = repeat

        env = jinja2.Environment(undefined=jinja2.StrictUndefined)
        self._input_tmpl = env.from_string(input_template)
        self._output_tmpl = (
            env.from_string(output_template) if output_template is not None else None
        )
        self._tools_tmpl = (
            env.from_string(tools_template) if tools_template is not None else None
        )
        # With a tools template the input is the request's shape: the row's
        # own fields (`messages` unless `input_schema` says otherwise) and
        # the `tools` beside them. The program reads the row back out with
        # `row_schema`, so the prompt is built from the row alone.
        self.row_schema = None
        if self._tools_tmpl is not None:
            self.row_schema = (
                self.input_schema
                if self.input_schema is not None
                else self.input_data_model.get_schema()
            )
            self.input_schema = _with_tools(self.row_schema)

    def _make_input(self, rendered: str, row: dict | None = None):
        if self._tools_tmpl is not None:
            fields = _json.loads(rendered)
            if TOOLS_FIELD in fields:
                raise ValueError(
                    f"`input_template` renders a `{TOOLS_FIELD}` field, which "
                    "`tools_template` fills: rename it."
                )
            tools = _json.loads(self._tools_tmpl.render(**row))
            if not isinstance(tools, list):
                raise ValueError(
                    "`tools_template` must render a JSON list of tool "
                    f"declarations; got {type(tools).__name__}."
                )
            for tool in tools:
                if (problem := tool_declaration_problem(tool)) is not None:
                    raise ValueError(
                        f"`tools_template` rendered a tool the API would refuse: {problem}. "
                        f"Each one is {TOOL_SHAPE}, as a Chat Completions request sends it."
                    )
            return JsonDataModel(
                schema=self.input_schema, json={**fields, TOOLS_FIELD: tools}
            )
        if self.input_schema is not None:
            return JsonDataModel(schema=self.input_schema, json=_json.loads(rendered))
        return self.input_data_model.model_validate_json(rendered)

    def _make_target(self, rendered: str):
        if self.output_schema is not None:
            return JsonDataModel(schema=self.output_schema, json=_json.loads(rendered))
        return self.output_data_model.model_validate_json(rendered)

    def _iter_rows(self):
        """Yield raw row dicts from the underlying source.

        Subclasses must implement this. Each yielded dict is passed as
        kwargs to the Jinja2 input/output templates, so its keys must be
        valid Python identifiers matching the template variables.
        """
        raise NotImplementedError

    def __iter__(self):
        """Render rows through the templates and yield batches.

        Yields `(x, y)` when an `output_template` is configured, or
        single-element `(x,)` batches when it isn't. Honors `limit` (caps
        raw rows), `repeat` (each raw example is emitted `repeat` times in
        a row), and `batch_size` (size of the yielded numpy object arrays;
        `None` accumulates everything into a single trailing batch). The
        trailing partial batch is always flushed at the end.
        """
        inputs_only = self._output_tmpl is None
        bs = self.batch_size
        x_buf, y_buf = [], []
        seen = 0
        for row in self._iter_rows():
            if self.limit is not None and seen >= self.limit:
                break
            seen += 1
            x = self._make_input(self._input_tmpl.render(**row), row)
            y = (
                None
                if inputs_only
                else self._make_target(self._output_tmpl.render(**row))
            )
            for _ in range(self.repeat):
                x_buf.append(x)
                if not inputs_only:
                    y_buf.append(y)
                if bs is not None and len(x_buf) >= bs:
                    yield _batch(x_buf, y_buf, inputs_only)
                    x_buf, y_buf = [], []
        if x_buf:
            yield _batch(x_buf, y_buf, inputs_only)

    def _total_batches(self, num_rows: int) -> int:
        """Number of batches given `num_rows` raw (pre-repeat) examples.

        `batch_size=None` collapses to a single batch.
        """
        n = num_rows * self.repeat
        if self.batch_size is None:
            return 1 if n > 0 else 0
        return (n + self.batch_size - 1) // self.batch_size

    def __call__(self):
        """Return a fresh generator over the dataset's batches."""
        return iter(self)

    def __len__(self):
        """Number of batches, if known. Override when the size is finite."""
        raise NotImplementedError


def _batch(x_buf, y_buf, inputs_only):
    x = np.array(x_buf, dtype="object")
    if inputs_only:
        return (x,)
    return (x, np.array(y_buf, dtype="object"))


def _with_tools(schema):
    """The row's `schema` with the `tools` of a Chat Completions request beside its fields."""
    out = dict(schema)
    out["properties"] = {
        **(schema.get("properties") or {}),
        TOOLS_FIELD: {
            "type": "array",
            "description": "The tools the model may call on this row, as a Chat Completions request declares them.",
            "items": synalinks.ChatCompletionTool.get_schema(),
        },
    }
    out["required"] = [*(schema.get("required") or []), TOOLS_FIELD]
    return out


def _coerce_schema(schema):
    """Accept a JSON Schema as either a dict or a JSON-encoded string."""
    if schema is None:
        return None
    if isinstance(schema, str):
        return _close_objects(_json.loads(schema))
    if isinstance(schema, dict):
        return _close_objects(schema)
    raise TypeError(
        f"`schema` must be a dict or JSON string; got {type(schema).__name__}."
    )


# Keywords whose value is itself a schema, or a list of them. Walked so a
# nested object is closed too — OpenAI checks every level, not just the root.
_SCHEMA_VALUES = ("items", "additionalItems", "contains", "not", "if", "then", "else")
_SCHEMA_LISTS = ("anyOf", "oneOf", "allOf", "prefixItems")
_SCHEMA_MAPS = ("properties", "$defs", "definitions", "patternProperties")


def _close_objects(schema):
    """Mark every object in `schema` closed, as strict structured output requires.

    A schema written by hand rarely says `additionalProperties: false`, and
    most providers do not care. OpenAI (and Azure, and the OpenAI-compatible
    providers the framework sends `strict: true` to) reject the request outright:

        Invalid schema for response_format 'structured_output':
        In context=(), 'additionalProperties' is required to be supplied
        and to be false.

    the framework turns that rejection into a `None` output rather than an error,
    so the model answers nothing on every row and the cell scores zero with
    zero tokens — which reads like a terrible model instead of a broken
    request. Closing the objects here, once, at the door of every dataset,
    keeps the same schema working on every provider: nobody minds a closed
    object, and the datasets never intended to allow extra keys anyway.

    An explicit `additionalProperties` is left alone, whatever it says.
    """
    if isinstance(schema, list):
        return [_close_objects(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    out = dict(schema)
    for key in _SCHEMA_MAPS:
        if isinstance(out.get(key), dict):
            out[key] = {k: _close_objects(v) for k, v in out[key].items()}
    for key in _SCHEMA_LISTS:
        if isinstance(out.get(key), list):
            out[key] = [_close_objects(v) for v in out[key]]
    for key in _SCHEMA_VALUES:
        if key in out:
            out[key] = _close_objects(out[key])
    if out.get("type") == "object" or "properties" in out:
        out.setdefault("additionalProperties", False)
    return out
