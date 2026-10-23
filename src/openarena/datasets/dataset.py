# License Apache 2.0: (c) 2026 Atena-Reply

import json as _json

import synalinks
from synalinks.src.datasets.dataset import TOOLS_FIELD  # noqa: F401 — the request's `tools` field, re-exported for the tools program


class Dataset(synalinks.datasets.Dataset):
    """The engine's streaming `Dataset`, with every hand-written schema closed at the door.

    Everything is the base class': `_iter_rows()` to implement, the three
    Jinja2 templates over a row (`input_template`, `output_template` and
    `tools_template`, the row's own tool declarations as a Chat Completions
    request carries them, with `row_schema` the row's shape without them),
    `limit`, `repeat`, `batch_size`, the `(x, y)` batches. The difference is
    `_close_objects`: an `input_schema` or `output_schema` written by hand
    is marked closed, object by object, before the base class reads it.
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
        super().__init__(
            input_data_model=input_data_model,
            input_schema=_coerce_schema(input_schema),
            input_template=input_template,
            output_data_model=output_data_model,
            output_schema=_coerce_schema(output_schema),
            output_template=output_template,
            tools_template=tools_template,
            batch_size=batch_size,
            limit=limit,
            repeat=repeat,
            **kwargs,
        )


def _coerce_schema(schema):
    """Accept a JSON Schema as either a dict or a JSON-encoded string, its objects closed."""
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
