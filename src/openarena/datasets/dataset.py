# License Apache 2.0: (c) 2026 Atena-Reply

import ast
import base64
import json as _json
import mimetypes
import os
import re

import jinja2
import synalinks
from synalinks.src.datasets.dataset import TOOLS_FIELD  # noqa: F401 — the request's `tools` field, re-exported for the tools program

# Where a relative image or audio path in a row is read from: the folder of a dataset that has one (`FolderDataset`).
MEDIA_DIR = "__media_dir__"
# The first bytes of the image formats providers take, for an image that comes without a file name.
_IMAGE_MAGIC = ((b"\x89PNG", "image/png"), (b"\xff\xd8\xff", "image/jpeg"), (b"GIF8", "image/gif"), (b"RIFF", "image/webp"))
_URL = re.compile(r"(https?|file|data):")


def _media(context, value) -> tuple[bytes | None, str]:
    """An image or audio value as `(bytes, file name)` when it holds the file, Hugging Face's undecoded
    `{bytes, path}`; else `(None, where it is)`, a URL, or a path made absolute against the dataset's folder."""
    if isinstance(value, dict):
        if value.get("bytes"):
            return value["bytes"], value.get("path") or ""
        value = value.get("path") or value.get("src")
    if not isinstance(value, str) or not value:
        raise ValueError(f"no image or audio in {value!r}")
    if _URL.match(value):
        return None, value
    return None, os.path.abspath(os.path.join(context.get(MEDIA_DIR) or "", value))


@jinja2.pass_context
def as_image(context, value) -> dict:
    """An image column's value as a chat message's `image_url` part: the file's bytes inlined, a URL or a file
    left for synalinks to read, one batch at a time."""
    raw, where = _media(context, value)
    if raw is not None:
        mime = mimetypes.guess_type(where)[0] or next((m for magic, m in _IMAGE_MAGIC if raw.startswith(magic)), "image/png")
        where = f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"
    elif not _URL.match(where):
        where = f"file://{where}"
    return {"type": "image_url", "image_url": {"url": where}}


def _audio_format(raw: bytes) -> str:
    """The format of audio bytes from their header: `flac`, `mp3`, `ogg`, else `wav`."""
    if raw.startswith(b"fLaC"):
        return "flac"
    if raw.startswith(b"ID3") or raw[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return "mp3"
    if raw.startswith(b"OggS"):
        return "ogg"
    return "wav"


@jinja2.pass_context
def as_audio(context, value) -> dict:
    """An audio column's value as an `input_audio` part in its file's `format` (`wav`, `mp3`, `flac`): the
    file's bytes inlined, a URL or a file left for synalinks to read, one batch at a time."""
    raw, where = _media(context, value)
    format = os.path.splitext(where.split("?")[0])[1].lstrip(".").lower() or (_audio_format(raw) if raw else "wav")
    if raw is not None:
        return {"type": "input_audio", "input_audio": {"data": base64.b64encode(raw).decode("ascii"), "format": format}}
    return {"type": "input_audio", "input_audio": {"url" if _URL.match(where) else "path": where, "format": format}}


# The templates' Jinja environment: an unknown variable is an error, as in synalinks. `fromjson` reads a column
# that holds JSON as text, the way many tool-calling datasets ship their tools and calls; `fromliteral` one
# written as Python writes a list (`"['$6', '$7']"`); `as_image` and `as_audio` make a column a message's part.
JINJA = jinja2.Environment(undefined=jinja2.StrictUndefined)
JINJA.filters["fromjson"] = _json.loads
JINJA.filters["fromliteral"] = ast.literal_eval
JINJA.filters["as_image"] = as_image
JINJA.filters["as_audio"] = as_audio


class RowTemplate:
    """A template that also sees the whole row as `row`: `row["Question Text"]` reads a column no identifier can name.

    A column called `row` keeps its own value.
    """

    def __init__(self, source: str):
        self.template = JINJA.from_string(source)

    def render(self, **row) -> str:
        return self.template.render({"row": row, **row})


class ToolsTemplate(RowTemplate):
    """The row's tools template, what it renders handed on in the wire shape (`tool_declarations`)."""

    def render(self, **row) -> str:
        return _json.dumps(tool_declarations(_json.loads(super().render(**row))))


# A parameter's type as Python names it, the way some datasets write it (`str`, `List[int]`, `int, optional`), in
# JSON Schema's words.
JSON_TYPES = {
    "str": "string", "string": "string",
    "int": "integer", "integer": "integer",
    "float": "number", "number": "number",
    "bool": "boolean", "boolean": "boolean",
    "list": "array", "tuple": "array", "array": "array",
    "dict": "object", "object": "object",
    "none": "null", "null": "null",
}


def tool_declarations(tools) -> list:
    """`tools` as a Chat Completions request declares them, from the shapes datasets ship them in.

    One declaration or a list of them. A bare `{name, description, parameters}` is wrapped as
    `{"type": "function", "function": ...}`, and parameters written as a mapping of fields made an object
    schema, a field required unless it has a `default` or its type says optional. Each field's type is put in
    JSON Schema's words, or left out when it names none. Anything else is handed on as it is, for the check
    that follows to name.
    """
    declared = []
    for tool in tools if isinstance(tools, list) else [tools]:
        if isinstance(tool, dict) and isinstance(tool.get("name"), str):
            # A flat declaration (`{"type": "function", "name": ...}`, as the Responses API writes it) keeps no `type` inside.
            tool = {"type": "function", "function": {key: value for key, value in tool.items() if key != "type"}}
        function = tool.get("function") if isinstance(tool, dict) else None
        if isinstance(function, dict) and isinstance(function.get("parameters"), dict):
            tool = {**tool, "function": {**function, "parameters": _parameters(function["parameters"])}}
        declared.append(tool)
    return declared


def _parameters(parameters: dict) -> dict:
    # A schema says its `type` in words; a mapping of fields may hold a field named `type`, itself a mapping.
    schema = isinstance(parameters.get("type"), (str, list)) or isinstance(parameters.get("properties"), dict)
    if not schema and parameters and all(isinstance(field, dict) for field in parameters.values()):
        optional = [name for name, field in parameters.items() if "default" in field or "optional" in str(field.get("type", "")).lower()]
        parameters = {"type": "object", "properties": parameters, "required": [name for name in parameters if name not in optional]}
    if isinstance(parameters.get("properties"), dict):
        parameters = {**parameters, "properties": {name: _typed(field) for name, field in parameters["properties"].items()}}
    return parameters


def _typed(field):
    """A field with its type in JSON Schema's words (one it cannot name left out), and an array its `items`, which providers require."""
    if not isinstance(field, dict) or not isinstance(field.get("type"), str):
        return field
    words = [JSON_TYPES[word] for word in re.findall(r"[a-z]+", field["type"].lower()) if word in JSON_TYPES]
    rest = {key: value for key, value in field.items() if key != "type"}
    if not words:
        return rest
    if words[0] == "array" and "items" not in rest:
        rest["items"] = {"type": words[1]} if len(words) > 1 else {}  # `List[int]`: integers; `list`: anything
    return {**rest, "type": words[0]}


class Dataset(synalinks.datasets.Dataset):
    """The engine's streaming `Dataset`, with every hand-written schema closed at the door.

    Everything is the base class': `_iter_rows()` to implement, the three
    Jinja2 templates over a row (`input_template`, `output_template` and
    `tools_template`, the row's own tool declarations as a Chat Completions
    request carries them, with `row_schema` the row's shape without them),
    `limit`, `repeat`, `batch_size`, the `(x, y)` batches. Two differences.
    `_close_objects`: an `input_schema` or `output_schema` written by hand
    is marked closed, object by object, before the base class reads it. And
    the templates are compiled in the engine's `JINJA`, as `RowTemplate`s:
    each sees the whole row as `row` beside its columns, and has `fromjson`,
    `fromliteral`, `as_image` and `as_audio`.
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
        # Compiled first, so a template that does not parse fails before the source is opened (a Hub dataset downloaded).
        compiled = (
            None if input_template is None else RowTemplate(input_template),
            None if output_template is None else RowTemplate(output_template),
            None if tools_template is None else ToolsTemplate(tools_template),
        )
        # synalinks compiles the templates in a Jinja environment of its own, which has no `fromjson`: it is handed
        # stand-ins, present where the templates are, and the engine's own replace them.
        super().__init__(
            input_data_model=input_data_model,
            input_schema=_coerce_schema(input_schema),
            input_template=None if input_template is None else "{}",
            output_data_model=output_data_model,
            output_schema=_coerce_schema(output_schema),
            output_template=None if output_template is None else "{}",
            tools_template=None if tools_template is None else "[]",
            batch_size=batch_size,
            limit=limit,
            repeat=repeat,
            **kwargs,
        )
        self.input_template, self.output_template, self.tools_template = input_template, output_template, tools_template
        self._input_tmpl, self._output_tmpl, self._tools_tmpl = compiled

    def _media_dir(self, folder) -> None:
        """Read a relative image or audio path in a row (`images/1.png`) from `folder`, the dataset's own."""
        for template in (self._input_tmpl, self._output_tmpl, self._tools_tmpl):
            if template is not None:
                template.template.globals[MEDIA_DIR] = str(folder)


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
