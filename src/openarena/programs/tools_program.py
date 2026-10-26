# License Apache 2.0: (c) 2026 Atena-Reply

"""Model-with-tools program: one call per row, with tool schemas the model may call.

Used for a dataset that declares a `tools:` block. It tests a model's tool
calling itself, with no tool server behind it: the model is handed the
dataset's `tool_schemas` (function declarations in the Chat Completions wire
format), and the row's own when the dataset renders them with a
`tools_template`, and asked once per row; its answer — a chat message whose
`tool_calls` say which functions it chose to call, with what arguments — is
the prediction. Nothing is executed. `tool_calls_accuracy` scores those calls
against the gold ones the output template renders.

Unlike the agent program there is no loop and no MCP server: a single turn,
the way a tool-calling benchmark is meant.
"""

from __future__ import annotations

import synalinks

from openarena.datasets.dataset import TOOLS_FIELD
from openarena.programs.utils import language_model as _language_model, safe_name


class RowToolsGenerator(synalinks.Generator):
    """A Generator whose input is a Chat Completions request: the row, and the `tools` it declares.

    The input carries the row's `tools` (rendered by the dataset's
    `tools_template`) beside its own fields, `messages` by default. The
    tools are sent as the request's tools, beside the dataset's, and the
    row is read back in its own shape (`row_schema`, the dataset's input
    schema before the tools joined it) so the prompt is built from the row
    alone: a row of chat messages keeps its own system prompt, as it would
    on the API.
    """

    def __init__(self, row_schema=None, **kwargs):
        super().__init__(**kwargs)
        self.row_schema = row_schema

    def _row(self, inputs):
        fields = {k: v for k, v in inputs.get_json().items() if k != TOOLS_FIELD}
        return synalinks.JsonDataModel(schema=self.row_schema, json=fields)

    async def call(self, inputs, tools=None, tool_schemas=None, tool_choice=None, training=False):
        if not inputs:
            return None
        row_tools = list(inputs.get(TOOLS_FIELD) or [])
        return await super().call(
            self._row(inputs),
            tools=tools,
            tool_schemas=list(tool_schemas or []) + row_tools,
            tool_choice=tool_choice,
            training=training,
        )

    async def compute_output_spec(self, inputs, tools=None, tool_schemas=None, tool_choice=None, training=False):
        return await super().compute_output_spec(
            synalinks.SymbolicDataModel(schema=self.row_schema),
            tools=tools,
            tool_schemas=tool_schemas,
            tool_choice=tool_choice,
            training=training,
        )

    def get_config(self):
        return {**super().get_config(), "row_schema": self.row_schema}


async def build_tools(model_id: str, dataset, tools_kwargs: dict, reward, metrics=None, lm_kwargs=None):
    """Build (and compile) a Generator that answers each row with the dataset's tool schemas in hand.

    `tools_kwargs` is the dataset's `tools:` block: `tool_schemas` (a list of
    `{"type": "function", "function": {...}}` declarations) and any other
    `Generator` kwarg, `instructions` for one. A dataset with a
    `tools_template` brings each row's declarations too, in the input's
    `tools` as a Chat Completions request carries them, sent beside the
    block's; then the block's `tool_schemas` may be left out. The output is the
    model's chat message, tool calls included, whatever the dataset's
    `output_schema` says: that schema only shapes the target, the gold
    `{"tool_calls": [...]}` the reward compares with.
    """
    kwargs = dict(tools_kwargs)
    tool_schemas = kwargs.pop("tool_schemas", None)
    per_row = getattr(dataset, "row_schema", None) is not None
    if not tool_schemas and not per_row:
        raise ValueError(
            "tools: `tool_schemas` must list at least one function declaration, or the dataset render each row's with `tools_template`."
        )
    inputs = synalinks.Input(
        schema=dataset.input_schema,
        data_model=dataset.input_data_model if dataset.input_schema is None else None,
    )
    if per_row:
        kwargs["row_schema"] = dataset.row_schema
    generator = RowToolsGenerator if per_row else synalinks.Generator
    outputs = await generator(
        language_model=_language_model(model_id, lm_kwargs),
        tool_schemas=list(tool_schemas or []),
        **kwargs,
    )(inputs)
    program = synalinks.Program(inputs=inputs, outputs=outputs, name=f"tools_{safe_name(model_id)}")
    program.compile(reward=reward, metrics=metrics or None)
    return program
