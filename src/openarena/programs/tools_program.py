# License Apache 2.0: (c) 2026 Atena-Reply

"""Model-with-tools program: one call per row, with tool schemas the model may call.

Used for a dataset that declares a `tools:` block. It tests a model's tool
calling itself, with no tool server behind it: the model is handed the
dataset's `tool_schemas` (function declarations in the Chat Completions wire
format) and asked once per row; its answer — a chat message whose
`tool_calls` say which functions it chose to call, with what arguments — is
the prediction. Nothing is executed. `tool_calls_accuracy` scores those calls
against the gold ones the output template renders.

Unlike the agent program there is no loop and no MCP server: a single turn,
the way a tool-calling benchmark is meant.
"""

from __future__ import annotations

import synalinks

from openarena.programs.utils import language_model as _language_model, safe_name


async def build_tools(model_id: str, dataset, tools_kwargs: dict, reward, metrics=None, lm_kwargs=None):
    """Build (and compile) a Generator that answers each row with the dataset's tool schemas in hand.

    `tools_kwargs` is the dataset's `tools:` block: `tool_schemas` (a list of
    `{"type": "function", "function": {...}}` declarations) and any other
    `Generator` kwarg, `instructions` for one. The output is the
    model's chat message, tool calls included, so the dataset must not set an
    output schema of its own.
    """
    kwargs = dict(tools_kwargs)
    tool_schemas = kwargs.pop("tool_schemas", None)
    if not tool_schemas:
        raise ValueError("tools: `tool_schemas` is required and must list at least one function declaration.")
    inputs = synalinks.Input(
        schema=dataset.input_schema,
        data_model=dataset.input_data_model if dataset.input_schema is None else None,
    )
    outputs = await synalinks.Generator(
        language_model=_language_model(model_id, lm_kwargs),
        tool_schemas=list(tool_schemas),
        **kwargs,
    )(inputs)
    program = synalinks.Program(inputs=inputs, outputs=outputs, name=f"tools_{safe_name(model_id)}")
    program.compile(reward=reward, metrics=metrics or None)
    return program
