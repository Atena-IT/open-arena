# License Apache 2.0: (c) 2026 Atena-Reply

"""Agent-based program: `FunctionCallingAgent` driven by MCP tools.

Used for any dataset that declares an `agent:` block in YAML. The tools
are loaded at build time; the agent then runs its own
call-tool/observe/decide loop per row. The trial runner in
`libraries/openarena/src/openarena/evaluate.py` calls `build_agent()`; everything downstream
(`program.evaluate(x=ds)`, reward computation, reduction) treats the result
as an opaque `Program`.
"""

import synalinks

from openarena.programs.utils import language_model as _language_model, safe_name


async def build_agent(
    model_id: str,
    dataset,
    agent_config: dict,
    reward,
    metrics=None,
    lm_kwargs=None,
):
    """Build (and compile) an agent-based program for one trial.

    `agent_config` is the per-dataset `agent:` block from YAML, with
    `mcp_servers` already resolved to a name → connection-config dict
    (the registry lookup is done in `src/config.py` before this call).

    Recognised keys consumed here:
      - `type` (str): only `"function_calling"` is supported today.
      - `mcp_servers` (dict): `{server_name: connection_dict}` passed to
        `MultiServerMCPClient`. Tool list is loaded eagerly via
        `client.get_tools()` so a missing server fails the trial fast
        rather than silently producing a tool-less agent.
      - `return_inputs` (bool): prepend the row's input fields to the answer,
        as `Generator(return_inputs=True)` does. Default False.
    `return_inputs_with_trajectory` (default True) keeps the
    agent's `messages` trajectory beside a structured answer; False returns
    the answer alone. Without an output schema the trajectory *is* the answer.
    All other keys are forwarded to `FunctionCallingAgent`,
    except `language_model` / `tools` / `data_model` / `schema` which are
    set from the model and dataset.
    """
    cfg = dict(agent_config)

    agent_type = cfg.pop("type", "function_calling")
    if agent_type != "function_calling":
        raise ValueError(
            f"agent: unsupported type {agent_type!r}. Supported: 'function_calling'."
        )

    return_inputs = bool(cfg.pop("return_inputs", False))

    server_connections = cfg.pop("mcp_servers", None)
    if not server_connections:
        raise ValueError(
            "agent: `mcp_servers` is required and must resolve to at least one "
            "server (declare servers under top-level `mcp_servers:` and "
            "reference them by name in the dataset's `agent.mcp_servers:` list)."
        )

    # Validate config shape BEFORE opening any MCP session — a bad YAML
    # shouldn't cost a subprocess spawn / network round-trip.
    reserved = {"language_model", "tools"}
    overlap = reserved & set(cfg)
    if overlap:
        raise ValueError(
            f"agent: keys {sorted(overlap)} are set automatically and cannot be "
            f"overridden in YAML."
        )

    # Final-answer schema/data_model: prefer explicit override in agent_config,
    # otherwise inherit from the dataset's output side. Inputs-only datasets
    # (no output_template) leave both unset → agent returns a free-form chat
    # answer via its final_generator.
    if "schema" not in cfg and "data_model" not in cfg:
        if dataset.output_schema is not None:
            cfg["schema"] = dataset.output_schema
        elif dataset.output_data_model is not None:
            cfg["data_model"] = dataset.output_data_model

    client = synalinks.MultiServerMCPClient(connections=server_connections)
    tools = await client.get_tools()
    if not tools:
        raise ValueError(
            f"agent: no MCP tools loaded from servers {sorted(server_connections)} "
            f"— check that each server is reachable and exposes at least one tool."
        )

    inputs = synalinks.Input(
        schema=dataset.input_schema,
        data_model=dataset.input_data_model if dataset.input_schema is None else None,
    )
    outputs = await synalinks.FunctionCallingAgent(
        language_model=_language_model(model_id, lm_kwargs),
        tools=tools,
        **cfg,
    )(inputs)
    if return_inputs:
        outputs = await synalinks.ops.concat(inputs, outputs, name="inputs_and_answer")

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name=f"agent_{safe_name(model_id)}",
    )
    program.compile(reward=reward, metrics=metrics or None)
    return program
