# License Apache 2.0: (c) 2026 Athena-Reply

"""The editable program graphs.

Two entry points the trial runner in `evaluate.py` calls:

- `build_program(...)` — Generator-based eval (Q&A, classification,
  summarization, ...). One LM call per row.
- `build_agent(...)` — `FunctionCallingAgent`-based eval driven by MCP
  tools loaded at build time. The agent runs its own
  call-tool/observe/decide loop per row.

The trial picks one based on whether the dataset declared an `agent:`
block in YAML; everything downstream (`program.evaluate(x=ds)`,
reward computation, reduction) treats both as opaque
`synalinks.Program` instances.
"""

import synalinks


async def build_program(model_id: str, dataset, generator_kwargs: dict, reward):
    """Build (and compile) a Generator-based synalinks program for one trial."""
    inputs = synalinks.Input(
        schema=dataset.input_schema,
        data_model=dataset.input_data_model if dataset.input_schema is None else None,
    )

    gen_kwargs = dict(generator_kwargs)
    if dataset.output_schema is not None and "schema" not in gen_kwargs:
        gen_kwargs["schema"] = dataset.output_schema

    outputs = await synalinks.Generator(
        language_model=model_id,
        **gen_kwargs,
    )(inputs)

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name=f"eval_{_safe_name(model_id)}",
    )
    program.compile(reward=reward)
    return program


async def build_agent(model_id: str, dataset, agent_config: dict, reward):
    """Build (and compile) an agent-based program for one trial.

    `agent_config` is the per-dataset `agent:` block from YAML, with
    `mcp_servers` already resolved to a name → connection-config dict
    (the registry lookup is done in evaluate.py before this call).

    Recognised keys consumed here:
      - `type` (str): only `"function_calling"` is supported today.
      - `mcp_servers` (dict): `{server_name: connection_dict}` passed to
        `synalinks.MultiServerMCPClient`. Tool list is loaded eagerly via
        `client.get_tools()` so a missing server fails the trial fast
        rather than silently producing a tool-less agent.
    All other keys are forwarded to `synalinks.FunctionCallingAgent`,
    except `language_model` / `tools` / `data_model` / `schema` which are
    set from the model and dataset.
    """
    cfg = dict(agent_config)

    agent_type = cfg.pop("type", "function_calling")
    if agent_type != "function_calling":
        raise ValueError(
            f"agent: unsupported type {agent_type!r}. Supported: 'function_calling'."
        )

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
        language_model=model_id,
        tools=tools,
        **cfg,
    )(inputs)

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name=f"agent_{_safe_name(model_id)}",
    )
    program.compile(reward=reward)
    return program


def _safe_name(model_id: str) -> str:
    return model_id.replace("/", "_").replace(":", "_")
