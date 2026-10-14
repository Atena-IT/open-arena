---
name: openarena-add-mcp-tools
description: Use when openarena should evaluate models that answer by calling tools — connecting an MCP (Model Context Protocol) server in mcp_servers.yaml over stdio, streamable HTTP, SSE or websocket, and writing an `agent:` dataset whose model calls those tools on every row; also for handing MCP tools to a Harbor environment's agent.
---

# openarena-add-mcp-tools

`configs/mcp_servers.yaml` holds connections to MCP tool servers, one per
entry. A dataset with an `agent:` block gives the model under test those
servers' tools: on every row it may call them, as many times as it needs,
before answering. The answer is scored by the dataset's reward as usual.

## The server entry

```yaml
mcp_servers:
  fetch:                               # a local process, spawned per run
    transport: stdio
    command: uvx                       # the executable
    args: [mcp-server-fetch]           # its arguments, one per item
    env: {}                            # added to the process's environment
    # cwd: /path/to/workdir
    description: Fetches a URL and returns it as markdown.
    tags: [web]
  docs:                                # a server already running somewhere
    transport: streamable_http         # streamable_http | sse | websocket
    url: http://localhost:8080/mcp
    headers:
      Authorization: Bearer <token>
```

`stdio` takes `command` (required), `args`, `env`, `cwd`; the other
transports take `url` (required) and `headers`. A field of the other
transport is an error. Secrets: prefer an `env` / header value the user
gives you over writing a token you were shown into the file.

## The dataset that uses it

```yaml
datasets:
  web-questions:
    type: local
    config:
      path: questions.jsonl            # {"question": "...", "answer": "..."}
      input_template: '{"question": {{ question | tojson }}}'
      output_template: '{"answer": {{ answer | tojson }}}'
      input_schema: {type: object, properties: {question: {type: string, description: The question to answer}}, required: [question]}
      output_schema: {type: object, properties: {answer: {type: string, description: The answer only}}, required: [answer]}
    agent:
      mcp_servers: [fetch]             # mcp_servers.yaml keys, at least one
    reward: judge                      # free-text answers: a judge (openarena-add-reward)
```

The evaluation's `language_models` are ordinary `models` keys: the models
that call the tools. A model has to support tool calling.

## Tools in a Harbor sandbox

An environment can hand servers to its agent: `mcp_servers: [fetch]` in
`environments.yaml` (openarena-add-environment). Harbor carries less than a
dataset's agent does:

- `stdio`, `sse` and `streamable_http` only; `websocket` is refused.
- No `headers`, and no stdio `env` or `cwd`: Harbor would drop them, so such
  a server is refused. Move a stdio server's variables to the environment's
  `env:`.
- A stdio command runs inside the sandbox: the image must have it (`uvx`
  above needs `uv` installed in the Dockerfile).
- An HTTP server must be reachable from the sandbox, not on `localhost`.

## Steps

1. Know the server: its launch command (stdio) or its URL. Do not start a
   long-running server yourself; a stdio one is spawned by the run.
2. Add the entry; add the dataset with `agent.mcp_servers`.
3. `uv run openarena check my-arena/configs` (a dataset naming a server that
   is not in `mcp_servers.yaml` is an error).
4. Run a few rows first (openarena-run-evaluation): a server that fails to
   start fails the cell, and its error is in the result.
