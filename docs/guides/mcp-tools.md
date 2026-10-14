# MCP tools

`configs/mcp_servers.yaml` holds connections to
[MCP](https://modelcontextprotocol.io) tool servers. A dataset with an
`agent:` block gives the model under test those servers' tools: on each row it
may call them, as often as it needs, before it answers. The answer is scored by
the dataset's reward as usual.

```yaml
mcp_servers:
  fetch:                               # a local process, started by the run
    transport: stdio
    command: uvx
    args: [mcp-server-fetch]
    env: {}
    description: Fetches a URL and returns it as markdown.
    tags: [web]
  docs:                                # a server already running
    transport: streamable_http
    url: http://localhost:8080/mcp
    headers:
      Authorization: Bearer <token>
```

| transport | fields |
|---|---|
| `stdio` | `command` (required), `args`, `env`, `cwd` |
| `streamable_http`, `sse`, `websocket` | `url` (required), `headers` |

A field of another transport is an error.

## A dataset that uses them

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
      mcp_servers: [fetch]             # mcp_servers.yaml names, at least one
    reward: judge
```

The evaluation lists ordinary models, which must support tool calling.

## Tools in a Harbor sandbox

An [environment](environments.md) can hand servers to its agent with
`mcp_servers: [...]`. Harbor carries less than a dataset's agent does:

- `stdio`, `sse` and `streamable_http` only; `websocket` is refused.
- No `headers`, and no stdio `env` or `cwd`. Put a stdio server's variables in
  the environment's `env:` instead.
- A stdio command runs inside the sandbox, so the image must have it.
- An HTTP server must be reachable from the sandbox, not on `localhost`.
