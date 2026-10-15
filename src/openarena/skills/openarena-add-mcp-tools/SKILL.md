---
name: openarena-add-mcp-tools
description: Use when openarena should evaluate models that answer by calling tools — connecting an MCP (Model Context Protocol) server in mcp_servers.yaml over stdio, streamable HTTP, SSE or websocket, and writing an `agent:` dataset whose model calls those tools on every row; also for handing MCP tools to a Harbor environment's agent.
---

# openarena-add-mcp-tools

`configs/mcp_servers.yaml` holds connections to MCP tool servers, one per
entry. A dataset with an `agent:` block gives the model under test those
servers' tools: on every row it may call them for up to `max_iterations`
turns, then writes its answer, which the dataset's reward scores as usual.
Full reference: `docs/guides/mcp-tools.md`.

Not the right tool when the test is only *which* function a model would call
(a `tools:` dataset, openarena-add-dataset), or an agent you built yourself
(openarena-compare-a2a-agents).

## The server entry

```yaml
mcp_servers:
  fetch:                               # the key also prefixes tool names: fetch_fetch
    transport: stdio                   # the default: spawn a process
    command: uvx                       # required: the executable
    args: [mcp-server-fetch]           # one item per argument; pin a version to compare runs later
    env:                               # the ONLY extra variables the process gets
      USER_AGENT: arena-eval
    # cwd: /abs/path/to/server
    description: Fetches a URL and returns it as markdown.
    tags: [web]
  docs:                                # a server already running somewhere
    transport: streamable_http         # streamable_http | sse | websocket
    url: http://localhost:8080/mcp     # required; http://, https://, ws:// or wss://
    headers:                           # streamable_http and sse only
      Authorization: Bearer ${DOCS_MCP_TOKEN}   # from the workspace's .env
```

- `stdio` reads `command` (required), `args`, `env`, `cwd`. `streamable_http`
  and `sse` read `url` (required) and `headers`; `websocket` reads `url` only.
  A field of another transport is an error.
- Any other field is an error: there is **no tool filter** (the model gets
  every tool of every listed server) and **no timeout** field. To hide tools,
  use the server's own flags in `args`.
- **A stdio server does not see `.env`.** It gets only `HOME`, `LOGNAME`,
  `PATH`, `SHELL`, `TERM`, `USER`, plus its own `env:`. A server that needs an
  API key gets it through `env:`, from the workspace's `.env`:
  `env: {API_KEY: "${API_KEY}"}`.
- **Secrets as `${NAME}`**: put the token or key in the workspace's `.env`
  and reference it in `headers` or `env` as `${NAME}`; the run replaces it
  when it reads its config, refuses to start if it is set nowhere, and
  `.open-arena/<evaluation>/config.yaml` keeps the reference, not the value.
  Never write a token you were shown into the file.
- **Each tool call opens a new MCP session** (for stdio, a new process).
  Nothing the server keeps in memory survives between two calls, even within
  a row; state on disk or in a database does.
- Relative paths in `command`, `args`, `cwd` are read from the folder
  `openarena` runs in, not `configs/`: prefer absolute paths.

## The dataset that uses it

```yaml
datasets:
  web-questions:
    type: local
    config:
      path: questions.jsonl            # {"question": "...", "answer": "..."}
      limit: 5                         # a few rows first; remove once it works
      input_template: '{"question": {{ question | tojson }}}'
      output_template: '{"answer": {{ answer | tojson }}}'
      input_schema: {type: object, properties: {question: {type: string, description: The question to answer}}, required: [question]}
      output_schema: {type: object, properties: {answer: {type: string, description: The answer only}}, required: [answer]}
    agent:
      mcp_servers: [fetch]             # mcp_servers.yaml keys, a list, at least one
      max_iterations: 8                # tool-calling turns per row (default 5)
      instructions: Fetch the pages you need first; do not answer from memory.
    reward: judge                      # free-text answers: a judge (openarena-add-reward)
    metrics:
      - {class: step_efficiency, language_model: gpt-4o-mini}
```

- `agent:` fields: `mcp_servers` (required), `max_iterations` (5),
  `instructions`, `final_instructions`, `use_chain_of_thought` (false),
  `return_inputs_with_trajectory` (true: the trajectory `messages` sits beside
  the answer), `return_inputs` (false). `language_model` and `tools` are set
  by the run and refused. Any other key is forwarded to the agent unchecked:
  a misspelling passes `check` and stops the run when it is prepared.
- The evaluation's `language_models` are ordinary `models` keys, the models
  that call the tools; each must support tool calling (`check` cannot know).
- Scoring: exact rewards read the answer fields only; judges (`lm_as_judge`,
  ...) and tool-use judges (`tool_correctness`, `tool_use`,
  `step_efficiency`, ...) read the trajectory too. `tool_calls_accuracy` does
  **not** apply to `agent:` datasets (it is for `tools:` datasets).
- A failing tool call does not fail the row: the model sees `error: <message>`
  and may retry.

## Tools in a Harbor sandbox

An environment can hand servers to its agent: `mcp_servers: [<key>, ...]` in
`environments.yaml` (openarena-add-environment). Harbor carries less than a
dataset's agent does:

- `stdio`, `sse` and `streamable_http` only; `websocket` is refused.
- No `headers`, and no stdio `env` or `cwd`: Harbor would drop them, so such
  a server is refused (`fetch` above, with its `env:`, would be: give the
  sandbox its own entry without one). Move a stdio server's variables to the
  environment's `env:`.
- A stdio command runs inside the sandbox: the image must have it (`uvx`
  needs `uv` installed in the Dockerfile).
- An HTTP server must be reachable from the sandbox, not on `localhost`.

## Steps

1. Know the server: its launch command (stdio), or its URL and transport
   (from the server's docs). Do not start a long-running server yourself; a
   stdio one is spawned by the run. An HTTP server must already be up.
2. Add the entry; add the dataset with `agent.mcp_servers`.
3. `uv run openarena check configs` from the workspace. It checks shapes and
   that `agent.mcp_servers` names exist; it does not connect, so a wrong URL,
   transport or command shows only at run time.
   `uv run openarena mcp-servers list --configs configs` shows what is saved.
4. Run a few rows first (openarena-run-evaluation). Every server's tool list
   is read while the run is prepared: a server that is down, a command not on
   `PATH` or a server with no tools fails **that cell**
   (`could not load the MCP tools of [...]: <why>`, or
   `no MCP tools loaded from servers [...]`), and the run goes on and exits 1.
   Fix it and rerun: the failed cells run again, no `--no-cache` needed. A
   misspelled `agent:` key still stops the whole run.
5. Read the result (openarena-read-results). The CLI does not trace into
   MLflow and does not save per-row trajectories; to read them, run from
   Python with `on_sample` or from the arena app. If every model scores alike
   and low, `max_iterations` is likely too low.
