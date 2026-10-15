---
name: openarena-compare-a2a-agents
description: Use when openarena should compare remote agents rather than models — agents speaking the Agent2Agent (A2A) protocol, each at its own URL — registering them in a2a_agents.yaml, writing an a2a dataset (an empty `a2a` block) whose rows are sent to each agent as one message, and an evaluation whose axis lists the agents.
---

# openarena-compare-a2a-agents

An A2A agent is a running service with an agent card at
`<url>/.well-known/agent-card.json`. An `a2a: {}` dataset sends each row to
each agent as one user message and scores the reply with the dataset's
reward. The agents are what is compared: the evaluation lists `a2a_agents`
keys in `language_models`, where it would list models. Depth:
`docs/guides/a2a-agents.md`.

## The agent entry

```yaml
a2a_agents:
  helper:
    url: http://localhost:9999         # required: http:// or https://; the card is read here
    # card_path: /.well-known/agent.json   # the card elsewhere, a path under url
    headers:                           # sent on every request (card, messages, task polls)
      Authorization: Bearer ${HELPER_TOKEN}   # HELPER_TOKEN set in the workspace's .env
    timeout_sec: 120                   # one whole row, polls included (default 300, > 0)
    description: The support agent, current version.
    tags: [support]
  helper-next:
    url: http://localhost:9998
    description: The support agent, release candidate.
```

Only these six fields exist: `url`, `card_path`, `headers`, `timeout_sec`,
`description`, `tags`. There is no streaming, retry or rate-limit field; any
other key is an error. Keep a token out of the file: put it in the
workspace's `.env` (`HELPER_TOKEN=...`) and reference it as `${NAME}`; the run
replaces it when it reads its config, refuses to start when it is set nowhere,
and the run's `config.yaml` keeps the reference. `openarena check` warns about
a variable set neither in the shell nor in `.env`. Still prefer a revocable
evaluation key. Quote a header value YAML would read as a number.

`url` is only where the card is read. Messages go to the address the card's
interfaces advertise. An agent behind a proxy or tunnel that advertises an
internal address (`localhost`, a container name) reads fine, then every
message fails to connect; fix the card on the agent's side.

## The dataset and the evaluation

What the agent must reply depends on the dataset's `output_template`:

- **With `output_template` and `output_schema`**: the agent must answer a
  JSON object (a reply that is a single fenced `json` block is unwrapped).
  Prose scores 0 on that row. The agent never sees the schema, so tell it the format
  in the input, as `reply_format` does below. Any reward then works against
  the reference (`exact_match`, a judge).
- **With no output template** (inputs only): the reply is kept as free text,
  and only a judge can score it, its criteria in the judge's `instructions`.
  Use this for agents that answer in prose.
- An `output_template` without an `output_schema` makes the expected answer a
  chat message: the agent must then answer `{"role": ..., "content": ...}`
  JSON. Give the schema or drop the template.

```yaml
datasets:
  support-questions:
    type: local
    config:
      path: support.jsonl              # {"question": "...", "expected": "..."}
      limit: 5                         # start small, read the answers, then grow
      input_template: '{"question": {{ question | tojson }}, "reply_format": "a JSON object with one string field, answer, and nothing else"}'
      output_template: '{"answer": {{ expected | tojson }}}'
      input_schema: {type: object, properties: {question: {type: string, description: The question to answer}, reply_format: {type: string, description: How to reply}}, required: [question, reply_format]}
      output_schema: {type: object, properties: {answer: {type: string, description: The answer only}}, required: [answer]}
    a2a: {}                            # always empty: the agents come from the evaluation
    reward: judge                      # a rewards.yaml key (openarena-add-reward)
  support-open:                        # inputs only: free-text replies, judge-scored
    type: local
    config:
      path: support.jsonl
      input_template: '{"question": {{ question | tojson }}}'
      input_schema: {type: object, properties: {question: {type: string, description: The question to answer}}, required: [question]}
    a2a: {}
    reward: judge
evaluations:
  support-agents:
    language_models: [helper, helper-next]   # a2a_agents keys, not models keys
    datasets: [support-questions]
    batch_size: 2                      # rows sent to one agent at once
```

The rendered input is sent as the text of one user message, with nothing else
(no system prompt, no schema, no earlier rows). The reply is a message (its
text) or a task, polled every second until it ends; a `completed` task gives
its artifacts' text. Only text parts are read.

Rules the run enforces:

- Every dataset of the evaluation must be `a2a`: mixing with model datasets
  is refused. Run agents and models as two evaluations.
- A row the agent does not answer scores 0, with a warning
  (`A2A agent <url>: <why>`), and the cell goes on: a non-JSON or non-object
  reply where JSON is needed, a task ending `failed`, `canceled`, `rejected`,
  `input required` or `auth required` (no second turn is sent), a timeout
  (`no answer within <timeout_sec>s`), a connection or HTTP error. Rows are
  not retried.
- A cell where **no** row got an answer (the agent down, a wrong token, a
  card advertising an unreachable address) fails with
  `no row got an answer: ...`, and the run exits 1. Rerunning the same
  command reruns the failed cells and keeps the scored ones.
- No streaming, even if the card offers it, and no rate limiting: keep
  `batch_size` within the agent's quota, low for production.
- Only the final answer is recorded: no trajectory, tool calls, tokens or
  cost for agents. Trajectory rewards and cost weights mean nothing here.

## Steps

1. Get each agent's URL from the user, and whether it needs a header.
   `curl -s <url>/.well-known/agent-card.json` (add `-H` for auth) shows its
   card, its skills and the interface URLs messages will go to; check those
   are reachable. Do not start agents yourself.
2. Ask whether the agent can answer a JSON object; if not, use an
   inputs-only dataset with a judge.
3. Add the entries, the dataset, its reward, and the evaluation.
4. `uv run openarena check my-arena/configs` (reads files only; it never
   contacts the agents).
5. Run it (openarena-run-evaluation):
   `uv run openarena evaluations run support-agents --configs my-arena/configs`.
   Rank with a leaderboard (openarena-rank-leaderboard) with `models: []`,
   which takes every agent scored; listing agent names in `models:` makes
   `check` warn that each is `not in models`.
