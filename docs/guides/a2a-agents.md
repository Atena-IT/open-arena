# A2A agents

openarena can compare remote agents rather than models: services speaking the
[Agent2Agent protocol](https://a2a-protocol.org), each at its own URL with an
agent card at `<url>/.well-known/agent-card.json`.

```yaml
a2a_agents:
  helper:
    url: http://localhost:9999         # required: the agent's base URL
    # card_path: /agent.json           # the card elsewhere, relative to url
    headers:
      Authorization: Bearer <token>
    timeout_sec: 120                   # one row, from sending to the last word (default 300)
    description: The support agent, current version.
  helper-next:
    url: http://localhost:9998
    description: The support agent, release candidate.
```

## A dataset and an evaluation over agents

An `a2a: {}` dataset sends each row to the agent as one user message, the
rendered `input_template`, and scores its reply. The evaluation lists agents
where it would list models:

```yaml
datasets:
  support-questions:
    type: local
    config:
      path: support.jsonl              # {"question": "...", "expected": "..."}
      input_template: '{"question": {{ question | tojson }}}'
      output_template: '{"answer": {{ expected | tojson }}}'
    a2a: {}                            # always empty: the agents come from the evaluation
    reward: judge
evaluations:
  support-agents:
    language_models: [helper, helper-next]   # a2a_agents names
    datasets: [support-questions]
```

The reply is a message, or a task the agent works on until it completes. A
judge reward suits most agents, whose replies are free text. An evaluation
cannot mix A2A datasets with model datasets: its axis is agents or models.

To see an agent's card and skills:

```shell
curl -s http://localhost:9999/.well-known/agent-card.json
```
