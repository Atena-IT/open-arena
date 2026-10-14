---
name: openarena-compare-a2a-agents
description: Use when openarena should compare remote agents rather than models — agents speaking the Agent2Agent (A2A) protocol, each at its own URL — registering them in a2a_agents.yaml, writing an a2a dataset (an empty `a2a` block) whose rows are sent to each agent as one message, and an evaluation whose axis lists the agents.
---

# openarena-compare-a2a-agents

An A2A agent is a service with an agent card at
`<url>/.well-known/agent-card.json`. An `a2a: {}` dataset sends each row to
the agent as one user message and scores its reply. The agents are what is
compared: the evaluation lists `a2a_agents` keys where it would list models.

## The agent entry

```yaml
a2a_agents:
  helper:
    url: http://localhost:9999         # required: the agent's base URL
    # card_path: /agent.json           # the card elsewhere, relative to url
    headers:                           # sent on every request
      Authorization: Bearer <token>
    timeout_sec: 120                   # one row, from sending to the agent's last word (default 300)
    description: The support agent, current version.
    tags: [support]
  helper-next:
    url: http://localhost:9998
    description: The support agent, release candidate.
```

## The dataset and the evaluation

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
    language_models: [helper, helper-next]   # a2a_agents keys, not models keys
    datasets: [support-questions]
```

The row is sent as the rendered input's JSON; the reply (a message, or a task
the agent completes) is the answer. A judge reward (openarena-add-reward)
fits most agents, whose replies are free text. An evaluation mixing an `a2a`
dataset with model datasets is refused: its axis is agents or models.

## Steps

1. Get each agent's URL from the user, and whether it needs a header.
   `curl -s <url>/.well-known/agent-card.json` shows its card and skills.
   Do not start agents yourself.
2. Add the entries, the dataset, and the evaluation.
3. `uv run openarena check my-arena/configs`.
4. Run it (openarena-run-evaluation); rank with a leaderboard whose `models`
   are the agents' keys (openarena-rank-leaderboard).
