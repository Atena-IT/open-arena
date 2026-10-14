# License Apache 2.0: (c) 2026 Atena-Reply

"""A2A-agent program: each row is answered by a remote agent speaking the Agent2Agent protocol.

Used for a dataset that declares an `a2a: {}` block. An A2A agent has no
model to pick: the agent is what is compared, so the sweep's axis names
agents (keys of `a2a_agents:`) where other kinds name models, and each
program is built for one of them. The `A2aAgent` module reads the agent's
card (`<url>/.well-known/agent-card.json`) on first use, opens a client on
the interface the card prefers, then sends each row as one user message and
waits for the answer: a message, or a task it polls until it ends. The text
of the answer (the message's parts, else the task's artifacts) is the output.

A dataset with an output template expects the agent to answer JSON matching
that schema (a fenced block is unwrapped); an inputs-only dataset takes the
text as a chat message.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any

import synalinks
from synalinks.src.backend import JsonDataModel, SymbolicDataModel
from synalinks.src.modules.module import Module

from openarena.programs.utils import safe_name

# What an inputs-only dataset gets back: the agent's text as one chat message.
CHAT_MESSAGE_SCHEMA = {
    "type": "object",
    "properties": {"role": {"type": "string"}, "content": {"type": "string"}},
    "required": ["role", "content"],
    "additionalProperties": False,
}

_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)
# How often a task that is still running is asked for again.
POLL_INTERVAL_SEC = 1.0


def http_client(connection: dict, timeout: float | None = None):
    """The `httpx.AsyncClient` every request to the agent goes through: its headers, its timeout."""
    import httpx

    return httpx.AsyncClient(headers=dict(connection.get("headers") or {}), timeout=timeout or connection.get("timeout_sec") or 300)


async def resolve_card(http, connection: dict):
    """The agent's card, from `card_path` under `url` (the well-known path by default)."""
    from a2a.client import A2ACardResolver

    return await A2ACardResolver(http, connection["url"]).get_agent_card(relative_card_path=connection.get("card_path"))


def describe_card(card) -> dict[str, Any]:
    """What the registry keeps of a card: who the agent is, what it offers, how it is reached."""
    interfaces = [
        {"url": i.url, "binding": i.protocol_binding, "version": i.protocol_version} for i in card.supported_interfaces
    ]
    return {
        "agent": " ".join(x for x in (card.name, card.version) if x) or None,
        "description": card.description or None,
        "streaming": bool(card.capabilities.streaming),
        "interfaces": interfaces,
        "skills": [{"id": s.id, "name": s.name, "description": s.description or None} for s in card.skills],
    }


def _terminal(state) -> bool:
    from a2a.types import TaskState

    return state in (
        TaskState.TASK_STATE_COMPLETED,
        TaskState.TASK_STATE_FAILED,
        TaskState.TASK_STATE_CANCELED,
        TaskState.TASK_STATE_REJECTED,
        TaskState.TASK_STATE_INPUT_REQUIRED,
        TaskState.TASK_STATE_AUTH_REQUIRED,
    )


def answer_text(response) -> str:
    """The text of the agent's answer: a message's parts, else a finished task's artifacts.

    A task that did not complete raises, with what the agent said about it.
    """
    from a2a.helpers.proto_helpers import get_artifact_text, get_message_text
    from a2a.types import TaskState

    if response.HasField("message"):
        return get_message_text(response.message).strip()
    task = response.task
    state = task.status.state
    if state != TaskState.TASK_STATE_COMPLETED:
        said = get_message_text(task.status.message).strip() if task.status.HasField("message") else ""
        name = TaskState.Name(state).removeprefix("TASK_STATE_").lower().replace("_", " ")
        raise ValueError(f"the agent's task ended {name}" + (f": {said}" if said else ""))
    texts = [get_artifact_text(a) for a in task.artifacts]
    text = "\n".join(t for t in texts if t).strip()
    if not text and task.status.HasField("message"):
        text = get_message_text(task.status.message).strip()
    return text


class A2aAgent(Module):
    """Answer each row with one message to an A2A agent.

    Args:
        url (str): The agent's base URL.
        headers (dict): HTTP headers sent on every request.
        card_path (str | None): Where the card is under `url`; None reads
            the well-known path.
        timeout_sec (float): How long one row may take.
        schema (dict | None): JSON schema of the output; None takes free text
            as a chat message.
    """

    def __init__(self, url, headers=None, card_path=None, timeout_sec=300, schema=None, **kwargs):
        super().__init__(**kwargs)
        self.url = url
        self.headers = dict(headers or {})
        self.card_path = card_path or None
        self.timeout_sec = float(timeout_sec)
        self.free_text = schema is None
        self.schema = schema or CHAT_MESSAGE_SCHEMA
        self._http = None
        self._client = None
        self.built = True

    @property
    def connection(self) -> dict[str, Any]:
        return {"url": self.url, "headers": self.headers, "card_path": self.card_path, "timeout_sec": self.timeout_sec}

    async def compute_output_spec(self, inputs, training=False):
        return SymbolicDataModel(schema=self.schema, name=self.name)

    async def _connect(self):
        """The client, opened on first use (on the sweep's own loop) and kept for the module's life."""
        if self._client is None:
            from a2a.client import ClientConfig, ClientFactory

            self._http = http_client(self.connection)
            card = await resolve_card(self._http, self.connection)
            self._client = ClientFactory(ClientConfig(streaming=False, httpx_client=self._http)).create(card)
        return self._client

    async def ask(self, text: str) -> str:
        """Send `text` as one user message; the text of the agent's answer."""
        from a2a.helpers.proto_helpers import new_text_message
        from a2a.types import GetTaskRequest, Role, SendMessageRequest

        client = await self._connect()
        deadline = time.monotonic() + self.timeout_sec
        async with asyncio.timeout(self.timeout_sec):
            last = None
            async for response in client.send_message(SendMessageRequest(message=new_text_message(text, role=Role.ROLE_USER))):
                last = response
            if last is None:
                raise ValueError("the agent sent no answer")
            # A task the agent is still working on is asked for until it ends.
            while last.HasField("task") and not _terminal(last.task.status.state):
                await asyncio.sleep(min(POLL_INTERVAL_SEC, max(0.0, deadline - time.monotonic())))
                task = await client.get_task(GetTaskRequest(id=last.task.id))
                last.task.CopyFrom(task)
        return answer_text(last)

    async def call(self, inputs, training=False):
        if inputs is None:
            return None
        text = await self.ask(json.dumps(inputs.get_json()))
        return JsonDataModel(json=self._shape(text), schema=self.schema, name=self.name)

    async def aclose(self) -> None:
        """Close the client and its connections, once the program that holds them is done."""
        client, http, self._client, self._http = self._client, self._http, None, None
        if client is not None:
            await client.close()
        if http is not None:
            await http.aclose()

    def _shape(self, text: str) -> dict:
        """The agent's text as the output the dataset expects."""
        if self.free_text:
            return {"role": "assistant", "content": text}
        candidate = text
        fenced = _FENCE.match(text)
        if fenced:
            candidate = fenced.group(1)
        try:
            data = json.loads(candidate)
        except ValueError as e:
            raise ValueError(f"the agent did not answer JSON for the dataset's output schema: {text[:200]!r}") from e
        if not isinstance(data, dict):
            raise ValueError("the agent's JSON answer is not an object")
        return data

    def get_config(self):
        return {
            "url": self.url,
            "headers": self.headers,
            "card_path": self.card_path,
            "timeout_sec": self.timeout_sec,
            "schema": None if self.free_text else self.schema,
            "name": self.name,
        }


async def build_a2a(agent: str, dataset, connection: dict, reward, metrics=None):
    """Build (and compile) a program that answers each row with the A2A agent `agent`.

    `agent` is the axis entry (an `a2a_agents` key) and `connection` its
    entry as the run's config resolved it: `{url, headers, card_path,
    timeout_sec}`. The program keeps the agent's client; `close_program`
    closes it.
    """
    inputs = synalinks.Input(
        schema=dataset.input_schema,
        data_model=dataset.input_data_model if dataset.input_schema is None else None,
    )
    schema = dataset.output_schema
    if schema is None and dataset.output_data_model is not None:
        schema = dataset.output_data_model.get_schema()
    module = A2aAgent(
        url=connection["url"],
        headers=connection.get("headers") or {},
        card_path=connection.get("card_path"),
        timeout_sec=connection.get("timeout_sec") or 300,
        schema=schema,
        name=f"a2a_{safe_name(agent)}",
    )
    outputs = await module(inputs)
    program = synalinks.Program(inputs=inputs, outputs=outputs, name=f"a2a_{safe_name(agent)}")
    program.compile(reward=reward, metrics=metrics or None)
    program._arena_a2a_module = module  # for `close_program`
    return program


async def close_program(program) -> None:
    """Close the client behind a program built by `build_a2a`; a no-op for any other program."""
    module = getattr(program, "_arena_a2a_module", None)
    if module is not None:
        await module.aclose()
