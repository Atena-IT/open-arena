# License Apache 2.0: (c) 2026 Atena-Reply

"""The A2A program: a remote agent answers each row, its client faked.

What the module makes of an agent's answer (free text as a chat message,
JSON for a schema, a fenced block unwrapped), how it asks (one message, a
task polled until it ends), what it keeps of a card, how the program is
built and closed.
"""

import asyncio
from types import SimpleNamespace

import pytest
from a2a import types as a2a
from a2a.helpers.proto_helpers import new_text_message
from synalinks.src.backend import JsonDataModel

from openarena.programs.a2a_program import (
    CHAT_MESSAGE_SCHEMA,
    A2aAgent,
    answer_text,
    build_a2a,
    close_program,
    describe_card,
)
from openarena.rewards import get

A_SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"], "additionalProperties": False}
Q_SCHEMA = {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"], "additionalProperties": False}


def _message(text, role=a2a.Role.ROLE_AGENT):
    return a2a.SendMessageResponse(message=new_text_message(text, role=role))


def _task(state, text=None, artifact=None, id="t1"):
    task = a2a.Task(id=id, status=a2a.TaskStatus(state=state))
    if text is not None:
        task.status.message.CopyFrom(new_text_message(text, role=a2a.Role.ROLE_AGENT))
    if artifact is not None:
        task.artifacts.append(a2a.Artifact(parts=[a2a.Part(text=artifact)]))
    return task


def test_the_answers_text_is_the_messages_or_a_finished_tasks_artifacts():
    assert answer_text(_message("  42 ")) == "42"
    done = a2a.SendMessageResponse(task=_task(a2a.TaskState.TASK_STATE_COMPLETED, artifact="the answer"))
    assert answer_text(done) == "the answer"
    said = a2a.SendMessageResponse(task=_task(a2a.TaskState.TASK_STATE_COMPLETED, text="done: 42"))
    assert answer_text(said) == "done: 42"  # no artifact: what the status says
    with pytest.raises(ValueError, match="the agent's task ended failed: out of tokens"):
        answer_text(a2a.SendMessageResponse(task=_task(a2a.TaskState.TASK_STATE_FAILED, text="out of tokens")))
    with pytest.raises(ValueError, match="the agent's task ended input required$"):
        answer_text(a2a.SendMessageResponse(task=_task(a2a.TaskState.TASK_STATE_INPUT_REQUIRED)))


class _Client:
    """The A2A client: `send_message` streams the responses given, `get_task` the task's later states."""

    def __init__(self, responses, later=()):
        self.responses, self.later, self.sent, self.closed = list(responses), list(later), [], False

    async def send_message(self, request):
        self.sent.append(request.message)
        for response in self.responses:
            yield response

    async def get_task(self, request):
        return self.later.pop(0)

    async def close(self):
        self.closed = True


def _agent(client, **kwargs):
    agent = A2aAgent(url="http://agent.example", schema=A_SCHEMA, timeout_sec=5, **kwargs)
    agent._client = client
    agent._http = SimpleNamespace(aclose=_closed(agent))
    return agent


def _closed(agent):
    async def aclose():
        agent.http_closed = True

    return aclose


def test_a_row_is_sent_as_one_user_message_and_a_running_task_is_polled_until_it_ends(monkeypatch):
    import openarena.programs.a2a_program as module

    monkeypatch.setattr(module, "POLL_INTERVAL_SEC", 0.0)
    client = _Client([_message("ignored"), _message('{"answer": "4"}')])
    agent = _agent(client)
    assert asyncio.run(agent.ask("2 + 2?")) == '{"answer": "4"}'  # the last response is the answer
    [sent] = client.sent
    assert sent.role == a2a.Role.ROLE_USER and sent.parts[0].text == "2 + 2?"
    running = a2a.SendMessageResponse(task=_task(a2a.TaskState.TASK_STATE_WORKING))
    polled = _agent(_Client([running], later=[_task(a2a.TaskState.TASK_STATE_WORKING), _task(a2a.TaskState.TASK_STATE_COMPLETED, artifact='{"answer": "4"}')]))
    assert asyncio.run(polled.ask("2 + 2?")) == '{"answer": "4"}'
    with pytest.raises(ValueError, match="the agent sent no answer"):
        asyncio.run(_agent(_Client([])).ask("?"))


def test_each_row_predicts_the_agents_json_or_nothing_with_why():
    row = JsonDataModel(json={"question": "2 + 2?"}, schema=Q_SCHEMA)
    agent = _agent(_Client([_message('```json\n{"answer": "4"}\n```')]))
    answer = asyncio.run(agent.call(row))
    assert answer.get_json() == {"answer": "4"} and answer.get_schema() == A_SCHEMA and agent.last_failure is None
    assert agent._client.sent[0].parts[0].text == '{"question": "2 + 2?"}'  # the row, as JSON
    for text, failure in (("four", "the agent did not answer JSON for the dataset's output schema: 'four'"), ("[1]", "the agent's JSON answer is not an object")):
        agent = _agent(_Client([_message(text)]))
        with pytest.warns(UserWarning, match="A2A agent http://agent.example"):
            assert asyncio.run(agent.call(row)) is None
        assert agent.last_failure == failure
    assert asyncio.run(agent.call(None)) is None
    free = A2aAgent(url="http://agent.example")
    free._client = _Client([_message("four")])
    assert asyncio.run(free.call(row)).get_json() == {"role": "assistant", "content": "four"} and free.schema == CHAT_MESSAGE_SCHEMA


def test_the_module_keeps_its_connection_and_closes_it():
    agent = A2aAgent(url="http://agent.example", headers={"X": "1"}, card_path="", timeout_sec="30", schema=A_SCHEMA, name="bot")
    assert agent.connection == {"url": "http://agent.example", "headers": {"X": "1"}, "card_path": None, "timeout_sec": 30.0}
    assert agent.get_config() == {**agent.connection, "schema": A_SCHEMA, "name": "bot"} and agent.built
    assert A2aAgent(url="u").get_config()["schema"] is None
    spec = asyncio.run(agent.compute_output_spec(None))
    assert spec.get_schema() == A_SCHEMA
    client = _Client([])
    agent = _agent(client)
    asyncio.run(agent.aclose())
    assert client.closed and agent.http_closed and agent._client is None and agent._http is None
    asyncio.run(agent.aclose())  # closing twice is fine


def test_a_card_is_kept_as_who_the_agent_is_and_how_it_is_reached():
    card = SimpleNamespace(
        name="Bot", version="1.2", description="", capabilities=SimpleNamespace(streaming=True),
        supported_interfaces=[SimpleNamespace(url="http://a/rpc", protocol_binding="JSONRPC", protocol_version="0.3")],
        skills=[SimpleNamespace(id="s1", name="Search", description=None)],
    )
    assert describe_card(card) == {
        "agent": "Bot 1.2", "description": None, "streaming": True,
        "interfaces": [{"url": "http://a/rpc", "binding": "JSONRPC", "version": "0.3"}],
        "skills": [{"id": "s1", "name": "Search", "description": None}],
    }


def test_the_program_is_built_for_one_agent_and_closed_after_the_grid():
    dataset = SimpleNamespace(input_schema=Q_SCHEMA, input_data_model=None, output_schema=A_SCHEMA, output_data_model=None)
    connection = {"url": "http://agent.example", "headers": {"X": "1"}, "card_path": None, "timeout_sec": None}
    program = asyncio.run(build_a2a("my/bot", dataset, connection, get("exact_match")))
    assert program.name == "a2a_my_bot" and program.compiled
    module = program._arena_a2a_module
    assert isinstance(module, A2aAgent) and module.name == "a2a_my_bot" and module.timeout_sec == 300.0 and module.schema == A_SCHEMA
    import synalinks

    chat = SimpleNamespace(input_schema=None, input_data_model=synalinks.ChatMessages, output_schema=None, output_data_model=synalinks.ChatMessage)
    assert asyncio.run(build_a2a("bot", chat, connection, get("exact_match")))._arena_a2a_module.schema == synalinks.ChatMessage.get_schema()
    client = _Client([])
    module._client, module._http = client, SimpleNamespace(aclose=_closed(module))
    asyncio.run(close_program(program))
    assert client.closed and module._client is None
    asyncio.run(close_program(SimpleNamespace()))  # any other program: nothing to close
