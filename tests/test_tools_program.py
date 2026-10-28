# License Apache 2.0: (c) 2026 Atena-Reply

"""A `tools:` dataset may render each row's tool declarations with `tools_template`: the input is then a Chat Completions request, `messages` and `tools`, and the model is sent that row's tools."""

import asyncio
import json
import os

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import pytest
import synalinks

from openarena.config.entries import DatasetEntry, tool_declaration_problem
from openarena.datasets.dataset import TOOLS_FIELD
from openarena.datasets.local_dataset import LocalDataset
from openarena.programs.tools_program import build_tools


def _tool(name):
    return {"type": "function", "function": {"name": name, "description": f"The {name}.", "parameters": {"type": "object", "properties": {}}}}


def _ask(text):
    return [{"role": "system", "content": "You are a weather bot."}, {"role": "user", "content": text}]


ROWS = [
    {"messages": _ask("Weather in Paris?"), "tools": [_tool("get_weather")], "calls": [{"name": "get_weather", "arguments": {"city": "Paris"}}]},
    {"messages": _ask("Time in Rome?"), "tools": [_tool("get_time"), _tool("get_weather")], "calls": [{"name": "get_time", "arguments": {}}]},
]
TEMPLATES = {
    "input_template": '{"messages": {{ messages | tojson }}}',
    "output_template": '{"tool_calls": {{ calls | tojson }}}',
    "output_schema": {"type": "object", "properties": {"tool_calls": {"type": "array"}}, "required": ["tool_calls"]},
}


def _dataset(tmp_path, rows=ROWS, **templates):
    path = tmp_path / "calls.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return LocalDataset(path=str(path), batch_size=2, **{**TEMPLATES, **templates})


def _fake_model(monkeypatch, seen):
    """The model under test, recording the tools and messages each request carries."""

    async def fake_call(self, messages, schema=None, tools=None, tool_schemas=None, **kwargs):
        seen.append(([t["function"]["name"] for t in tool_schemas], messages.get_json()["messages"]))
        message = synalinks.ChatMessage(role="assistant", content="")
        return synalinks.JsonDataModel(json=message.get_json(), schema=message.get_schema())

    monkeypatch.setattr(synalinks.LanguageModel, "__call__", fake_call)


def test_each_input_is_the_request_the_row_makes_its_messages_and_its_tools(tmp_path):
    ds = _dataset(tmp_path, tools_template="{{ tools | tojson }}")
    [(x, _)] = list(ds)
    assert [row.get_json() for row in x] == [
        {"messages": _ask("Weather in Paris?"), TOOLS_FIELD: [_tool("get_weather")]},
        {"messages": _ask("Time in Rome?"), TOOLS_FIELD: [_tool("get_time"), _tool("get_weather")]},
    ]
    assert TOOLS_FIELD == "tools"  # the request's field, not the kwarg's name
    assert ds.input_schema["properties"][TOOLS_FIELD]["items"] == synalinks.ChatCompletionTool.get_schema()
    assert ds.row_schema == synalinks.ChatMessages.get_schema()


def test_a_rows_tools_in_the_shapes_datasets_ship_reach_the_request_in_the_wire_shape(tmp_path):
    # As xLAM ships them: JSON text, a bare declaration, its parameters a mapping of fields typed the Python way.
    fields = {"city": {"type": "str"}, "days": {"type": "List[int]", "default": [1]}, "unit": {"type": "str, optional"}, "when": {"type": "Moment"}, "type": {"type": "str", "default": "game"}}
    bare = {"name": "get_weather", "description": "The weather.", "parameters": fields}  # a field named `type`, as xLAM has
    flat = {"type": "function", **bare}  # as the Responses API declares a tool: no `function` envelope
    rows = [{**ROWS[0], "tools": json.dumps([bare])}, {**ROWS[1], "tools": json.dumps(flat)}]  # a list, then one alone
    [(x, _)] = list(_dataset(tmp_path, rows=rows, tools_template="{{ tools | fromjson | tojson }}"))
    properties = {
        "city": {"type": "string"},
        "days": {"type": "array", "items": {"type": "integer"}, "default": [1]},  # providers refuse an array without `items`
        "unit": {"type": "string"},
        "when": {},
        "type": {"type": "string", "default": "game"},
    }
    wire = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "The weather.",
            "parameters": {"type": "object", "properties": properties, "required": ["city", "when"]},
        },
    }
    assert [row.get_json()[TOOLS_FIELD] for row in x] == [[wire], [wire]]
    # The wire shape, and a parameters schema already written as one, are handed on as they are.
    [(x, _)] = list(_dataset(tmp_path, tools_template="{{ tools | tojson }}"))
    assert x[0].get_json()[TOOLS_FIELD] == ROWS[0]["tools"]


def test_a_declaration_the_api_would_refuse_is_refused_when_the_row_is_read(tmp_path):
    with pytest.raises(ValueError, match='a tool declaration is {"type": "function", "function": {...}}'):
        list(_dataset(tmp_path, tools_template="{{ [{'parameters': {}}] | tojson }}"))  # no name: nothing to wrap
    with pytest.raises(ValueError, match="rename it"):
        list(_dataset(tmp_path, tools_template="[]", input_template='{"messages": [], "tools": []}'))
    assert tool_declaration_problem(_tool("f")) is None
    assert tool_declaration_problem({"type": "function", "function": {}}) == "a tool declaration's `function` names the function (`name`)"


def test_the_model_is_sent_the_rows_tools_beside_the_datasets_and_the_rows_own_system_prompt(tmp_path, monkeypatch):
    seen = []
    _fake_model(monkeypatch, seen)
    ds = _dataset(tmp_path, tools_template="{{ tools | tojson }}")
    program = asyncio.run(build_tools("openai/gpt-4o-mini", ds, {"tool_schemas": [_tool("search")]}, None))
    [(x, _)] = list(ds)
    for row in x:
        asyncio.run(program(row))
    assert [names for names, _ in seen] == [["search", "get_weather"], ["search", "get_time", "get_weather"]]
    # The request's messages are the row's, as on the API: its own system prompt, no second one in front, no tools in the text.
    assert [[m["role"] for m in messages] for _, messages in seen] == [["system", "user"], ["system", "user"]]
    assert seen[0][1][0]["content"] == "You are a weather bot."
    assert all("get_weather" not in json.dumps(messages) for _, messages in seen)


def test_a_row_of_plain_fields_is_asked_behind_the_instructions_with_its_tools(tmp_path, monkeypatch):
    seen = []
    _fake_model(monkeypatch, seen)
    ds = _dataset(
        tmp_path,
        tools_template="{{ tools | tojson }}",
        input_template='{"question": {{ messages[-1].content | tojson }}}',
        input_schema={"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]},
    )
    program = asyncio.run(build_tools("openai/gpt-4o-mini", ds, {"instructions": "Call the function that answers."}, None))
    [(x, _)] = list(ds)
    asyncio.run(program(x[0]))
    [(names, messages)] = seen
    assert names == ["get_weather"]
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "Call the function that answers." in messages[0]["content"]
    assert "Weather in Paris?" in messages[1]["content"] and "get_weather" not in messages[1]["content"]


def test_a_tools_dataset_needs_declarations_from_the_block_or_the_rows_in_the_wire_shape():
    base = {"type": "local", "config": {"path": "calls.jsonl", **TEMPLATES}, "reward": "tool-calls"}
    with pytest.raises(ValueError, match="tools_template"):
        DatasetEntry(**base, tools={})
    assert DatasetEntry(**{**base, "config": {**base["config"], "tools_template": "{{ tools | tojson }}"}}, tools={}).kind == "tools"
    with pytest.raises(ValueError, match="only offers tools to a `tools:` dataset"):
        DatasetEntry(**{**base, "config": {**base["config"], "tools_template": "{{ tools | tojson }}"}})
    with pytest.raises(ValueError, match="wrap it as"):
        DatasetEntry(**base, tools={"tool_schemas": [{"name": "get_weather", "parameters": {}}]})
    with pytest.raises(ValueError, match="is a list of function declarations, not int"):
        DatasetEntry(**base, tools={"tool_schemas": 1})
    assert DatasetEntry(**base, tools={"tool_schemas": [_tool("get_weather")]}).kind == "tools"


def test_at_the_wire_each_request_carries_its_rows_tools_and_the_reward_scores_the_calls(tmp_path, monkeypatch):
    """Through the real `LanguageModel`: `litellm.acompletion` sees the row's tools in the request's `tools`, with the block's beside them."""
    import litellm

    from openarena.rewards import get

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    requests = []

    async def fake_acompletion(**kwargs):
        requests.append(kwargs)
        # The model calls the row's first own tool, with the city the gold expects.
        name = next(t["function"]["name"] for t in kwargs["tools"] if t["function"]["name"] != "search")
        arguments = json.dumps({"city": "Paris"} if name == "get_weather" else {})
        return litellm.ModelResponse(
            model="gpt-4o-mini",
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            choices=[{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": None,
                "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": name, "arguments": arguments}}],
            }}],
        )

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    ds = _dataset(tmp_path, tools_template="{{ tools | tojson }}")
    program = asyncio.run(build_tools("openai/gpt-4o-mini", ds, {"tool_schemas": [_tool("search")]}, get("tool_calls_accuracy")))
    result = asyncio.run(program.evaluate(x=ds(), return_dict=True))
    assert result["reward"] == 1.0
    assert [[t["function"]["name"] for t in r["tools"]] for r in requests] == [["search", "get_weather"], ["search", "get_time", "get_weather"]]
    assert [[m["role"] for m in r["messages"]] for r in requests] == [["system", "user"], ["system", "user"]]
    assert requests[0]["messages"][0]["content"] == "You are a weather bot."
