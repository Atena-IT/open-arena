# License Apache 2.0: (c) 2026 Atena-Reply

"""The programs a dataset is answered by, as the sweep builds them, and what the config hands the builders.

`build_program` for a `generator:` dataset, `build_agent` for an `agent:` one
(its MCP client faked: the servers are never opened), the layout of what a
prediction holds beside the answer, the keys an output template renders, and
`Config.resolved_agent`, which turns a dataset's `agent:` block into what the
builder reads.
"""

import asyncio
import os
from types import SimpleNamespace

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import pytest
import synalinks

from openarena.config import Config, dataset_kind, flatten_dataset
from openarena.datasets.dataset import _close_objects
from openarena.programs.agent_program import McpToolsUnavailable, _reason, build_agent
from openarena.programs.generator_program import build_program
from openarena.programs.utils import answer_layout, safe_name, template_keys
from openarena.rewards import get

Q_SCHEMA = _close_objects({"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]})
A_SCHEMA = _close_objects({"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]})


def _dataset(input_schema=Q_SCHEMA, output_schema=A_SCHEMA, **fields):
    return SimpleNamespace(
        **{
            "input_schema": input_schema,
            "input_data_model": None if input_schema is not None else synalinks.ChatMessages,
            "output_schema": output_schema,
            "output_data_model": None if output_schema is not None else synalinks.ChatMessage,
            "output_template": '{"answer": {{ answer | tojson }}}',
            **fields,
        }
    )


def _modules(program, kind):
    return [m for m in program._flatten_modules(recursive=True) if isinstance(m, kind)]


def _output_fields(program):
    outputs = program.outputs[0] if isinstance(program.outputs, list) else program.outputs
    return list(outputs.get_schema()["properties"])


# -- generator -------------------------------------------------------------------


def test_a_generator_program_answers_in_the_datasets_output_schema_with_the_reward_compiled():
    reward = get("exact_match")
    program = asyncio.run(build_program("vllm/Qwen/Qwen3-4B", _dataset(), {"instructions": "Answer."}, reward))
    assert program.name == "eval_vllm_Qwen_Qwen3-4B" and program.compiled and program.reward is reward
    [generator] = _modules(program, synalinks.Generator)
    assert type(generator) is synalinks.Generator and generator.schema == A_SCHEMA and generator.instructions == "Answer."
    assert generator.language_model.model.endswith("vllm/Qwen/Qwen3-4B") and generator.language_model.default_kwargs == {}
    assert _output_fields(program) == ["answer"]
    # The block's own schema wins over the dataset's.
    own = _close_objects({"type": "object", "properties": {"reply": {"type": "string"}}})
    program = asyncio.run(build_program("openai/gpt-4o-mini", _dataset(), {"schema": own}, reward))
    assert _modules(program, synalinks.Generator)[0].schema == own


def test_chain_of_thought_builds_the_frameworks_module_with_thinking_before_the_answer():
    program = asyncio.run(build_program("openai/gpt-4o-mini", _dataset(), {"chain_of_thought": True}, get("exact_match")))
    assert _modules(program, synalinks.ChainOfThought)
    assert _output_fields(program) == ["thinking", "answer"]


def test_a_models_params_build_its_language_model_and_openrouter_gets_its_routing():
    metrics = [synalinks.metrics.MeanMetricWrapper(get("exact_match"), name="em")]
    program = asyncio.run(build_program("openai/gpt-4o-mini", _dataset(), {}, get("exact_match"), metrics=metrics, lm_kwargs={"temperature": 0.0, "reasoning_effort": "low"}))
    [generator] = _modules(program, synalinks.Generator)
    assert generator.language_model.default_kwargs == {"temperature": 0.0, "reasoning_effort": "low"}
    assert [m.name for m in program._compile_metrics._user_metrics] == ["em"]
    routed = asyncio.run(build_program("openrouter/deepseek/deepseek-r1", _dataset(), {}, get("exact_match")))
    assert _modules(routed, synalinks.Generator)[0].language_model.default_kwargs == {"extra_body": {"provider": {"require_parameters": True}}}


# -- agent -----------------------------------------------------------------------


async def add(a: int, b: int):
    """Add two numbers.

    Args:
        a (int): The first number.
        b (int): The second number.
    """
    return {"sum": a + b}


def _mcp(monkeypatch, tools=None, error=None, seen=None):
    class Client:
        def __init__(self, connections):
            (seen if seen is not None else []).append(connections)

        async def get_tools(self):
            if error is not None:
                raise error
            return tools if tools is not None else [synalinks.Tool(add)]

    monkeypatch.setattr(synalinks, "MultiServerMCPClient", Client)


SERVERS = {"calc": {"transport": "stdio", "command": "calc-mcp", "args": []}}


def test_an_agent_program_runs_the_servers_tools_and_answers_in_the_datasets_schema(monkeypatch):
    seen = []
    _mcp(monkeypatch, seen=seen)
    reward = get("exact_match")
    program = asyncio.run(build_agent("openai/gpt-4o-mini", _dataset(), {"mcp_servers": SERVERS, "max_iterations": 3}, reward, lm_kwargs={"temperature": 0.0}))
    assert seen == [SERVERS] and program.name == "agent_openai_gpt-4o-mini" and program.reward is reward
    [agent] = _modules(program, synalinks.FunctionCallingAgent)
    assert agent.schema == A_SCHEMA and agent.max_iterations == 3 and [getattr(t, "name", t) for t in agent.tools] == ["add"]
    assert agent.language_model.default_kwargs == {"temperature": 0.0}
    fields = _output_fields(program)
    assert "answer" in fields and "messages" in fields  # the trajectory beside the answer
    alone = asyncio.run(build_agent("openai/gpt-4o-mini", _dataset(), {"mcp_servers": SERVERS, "return_inputs_with_trajectory": False}, reward))
    assert _output_fields(alone) == ["answer"]
    with_inputs = asyncio.run(build_agent("openai/gpt-4o-mini", _dataset(), {"mcp_servers": SERVERS, "return_inputs": True}, reward))
    assert _output_fields(with_inputs)[0] == "question" and "answer" in _output_fields(with_inputs)


def test_without_an_output_schema_the_agent_answers_freely_and_a_block_schema_wins(monkeypatch):
    _mcp(monkeypatch)
    free = asyncio.run(build_agent("openai/gpt-4o-mini", _dataset(output_schema=None), {"mcp_servers": SERVERS}, get("exact_match")))
    [agent] = _modules(free, synalinks.FunctionCallingAgent)
    assert agent.schema is None or agent.schema == synalinks.ChatMessage.get_schema()
    own = _close_objects({"type": "object", "properties": {"reply": {"type": "string"}}})
    chosen = asyncio.run(build_agent("openai/gpt-4o-mini", _dataset(), {"mcp_servers": SERVERS, "schema": own}, get("exact_match")))
    assert _modules(chosen, synalinks.FunctionCallingAgent)[0].schema == own


def test_an_agent_block_is_checked_before_any_server_is_opened(monkeypatch):
    _mcp(monkeypatch, error=AssertionError("no server may be opened"))
    with pytest.raises(ValueError, match="unsupported type 'react'"):
        asyncio.run(build_agent("m", _dataset(), {"type": "react", "mcp_servers": SERVERS}, None))
    with pytest.raises(ValueError, match="`mcp_servers` is required"):
        asyncio.run(build_agent("m", _dataset(), {"mcp_servers": {}}, None))
    with pytest.raises(ValueError, match=r"keys \['language_model', 'tools'\] are set automatically"):
        asyncio.run(build_agent("m", _dataset(), {"mcp_servers": SERVERS, "tools": [], "language_model": "x"}, None))


def test_servers_that_give_no_tools_fail_the_cell_with_why(monkeypatch):
    _mcp(monkeypatch, error=ExceptionGroup("unhandled errors in a TaskGroup", [FileNotFoundError("[Errno 2] No such file or directory: 'calc-mcp'")]))
    with pytest.raises(McpToolsUnavailable, match=r"could not load the MCP tools of \['calc'\]: FileNotFoundError: \[Errno 2\] No such file or directory: 'calc-mcp'"):
        asyncio.run(build_agent("m", _dataset(), {"mcp_servers": SERVERS}, None))
    _mcp(monkeypatch, tools=[])
    with pytest.raises(McpToolsUnavailable, match=r"no MCP tools loaded from servers \['calc'\]"):
        asyncio.run(build_agent("m", _dataset(), {"mcp_servers": SERVERS}, None))
    nested = ExceptionGroup("outer", [ExceptionGroup("inner", [RuntimeError("down\n  for  good")]), ValueError("second")])
    assert _reason(nested) == "RuntimeError: down for good" and _reason(ValueError("x")) == "ValueError: x"


# -- what a prediction holds -------------------------------------------------------


def test_the_answer_fields_and_what_the_program_returns_beside_them():
    ds = _dataset()
    assert answer_layout(ds, "model", {}) == (["answer"], [])
    assert answer_layout(ds, "model", {"chain_of_thought": True}) == (["answer"], ["thinking"])
    assert answer_layout(ds, "model", {"return_inputs": True}) == (["answer"], ["question"])
    assert answer_layout(ds, "model", {"schema": {"properties": {"reply": {}}}}) == (["reply"], [])
    assert answer_layout(_dataset(output_schema=None), "model", {}) == (list(synalinks.ChatMessage.get_schema()["properties"]), [])
    assert answer_layout(ds, "agent", {}) == (["answer"], ["messages"])
    assert answer_layout(ds, "agent", {"return_inputs_with_trajectory": False, "return_inputs": True}) == (["answer"], ["question"])
    assert answer_layout(_dataset(output_schema=None), "agent", {}) == (list(synalinks.ChatMessage.get_schema()["properties"]), ["messages"])
    inputs_only = _dataset(output_schema=None, output_data_model=None, output_template=None)
    assert answer_layout(inputs_only, "agent", {}) == (["messages"], [])  # no target at all: the trajectory is the answer
    assert answer_layout(ds, "tools", {})[0] == list(synalinks.ChatMessage.get_schema()["properties"])  # a tool call is a chat message, whatever the target
    assert answer_layout(ds, "a2a", {"return_inputs": True}) == ([], [])


def test_a_field_name_the_program_would_rename_is_refused_before_the_run():
    thinking = _dataset(output_schema={"type": "object", "properties": {"thinking": {}, "answer": {}}})
    with pytest.raises(ValueError, match="already has a `thinking` field"):
        answer_layout(thinking, "model", {"chain_of_thought": True})
    same = _dataset(input_schema={"type": "object", "properties": {"answer": {}}})
    with pytest.raises(ValueError, match=r"input field\(s\) \['answer'\] have the same name as the answer"):
        answer_layout(same, "model", {"return_inputs": True})
    trajectory = _dataset(input_schema={"type": "object", "properties": {"messages": {}}})
    with pytest.raises(ValueError, match="the same name as the agent trajectory"):
        answer_layout(trajectory, "agent", {"return_inputs": True})


def test_the_keys_an_output_template_renders():
    assert template_keys('{"answer": {{ a | tojson }}, "unit": "cm"}') == ["answer", "unit"]
    assert template_keys('{% if x %}{"answer": 1}{% else %}{"answer": 2, "note": "n"}{% endif %}') == ["answer", "note"]
    assert template_keys('{"answer": {{ a }}') == ["answer"]  # not JSON around its placeholders: read off its pairs
    assert template_keys(None) == [] and template_keys("") == []
    assert safe_name("vllm/Qwen/Qwen3-4B:latest") == "vllm_Qwen_Qwen3-4B_latest"


# -- the config's side ---------------------------------------------------------------


def _config(**dataset):
    return Config.model_validate(
        {
            "evaluation": {"language_models": ["m"], "datasets": ["ds"]},
            "models": {"m": {"model": "openai/gpt-4o-mini"}},
            "rewards": {"exact": {"type": "exact_match"}},
            "mcp_servers": {"calc": {"transport": "stdio", "command": "calc-mcp", "args": ["--fast"]}, "web": {"transport": "streamable_http", "url": "http://localhost:8080/mcp"}},
            "datasets": {"ds": {"type": "local", "config": {"path": "x.jsonl", "input_template": '{"q": {{ q | tojson }}}'}, "reward": "exact", **dataset}},
        }
    )


def test_an_agent_block_is_resolved_to_its_servers_connections_and_the_rest_kept():
    cfg = _config(agent={"mcp_servers": ["calc", "web"], "max_iterations": 2, "return_inputs": True})
    resolved = cfg.resolved_agent("ds")
    assert resolved == {
        "type": "function_calling",
        "mcp_servers": {"calc": cfg.mcp_servers["calc"].connection(), "web": cfg.mcp_servers["web"].connection()},
        "max_iterations": 2,
        "return_inputs": True,
    }
    assert resolved["mcp_servers"]["calc"]["command"] == "calc-mcp" and resolved["mcp_servers"]["web"]["url"] == "http://localhost:8080/mcp"
    assert cfg.dataset_kind("ds") == "agent" and _config().resolved_agent("ds") is None and _config().dataset_kind("ds") == "model"


def test_a_registry_entry_flattens_to_what_the_provider_and_the_sweep_read():
    entry = {
        "type": "local",
        "description": "kept by the app, not the engine",
        "tags": ["x"],
        "config": {"path": "x.jsonl", "input_template": "{}"},
        "generator": {"instructions": "Answer."},
        "reward": "exact",
        "metrics": ["answer_accuracy"],
    }
    assert flatten_dataset(entry) == {"type": "local", "path": "x.jsonl", "input_template": "{}", "generator": {"instructions": "Answer."}, "reward": "exact", "metrics": ["answer_accuracy"]}
    assert dataset_kind(entry) == "model" and dataset_kind({**entry, "tools": {}}) == "tools"
    cfg = _config(generator={"instructions": "Answer."}, metrics=[{"class": "answer_accuracy", "alias": "acc"}])
    assert cfg.dataset_entry("ds") == {"type": "local", "path": "x.jsonl", "input_template": '{"q": {{ q | tojson }}}', "generator": {"instructions": "Answer."}, "reward": "exact", "metrics": [{"class": "answer_accuracy", "alias": "acc"}]}
    assert cfg.generator_kwargs("ds") == {"instructions": "Answer."} and cfg.model_id("m") == "openai/gpt-4o-mini" and cfg.model_id("not-a-model") == "not-a-model"
