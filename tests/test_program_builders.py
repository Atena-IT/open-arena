# License Apache 2.0: (c) 2026 Atena-Reply

"""The programs a dataset is answered by, as the sweep builds them, and what the config hands the builders.

`build_program` for a `generator:` dataset, `build_agent` for an `agent:` one
(its MCP client faked: the servers are never opened), the layout of what a
prediction holds beside the answer, the keys an output template renders, and
`Config.resolved_agent`, which turns a dataset's `agent:` block into what the
builder reads.
"""

import asyncio
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import litellm
import pytest
import synalinks

from openarena.config import Config, dataset_kind, flatten_dataset
from openarena.config.model_types import REASONING_EFFORTS
from openarena.datasets.dataset import _close_objects
from openarena.programs.agent_program import McpToolsUnavailable, _reason, build_agent
from openarena.programs.generator_program import build_program
from openarena.programs.tools_program import build_tools
from openarena.programs.utils import answer_layout, language_model, safe_name, template_keys, with_model_effort
from openarena.rewards import build_models, get

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


# -- the model's reasoning effort -------------------------------------------------


# litellm lists no such model: a test registers it where it needs a reasoning one.
THINKER = "openrouter/acme/thinker-1"


@pytest.fixture
def litellm_calls(monkeypatch):
    """What reaches `litellm.acompletion`, stubbed so that nothing leaves the machine."""
    calls = []

    async def acompletion(**kwargs):
        calls.append(kwargs)
        properties = kwargs["response_format"]["json_schema"]["schema"]["properties"]
        message = {"role": "assistant", "content": json.dumps({field: "x" for field in properties}), "reasoning_content": "the trace"}
        return litellm.ModelResponse(
            choices=[{"index": 0, "finish_reason": "stop", "message": message}],
            usage={"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        )

    monkeypatch.setattr(litellm, "acompletion", acompletion)
    return calls


def _registered(monkeypatch):
    monkeypatch.setitem(litellm.model_cost, THINKER, {"litellm_provider": "openrouter", "mode": "chat", "supports_reasoning": True})


def _sent(program, calls, key="reasoning_effort"):
    """The `key` the program's one call sent to litellm; `"<absent>"` when it sent none."""
    asyncio.run(program(synalinks.JsonDataModel(json={"question": "q"}, schema=Q_SCHEMA)))
    [call] = calls
    calls.clear()
    return call.get(key, "<absent>")


def _generated(model, chain_of_thought=False, lm_kwargs=None, **block):
    program = asyncio.run(build_program(model, _dataset(), {"chain_of_thought": chain_of_thought, **block}, get("exact_match"), lm_kwargs=lm_kwargs))
    [module] = _modules(program, synalinks.ChainOfThought if chain_of_thought else synalinks.Generator)
    return program, module


@pytest.mark.parametrize("chain_of_thought", [False, True])
@pytest.mark.parametrize("effort", ["low", "medium", "high"])
def test_the_level_set_on_a_model_is_the_one_its_generator_sends(monkeypatch, litellm_calls, effort, chain_of_thought):
    _registered(monkeypatch)
    program, module = _generated(THINKER, chain_of_thought, {"reasoning_effort": effort})
    assert module.reasoning_effort == effort and _sent(program, litellm_calls) == effort


@pytest.mark.parametrize("chain_of_thought", [False, True])
def test_off_on_a_model_turns_a_local_models_thinking_off(litellm_calls, chain_of_thought):
    program, module = _generated("ollama/qwen3:8b", chain_of_thought, {"reasoning_effort": "disable"})
    assert module.reasoning_effort == "disable" and _sent(program, litellm_calls, "think") is False


@pytest.mark.parametrize("chain_of_thought", [False, True])
def test_a_level_the_generator_block_sets_wins_over_the_models(monkeypatch, litellm_calls, chain_of_thought):
    _registered(monkeypatch)
    program, module = _generated(THINKER, chain_of_thought, {"reasoning_effort": "high"}, reasoning_effort="medium")
    assert module.reasoning_effort == "medium" and _sent(program, litellm_calls) == "medium"


def test_a_model_with_no_level_keeps_the_generators_own_default(monkeypatch, litellm_calls):
    _registered(monkeypatch)
    for lm_kwargs in (None, {}, {"temperature": 0.0}):
        program, generator = _generated(THINKER, lm_kwargs=lm_kwargs)
        assert generator.reasoning_effort is None and _sent(program, litellm_calls) is None
        program, thinker = _generated(THINKER, True, lm_kwargs)
        assert thinker.reasoning_effort == "low" and _sent(program, litellm_calls) == "low"


def test_a_model_the_framework_does_not_know_as_a_reasoner_still_sends_no_level(litellm_calls):
    for chain_of_thought in (False, True):
        program, module = _generated(THINKER, chain_of_thought, {"reasoning_effort": "high"})
        assert module.reasoning_effort == "high" and _sent(program, litellm_calls) == "<absent>"


def _answered(program, calls):
    """What the program's one call answered, as json, and the fields that call's schema asked for."""
    answered = asyncio.run(program(synalinks.JsonDataModel(json={"question": "q"}, schema=Q_SCHEMA)))
    [call] = calls
    calls.clear()
    return answered.get_json(), list(call["response_format"]["json_schema"]["schema"]["properties"])


def test_a_model_that_reasons_at_a_level_keeps_its_trace_in_thinking_without_the_toggle(monkeypatch, litellm_calls):
    _registered(monkeypatch)
    program = asyncio.run(build_program(THINKER, _dataset(), {}, get("exact_match"), lm_kwargs={"reasoning_effort": "low"}))
    assert _modules(program, synalinks.ChainOfThought) and _output_fields(program) == ["thinking", "answer"]
    answered, asked = _answered(program, litellm_calls)
    # The request is the plain generator's, `thinking` left out of it; the trace fills it after.
    assert asked == ["answer"] and answered == {"thinking": "the trace", "answer": "x"}


PLAIN = "openrouter/acme/plain-1"  # litellm lists no such model


@pytest.mark.parametrize(
    "model, lm_kwargs, block, kept",
    [
        (THINKER, {"reasoning_effort": "low"}, {}, True),
        (THINKER, {"reasoning_effort": "disable"}, {"reasoning_effort": "high"}, True),  # the block's level wins
        (THINKER, {"reasoning_effort": "disable"}, {}, False),
        (THINKER, {"reasoning_effort": "high"}, {"reasoning_effort": "disable"}, False),
        (THINKER, None, {}, False),
        (PLAIN, {"reasoning_effort": "high"}, {}, False),  # the framework sends it no level, so no trace comes back
    ],
)
def test_only_a_model_that_reasons_at_a_level_keeps_its_trace(monkeypatch, model, lm_kwargs, block, kept):
    _registered(monkeypatch)
    program = asyncio.run(build_program(model, _dataset(), dict(block), get("exact_match"), lm_kwargs=lm_kwargs))
    assert bool(_modules(program, synalinks.ChainOfThought)) is kept
    assert _output_fields(program) == (["thinking", "answer"] if kept else ["answer"])


def test_a_dataset_whose_output_has_its_own_thinking_field_is_left_as_it_is(monkeypatch):
    _registered(monkeypatch)
    own = _close_objects({"type": "object", "properties": {"thinking": {"type": "string"}, "answer": {"type": "string"}}, "required": ["thinking", "answer"]})
    program = asyncio.run(build_program(THINKER, _dataset(output_schema=own), {}, get("exact_match"), lm_kwargs={"reasoning_effort": "low"}))
    assert not _modules(program, synalinks.ChainOfThought) and _output_fields(program) == ["thinking", "answer"]


def test_a_dataset_that_returns_an_input_named_thinking_is_left_as_it_is_unless_the_flag_asks_for_it(monkeypatch):
    _registered(monkeypatch)
    named = _dataset(input_schema=_close_objects({"type": "object", "properties": {"thinking": {"type": "string"}}, "required": ["thinking"]}))
    program = asyncio.run(build_program(THINKER, named, {"return_inputs": True}, get("exact_match"), lm_kwargs={"reasoning_effort": "low"}))
    assert not _modules(program, synalinks.ChainOfThought)  # no trace kept, no error
    with pytest.raises(ValueError, match="return inputs"):
        asyncio.run(build_program(THINKER, named, {"return_inputs": True, "chain_of_thought": True}, get("exact_match")))


def test_max_is_sent_as_the_highest_level_a_generator_takes(monkeypatch, litellm_calls):
    _registered(monkeypatch)
    program, module = _generated(THINKER, lm_kwargs={"reasoning_effort": "max"})
    assert module.reasoning_effort == "xhigh" and _sent(program, litellm_calls) == "xhigh"


@pytest.mark.parametrize("effort", REASONING_EFFORTS)
def test_every_level_a_model_may_carry_builds_a_program(effort):
    # `synalinks.Generator` raises on a level it does not know: a level added to the arena is caught here, not in a run.
    for chain_of_thought in (False, True):
        _generated("openai/gpt-4o-mini", chain_of_thought, {"reasoning_effort": effort})


def test_a_tools_program_and_an_agent_take_the_models_level_too(monkeypatch):
    _mcp(monkeypatch)
    tools = {"tool_schemas": [{"type": "function", "function": {"name": "add", "parameters": {"type": "object", "properties": {}}}}]}
    for block, expected in (({}, "high"), ({"reasoning_effort": "low"}, "low")):
        program = asyncio.run(build_tools("openai/gpt-4o-mini", _dataset(), {**tools, **block}, get("exact_match"), lm_kwargs={"reasoning_effort": "high"}))
        assert _modules(program, synalinks.Generator)[0].reasoning_effort == expected
        program = asyncio.run(build_agent("openai/gpt-4o-mini", _dataset(), {"mcp_servers": SERVERS, **block}, get("exact_match"), lm_kwargs={"reasoning_effort": "high"}))
        [agent] = _modules(program, synalinks.FunctionCallingAgent)
        assert [agent.reasoning_effort, agent.tool_calls_generator.reasoning_effort, agent.final_generator.reasoning_effort] == [expected] * 3
    program = asyncio.run(build_tools("openai/gpt-4o-mini", _dataset(), tools, get("exact_match"), lm_kwargs={"temperature": 0.0}))
    assert _modules(program, synalinks.Generator)[0].reasoning_effort is None


def test_the_models_level_fills_a_block_that_leaves_it_unset_and_never_edits_the_block():
    block = {"instructions": "Answer.", "reasoning_effort": None}
    assert with_model_effort({"reasoning_effort": "low"}, block) == {"instructions": "Answer.", "reasoning_effort": "low"}
    assert block["reasoning_effort"] is None
    assert with_model_effort({"reasoning_effort": "low"}, {"reasoning_effort": "high"}) == {"reasoning_effort": "high"}
    plain = {"instructions": "Answer."}
    assert with_model_effort(None, plain) is plain and with_model_effort({"temperature": 0.0}, plain) is plain


# -- `disable` on OpenRouter, read off the request body ------------------------------


# litellm lists neither: the first is registered as a reasoning model by the test, the second is not.
KNOWN, UNKNOWN = THINKER, "openrouter/acme/plain-1"


@pytest.fixture
def wire(monkeypatch):
    """The JSON bodies posted to a local stand-in for the provider, whatever the provider: nothing leaves the machine."""
    bodies = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            bodies.append(body)
            properties = body["response_format"]["json_schema"]["schema"]["properties"]
            message = {"role": "assistant", "content": json.dumps({field: "x" for field in properties})}
            reply = {
                "id": "gen-1",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
            }
            data = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    monkeypatch.setenv("OPENROUTER_API_BASE", f"{base}/api/v1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_API_BASE", f"{base}/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    yield bodies
    server.shutdown()
    server.server_close()


def _posted(model, bodies, chain_of_thought, lm_kwargs):
    """The body of the one request a program for `model` posts."""
    program, _ = _generated(model, chain_of_thought, lm_kwargs)
    asyncio.run(program(synalinks.JsonDataModel(json={"question": "q"}, schema=Q_SCHEMA)))
    [body] = bodies
    bodies.clear()
    return body


@pytest.mark.parametrize("chain_of_thought", [False, True])
@pytest.mark.parametrize("known", [True, False])
def test_off_turns_openrouter_reasoning_off_whether_or_not_litellm_knows_the_model(monkeypatch, wire, known, chain_of_thought):
    if known:
        _registered(monkeypatch)
    body = _posted(KNOWN if known else UNKNOWN, wire, chain_of_thought, {"reasoning_effort": "disable"})
    assert body["reasoning"] == {"enabled": False} and "reasoning_effort" not in body
    assert body["provider"] == {"require_parameters": True}


@pytest.mark.parametrize("chain_of_thought", [False, True])
@pytest.mark.parametrize("entry", [{"supports_reasoning": False}, {}], ids=["flag false", "no flag"])
def test_off_adds_no_reasoning_object_for_a_model_litellm_lists_as_not_reasoning(monkeypatch, wire, entry, chain_of_thought):
    # OpenRouter refuses `reasoning` on an endpoint that has no such parameter (404 under require_parameters),
    # so a model with nothing to turn off must not be sent the switch, from its own level or from the block's.
    monkeypatch.setitem(litellm.model_cost, UNKNOWN, {"litellm_provider": "openrouter", "mode": "chat", **entry})
    body = _posted(UNKNOWN, wire, chain_of_thought, {"reasoning_effort": "disable"})
    assert "reasoning" not in body and "reasoning_effort" not in body
    assert body["provider"] == {"require_parameters": True}
    program, _ = _generated(UNKNOWN, chain_of_thought, lm_kwargs=None, reasoning_effort="disable")
    asyncio.run(program(synalinks.JsonDataModel(json={"question": "q"}, schema=Q_SCHEMA)))
    [body] = wire
    wire.clear()
    assert "reasoning" not in body and "reasoning_effort" not in body


def test_off_reads_a_variant_id_off_its_base_model(monkeypatch, wire):
    monkeypatch.setitem(litellm.model_cost, UNKNOWN, {"litellm_provider": "openrouter", "mode": "chat", "supports_reasoning": False})
    assert "reasoning" not in _posted(UNKNOWN + ":free", wire, False, {"reasoning_effort": "disable"})
    # A variant the table lists itself is read as listed, not off its base.
    monkeypatch.setitem(litellm.model_cost, UNKNOWN + ":think", {"litellm_provider": "openrouter", "mode": "chat", "supports_reasoning": True})
    assert _posted(UNKNOWN + ":think", wire, False, {"reasoning_effort": "disable"})["reasoning"] == {"enabled": False}


@pytest.mark.parametrize("chain_of_thought", [False, True])
@pytest.mark.parametrize("known", [True, False])
@pytest.mark.parametrize("effort", ["low", "medium", "high"])
def test_any_other_level_adds_no_reasoning_object_to_an_openrouter_request(monkeypatch, wire, effort, known, chain_of_thought):
    if known:
        _registered(monkeypatch)
    body = _posted(KNOWN if known else UNKNOWN, wire, chain_of_thought, {"reasoning_effort": effort})
    assert "reasoning" not in body and body.get("reasoning_effort") == (effort if known else None)
    assert body["provider"] == {"require_parameters": True}


def test_a_model_with_no_level_adds_no_reasoning_object(wire):
    assert "reasoning" not in _posted(UNKNOWN, wire, False, None)
    assert "reasoning" not in _posted(UNKNOWN, wire, True, {"temperature": 0.0})


def test_the_level_a_generator_block_sets_decides_the_reasoning_object_not_the_models(monkeypatch, wire):
    _registered(monkeypatch)
    # Model Off, block high: the block wins, nothing turns reasoning off.
    program, _ = _generated(KNOWN, lm_kwargs={"reasoning_effort": "disable"}, reasoning_effort="high")
    asyncio.run(program(synalinks.JsonDataModel(json={"question": "q"}, schema=Q_SCHEMA)))
    [body] = wire
    wire.clear()
    assert "reasoning" not in body and body["reasoning_effort"] == "high"
    # Model high (or no level), block Off: the block's Off turns reasoning off.
    for lm_kwargs in ({"reasoning_effort": "high"}, None):
        program, _ = _generated(KNOWN, lm_kwargs=lm_kwargs, reasoning_effort="disable")
        asyncio.run(program(synalinks.JsonDataModel(json={"question": "q"}, schema=Q_SCHEMA)))
        [body] = wire
        wire.clear()
        assert body["reasoning"] == {"enabled": False} and "reasoning_effort" not in body


def test_a_provider_other_than_openrouter_gets_no_reasoning_object(wire):
    body = _posted("openai/gpt-4o-mini", wire, False, {"reasoning_effort": "disable"})
    assert "reasoning" not in body and "reasoning_effort" not in body and "provider" not in body
    assert language_model("ollama/qwen3:8b", {"reasoning_effort": "disable"}).default_kwargs == {"reasoning_effort": "disable"}


def test_the_reasoning_switch_keeps_the_routing_and_the_entrys_own_extra_body():
    off = language_model(KNOWN, {"reasoning_effort": "disable", "extra_body": {"provider": {"order": ["a"]}, "seed": 1}})
    assert off.default_kwargs["extra_body"] == {
        "provider": {"order": ["a"], "require_parameters": True},
        "seed": 1,
        "reasoning": {"enabled": False},
    }
    assert "reasoning" not in language_model(KNOWN, {"reasoning_effort": "high"}).default_kwargs["extra_body"]


def test_every_model_built_from_an_entry_carries_the_switch_the_builders_and_the_judges(monkeypatch):
    _mcp(monkeypatch)
    off = {"reasoning_effort": "disable"}
    tools = {"tool_schemas": [{"type": "function", "function": {"name": "add", "parameters": {"type": "object", "properties": {}}}}]}
    models = [
        _generated(KNOWN, lm_kwargs=off)[1].language_model,
        _modules(asyncio.run(build_tools(KNOWN, _dataset(), tools, get("exact_match"), lm_kwargs=off)), synalinks.Generator)[0].language_model,
        _modules(asyncio.run(build_agent(KNOWN, _dataset(), {"mcp_servers": SERVERS}, get("exact_match"), lm_kwargs=off)), synalinks.FunctionCallingAgent)[0].language_model,
        build_models({"language_model": {"model": KNOWN, **off}})["language_model"],
    ]
    assert [m.default_kwargs["extra_body"]["reasoning"] for m in models] == [{"enabled": False}] * 4


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
