# License Apache 2.0: (c) 2026 Atena-Reply

"""Helpers shared by the program builders in this package."""


def safe_name(model_id: str) -> str:
    """Turn a model id like `vllm/Qwen/Qwen3-4B` into a valid program name."""
    return model_id.replace("/", "_").replace(":", "_")


def with_provider_routing(model_id: str, lm_kwargs: dict | None) -> dict:
    """`lm_kwargs` plus the OpenRouter routing every structured call needs.

    the framework asks an `openrouter/...` model for structured output as a forced
    `structured_output` tool call. OpenRouter by default may route to a host
    that does not support tools and silently drops them: the model then answers
    in prose and imitates the call in text (DeepSeek-R1 writes
    `!function_call:{...}`), which fails to parse on every retry.
    `require_parameters` restricts routing to hosts that honour every parameter
    sent, so either the tools are honoured or the call fails at once.
    """
    kwargs = dict(lm_kwargs or {})
    if model_id.startswith("openrouter/"):
        extra_body = dict(kwargs.get("extra_body") or {})
        provider = dict(extra_body.get("provider") or {})
        provider.setdefault("require_parameters", True)
        extra_body["provider"] = provider
        kwargs["extra_body"] = extra_body
    return kwargs


def language_model(model_id: str, lm_kwargs: dict | None = None):
    """What to pass as `language_model=`.

    the framework resolves a bare id itself, and a `LanguageModel` is only built
    when there are options to carry (per-model `reasoning_effort`, or the
    OpenRouter routing of `with_provider_routing`).
    """
    lm_kwargs = with_provider_routing(model_id, lm_kwargs)
    if not lm_kwargs:
        return model_id
    import synalinks

    return synalinks.LanguageModel(model=model_id, **lm_kwargs)


def _fields(schema=None, data_model=None) -> list[str]:
    if schema is None and data_model is not None:
        schema = data_model.get_schema()
    return list((schema or {}).get("properties", {}))


def template_keys(template: str | None) -> list[str]:
    """The top-level keys an output template renders: the fields every target has.

    Each `{{ ... }}` is read as a value and the rest parsed as JSON; a
    template that is not JSON around its placeholders (a `{% ... %}` block,
    say) has its keys read off its `"key":` pairs instead. Mirrors
    `topLevelKeys` in the dataset form. `[]` without a template.
    """
    import json
    import re

    if not template:
        return []
    if "{%" not in template:
        try:
            value = json.loads(re.sub(r"{{[\s\S]*?}}", "null", template))
        except ValueError:
            value = None
        if isinstance(value, dict):
            return list(value)
    keys: list[str] = []
    for key in re.findall(r'"([A-Za-z_][\w-]*)"\s*:', template):
        if key not in keys:
            keys.append(key)
    return keys


def answer_layout(dataset, kind: str, block: dict | None) -> tuple[list[str], list[str]]:
    """`(answer fields, fields the program returns beside them)` for a dataset.

    The fields beside the answer are what `return_inputs` (the row's inputs),
    a chain of thought (`thinking`) and an agent's kept trajectory
    (`messages`) add to each prediction. A
    reward comparing whole predictions has to leave them out, and none may
    share a name with an answer field: `ops.concat` would rename the
    answer's copy to `<name>_1`, and a reward would read the wrong value.
    Raises `ValueError` on such a clash.
    """
    import synalinks

    block = block or {}
    chat_message = _fields(data_model=synalinks.ChatMessage)
    added: list[str] = []
    if kind in ("model", "tools"):
        schema = block.get("schema") or (dataset.output_schema if kind == "model" else None)
        answer = _fields(schema, block.get("data_model")) or chat_message
        if kind == "model" and block.get("chain_of_thought"):
            if "thinking" in answer:
                raise ValueError(
                    "chain of thought: the output schema already has a `thinking` field, "
                    "which the chain of thought adds itself. Rename it in the output "
                    "template (and `output_schema`), or turn chain of thought off."
                )
            added.append("thinking")
    elif kind == "agent":
        answer = _fields(
            block.get("schema") or dataset.output_schema,
            block.get("data_model") or dataset.output_data_model,
        )
        if not answer:
            answer = ["messages"]  # no schema: the trajectory is the answer
        elif block.get("return_inputs_with_trajectory", True):
            added.append("messages")
    else:
        return [], []
    if block.get("return_inputs"):
        inputs = _fields(dataset.input_schema, dataset.input_data_model)
        clash = sorted(set(inputs) & set(answer + added))
        if clash:
            beside = "the answer" if set(clash) & set(answer) else "the agent trajectory"
            raise ValueError(
                f"return inputs: input field(s) {clash} have the same name as {beside}, "
                "which would be renamed and scored wrongly. Rename them in the input "
                "template (and `input_schema`), or turn off returning the inputs."
            )
        added = inputs + added
    return answer, added
