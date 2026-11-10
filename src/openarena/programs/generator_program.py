# License Apache 2.0: (c) 2026 Atena-Reply

"""Generator-based program: one LM call per row.

Used for Q&A, classification, summarization, ... evals — any dataset that
declares a `generator:` block in YAML. The trial runner in
`libraries/openarena/src/openarena/evaluate.py` calls `build_program()`; everything downstream
(`program.evaluate(x=ds)`, reward computation, reduction) treats the result
as an opaque `Program`.
"""

import synalinks

from openarena.programs.utils import answer_layout, keeps_native_trace, language_model as _language_model, safe_name, with_model_effort


async def build_program(
    model_id: str,
    dataset,
    generator_kwargs: dict,
    reward,
    metrics=None,
    lm_kwargs=None,
):
    """Build (and compile) a Generator-based program for one trial.

    `metrics` is an optional list of pre-resolved `metrics.Metric`
    instances forwarded to `program.compile(metrics=...)`. They ride along
    the primary `program.evaluate()` pass and surface in its return dict.

    `lm_kwargs` are per-model options for `LanguageModel` (a
    `reasoning_effort`, say). Without them, the bare id is handed to
    the framework exactly as before. The model's `reasoning_effort` is also
    the module's own, unless the `generator:` block sets one (`with_model_effort`).

    `chain_of_thought: true` in the generator block builds the framework's
    `ChainOfThought` instead of a plain `Generator`: it puts a `thinking`
    field before the dataset's own output fields so the
    model reasons before it answers — without the dataset's schema having to
    make room for it. `answer_layout` counts `thinking` beside the answer, so
    a rule-based reward still scores the answer alone.

    A model that reasons natively at a level gets it with or without the flag
    (`keeps_native_trace`): its request is the same, the framework fills `thinking`
    from the reasoning trace instead of asking the model to write it, and the
    trace is kept with the row. The flag stays what makes any other model write it.
    A dataset whose own output has a `thinking` field is left as it is.
    """
    inputs = synalinks.Input(
        schema=dataset.input_schema,
        data_model=dataset.input_data_model if dataset.input_schema is None else None,
    )

    gen_kwargs = dict(generator_kwargs)
    thinking = gen_kwargs.pop("chain_of_thought", False) or (
        keeps_native_trace(model_id, lm_kwargs, gen_kwargs) and "thinking" not in answer_layout(dataset, "model", gen_kwargs)[0]
    )
    if thinking:
        answer_layout(dataset, "model", {**gen_kwargs, "chain_of_thought": True})
    module = synalinks.ChainOfThought if thinking else synalinks.Generator
    if dataset.output_schema is not None and "schema" not in gen_kwargs:
        gen_kwargs["schema"] = dataset.output_schema

    outputs = await module(
        language_model=_language_model(model_id, lm_kwargs, gen_kwargs),
        **with_model_effort(lm_kwargs, gen_kwargs),
    )(inputs)

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name=f"eval_{safe_name(model_id)}",
    )
    program.compile(reward=reward, metrics=metrics or None)
    return program
