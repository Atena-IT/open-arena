# License Apache 2.0: (c) 2026 Atena-Reply

"""Generator-based program: one LM call per row.

Used for Q&A, classification, summarization, ... evals — any dataset that
declares a `generator:` block in YAML. The trial runner in
`libraries/openarena/src/openarena/evaluate.py` calls `build_program()`; everything downstream
(`program.evaluate(x=ds)`, reward computation, reduction) treats the result
as an opaque `Program`.
"""

import synalinks

from openarena.programs.utils import language_model as _language_model, safe_name


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
    the framework exactly as before.

    `chain_of_thought: true` in the generator block builds the framework's
    `ChainOfThought` instead of a plain `Generator`: it puts a `thinking`
    field before the dataset's own output fields so the
    model reasons before it answers — without the dataset's schema having to
    make room for it. `answer_layout` counts `thinking` beside the answer, so
    a rule-based reward still scores the answer alone.
    """
    inputs = synalinks.Input(
        schema=dataset.input_schema,
        data_model=dataset.input_data_model if dataset.input_schema is None else None,
    )

    gen_kwargs = dict(generator_kwargs)
    module = (
        synalinks.ChainOfThought if gen_kwargs.pop("chain_of_thought", False) else synalinks.Generator
    )
    if dataset.output_schema is not None and "schema" not in gen_kwargs:
        gen_kwargs["schema"] = dataset.output_schema

    outputs = await module(
        language_model=_language_model(model_id, lm_kwargs),
        **gen_kwargs,
    )(inputs)

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name=f"eval_{safe_name(model_id)}",
    )
    program.compile(reward=reward, metrics=metrics or None)
    return program
