# License Apache 2.0: (c) 2026 Atena-Reply

"""The vote samples: each row recorded once, numbered by its place in the dataset."""

import asyncio
from types import SimpleNamespace

from synalinks.src.backend import JsonDataModel

from openarena.evaluate import _tee_samples

SCHEMA = {"type": "object", "properties": {"q": {"type": "string"}}}


def _batch(*qs):
    return [JsonDataModel(schema=SCHEMA, json={"q": q}) for q in qs]


def test_the_build_time_call_on_the_first_batch_is_not_recorded():
    """The framework builds the program on the first batch, metrics included, then
    evaluates that batch again: only the evaluation is recorded, so row 0 is
    the first row once and every later index is where the row really is."""
    rows = []
    compiled = SimpleNamespace(built=False)

    async def inner(x, y, y_pred):
        compiled.built = True  # what building the metrics does
        return {}

    program = SimpleNamespace(compute_metrics=inner, _compile_metrics=compiled)
    _tee_samples(program, rows.append, limit=10)

    async def evaluate():
        first = _batch("a", "b")
        await program.compute_metrics(first, first, first)  # the build
        await program.compute_metrics(first, first, first)
        second = _batch("c", "d")
        await program.compute_metrics(second, second, second)

    asyncio.run(evaluate())
    assert [(r["index"], r["input"]["q"]) for r in rows] == [(0, "a"), (1, "b"), (2, "c"), (3, "d")]


def test_each_row_carries_its_score_and_a_failed_row_is_kept():
    """The batch's per-row rewards are on the program when metrics are computed:
    each row records its own, and a row the program answered nothing for is
    kept with no output, so a run's answers show its failures too."""
    rows = []

    async def inner(x, y, y_pred):
        return {}

    program = SimpleNamespace(compute_metrics=inner, _compile_metrics=SimpleNamespace(built=True))
    _tee_samples(program, rows.append, limit=10)

    async def evaluate():
        batch = _batch("a", "b")
        program._per_sample_rewards = [1.0, 0.0]
        await program.compute_metrics(batch, batch, [batch[0], None])

    asyncio.run(evaluate())
    assert [(r["index"], r["output"], r["reward"]) for r in rows] == [(0, {"q": "a"}, 1.0), (1, None, 0.0)]


def test_a_program_built_from_the_start_has_its_first_batch_recorded():
    """A2A, embedding and Harbor modules are built from the start: `evaluate`
    makes no build-time call, the metrics build on the first real batch, and
    that batch is recorded, not mistaken for the build."""
    rows = []
    compiled = SimpleNamespace(built=False)

    async def inner(x, y, y_pred):
        compiled.built = True
        return {}

    program = SimpleNamespace(compute_metrics=inner, _compile_metrics=compiled, _flatten_modules=lambda: [SimpleNamespace(built=True)])
    _tee_samples(program, rows.append, limit=10)
    batch = _batch("a", "b")
    asyncio.run(program.compute_metrics(batch, batch, batch))
    assert [r["index"] for r in rows] == [0, 1]
