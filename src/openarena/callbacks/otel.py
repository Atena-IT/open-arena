# License Apache 2.0: (c) 2026 Atena-Reply

"""The callback that gives a cell's program, and its judges, the OpenTelemetry hook."""

import synalinks
from synalinks.src import tree

from openarena.hooks.otel import attach


def _judges(program):
    """The judge programs a compiled program's reward and metrics hold (an LM, agent or rubric judge)."""
    compiled = getattr(program, "_compile_metrics", None)
    metrics = getattr(compiled, "_user_metrics", None) or []
    for item in [getattr(program, "reward", None), *tree.flatten(metrics)]:
        # A reward metric wraps its reward function in `_fn`.
        judge = getattr(getattr(item, "_fn", item), "program", None)
        if hasattr(judge, "_flatten_modules"):
            yield judge


class OtelCallback(synalinks.callbacks.Callback):
    """Attach `OtelHook` to every module of the program and of its judges, as its evaluation begins.

    The hook itself does nothing outside `use_tracing`, so a run that names no
    destination pays nothing but the attach.
    """

    def on_test_begin(self, logs=None):
        attach(self.program)
        for judge in _judges(self.program):
            attach(judge)
