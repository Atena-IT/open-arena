# License Apache 2.0: (c) 2026 Atena-Reply

"""Run the open-arena evaluation sweep.

The program graphs live in `libraries/openarena/src/openarena/programs/`. Two entry points there:
`build_program()` (Generator-based, used when a dataset declares
`generator:`) and `build_agent()` (FunctionCallingAgent + MCP tools,
used when a dataset declares `agent:`). The two are mutually
exclusive per dataset; this file dispatches between them based on
which block is set in YAML.

This module is the engine-side sweep: `run_sweep()` plus the result
rendering / TSV helpers. The user-facing `arena` CLI now lives in the
`open-arena-cli` package (`open_arena_cli.main`); its default sweep
command imports `run_sweep()` and the rendering helpers from here.

Run a sweep with:

    uv run arena -c config.yaml
    uv run arena -v                # show the progress bar
"""

import copy
import functools
import json
import traceback
import warnings
from pathlib import Path
from typing import Callable


import synalinks

synalinks.disable_keras_backend()  # MUST precede `import keras_tuner`

import keras_tuner.errors  # noqa: E402
from keras_tuner.src.engine import trial as trial_module  # noqa: E402
from keras_tuner.src.engine import tuner_utils  # noqa: E402

from synalinks.src.utils.naming import to_snake_case  # noqa: E402
from synalinks.src.utils.keras_tuner_utils import _run_trial_async  # noqa: E402

from openarena.config import Config, MetricEntry  # noqa: E402
from openarena.datasets import load_dataset_from_config  # noqa: E402
from openarena.lookups.prices import ensure_priced  # noqa: E402
from openarena.programs import build_a2a, build_agent, build_embedding, build_harbor, build_program, build_tools  # noqa: E402
from openarena.programs.harbor_program import no_answer_guard  # noqa: E402
from openarena.programs.a2a_program import close_program  # noqa: E402
from openarena.programs.utils import answer_layout, template_keys  # noqa: E402
from openarena.rewards import _REWARD_TYPES, get as get_reward  # noqa: E402


class SweepCancelled(keras_tuner.errors.FatalError):
    """Raised from an `on_event` callback to abort the sweep.

    Subclasses keras-tuner's `FatalError` so the tuner re-raises it instead
    of recording a failed trial and moving on to the next model.
    """


# What a cell has spent so far, forwarded with every batch so a run page can
# show where the money goes while it goes: the cost metrics (USD, cumulative
# over the cell, as the framework counts them) and the tokens behind them.
USAGE_METRICS = ("cost", "reward_cost", "embedding_cost", "input_tokens", "output_tokens")


def _usage(logs) -> dict[str, float]:
    return {
        k: float(logs[k])
        for k in USAGE_METRICS
        if isinstance(logs.get(k), (int, float))
    }


class _ProgressCallback(synalinks.callbacks.Callback):
    """Forward `program.evaluate` batch events to `on_event` for one cell."""

    def __init__(self, on_event, ds_name, model, ds):
        super().__init__()
        self._on_event = on_event
        self._ds_name = ds_name
        self._model = model
        self._ds = ds
        self._batches_total = None

    def _emit(self, event, **fields):
        self._on_event(
            {"event": event, "dataset": self._ds_name, "model": self._model, **fields}
        )

    def on_test_begin(self, logs=None):
        self._batches_total = _num_batches(self._ds)

    def on_test_batch_end(self, batch, logs=None):
        logs = logs or {}
        reward = logs.get("reward")
        self._emit(
            "batch",
            batch=int(batch) + 1,
            batches_total=self._batches_total,
            reward=float(reward) if isinstance(reward, (int, float)) else None,
            usage=_usage(logs),
        )

    def on_test_end(self, logs=None):
        metrics = {
            k: float(v) for k, v in (logs or {}).items() if isinstance(v, (int, float))
        }
        self._emit("cell_end", metrics=metrics)


class _RunMonitor(synalinks.callbacks.Monitor):
    """The framework's MLflow callback, pinned to the sweep's experiment.

    The default `Monitor` the framework adds when observability is on names
    its experiment after the program (`eval_<model>`), which scatters one
    sweep across experiments. This one logs every trial's evaluate metrics
    into the run's own experiment, one MLflow run per `(dataset, model)`,
    named and tagged so the MLflow UI reads like the arena.
    """

    def __init__(self, experiment_name, tracking_uri, ds_name, model, tags, on_event=None):
        self._on_event = on_event
        super().__init__(
            experiment_name=experiment_name,
            tracking_uri=tracking_uri,
            tags=tags,
            log_program_plot=False,
            log_program_model=False,
        )
        self._ds_name = ds_name
        self._model = model

    def on_test_begin(self, logs=None):
        model = self._model
        self.run_name = f"{self._ds_name} · {model}"
        self.tags = {**self.tags, "dataset": self._ds_name, "language_model": model}
        self._close_stale_run()
        # Tracing must never decide whether a cell gets evaluated: an MLflow
        # problem is worth a line in the log, not a failed trial.
        try:
            super().on_test_begin(logs)
        except Exception as e:  # noqa: BLE001
            self._run = None
            warnings.warn(
                f"MLflow: no run for {self.run_name} ({e}); evaluating untraced."
            )
            return
        # Which MLflow run holds this cell's traces, so the app links a cell (and its rows) straight to them.
        info = getattr(getattr(self, "_run", None), "info", None)
        if info is not None and self._on_event is not None:
            self._on_event({
                "event": "cell_traced", "dataset": self._ds_name, "model": self._model,
                "mlflow_run_id": info.run_id, "mlflow_experiment_id": info.experiment_id,
            })

    def on_test_end(self, logs=None):
        try:
            super().on_test_end(logs)
        except Exception as e:  # noqa: BLE001
            self._run = None
            warnings.warn(f"MLflow: could not close the run for {self.run_name} ({e}).")

    def _close_stale_run(self):
        """End a run that a previous trial left open.

        `Trainer.evaluate` calls `on_test_begin` and `on_test_end` with no
        `finally` between them, so a trial that raises mid-batch never closes
        its MLflow run. The next trial then meets a globally active run and
        `start_run` refuses it — "Run with UUID ... is already active" — so
        that trial fails too, and three of those in a row make keras-tuner
        abort the sweep. One bad batch would take the whole matrix with it.

        Every run active on this thread belongs to this sweep (one cell is
        evaluated at a time), so closing what is open is safe here.
        """
        import mlflow

        try:
            for _ in range(8):  # bounded: nested runs, not an unbounded stack
                if mlflow.active_run() is None:
                    break
                mlflow.end_run()
            else:
                return
        except Exception:  # noqa: BLE001 — tracing must not break the sweep
            pass
        self._run = None


def _num_batches(ds):
    try:
        return int(len(ds))
    except (NotImplementedError, TypeError):
        return None


# Default directory for trial state + output TSVs. Override per run with
# `--state-dir` so multiple runs can be kept side by side without clobbering
# each other's trial cache.
DEFAULT_STATE_DIR = Path(".open-arena")
TSV_NAME = "last_run.tsv"
FRONTIER_TSV_NAME = "frontier.tsv"


_MASK_KEYS = ("in_mask", "out_mask", "in_mask_pattern", "out_mask_pattern")


def _scope_to_answer(spec: dict, target: list[str]) -> dict:
    """A deterministic reward's `spec`, masked to the fields of the target.

    `target` is what the dataset's output template renders
    (`template_keys`), so the fields every target row has. A rule-based
    reward (exact match, ...) compares the prediction to the target field by
    field; masked to the target's own fields it never sees what else the
    program returns (returned inputs, a chain of thought, an agent's
    trajectory), whatever that is. A judge keeps seeing all of it, that is
    what it is for, and a mask the user set is theirs.
    """
    if not target or any(spec.get(k) for k in _MASK_KEYS):
        return spec
    from openarena.rewards import SELF_SCORED, category

    if spec.get("name") in SELF_SCORED:
        return spec  # it reads a score the program produced, not the target's fields
    try:
        deterministic = category(spec.get("name")) == "deterministic"
    except KeyError:
        return spec  # an unknown reward fails where it is built, with the list of known ones
    if not deterministic:
        return spec
    return {**spec, "in_mask": list(target)}


def _beside(entry: dict, base: Path) -> dict:
    """A Harbor task folders dataset's `path` read against the run's config: the run copies the tasks beside it (`tasks/`)."""
    if entry.get("type") == "harbor_tasks" and entry.get("path") and not Path(entry["path"]).is_absolute():
        return {**entry, "path": str(base / entry["path"])}
    return entry


def _instantiate_metrics(entries: list[MetricEntry], *, context: str, target=None):
    """Resolve validated `MetricEntry` rows into `[(alias, Metric, direction, is_objective)]`.

    Identifiers resolve in two registries: the project rewards (auto-wrapped
    in `MeanMetricWrapper` so they ride the primary `evaluate()` pass — no
    extra LM calls per "reward") and `metrics.get(...)` for plain
    metrics. `name=` on the wrapper flows to the wrapping metric (whose
    `.name` is the evaluate() result-dict key), not the inner reward —
    that's how the user's `alias:` survives into the matrix.
    """
    out = []
    for entry in entries:
        key = to_snake_case(entry.class_name)
        if key in _REWARD_TYPES:
            spec = {"name": key, **entry.kwargs}
            reward = get_reward(_scope_to_answer(spec, target) if target else spec)
            instance = synalinks.metrics.MeanMetricWrapper(
                reward,
                name=entry.alias or key,
            )
        elif entry.kwargs:
            instance = synalinks.metrics.get(
                {"class_name": entry.class_name, "config": entry.kwargs}
            )
        else:
            instance = synalinks.metrics.get(entry.class_name)
        if instance is None:
            raise ValueError(
                f"{context}: identifier {entry.class_name!r} resolved to None — "
                f"not a known metric or reward class."
            )
        resolved_alias = (
            entry.alias or getattr(instance, "name", None) or entry.class_name
        )
        if resolved_alias == "reward":
            raise ValueError(
                f"{context}: metric alias 'reward' collides with the primary "
                f"metric. Pick a different `alias:`."
            )
        out.append((resolved_alias, instance, entry.direction, entry.objective))
    return out


def _merge_metrics(*lists):
    """Concatenate metric lists, deduping by alias (first wins).

    Per-dataset entries override globals on alias collision, since the
    per-dataset list is passed second.
    """
    seen = {}
    for lst in lists:
        for entry in lst:
            alias = entry[0]
            if alias not in seen:
                seen[alias] = entry
    return list(seen.values())


def _pareto_frontier(cells, ds_name, model_ids, objectives):
    """Return the set of `model_ids` on the Pareto frontier for `ds_name`.

    `objectives` is a list of `(alias, direction)` pairs the dataset's
    tuner used. A model is on the frontier iff no other model dominates
    it on every objective. Domination: B dominates A iff B is at least as
    good as A on every objective AND strictly better on at least one
    (where "better" is max-direction-aware).

    Models with any missing objective value (failed trial / metric not
    computed) are excluded from the frontier — we can't compare them.
    """
    candidates = []
    for m in model_ids:
        scores = {}
        ok = True
        for alias, _ in objectives:
            v = cells.get((m, ds_name, alias))
            if v is None:
                ok = False
                break
            scores[alias] = v
        if ok:
            candidates.append((m, scores))

    dominated: set[str] = set()
    for ma, sa in candidates:
        if ma in dominated:
            continue
        for mb, sb in candidates:
            if mb == ma or mb in dominated:
                continue
            # Does B dominate A?
            at_least_as_good = True
            strictly_better = False
            for alias, direction in objectives:
                if direction == "max":
                    if sb[alias] < sa[alias]:
                        at_least_as_good = False
                        break
                    if sb[alias] > sa[alias]:
                        strictly_better = True
                else:  # min
                    if sb[alias] > sa[alias]:
                        at_least_as_good = False
                        break
                    if sb[alias] < sa[alias]:
                        strictly_better = True
            if at_least_as_good and strictly_better:
                dominated.add(ma)
                break

    return [m for m, _ in candidates if m not in dominated]


def _as_json(item):
    """The JSON behind one data model, or `None` when there is none.

    `SymbolicDataModel.get_json()` raises by design: the metrics are built
    once with symbolic placeholders before any row is evaluated, and that is
    how those probe calls are told apart from real batches.
    """
    if item is None:
        return None
    try:
        value = item.get_json()
    except Exception:  # noqa: BLE001 — symbolic model, or no json to give
        return None
    return value if isinstance(value, dict) else None


def _tee_samples(program, sink, *, limit):
    """Copy the first `limit` rows' (input, target, prediction, score) out of `evaluate`.

    `Trainer.compute_metrics(x, y, y_pred)` is the only hook in the evaluate
    loop that sees all three at once — rewards and metrics are handed
    `(y_true, y_pred)` without the input, and callbacks get neither — so the
    recorder wraps it on the program instance and passes everything through
    untouched. The batch's per-row rewards are on the program by then
    (`Trainer.test_on_batch` computes them first), so each row carries its
    `reward`. A row the program answered nothing for (the provider failed)
    is kept with `output: null`: it scored 0, and the run's answers say so.

    Rows are numbered from 0 in dataset order. Every model evaluates the same
    dataset in the same order, so the same index means the same row across
    models, which is what lets two models' answers be shown side by side.

    the framework also calls `compute_metrics` once while it builds the program,
    on the real first batch, before evaluating that batch again. That call is
    skipped: recorded, it would number the first batch twice and shift every
    later index by one batch — by a different amount per batch size, so two
    runs' "row 4" would be different questions. It only does so for a program
    with an unbuilt module; one built from the start has its first real batch
    recorded like the others.
    """
    inner = program.compute_metrics
    index = 0
    # Whether `evaluate` will make that build-time call: only when a module is unbuilt, its own test. A
    # module built from the start (A2A, embedding, Harbor) gets none, and its metrics build on the first real
    # batch, which must be recorded. A program that cannot say is assumed to build.
    flatten = getattr(program, "_flatten_modules", None)
    builds = flatten is None or not all(m.built for m in flatten())

    async def compute_metrics(x, y, y_pred):
        nonlocal index
        rows = len(x) if x is not None else 0
        # The build-time call comes while the compiled metrics are unbuilt
        # (that is why the framework makes it); every evaluated batch comes after.
        compiled = getattr(program, "_compile_metrics", None)
        building = builds and compiled is not None and not compiled.built
        # A symbolic batch carries no json at all; a real one always does.
        if not building and index < limit and rows > 0 and _as_json(x[0]) is not None:
            rewards = list(getattr(program, "_per_sample_rewards", None) or [])
            for i in range(rows):
                if index >= limit:
                    break
                output = (
                    _as_json(y_pred[i])
                    if y_pred is not None and i < len(y_pred)
                    else None
                )
                reward = rewards[i] if len(rewards) == rows else None
                sink(
                    {
                        "index": index,
                        "input": _as_json(x[i]),
                        "target": _as_json(y[i])
                        if y is not None and i < len(y)
                        else None,
                        "output": output,
                        "reward": float(reward) if isinstance(reward, (int, float)) else None,
                    }
                )
                # Advance even on a row the provider failed: the index is the
                # row's position, and it has to mean the same thing for every
                # model or the pairing breaks.
                index += 1
        return await inner(x, y, y_pred)

    program.compute_metrics = compute_metrics


def _hand_out(oracle, models):
    """Make the grid oracle issue one trial per model in `models`, all at once.

    `GridSearchOracle` derives each trial's values from the trial that ended
    before it and answers IDLE meanwhile, which is why keras-tuner runs a grid
    strictly one trial after another. This grid has one axis with values we
    already know, so nothing needs deriving: the oracle hands them out from a
    list. Retried trials (`_resume_grid`) come first, from kt's own retry
    queue, and are not in `models`.
    """
    queue = list(models)

    def populate_space(_trial_id):
        if not queue:
            return {"status": trial_module.TrialStatus.STOPPED, "values": None}
        return {
            "status": trial_module.TrialStatus.RUNNING,
            "values": {"language_model": queue.pop(0)},
        }

    oracle.populate_space = populate_space


async def _run_cell(tuner, trial, **fit_kwargs) -> None:
    """One trial, awaited: what `Tuner._try_run_and_update_trial` does, without the synchronous run.

    keras-tuner runs a trial synchronously on the calling thread; awaiting
    the framework's own `_run_trial_async` instead keeps the sweep on its own
    event loop. Outcomes land
    on the trial the way kt records them: COMPLETED with its metrics, FAILED
    for a `FailedTrialError`, INVALID for any other error, while a
    `FatalError` (cancellation, the consecutive-failure limit) propagates and
    ends the sweep, as before.
    """
    tuner.on_trial_begin(trial)
    try:
        results = await _run_trial_async(tuner, trial, **fit_kwargs)
        objective = tuner.oracle.objective
        tuner_utils.validate_trial_results(results, objective, "Tuner.run_trial()")
        tuner.oracle.update_trial(
            trial.trial_id,
            tuner_utils.convert_to_metrics_dict(results, objective),
            step=tuner_utils.get_best_step(results, objective),
        )
        trial.status = trial_module.TrialStatus.COMPLETED
    except keras_tuner.errors.FatalError:
        raise
    except Exception as e:  # noqa: BLE001 - a failed cell is a result, recorded on its trial
        failed = isinstance(e, keras_tuner.errors.FailedTrialError)
        trial.status = (
            trial_module.TrialStatus.FAILED
            if failed
            else trial_module.TrialStatus.INVALID
        )
        trial.message = traceback.format_exc()
    tuner.on_trial_end(trial)


def _resume_grid(oracle):
    """Let a reloaded grid tuner finish the cells its previous attempt missed.

    `GridSearchOracle` walks the grid with a cursor it keeps in memory only
    — `_ordered_ids` (the trials it has issued, in hyperparameter order) and
    `_populate_next` (the ones whose successor is still to be derived).
    Neither is part of `get_state`, so a tuner rebuilt from an existing
    `state_dir` comes back with the trials but an empty cursor, and its
    `populate_space` then either declares the search over (no trial to
    derive a successor from) or raises `KeyError` on the first one it does
    derive, because `_ordered_ids` has never heard of that trial. Rebuilding
    the cursor from the reloaded trials, ordered the way the oracle itself
    orders them, is what makes a resumed sweep reach the models the
    interrupted attempt never got to.

    Trials that did not complete — the one in flight when the process died,
    plus any that errored — go back on the oracle's retry queue, so those
    cells are re-run instead of carrying a stale failure into the matrix.

    Returns `(scored, retrying)`: the models whose cell is already
    COMPLETED (nothing to do for them), and those queued for another go.
    """
    trials = list(oracle.trials.values())
    if not trials:
        return [], []  # fresh project, or `overwrite=True` just wiped it
    ordered = sorted(
        trials,
        key=functools.cmp_to_key(
            lambda a, b: oracle._compare(
                a.hyperparameters.values, b.hyperparameters.values
            )
        ),
    )
    oracle._ordered_ids = type(oracle._ordered_ids)()
    for trial in ordered:
        oracle._ordered_ids.insert(trial.trial_id)
    oracle._populate_next = [trial.trial_id for trial in ordered]

    scored, retrying = [], []
    for trial in ordered:
        model = trial.hyperparameters.values.get("language_model")
        if model is None:
            continue  # stale cache from a different HP space
        if trial.status == "COMPLETED":
            scored.append(model)
        else:
            if trial.trial_id not in oracle._retry_queue:
                oracle._retry_queue.append(trial.trial_id)
            retrying.append(model)
    return scored, retrying


def _trial_errors(oracle) -> dict[str, str]:
    """`{model: why}` for each trial that did not complete: the last line of its traceback, the exception itself."""
    errors = {}
    for trial in oracle.trials.values():
        model = trial.hyperparameters.get("language_model")
        message = getattr(trial, "message", None)
        if model is None or not message or trial.status == "COMPLETED":
            continue
        lines = [line.strip() for line in str(message).splitlines() if line.strip()]
        errors[model] = lines[-1] if lines else str(message)
    return errors


def _collect_dataset_cells(oracle, ds_name, metric_keys, cells, statuses):
    """Fold one dataset-tuner's trials into the shared `cells` / `statuses` maps.

    Iterates *every* trial on the oracle (not `get_best_trials`, which
    silently drops FAILED/INVALID cells). Each tuner runs one trial per
    model_id (grid over a single `Choice("language_model", ...)` axis),
    so the (model_id, ds_name) coordinate is unique — no canonical-trial
    tie-breaking needed. The oracle's own objective + direction already
    rank trials internally; this just transfers metric values into the
    flat cell map for matrix / TSV rendering.
    """
    for trial in oracle.trials.values():
        m = trial.hyperparameters.get("language_model")
        if m is None:
            # Stale cache from a different HP space; skip.
            continue
        statuses[(m, ds_name)] = trial.status
        for k in metric_keys:
            try:
                cells[(m, ds_name, k)] = trial.metrics.get_best_value(k)
            except (KeyError, ValueError):
                cells[(m, ds_name, k)] = None


def _render_pareto_frontier(rows, ds_name, objectives, frontier_models):
    """Print one dataset's Pareto frontier as a markdown table.

    Pivots `rows` to `{model: {axis: value}}` for `ds_name` on the fly,
    then prints only the on-frontier models with one column per
    objective axis (annotated with ↑ / ↓ for direction). The full
    metrics matrix is in `.open-arena/last_run.tsv` (and the optional
    `--json` output); this is the human-facing summary for
    multi-objective datasets where a single "best model" doesn't exist.
    """
    axis_keys = [alias for alias, _ in objectives]
    arrow = {"max": "↑", "min": "↓"}
    obj_label = ", ".join(
        f"{alias} {arrow[direction]}" for alias, direction in objectives
    )
    print(f"### {ds_name} — Pareto frontier  ({obj_label})")
    print()

    if not frontier_models:
        print(
            "_(no Pareto frontier: all candidate trials had missing objective values)_"
        )
        print()
        return

    axis_set = set(axis_keys)
    by_model: dict[str, dict[str, float | None]] = {}
    for r in rows:
        if r["dataset"] != ds_name or r["metric"] not in axis_set:
            continue
        by_model.setdefault(r["model"], {})[r["metric"]] = r["value"]

    headers = ["language_model", *axis_keys]
    table_rows = []
    for m in frontier_models:
        scores = by_model.get(m, {})
        row = [m]
        for k in axis_keys:
            v = scores.get(k)
            row.append(f"{v:.4f}" if v is not None else "—")
        table_rows.append(row)

    widths = [
        max(len(headers[i]), *(len(r[i]) for r in table_rows))
        for i in range(len(headers))
    ]

    def _fmt(row):
        first = row[0].ljust(widths[0])
        rest = [c.rjust(widths[i + 1]) for i, c in enumerate(row[1:])]
        return "| " + " | ".join([first, *rest]) + " |"

    sep = ["-" * widths[0]] + [
        "-" * (widths[i] - 1) + ":" for i in range(1, len(headers))
    ]
    print(_fmt(headers))
    print("| " + " | ".join(sep) + " |")
    for row in table_rows:
        print(_fmt(row))
    print()


def _build_rows(
    cells,
    statuses,
    model_ids,
    dataset_names,
    metric_keys,
    *,
    direction_by_ds,
    metric_directions,
    frontier_by_ds,
):
    """Flatten the sweep state into one canonical long-form row list.

    Each row is a single `(model, dataset, metric)` cell with everything
    needed to render it independently: the metric `direction:`, the
    trial `status` (so consumers can distinguish "didn't run" from
    "ran and got 0.0"), and the `on_frontier` flag for the row's
    `(model, dataset)` pair. TSV and JSON output both stream from this
    list — no second pivoted view.
    """
    rows = []
    for d in dataset_names:
        frontier = set(frontier_by_ds.get(d, ()))
        primary_dir = direction_by_ds.get(d, "max")
        for m in model_ids:
            status = statuses.get((m, d))
            on_frontier = m in frontier
            for k in metric_keys:
                direction = (
                    primary_dir if k == "reward" else metric_directions.get(k, "max")
                )
                rows.append(
                    {
                        "model": m,
                        "dataset": d,
                        "metric": k,
                        "value": cells.get((m, d, k)),
                        "status": status,
                        "direction": direction,
                        "on_frontier": on_frontier,
                    }
                )
    return rows


def _write_tsv(rows, path: Path) -> None:
    """Write the long-form rows as TSV: `model\\tdataset\\tmetric\\tvalue\\tdirection`.

    Cells with `value is None` (failed / missing) are skipped — TSV is
    the happy-path view; consumers wanting failed cells use the JSON.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("model\tdataset\tmetric\tvalue\tdirection\n")
        for r in rows:
            if r["value"] is None:
                continue
            f.write(
                f"{r['model']}\t{r['dataset']}\t{r['metric']}\t"
                f"{r['value']:.6f}\t{r['direction']}\n"
            )


def _write_frontier_tsv(rows, objectives_by_ds, path: Path) -> None:
    """Write the Pareto frontier in long format:
    `dataset\\tmodel\\taxis\\tdirection\\tvalue`.

    Filters `rows` to (multi-objective dataset, on-frontier model,
    objective-axis metric). Single-objective datasets are omitted —
    their frontier is trivially the best row already in `last_run.tsv`.
    Empty file (header only) when no dataset has more than one objective.
    """
    axis_by_ds = {
        d: {a for a, _ in objs} for d, objs in objectives_by_ds.items() if len(objs) > 1
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("dataset\tmodel\taxis\tdirection\tvalue\n")
        for r in rows:
            if not r["on_frontier"] or r["value"] is None:
                continue
            if r["metric"] not in axis_by_ds.get(r["dataset"], ()):
                continue
            f.write(
                f"{r['dataset']}\t{r['model']}\t{r['metric']}\t"
                f"{r['direction']}\t{r['value']:.6f}\n"
            )


async def run_sweep(
    config_path: str,
    *,
    no_cache: bool = False,
    verbose: int = 0,
    state_dir: Path = DEFAULT_STATE_DIR,
    on_event: Callable[[dict], None] | None = None,
    on_sample: Callable[[dict], None] | None = None,
    samples_per_cell: int = 0,
    batch_size: int | None = None,
    mlflow_experiment: str | None = None,
    mlflow_tracking_uri: str | None = None,
    mlflow_tags: dict | None = None,
) -> dict:
    """Run the full sweep and return the JSON-serializable result dict.

    `on_event`, when given, receives progress dicts as the sweep advances:
    `sweep_start` (models, datasets), `dataset_start`, `dataset_resumed`
    (models already scored / queued for a retry, only when a previous
    attempt left state behind), `dataset_ready` (batches), `cell_start`
    (model), `batch` (batch, batches_total, running reward), `cell_end`
    (metrics), `dataset_end` (statuses per model) and `sweep_end`. Raising
    `SweepCancelled` from it aborts the sweep.

    `on_sample`, with `samples_per_cell > 0`, receives the first rows of each
    `(model, dataset)` cell as `{dataset, model, index, input, target,
    output}` — what the model was asked and what it answered, which the
    aggregate matrix throws away. Rows are numbered in dataset order, so the
    same `index` is the same row for every model.

    `mlflow_experiment`, when given (and observability is enabled), attaches
    an MLflow `Monitor` callback that logs each trial's evaluate metrics as
    one run of that experiment, tagged with `mlflow_tags`, the dataset and
    the model. Without it the framework still adds its default Monitor, which
    files metrics under an experiment named after the program.

    Programmatic entry point (e.g. for an API endpoint that needs the
    matrix without going through stdout). One `GridSearch` tuner
    runs per dataset over `Choice("language_model", ...)` — model is the
    only HP axis; the dataset is a fixed search context. Each tuner
    persists trial state under `<state_dir>/<dataset>/` (default
    `.open-arena`, override with `state_dir`), the same cache the CLI
    reads, and distinct `state_dir`s never collide.

    That cache is also what makes a sweep resumable: point a new call at
    the `state_dir` of an interrupted one (with `no_cache=False`, which is
    the default) and every cell that completed is skipped — its scores are
    read back from the trial — while the cells that failed or were in
    flight are re-run along with the ones never reached. Programs are only
    built for the cells that still have to run. See `_resume_grid`.

    Programs are built eagerly per `(model, dataset)` cell, then the cells
    run in turn, model-major, on this event loop (`_run_cell`), each with a
    view of its dataset batched to the run's `batch_size` (else the
    dataset's own): the batch is what the framework sends at once.

    `no_cache=True` drops every per-dataset cache before starting — use
    it when the model list or any dataset's reward / metric set changed.
    It is also what makes a run un-resumable, so a resume must never set it.
    `verbose` is forwarded to `Program.evaluate` (0 silent,
    1 progress bar, 2 per-batch).
    """
    cfg = Config.load(config_path)
    emit = on_event or (lambda _event: None)

    # The run's spend is litellm's price of each call: every model it may call
    # (the ones under test, the judges) is priced from the first source that
    # knows it, not only litellm's live table (lookups/prices.py).
    for entry in cfg.models.values():
        ensure_priced(entry.model, "embedding" if entry.type == "embedding" else "chat")

    # Optional process-wide reproducibility seed. `set_seed`
    # seeds both `numpy.random` and `random` — LM sampling temperature is
    # not affected (that's the provider's RNG).
    if cfg.seed is not None:
        synalinks.set_seed(int(cfg.seed))

    # Apply optional process-wide framework defaults BEFORE building any
    # rewards / programs, so reward specs that omit `language_model:` or
    # `embedding_model:` pick these up. The framework persists the identifier
    # into the framework's config file in the home folder as a side effect — that's the
    # library's documented behavior for string identifiers.
    if cfg.default_language_model is not None:
        synalinks.set_default_language_model(cfg.default_language_model)
    if cfg.default_embedding_model is not None:
        synalinks.set_default_embedding_model(cfg.default_embedding_model)

    model_ids = cfg.language_models
    dataset_names = cfg.selected_dataset_names()
    emit(
        {
            "event": "sweep_start",
            "models": list(model_ids),
            "datasets": list(dataset_names),
        }
    )
    # Every dataset is built from the config parsed once above, each from its
    # flat entry (`Config.dataset_entry`: a registry entry's `config`, its
    # provider kwargs and templates, beside its kind block). Re-reading
    # the file per dataset is O(datasets × filesize) — for a 12 MB /
    # 450-dataset config (~21 s/parse) that alone added hours to startup.
    datasets = {
        n: load_dataset_from_config(
            {"datasets": {n: _beside(cfg.dataset_entry(n), Path(config_path).parent)}}, name=n, yaml_path=str(config_path)
        )
        for n in dataset_names
    }

    # Pre-resolve the global metric list so unknown identifiers fail fast,
    # before the first trial.
    global_metrics = _instantiate_metrics(cfg.metrics, context="metrics")

    # One `GridSearch` tuner per dataset — model is the only HP
    # axis, dataset is a fixed search context. Per-dataset `project_name`
    # partitions the trial cache so changing one dataset's config doesn't
    # invalidate the others.
    cells: dict[tuple[str, str, str], float | None] = {}
    statuses: dict[tuple[str, str], str] = {}
    direction_by_ds: dict[str, str] = {}
    # Aliases collected in iteration order across datasets. Drives the
    # column order of the result matrix; per-alias direction is stored
    # here (not in `direction_by_ds`, which keys on dataset for the
    # primary reward column whose direction varies per dataset).
    metric_directions: dict[str, str] = {}
    metric_aliases: list[str] = []
    # `{ds: [(alias, direction), ...]}` for that dataset's tuner objectives.
    # Reused after the sweep to compute the per-dataset Pareto frontier.
    objectives_by_ds: dict[str, list[tuple[str, str]]] = {}

    # Every dataset is prepared — tuner, programs, callbacks — before any
    # trial runs, so the grid can then be walked model by model: all of one
    # model's cells, then the next model's. Dataset-major would make a local
    # server reload every model for every dataset.
    prepared: list[dict] = []
    for ds_name in dataset_names:
        emit({"event": "dataset_start", "dataset": ds_name})
        ds = datasets[ds_name]
        gen_kwargs = cfg.generator_kwargs(ds_name)
        agent_cfg = cfg.resolved_agent(ds_name)
        kind = cfg.dataset_kind(ds_name)
        # What each prediction holds: fails the run here, before any trial,
        # when returned inputs would clash with the answer's field names.
        block = {"model": gen_kwargs, "agent": agent_cfg}.get(kind)
        if kind == "tools":
            block = cfg.tools_kwargs(ds_name)
        answer_layout(ds, kind, block)
        # A rule-based reward compares the fields the output template renders.
        target = template_keys(getattr(ds, "output_template", None))
        reward = get_reward(_scope_to_answer(cfg.reward_spec(ds_name), target))
        reward_direction = cfg.reward_direction(ds_name)
        direction_by_ds[ds_name] = reward_direction
        ds_metrics = _merge_metrics(
            _instantiate_metrics(
                cfg.dataset_metrics(ds_name), context=f"datasets.{ds_name}.metrics", target=target
            ),
            global_metrics,
        )
        # `ds_metrics` (and its instances) is used only for the aliases /
        # directions / objectives metadata below. It must NOT be reused as the
        # per-model programs' metrics: operational metrics are stateful — each
        # binds to its program's LanguageModel on compile, and bind_program
        # OVERWRITES the previous binding — so a single instance shared across
        # the per-model programs ends up bound to the LAST model only, and every
        # other model's trial then reads that LM's counters and records zero ops
        # (tokens/latency/cost). Each program gets its own freshly-instantiated
        # set instead (below).

        # `programs` is filled *after* the tuner exists: on a resume the
        # tuner's reloaded trials say which cells are already scored, and
        # building a program is real work (an agent dataset opens every MCP
        # server it declares). kt's HP-discovery call and every real trial
        # then resolve `hypermodel(hp)` to a pre-built Program by lookup —
        # no async acrobatics inside a sync callback.
        programs: dict[str, synalinks.Program] = {}

        def hypermodel(hp, programs=programs):
            # kt's discovery probe calls this once at tuner construction, before
            # any program exists, and discards the result; every real trial
            # then resolves to a pre-built program.
            return programs.get(hp.Choice("language_model", values=model_ids))

        objectives = [synalinks.tuners.Objective("reward", direction=reward_direction)]
        objective_axes = [("reward", reward_direction)]
        for alias, _, direction, is_obj in ds_metrics:
            if is_obj:
                objectives.append(
                    synalinks.tuners.Objective(alias, direction=direction)
                )
                objective_axes.append((alias, direction))
        objectives_by_ds[ds_name] = objective_axes

        tuner = synalinks.tuners.GridSearch(
            hypermodel,
            objective=objectives if len(objectives) > 1 else objectives[0],
            max_trials=len(model_ids),
            directory=str(state_dir),
            project_name=ds_name,
            overwrite=no_cache,
            # kt gives up after 3 failures in a row, which on a sweep means
            # three flaky cells can cost the whole matrix — and a cell that
            # failed is a result too (`_collect_dataset_cells` records it).
            # Give up only once every model in the grid has failed back to
            # back, which is the case where something is genuinely broken.
            max_consecutive_failed_trials=max(3, len(model_ids)),
        )
        # `overwrite=no_cache` above wiped the project when the caller asked
        # for a clean run; otherwise the tuner has just reloaded whatever the
        # previous attempt left behind, and this decides what is left to do.
        scored, retrying = _resume_grid(tuner.oracle)
        pending = [m for m in model_ids if m not in scored]
        if scored or retrying:
            emit(
                {
                    "event": "dataset_resumed",
                    "dataset": ds_name,
                    "scored": scored,
                    "retrying": retrying,
                }
            )

        for model_id in pending:
            # Fresh, per-model metric instances (see note above): each program's
            # operational metrics must bind to that program's own LanguageModel.
            model_metrics = [
                m
                for _, m, _, _ in _merge_metrics(
                    _instantiate_metrics(
                        cfg.dataset_metrics(ds_name),
                        context=f"datasets.{ds_name}.metrics",
                        target=target,
                    ),
                    _instantiate_metrics(cfg.metrics, context="metrics", target=target),
                )
            ]
            lm_kwargs = cfg.lm_kwargs(model_id)
            # The axis names the model (a registry entry); the program calls its id.
            calls = cfg.model_id(model_id)
            if kind == "a2a":
                # The axis names the agent itself; there is no model id to call.
                programs[model_id] = await build_a2a(
                    model_id, ds, cfg.a2a_connection(model_id), reward,
                    metrics=model_metrics or None,
                )
            elif kind == "harbor":
                # Each row asked of a Harbor agent in a sandbox; the environments' folders were copied beside the run's config.
                # The reward goes along, serialized into a generated task's default tests/ (it scores here, on the arena's side).
                spec = {**cfg.harbor_spec(ds_name), "dataset": ds_name, "reward": cfg.reward_spec(ds_name), "reward_name": cfg.datasets[ds_name].reward}
                if spec.get("tasks"):
                    # Each task scored by the arena's reward system lists its rewards in tests/rewards.yaml: read and resolved once.
                    from openarena.datasets.harbor_tasks_dataset import task_folders
                    from openarena.programs.harbor_program import task_reward

                    entry = _beside(cfg.dataset_entry(ds_name), Path(config_path).parent)
                    spec["task_rewards"] = {
                        folder.name: scoring
                        for folder in task_folders(entry["path"], entry.get("tasks"))
                        if (scoring := task_reward(folder, cfg._resolve)) is not None
                    }
                programs[model_id] = await build_harbor(
                    calls, ds, spec, reward,
                    metrics=model_metrics or None,
                    contexts=Path(config_path).parent / "environments",
                    work_dir=Path(state_dir) / "harbor",
                    lm_kwargs=lm_kwargs,
                    model_name=model_id,
                )
            elif kind == "embedding":
                programs[model_id] = await build_embedding(
                    calls, ds, cfg.embedding_kwargs(ds_name), reward,
                    metrics=model_metrics or None,
                )
            elif kind == "tools":
                programs[model_id] = await build_tools(
                    calls, ds, cfg.tools_kwargs(ds_name), reward,
                    metrics=model_metrics or None, lm_kwargs=lm_kwargs,
                )
            elif agent_cfg is not None:
                programs[model_id] = await build_agent(
                    calls,
                    ds,
                    agent_cfg,
                    reward,
                    metrics=model_metrics or None,
                    lm_kwargs=lm_kwargs,
                )
            else:
                programs[model_id] = await build_program(
                    calls,
                    ds,
                    gen_kwargs,
                    reward,
                    metrics=model_metrics or None,
                    lm_kwargs=lm_kwargs,
                )
            if on_sample is not None and samples_per_cell > 0:
                _tee_samples(
                    programs[model_id],
                    lambda row, d=ds_name, m=model_id: on_sample(
                        {**row, "dataset": d, "model": m}
                    ),
                    limit=samples_per_cell,
                )

        batches_total = _num_batches(ds)
        emit(
            {
                "event": "dataset_ready",
                "dataset": ds_name,
                "batches_total": batches_total,
            }
        )

        prepared.append(
            {
                "name": ds_name,
                "ds": ds,
                "tuner": tuner,
                "pending": pending,
                "retrying": retrying,
                "ds_metrics": ds_metrics,
                "programs": programs,
            }
        )

    # One trial per pending cell, issued up front (`_hand_out`), then the
    # cells in turn, model-major; a dataset is viewed per cell so each cell
    # gets its own batch size.
    active = [item for item in prepared if item["pending"]]
    work: list[dict] = []
    for item in active:
        tuner = item["tuner"]
        tuner.on_search_begin()
        _hand_out(
            tuner.oracle, [m for m in item["pending"] if m not in item["retrying"]]
        )
        for n in range(len(item["pending"])):
            trial = tuner.oracle.create_trial(f"{tuner.tuner_id}/{n}")
            if trial.status != trial_module.TrialStatus.RUNNING:
                break
            work.append(
                {
                    "item": item,
                    "trial": trial,
                    "model": trial.hyperparameters.values["language_model"],
                }
            )
    work.sort(key=lambda c: (model_ids.index(c["model"]), prepared.index(c["item"])))

    async def run_cell(cell: dict) -> None:
        item, model_id = cell["item"], cell["model"]
        view = copy.copy(item["ds"])
        # The run's `batch_size` (else its evaluation's) overrides the dataset's own; with neither, one row.
        view.batch_size = batch_size or cfg.batch_size or view.batch_size or 1
        callbacks = [_ProgressCallback(emit, item["name"], model_id, view)]
        # A Harbor cell none of whose rows got an answer fails, ahead of the progress callback's `cell_end`.
        if (guard := no_answer_guard(item["programs"][model_id])) is not None:
            callbacks.insert(0, guard)
        if mlflow_experiment and synalinks.is_observability_enabled():
            callbacks.append(
                _RunMonitor(
                    mlflow_experiment,
                    mlflow_tracking_uri,
                    item["name"],
                    model_id,
                    mlflow_tags or {},
                    on_event=emit,
                )
            )
        emit(
            {
                "event": "cell_start",
                "dataset": item["name"],
                "model": model_id,
                "batches_total": _num_batches(view),
            }
        )
        await _run_cell(
            item["tuner"], cell["trial"], x=view, verbose=verbose, callbacks=callbacks
        )

    for cell in work:
        await run_cell(cell)
    for item in active:
        item["tuner"].on_search_end()
    # An A2A program keeps its agent's client for the grid; close them now that every cell has run.
    for item in prepared:
        for program in item["programs"].values():
            await close_program(program)

    for item in prepared:
        ds_name, ds_metrics = item["name"], item["ds_metrics"]
        ds_metric_keys = ["reward", *(alias for alias, _, _, _ in ds_metrics)]
        _collect_dataset_cells(
            item["tuner"].oracle, ds_name, ds_metric_keys, cells, statuses
        )
        emit(
            {
                "event": "dataset_end",
                "dataset": ds_name,
                "statuses": {m: statuses.get((m, ds_name)) for m in model_ids},
                "rewards": {m: cells.get((m, ds_name, "reward")) for m in model_ids},
                # Why a cell failed, so the run can say it rather than only that it did.
                "errors": _trial_errors(item["tuner"].oracle),
            }
        )
        for alias, _, direction, _ in ds_metrics:
            if alias not in metric_directions:
                metric_aliases.append(alias)
                metric_directions[alias] = direction

    metric_keys = ["reward", *metric_aliases]

    # Per-dataset Pareto frontier over each tuner's objective axes. Empty
    # list when the dataset has only one objective (frontier collapses to
    # the best row of the matrix — see ordering already in the TSV).
    frontier_by_ds = {
        ds_name: _pareto_frontier(cells, ds_name, model_ids, axes)
        if len(axes) > 1
        else []
        for ds_name, axes in objectives_by_ds.items()
    }

    rows = _build_rows(
        cells,
        statuses,
        model_ids,
        dataset_names,
        metric_keys,
        direction_by_ds=direction_by_ds,
        metric_directions=metric_directions,
        frontier_by_ds=frontier_by_ds,
    )
    meta = {
        "config": str(config_path),
        "models": list(model_ids),
        "datasets": list(dataset_names),
        "metrics": list(metric_keys),
        "directions": dict(direction_by_ds),
        "metric_directions": dict(metric_directions),
        "objectives_by_ds": {
            ds: [[a, d] for a, d in axes] for ds, axes in objectives_by_ds.items()
        },
        "frontier_by_ds": {ds: list(models) for ds, models in frontier_by_ds.items()},
    }
    emit({"event": "sweep_end"})
    return {"meta": meta, "rows": rows}


def _emit_json(result: dict, dest) -> None:
    """Write `result` as pretty-printed JSON with a trailing newline."""
    json.dump(result, dest, indent=2)
    dest.write("\n")
