# License Apache 2.0: (c) 2026 Athena-Reply

"""Run the open-arena evaluation sweep.

The program graph lives in `src/program.py`. Two entry points there:
`build_program()` (Generator-based, used when a dataset declares
`generator:`) and `build_agent()` (FunctionCallingAgent + MCP tools,
used when a dataset declares `agent:`). The two are mutually
exclusive per dataset; this file dispatches between them based on
which block is set in YAML. The `arena` console script (installed
by `uv sync` via the `[project.scripts]` entry in `pyproject.toml`)
resolves to `main()` below.

Run with:

    uv run arena -c config.yaml
    uv run arena -v                # show synalinks progress bar
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import synalinks

synalinks.disable_keras_backend()  # MUST precede `import keras_tuner`

import click  # noqa: E402
from synalinks.src.utils.naming import to_snake_case  # noqa: E402

from src.config import Config, MetricEntry  # noqa: E402
from src.datasets import load_dataset_from_yaml  # noqa: E402
from src.program import build_agent, build_program  # noqa: E402
from src.rewards import _REWARD_TYPES, get as get_reward  # noqa: E402


# Default directory for trial state + output TSVs. Override per run with
# `--state-dir` so multiple runs can be kept side by side without clobbering
# each other's trial cache.
DEFAULT_STATE_DIR = Path(".open-arena")
TSV_NAME = "last_run.tsv"
FRONTIER_TSV_NAME = "frontier.tsv"


def _instantiate_metrics(entries: list[MetricEntry], *, context: str):
    """Resolve validated `MetricEntry` rows into `[(alias, Metric, direction, is_objective)]`.

    Identifiers resolve in two registries: the project rewards (auto-wrapped
    in `MeanMetricWrapper` so they ride the primary `evaluate()` pass — no
    extra LM calls per "reward") and `synalinks.metrics.get(...)` for plain
    metrics. `name=` on the wrapper flows to the wrapping metric (whose
    `.name` is the evaluate() result-dict key), not the inner reward —
    that's how the user's `alias:` survives into the matrix.
    """
    out = []
    for entry in entries:
        key = to_snake_case(entry.class_name)
        if key in _REWARD_TYPES:
            reward = get_reward({"name": key, **entry.kwargs})
            instance = synalinks.metrics.MeanMetricWrapper(
                reward, name=entry.alias or key,
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
                f"not a known synalinks metric or reward class."
            )
        resolved_alias = entry.alias or getattr(instance, "name", None) or entry.class_name
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
    obj_label = ", ".join(f"{alias} {arrow[direction]}" for alias, direction in objectives)
    print(f"### {ds_name} — Pareto frontier  ({obj_label})")
    print()

    if not frontier_models:
        print("_(no Pareto frontier: all candidate trials had missing objective values)_")
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

    widths = [max(len(headers[i]), *(len(r[i]) for r in table_rows)) for i in range(len(headers))]

    def _fmt(row):
        first = row[0].ljust(widths[0])
        rest = [c.rjust(widths[i + 1]) for i, c in enumerate(row[1:])]
        return "| " + " | ".join([first, *rest]) + " |"

    sep = ["-" * widths[0]] + ["-" * (widths[i] - 1) + ":" for i in range(1, len(headers))]
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
                direction = primary_dir if k == "reward" else metric_directions.get(k, "max")
                rows.append({
                    "model": m,
                    "dataset": d,
                    "metric": k,
                    "value": cells.get((m, d, k)),
                    "status": status,
                    "direction": direction,
                    "on_frontier": on_frontier,
                })
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
        d: {a for a, _ in objs}
        for d, objs in objectives_by_ds.items() if len(objs) > 1
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
) -> dict:
    """Run the full sweep and return the JSON-serializable result dict.

    Programmatic entry point (e.g. for an API endpoint that needs the
    matrix without going through stdout). One `synalinks.tuners.GridSearch`
    runs per dataset over `Choice("language_model", ...)` — model is the
    only HP axis; the dataset is a fixed search context. Each tuner
    persists trial state under `<state_dir>/<dataset>/` (default
    `.open-arena`, override with `state_dir`), the same cache the CLI
    reads, so resumed runs work and distinct `state_dir`s never collide.

    Programs are built eagerly per `(model, dataset)` cell inside the
    async context, then handed to the tuner via a sync `hypermodel`
    closure that just looks them up — no awaitable acrobatics needed
    around kt's sync HP-discovery probe. `tuner.search(...)` itself is
    safe to call from inside `asyncio.run(...)` because the synalinks
    tuner subclass uses `run_maybe_nested` to bridge to its async
    `_run_trial_async`.

    `no_cache=True` drops every per-dataset cache before starting — use
    it when the model list or any dataset's reward / metric set changed.
    `verbose` is forwarded to `synalinks.Program.evaluate` (0 silent,
    1 progress bar, 2 per-batch).
    """
    cfg = Config.load(config_path)

    # Optional process-wide reproducibility seed. `synalinks.set_seed`
    # seeds both `numpy.random` and `random` — LM sampling temperature is
    # not affected (that's the provider's RNG).
    if cfg.seed is not None:
        synalinks.set_seed(int(cfg.seed))

    # Apply optional process-wide synalinks defaults BEFORE building any
    # rewards / programs, so reward specs that omit `language_model:` or
    # `embedding_model:` pick these up. Synalinks persists the identifier
    # into `~/.synalinks/synalinks.json` as a side effect — that's the
    # library's documented behavior for string identifiers.
    if cfg.default_language_model is not None:
        synalinks.set_default_language_model(cfg.default_language_model)
    if cfg.default_embedding_model is not None:
        synalinks.set_default_embedding_model(cfg.default_embedding_model)

    model_ids = cfg.experiments.language_models
    dataset_names = cfg.selected_dataset_names()
    datasets = {n: load_dataset_from_yaml(config_path, name=n) for n in dataset_names}

    # Pre-resolve the global metric list so unknown identifiers fail fast,
    # before the first trial.
    global_metrics = _instantiate_metrics(cfg.metrics, context="metrics")

    # One `synalinks.tuners.GridSearch` per dataset — model is the only HP
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

    for ds_name in dataset_names:
        ds = datasets[ds_name]
        gen_kwargs = cfg.generator_kwargs(ds_name)
        agent_cfg = cfg.resolved_agent(ds_name)
        reward = get_reward(cfg.reward_spec(ds_name))
        reward_direction = cfg.reward_direction(ds_name)
        direction_by_ds[ds_name] = reward_direction
        ds_metrics = _merge_metrics(
            _instantiate_metrics(
                cfg.dataset_metrics(ds_name), context=f"datasets.{ds_name}.metrics"
            ),
            global_metrics,
        )
        metric_instances = [m for _, m, _, _ in ds_metrics]

        # Build one program per language_model upfront. kt's HP-discovery
        # call and every real trial then resolve `hypermodel(hp)` to a
        # pre-built Program by lookup — no async acrobatics inside a sync
        # callback.
        programs: dict[str, synalinks.Program] = {}
        for model_id in model_ids:
            if agent_cfg is not None:
                programs[model_id] = await build_agent(
                    model_id, ds, agent_cfg, reward,
                    metrics=metric_instances or None,
                )
            else:
                programs[model_id] = await build_program(
                    model_id, ds, gen_kwargs, reward,
                    metrics=metric_instances or None,
                )

        def hypermodel(hp, programs=programs):
            model_id = hp.Choice("language_model", values=model_ids)
            return programs[model_id]

        objectives = [synalinks.tuners.Objective("reward", direction=reward_direction)]
        objective_axes = [("reward", reward_direction)]
        for alias, _, direction, is_obj in ds_metrics:
            if is_obj:
                objectives.append(synalinks.tuners.Objective(alias, direction=direction))
                objective_axes.append((alias, direction))
        objectives_by_ds[ds_name] = objective_axes

        tuner = synalinks.tuners.GridSearch(
            hypermodel,
            objective=objectives if len(objectives) > 1 else objectives[0],
            max_trials=len(model_ids),
            directory=str(state_dir),
            project_name=ds_name,
            overwrite=no_cache,
        )
        # The compiled programs have no optimizer (sweep is evaluate-only),
        # so synalinks's tuner dispatches `_run_trial_async` to
        # `program.evaluate(x=ds, verbose=...)` instead of `fit()` — no
        # wasted training-loop cycles.
        tuner.search(x=ds, verbose=verbose)

        ds_metric_keys = ["reward", *(alias for alias, _, _, _ in ds_metrics)]
        _collect_dataset_cells(tuner.oracle, ds_name, ds_metric_keys, cells, statuses)

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
        if len(axes) > 1 else []
        for ds_name, axes in objectives_by_ds.items()
    }

    rows = _build_rows(
        cells, statuses, model_ids, dataset_names, metric_keys,
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
    return {"meta": meta, "rows": rows}


def _emit_json(result: dict, dest) -> None:
    """Write `result` as pretty-printed JSON with a trailing newline."""
    json.dump(result, dest, indent=2)
    dest.write("\n")


@click.group(
    invoke_without_command=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.pass_context
@click.option(
    "-c",
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    default="config.yaml",
    show_default=True,
    help="Path to the YAML config.",
)
@click.option(
    "--state-dir",
    "state_dir",
    type=click.Path(file_okay=False, writable=True, path_type=Path),
    default=DEFAULT_STATE_DIR,
    show_default=True,
    help=(
        "Directory for trial state and the output TSVs. Use a distinct "
        "directory per run to keep multiple runs side by side without "
        "their trial caches colliding."
    ),
)
@click.option(
    "--no-cache",
    is_flag=True,
    help=(
        "Discard every per-dataset trial cache under the state dir before "
        "running. Use this when the model list or any dataset's reward / "
        "metric set has changed since the last run."
    ),
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help=(
        "Show the per-trial progress bar. Repeat for more detail: `-v` is "
        "synalinks's progress bar (verbose=1), `-vv` is per-batch lines "
        "(verbose=2, recommended when piping to a log file)."
    ),
)
@click.option(
    "--json",
    "json_out",
    type=click.Path(dir_okay=False, writable=True, allow_dash=True),
    default=None,
    help=(
        "Write the result matrix as JSON to PATH (use `-` for stdout). "
        "Suppresses the markdown tables and TSV when set to `-`; otherwise "
        "writes alongside them."
    ),
)
def main(
    ctx,
    config_path: str,
    state_dir: Path,
    no_cache: bool,
    verbose: int,
    json_out: str | None,
) -> None:
    """Run the local sweep or API subcommands."""
    if ctx.invoked_subcommand is None:
        asyncio.run(_run(config_path, state_dir, no_cache, verbose, json_out))


@main.command("serve")
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8000, show_default=True, type=int)
def serve_api(host: str, port: int) -> None:
    """Serve the Open Arena REST API."""
    import uvicorn

    uvicorn.run("src.api.app:app", host=host, port=port, reload=False)


@main.command("request")
@click.argument("method")
@click.argument("path")
@click.option("--server", "server_url", default="http://127.0.0.1:8000", show_default=True)
@click.option("--token", default=lambda: os.getenv("OPEN_ARENA_API_TOKEN", "open-arena-dev-token"), show_default="OPEN_ARENA_API_TOKEN or open-arena-dev-token")
@click.option("--file", "file_path", type=click.Path(exists=True, dir_okay=False, readable=True))
def request_api(method: str, path: str, server_url: str, token: str, file_path: str | None) -> None:
    """Send an authenticated API request and print JSON."""
    from src.api.client import ArenaAPIClient

    client = ArenaAPIClient(server_url, token=token)
    result = client.request_file(method, path, file_path)
    if result is not None:
        click.echo(json.dumps(result, indent=2))

async def _run(
    config_path: str,
    state_dir: Path,
    no_cache: bool,
    verbose: int,
    json_out: str | None,
) -> None:
    result = await run_sweep(
        config_path, no_cache=no_cache, verbose=verbose, state_dir=state_dir
    )
    meta, rows = result["meta"], result["rows"]

    # `--json -` is the API-style invocation: emit one JSON document on
    # stdout and skip the human-facing table / TSV outputs entirely.
    if json_out == "-":
        _emit_json(result, sys.stdout)
        return

    for ds_name, objectives in meta["objectives_by_ds"].items():
        if len(objectives) > 1:
            _render_pareto_frontier(
                rows, ds_name, objectives, meta["frontier_by_ds"].get(ds_name, []),
            )

    tsv_path = state_dir / TSV_NAME
    _write_tsv(rows, tsv_path)
    print(f"wrote {tsv_path}")

    # Only mention frontier.tsv when there's at least one multi-objective
    # dataset — otherwise it'd be header-only and the print noise is misleading.
    if any(len(o) > 1 for o in meta["objectives_by_ds"].values()):
        frontier_path = state_dir / FRONTIER_TSV_NAME
        _write_frontier_tsv(rows, meta["objectives_by_ds"], frontier_path)
        print(f"wrote {frontier_path}")

    if json_out:
        json_path = Path(json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open("w") as f:
            _emit_json(result, f)
        print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
