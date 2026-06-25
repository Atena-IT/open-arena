# License Apache 2.0: (c) 2026 Athena-Reply
"""``open_arena_cli.main`` -- Top-level Click group for the ``arena`` console script.

The ``main`` group hosts:
- Default subcommand (sweep) -- **lazy-imports** the engine; requires ``open-arena``
- ``serve`` -- **lazy-imports** the engine; requires ``open-arena``
- ``request`` -- thin HTTP command; requires only ``open_arena_core``
- All resource sub-groups from ``open_arena_cli.groups`` (env/verifier/leaderboard/run/discover)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import click

from open_arena_cli.groups import CLI_GROUPS


def _require_engine(feature: str) -> None:
    """Raise a friendly ClickException if the engine is not installed."""
    try:
        import src.evaluate  # noqa: F401
    except ImportError as exc:
        msg = (
            f"{feature} requires the full Open Arena engine." + chr(10) +
            "Install it with:  pip install open-arena" + chr(10) +
            f"(original error: {exc})"
        )
        raise click.ClickException(msg) from exc


DEFAULT_STATE_DIR = Path(".open-arena")


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
        "running. Use this when the model list or any dataset reward / "
        "metric set has changed since the last run."
    ),
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help=(
        "Show the per-trial progress bar. Repeat for more detail: -v is "
        "synalinks progress bar (verbose=1), -vv is per-batch lines "
        "(verbose=2, recommended when piping to a log file)."
    ),
)
@click.option(
    "--json",
    "json_out",
    type=click.Path(dir_okay=False, writable=True, allow_dash=True),
    default=None,
    help=(
        "Write the result matrix as JSON to PATH (use - for stdout). "
        "Suppresses the markdown tables and TSV when set to -; otherwise "
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
    """Run the local sweep or API subcommands.

    When invoked without a subcommand the local evaluation sweep runs.
    This requires the full engine: pip install open-arena.
    """
    if ctx.invoked_subcommand is None:
        _require_engine("Local sweep")
        import asyncio
        from src.evaluate import (
            run_sweep,
            _emit_json,
            _render_pareto_frontier,
            _write_tsv,
            _write_frontier_tsv,
            TSV_NAME,
            FRONTIER_TSV_NAME,
        )

        async def _run() -> None:
            result = await run_sweep(
                config_path, no_cache=no_cache, verbose=verbose, state_dir=state_dir
            )
            meta, rows = result["meta"], result["rows"]
            if json_out == "-":
                import sys
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

        asyncio.run(_run())


@main.command("serve")
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8000, show_default=True, type=int)
def serve_api(host: str, port: int) -> None:
    """Serve the Open Arena REST API.

    Requires the full engine: pip install open-arena.
    """
    _require_engine("arena serve")
    import uvicorn

    uvicorn.run("src.api.app:app", host=host, port=port, reload=False)


@main.command("request")
@click.argument("method")
@click.argument("path")
@click.option("--server", "server_url", default="http://127.0.0.1:8000", show_default=True)
@click.option(
    "--token",
    default=lambda: os.getenv("OPEN_ARENA_API_TOKEN", "open-arena-dev-token"),
    show_default="OPEN_ARENA_API_TOKEN or open-arena-dev-token",
)
@click.option("--file", "file_path", type=click.Path(exists=True, dir_okay=False, readable=True))
def request_api(method: str, path: str, server_url: str, token: str, file_path: str | None) -> None:
    """Send an authenticated API request and print JSON.

    This is a thin command -- works without the engine installed.
    """
    from open_arena_core.client import ArenaAPIClient

    client = ArenaAPIClient(server_url, token=token)
    result = client.request_file(method, path, file_path)
    if result is not None:
        click.echo(json.dumps(result, indent=2))


# Register the resource sub-groups so they are all reachable
# under the arena console-script entry-point:
#   arena env list|get|create|delete
#   arena verifier list|get|create|delete
#   arena leaderboard list|get|create|delete|models|environments|entries
#   arena run submit|get|results|list
#   arena discover metric-kinds|aggregations|model-providers|dataset-providers
for _g in CLI_GROUPS:
    main.add_command(_g)


if __name__ == "__main__":
    main()
