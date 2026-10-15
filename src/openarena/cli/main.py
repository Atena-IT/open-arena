# License Apache 2.0: (c) 2026 Atena-Reply

"""The `openarena` command line, on click, printed with rich.

    openarena init [DIR] [--template NAME] [--force]
    openarena check [PATH] [--configs DIR]
    openarena evaluations list [--configs DIR]
    openarena evaluations run NAME [--configs DIR] [--state-dir DIR] [--no-cache] [--batch-size N] [--json FILE]
    openarena leaderboards list [--configs DIR]
    openarena leaderboards run NAME [--configs DIR] [--state-dir DIR] [--no-eol] [--json FILE]

`init` writes a ready-to-use workspace into DIR (the current folder by
default), laid out as the app keeps one: `configs/`, one registry per file
(models, rewards, datasets, MCP servers, A2A agents, environments,
evaluations, leaderboards), each field explained in comments and what is
optional given as commented examples; `tasks/<name>/`, Harbor task folders
(`task.toml`, `instruction.md`, `environment/`, `tests/test.sh`,
`solution/solve.sh`) that a `harbor_tasks` dataset runs; and
`environments/<name>/environment/`, an environment's build context; and,
for a coding agent, `AGENTS.md` (the guide it reads first, `CLAUDE.md` a
link to it), the engine's Agent Skills in
`.agents/skills/`, a `README.md`, `.env.template` and `.gitignore`. Every
file of `templates/<NAME>/` is copied, then every `skills/openarena-*`. It refuses to overwrite a file
already there unless `--force` is given, and writes nothing in that case.

`check` validates a configs folder, or one registry file, with the engine's
strict schema and prints each problem with its line and how to fix it
(`config/check.py`).

`evaluations run` launches the evaluation saved as NAME in the `configs/`
folder (`./configs` by default): the evaluation and the entries it names
are gathered into one config (the workspace's `.env` read first, never over
a variable already set), written to `<state-dir>/<NAME>/config.yaml`,
and swept; the result lands beside it in `result.json`. A Harbor
environment's task folder (`environments/<name>/` beside `configs/`, as in
a workspace) is copied beside the config, the build context its tasks get,
and so is a `harbor_tasks` dataset's folder of tasks (`tasks/`).
Trial state stays in
that folder, so running NAME again resumes where it stopped, unless
`--no-cache` is given.

`leaderboards run` reads the leaderboard saved as NAME (`leaderboards.yaml`:
a scope of datasets and models, and each metric's share of the score) over
every `result.json` under the state dir, newest first per cell, and prints
the ranking the leaderboard page shows (`src/leaderboard.py`).

Run it as a tool (`uv tool install`), or with `uv run` in this project:

    uv run openarena init my-arena
    uv run openarena check my-arena/configs
    uv run openarena evaluations run smoke --configs my-arena/configs
    uv run openarena leaderboards run overall --configs my-arena/configs
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

import click
import yaml
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from openarena.config.files import evaluation_config, load_yaml, registries

TEMPLATES = Path(__file__).parent / "templates"
# The engine's Agent Skills, one per action; `init` copies them into the workspace's `.agents/skills/`.
SKILLS = Path(__file__).parent.parent / "skills"
AGENT_SKILLS = Path(".agents") / "skills"
STATE_DIR = Path(".open-arena")

console = Console()


def templates() -> list[str]:
    """The template names, one folder each."""
    return sorted(p.name for p in TEMPLATES.iterdir() if p.is_dir())


def init(dest: Path, *, template: str = "default", force: bool = False) -> list[Path]:
    """Copy the workspace `templates/<template>/` into `dest`; the files written, in order.

    Raises `ValueError` for an unknown template, `FileExistsError` naming
    every file in the way when `force` is off.
    """
    source = TEMPLATES / template
    if not source.is_dir():
        raise ValueError(f"no template {template!r}; choose one of: {', '.join(templates())}")
    files = sorted(p for p in source.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    targets = [dest / f.relative_to(source) for f in files]
    # Every openarena skill, as the workspace's own: what a coding agent working in it reads.
    skills = sorted(p for p in SKILLS.glob("openarena-*/**/*") if p.is_file() and "__pycache__" not in p.parts)
    files += skills
    targets += [dest / AGENT_SKILLS / f.relative_to(SKILLS) for f in skills]
    # A link counts as in the way even when it leads nowhere: `exists()` alone would miss it.
    taken = [t for t in targets if t.exists() or t.is_symlink()]
    if taken and not force:
        raise FileExistsError(
            f"already in {dest}: " + ", ".join(str(t.relative_to(dest)) for t in taken) + " (use --force to overwrite)"
        )
    for f, t in zip(files, targets):
        t.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f, t)
        # Harbor runs a task's scripts as they are: an installed package may not have kept the bit.
        if t.suffix == ".sh":
            t.chmod(0o755)
    _link_claude_md(dest)
    return targets


def _link_claude_md(dest: Path) -> None:
    """CLAUDE.md as a link to AGENTS.md, one guide under both names, as the template holds it.

    A packaged template stores the link as a plain copy, so it is made again
    here; where links cannot be made (Windows without the right), the copy stays.
    """
    agents, claude = dest / "AGENTS.md", dest / "CLAUDE.md"
    if not agents.is_file():
        return
    try:
        if claude.is_symlink() or claude.exists():
            claude.unlink()
        claude.symlink_to("AGENTS.md")
    except OSError:
        shutil.copyfile(agents, claude)


def load_env(workspace: Path) -> bool:
    """The workspace's `.env` (beside `configs/`) into the environment, never over a variable already set; whether there was one."""
    from dotenv import load_dotenv

    path = workspace / ".env"
    return path.is_file() and load_dotenv(path, override=False)


def _entries(path: Path) -> list[str]:
    """The entry names of a registry file, `{section: {name: ...}}`."""
    raw = load_yaml(path) or {}
    return [name for body in raw.values() if isinstance(body, dict) for name in body]


def _section(configs: Path, section: str) -> dict[str, dict]:
    """The `{name: entry}` of one registry of `configs`."""
    return {name: body or {} for name, body in registries(configs).get(section, {}).items()}


def _saved(configs: Path, section: str, name: str) -> dict:
    """One entry of a registry; `KeyError` naming the choices when it is not there."""
    entries = _section(configs, section)
    if name not in entries:
        noun = section.rstrip("s")
        raise KeyError(f"no {noun} {name!r} in {configs}; choose one of: {', '.join(sorted(entries)) or '(none)'}")
    return entries[name]


def finished_runs(state_dir: Path) -> list[dict]:
    """Every run under `state_dir` with a `result.json`, newest first, as `leaderboard.board` reads them."""
    runs = []
    for result_path in state_dir.glob("*/result.json"):
        config_path = result_path.with_name("config.yaml")
        config = yaml.safe_load(config_path.read_text()) if config_path.is_file() else {}
        models = (config or {}).get("models") or {}
        runs.append(
            {
                "id": result_path.parent.name,
                "mtime": result_path.stat().st_mtime,
                "result": json.loads(result_path.read_text()),
                "model_ids": {n: m["model"] for n, m in models.items() if isinstance(m, dict) and "model" in m},
            }
        )
    runs.sort(key=lambda r: r["mtime"], reverse=True)
    return runs


def _progress(event: dict) -> None:
    kind = event.get("event")
    if kind == "dataset_start":
        console.print(f"[bold]{event.get('dataset', '')}[/]")
    elif kind == "cell_start":
        console.print(f"  [cyan]{event.get('model', '')}[/] running…")
    elif kind == "cell_end":
        metrics = event.get("metrics") or {}
        shown = ", ".join(f"{k} {v:.4f}" for k, v in metrics.items() if isinstance(v, (int, float)))
        console.print(f"  [cyan]{event.get('model', '')}[/] done {shown}")
    elif kind == "dataset_end":
        # Why a cell failed, said where it happened: a key, a model server, a template.
        for model, error in (event.get("errors") or {}).items():
            console.print(f"  [cyan]{model}[/] [red]failed[/]: {escape(str(error))}", soft_wrap=True)


def _results(result: dict) -> None:
    """One table per dataset: a row per model, a column per metric."""
    metrics = result["meta"]["metrics"]
    for dataset in result["meta"]["datasets"]:
        table = Table(title=dataset)
        table.add_column("model", style="cyan")
        for metric in metrics:
            table.add_column(metric, justify="right")
        for model in result["meta"]["models"]:
            cells = {r["metric"]: r for r in result["rows"] if r["dataset"] == dataset and r["model"] == model}
            values = [cells.get(m, {}).get("value") for m in metrics]
            table.add_row(model, *(f"{v:.4f}" if v is not None else "—" for v in values))
        console.print(table)


def _fmt(value: float) -> str:
    """Four decimals, as the results table has; a whole count (days, tokens) as one, never a reward's 0 or 1."""
    if value == int(value) and abs(value) >= 2:
        return f"{int(value):,}"
    return f"{value:.4f}"


def _ranking(name: str, leaderboard: dict, ranking: dict) -> None:
    """The ranked table, then the models in scope that could not be ranked and why."""
    title = leaderboard.get("title") or name
    console.print(f"[bold]{title}[/]" + (f" — {leaderboard['description']}" if leaderboard.get("description") else ""))
    console.print(f"Datasets: {', '.join(ranking['datasets']) or '(none)'}")
    console.print("Shares: " + ", ".join(f"{m} {share:.0f}%" for m, share in ranking["shares"].items()))
    table = Table()
    table.add_column("#", justify="right")
    table.add_column("model", style="cyan")
    table.add_column("score", justify="right", style="bold")
    for axis in ranking["axes"]:
        arrow = "↑" if axis["direction"] == "max" else "↓"
        table.add_column(f"{axis['name']} {arrow}", justify="right")
    table.add_column("frontier", justify="center")
    for row in ranking["rows"]:
        table.add_row(
            str(row["rank"]),
            row["model"],
            f"{row['score']:.4f}",
            *(_fmt(row["values"][a["name"]]) for a in ranking["axes"]),
            "★" if row["on_frontier"] else "",
        )
    console.print(table)
    for model in ranking["skipped"]:
        reasons = [
            f"not scored on {g['dataset']}" if g["missing"] is None else f"{g['dataset']} lacks {', '.join(g['missing'])}"
            for g in ranking["gaps"].get(model, [])
        ]
        console.print(f"[yellow]Not ranked[/] {model}: {'; '.join(reasons) or 'not scored in scope'}")
    for out in ranking.get("excluded") or []:
        broken = "; ".join(f"{b['metric']} {'at most' if b['op'] == 'lte' else 'at least'} {b['value']:g}" for b in out["failed"])
        console.print(f"[yellow]Over budget[/] {out['model']}: {broken}")


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
    console.print(f"Wrote [bold]{path}[/]")


configs_option = click.option(
    "--configs",
    default="configs",
    show_default=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="The configs folder the entries are saved in.",
)
state_dir_option = click.option(
    "--state-dir",
    default=str(STATE_DIR),
    show_default=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Where runs keep their config, trial state and result, one folder per evaluation.",
)


def _listing(configs: Path, section: str, columns: list[str], *, scopes: tuple[str, ...] = ()) -> None:
    """One row per entry of `section`; an empty list in a `scopes` column reads as `all`, as a saved scope means."""
    try:
        entries = _section(configs, section)
    except FileNotFoundError as e:
        raise click.ClickException(str(e)) from None
    table = Table()
    table.add_column("name", style="cyan")
    for column in columns:
        table.add_column(column)
    for name, entry in entries.items():
        cells = []
        for column in columns:
            value = entry.get(column)
            if column in scopes and not value:
                cells.append("[dim]all[/]")
            elif isinstance(value, list):
                cells.append(", ".join(map(str, value)))
            else:
                cells.append(str(value or ""))
        table.add_row(name, *cells)
    console.print(table)


@click.group()
def cli() -> None:
    """Open arena: evaluate models on datasets, and rank them on leaderboards."""


def _report(report) -> None:
    """One file's check: a line when it is clean, else each problem with where it is, what is wrong and how to fix it."""

    noun = report.section.rstrip("s").replace("_", " ")
    count = f"{len(report.names)} {noun}{'' if len(report.names) == 1 else 's'}"
    if not report.problems:
        console.print(f"[green]✓[/] {escape(str(report.path))}: {count}, no problems")
        return
    errors, warnings = len(report.errors), len(report.warnings)
    tally = [f"{n} {word}{'' if n == 1 else 's'}" for n, word in ((errors, "error"), (warnings, "warning")) if n]
    mark = "[red]✗[/]" if errors else "[yellow]![/]"
    # A file named after no registry holds none: only its problems are counted.
    said = ", ".join(tally) if report.section == "registry" else f"{count}, {', '.join(tally)}"
    console.print(f"{mark} {escape(str(report.path))}: {said}")
    pad = " " * 11
    for problem in report.problems:
        level = "[red]error  [/]" if problem.level == "error" else "[yellow]warning[/]"
        line = f"line {problem.line}  " if problem.line else ""
        console.print()
        console.print(f"  {level}  [dim]{line}[/][cyan]{escape(problem.where)}[/]", soft_wrap=True)
        console.print(f"{pad}{escape(problem.message)}", soft_wrap=True)
        if problem.fix:
            first, *rest = problem.fix.split("\n")
            console.print(f"     [green]fix[/]   {escape(first)}", soft_wrap=True)
            for more in rest:
                console.print(f"{pad}  {escape(more)}", soft_wrap=True, highlight=False)
    console.print()


# Every registry a `configs/` folder holds: its command name, what it is, the columns `list` shows.
REGISTRIES = {
    "models": ("models", "Models: the litellm id each name calls, and its params.", ["model", "type", "description"], ()),
    "rewards": ("rewards", "Rewards: a catalog reward, its kwargs, which way is better.", ["type", "direction", "description"], ()),
    "datasets": ("datasets", "Datasets: a provider, the Jinja templates over its rows, the reward that scores it.", ["type", "reward", "description"], ()),
    "mcp_servers": ("mcp-servers", "MCP servers: the tool servers agent datasets call.", ["transport", "command", "url"], ()),
    "a2a_agents": ("a2a-agents", "A2A agents: the remote agents an a2a sweep compares.", ["url", "description"], ()),
    "environments": ("environments", "Environments: the Harbor sandboxes a task's agent works in.", ["docker_image", "cpus", "memory_mb", "network_mode"], ()),
    "evaluations": ("evaluations", "Saved sweeps: every listed model on every listed dataset.", ["language_models", "datasets", "description"], ()),
    "leaderboards": (
        "leaderboards",
        "Saved ways of ranking the scores of finished runs.",
        ["title", "datasets", "models", "description"],
        ("datasets", "models"),
    ),
}


def _registry_group(section: str) -> click.Group:
    """`openarena <registry>` with its `list`; a registry's own commands are added to what this returns."""
    command, about, columns, scopes = REGISTRIES[section]
    group = click.Group(command, help=about)

    @group.command("list")
    @configs_option
    def list_command(configs: Path) -> None:
        _listing(configs, section, columns, scopes=scopes)

    list_command.help = f"List the {command.replace('-', ' ')} saved in the configs folder."

    cli.add_command(group)
    return group


@cli.command("init")
@click.argument("dest", default=".", type=click.Path(file_okay=False, path_type=Path))
@click.option(
    "--template",
    default="default",
    show_default=True,
    type=click.Choice(templates()),
    help="The template to start from.",
)
@click.option("--force", is_flag=True, help="Overwrite files already there.")
def init_command(dest: Path, template: str, force: bool) -> None:
    """Write a ready-to-use workspace into DEST (default: here): configs/, tasks/ and environments/."""
    try:
        written = init(dest, template=template, force=force)
    except FileExistsError as e:
        raise click.ClickException(str(e)) from None

    console.print(f"Wrote [bold]{dest}[/] from the [cyan]{template}[/] template:")
    table = Table()
    table.add_column("file", style="green")
    table.add_column("entries")
    skills = sorted({p.relative_to(dest / AGENT_SKILLS).parts[0] for p in written if p.is_relative_to(dest / AGENT_SKILLS)})
    for path in written:
        if path.is_relative_to(dest / AGENT_SKILLS):
            continue
        entries = _entries(path) if path.parent.name == "configs" and path.suffix == ".yaml" else []
        table.add_row(str(path.relative_to(dest)), ", ".join(entries))
    if skills:
        table.add_row(f"{AGENT_SKILLS}/", f"{len(skills)} skills: " + ", ".join(s.removeprefix("openarena-") for s in skills))
    console.print(table)

    steps = Text()
    commands = ([f"cd {dest}"] if dest.resolve() != Path.cwd() else []) + [
        "cp .env.template .env      # the keys your models need",
        "openarena check configs",
        "openarena evaluations run smoke",
    ]
    for i, command in enumerate(commands):
        steps.append(("\n" if i else "") + "$ ", style="bold cyan")
        steps.append(command, style="bold")
    console.print(Panel(steps, title="Next steps", title_align="left", border_style="cyan", padding=(1, 2), expand=False))
    console.print("Then start your coding agent (Claude Code, Codex, Cursor, ...) in the folder: AGENTS.md and .agents/skills/ guide it.")


for _section_name in REGISTRIES:
    _registry_group(_section_name)
evaluations = cli.commands["evaluations"]
leaderboards = cli.commands["leaderboards"]


@cli.command("check")
@click.argument("path", default="configs", type=click.Path(path_type=Path))
@click.option(
    "--configs",
    type=click.Path(file_okay=False, path_type=Path),
    help="For a file: the folder whose registries its references are checked against (default: the file's own).",
)
def check_command(path: Path, configs: Path | None) -> None:
    """Check PATH: a configs folder, every registry file in it (default: ./configs), or one registry file.

    Each problem is printed with its line, what is wrong and how to fix the
    YAML. A file's registry is read off its name only (`datasets.yaml`); a
    file with any other name is an error. Exits 1 on an error.
    """
    from openarena.config.check import check_file, check_folder

    if path.is_dir():
        reports = check_folder(path)
        if not reports:
            raise click.ClickException(f"no registry files in {path}; write some with `openarena init`")
    elif path.is_file():
        reports = [check_file(path, beside=configs)]
    else:
        raise click.ClickException(f"nothing at {path}; pass a configs folder or a registry file")
    for report in reports:
        _report(report)
    if any(r.errors for r in reports):
        raise SystemExit(1)


@evaluations.command("run")
@click.argument("name")
@configs_option
@state_dir_option
@click.option("--no-cache", is_flag=True, help="Start over instead of resuming a previous run.")
@click.option("--batch-size", type=click.IntRange(min=1), help="Override the evaluation's batch size.")
@click.option("--json", "json_out", type=click.Path(dir_okay=False, path_type=Path), help="Also write the result as JSON.")
@click.option("-v", "--verbose", count=True, help="Show progress (-v bar, -vv per batch).")
def evaluations_run(
    name: str, configs: Path, state_dir: Path, no_cache: bool, batch_size: int | None, json_out: Path | None, verbose: int
) -> None:
    """Run the evaluation saved as NAME."""
    from openarena.config import Config

    # The workspace's keys and endpoints, before anything reads them (a provider client, tracing).
    if load_env(configs.parent):
        console.print(f"Read [bold]{configs.parent / '.env'}[/]")
    try:
        raw = evaluation_config(configs, name)
        Config.model_validate(raw)
    except (FileNotFoundError, ValueError) as e:
        raise click.ClickException(str(e)) from None
    except KeyError as e:
        raise click.ClickException(e.args[0]) from None

    run_dir = state_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    # Each environment's task folder beside the config, as the app snapshots it: what the tasks build from.
    for environment in raw.get("environments") or {}:
        source, copy = configs.parent / "environments" / environment, run_dir / "environments" / environment
        # As it is now: a file removed since the last run must not linger in this one.
        shutil.rmtree(copy, ignore_errors=True)
        if source.is_dir():
            shutil.copytree(source, copy)
    # A `harbor_tasks` dataset's task folders, read from its `path` relative to the config, as the app copies them.
    for dataset in (raw.get("datasets") or {}).values():
        tasks = (dataset.get("config") or {}).get("path") if dataset.get("type") == "harbor_tasks" else None
        if tasks and not Path(tasks).is_absolute():
            source, copy = configs.parent / tasks, run_dir / tasks
            shutil.rmtree(copy, ignore_errors=True)
            if source.is_dir():
                shutil.copytree(source, copy)
    console.print(f"Running [bold]{name}[/] ({config_path})")

    # Imported here: the engine's imports are heavy, and `init` does not need them.
    from openarena.evaluate import run_sweep

    result = asyncio.run(
        run_sweep(
            str(config_path),
            no_cache=no_cache,
            verbose=verbose,
            state_dir=run_dir,
            on_event=_progress,
            batch_size=batch_size,
        )
    )
    (run_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    _results(result)
    if json_out is not None:
        _write_json(json_out, result)


@leaderboards.command("run")
@click.argument("name")
@configs_option
@state_dir_option
@click.option("--no-eol", is_flag=True, help="Leave out time to end of life, which looks dates up online.")
@click.option("--json", "json_out", type=click.Path(dir_okay=False, path_type=Path), help="Also write the ranking as JSON.")
def leaderboards_run(name: str, configs: Path, state_dir: Path, no_eol: bool, json_out: Path | None) -> None:
    """Rank the models on the leaderboard saved as NAME, from the finished runs."""
    from openarena import leaderboard as lb

    try:
        saved = _saved(configs, "leaderboards", name)
    except FileNotFoundError as e:
        raise click.ClickException(str(e)) from None
    except KeyError as e:
        raise click.ClickException(e.args[0]) from None
    runs = finished_runs(state_dir)
    if not runs:
        raise click.ClickException(f"no finished runs under {state_dir}; run one with `openarena evaluations run`")
    data = lb.board(runs, eol=not no_eol)
    ranking = lb.rank(
        data,
        datasets=saved.get("datasets") or None,
        models=saved.get("models") or None,
        shares=saved.get("shares") or None,
        budgets=saved.get("budgets") or None,
        group_shares=saved.get("group_shares") or None,
        dataset_shares=saved.get("dataset_shares") or None,
    )
    _ranking(name, saved, ranking)
    if json_out is not None:
        _write_json(json_out, {"leaderboard": {"name": name, **saved}, **ranking})


if __name__ == "__main__":
    cli()
