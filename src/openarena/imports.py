# License Apache 2.0: (c) 2026 Atena-Reply

"""Benchmarks scored elsewhere, read as one more run of the board.

An import has the shape of a run's `result.json`: `rows` of `{model,
dataset, metric, value, direction}`, each `(model, dataset)` cell with a
`reward` row, plus optional `ci_low`, `ci_high` and `n` per row, and a
`meta` naming the import and where it came from. Anything else is refused
here; `openarena.import_readers` reads the shapes other tools write into
this one. Datasets are read as
`<import>/<dataset>`, so an imported one never collides with a dataset of
the workspace.
"""

from __future__ import annotations

import json
import logging
import math
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from openarena.config.entries import Direction, Name, check_shares, check_weights
from openarena.config.tags import Tags
from openarena.leaderboard import default_datasets

_log = logging.getLogger(__name__)

# ponytail: fixed caps, a setting if a real import outgrows them.
MAX_BYTES = 20 * 1024 * 1024
MAX_ROWS = 200_000

Text = Annotated[str, Field(min_length=1, max_length=200)]
# Shown as a link in the app, so only the web's own schemes.
Url = Annotated[str, Field(max_length=500, pattern=r"^https?://\S+$")]


class ImportMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Name = Field(description="The import's name, and the prefix of its datasets.")
    title: Text
    description: str | None = Field(default=None, max_length=2000)
    source: Text | None = Field(
        default=None, description="The tool or run the scores come from."
    )
    protocol: Text | None = None
    hash: Text | None = Field(
        default=None,
        description="The protocol's or the run's hash, for tracing it back.",
    )
    provisional: bool = False
    tags: Tags = Field(
        default_factory=list,
        description="Labels for browsing, as on every arena entry; the import's leaderboard carries them.",
    )
    url: Url | None = Field(
        default=None, description="Where the results or their report live."
    )
    evaluated_at: Text | None = Field(
        default=None,
        description="When the scores were computed, ISO 8601 (`2026-10-05`).",
    )
    version: Text | None = Field(
        default=None, description="The benchmark's or the harness's version."
    )
    license: Text | None = Field(
        default=None, description="Under what terms the results may be shared."
    )
    authors: Text | None = Field(
        default=None, description="Who ran it, or whom to ask."
    )


class ImportDataset(BaseModel):
    """What one dataset (task) of the import is, beyond its scores."""

    model_config = ConfigDict(extra="forbid")

    description: str | None = Field(default=None, max_length=2000)
    group: Text | None = Field(
        default=None,
        description="The suite or criterion it belongs to (`italian_knowledge`).",
    )
    tags: Tags = Field(
        default_factory=list,
        description="Language, domain, license status of the data (`italian`, `non-commercial`).",
    )
    url: Url | None = None
    license: Text | None = Field(default=None, description="The dataset's own license.")


class ImportModel(BaseModel):
    """What one model of the import is, beyond its scores."""

    model_config = ConfigDict(extra="forbid")

    model_id: Text | None = Field(
        default=None,
        description="Its litellm id when it is one (`openrouter/qwen/qwen3-32b`), to match an arena model later.",
    )
    description: str | None = Field(default=None, max_length=2000)
    tags: Tags = Field(
        default_factory=list,
        description="Size, family, settings (`27b`, `reasoning-high`).",
    )
    url: Url | None = None


class ImportLeaderboard(BaseModel):
    """How to read the imported scores as a leaderboard: what an export carries, so a re-import gives the same board."""

    model_config = ConfigDict(extra="forbid")

    name: Name | None = Field(
        default=None,
        description="The saved leaderboard's name; the import's when absent.",
    )
    title: Text | None = None
    description: str | None = Field(default=None, max_length=2000)
    tags: Tags = Field(default_factory=list)
    datasets: list[Text] = Field(
        default_factory=list,
        description="Datasets of `rows` in scope; empty means all of them.",
    )
    models: list[Text] = Field(
        default_factory=list,
        description="Models of `rows` in scope; empty means all of them.",
    )
    shares: dict[str, float] = Field(
        default_factory=dict,
        description="Percent of the score per metric; empty means the default preset.",
    )
    group_shares: dict[str, float] = Field(
        default_factory=dict,
        description="Weight per group (`datasets.<name>.group`, or a dataset in no group); left out weighs 1, 0 leaves it out of the score.",
    )
    dataset_shares: dict[str, float] = Field(
        default_factory=dict,
        description="Weight per dataset of `rows` within its group; left out weighs 1, 0 leaves it out of the score.",
    )

    # The rules a saved leaderboard is held to, checked before the import is stored rather than after.
    @field_validator("shares")
    @classmethod
    def _shares(cls, shares: dict[str, float]) -> dict[str, float]:
        return check_shares(shares)

    @field_validator("group_shares", "dataset_shares")
    @classmethod
    def _weights(cls, weights: dict[str, float]) -> dict[str, float]:
        return check_weights(weights)


class ImportChart(BaseModel):
    """A chart the import asks the app to draw, from a fixed vocabulary: a spec, never code.

    Values come from `rows`, aggregated per model over the datasets in the
    board's scope, so a chart can only show what was imported.

    - `bar`: one bar per model.
    - `stacked-bar`: that bar cut into segments by `stack`, each dataset's
      `group` (from `datasets`) or the dataset itself.
    - `heatmap`: models against `stack` keys, one cell per pair.
    - `scatter`: one point per model, `x` against `metric`, with the Pareto frontier.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["bar", "stacked-bar", "heatmap", "scatter"]
    title: Text
    description: str | None = Field(default=None, max_length=500)
    metric: Text = Field(description="A metric of `rows` (`reward` included): the bars, the cells, or the scatter's y.")
    x: Text | None = Field(default=None, description="`scatter` only: the metric on the x axis.")
    aggregate: Literal["sum", "mean"] = Field(default="mean", description="How a model's cells add up: `sum` for time or cost, `mean` for a score.")
    stack: Literal["group", "dataset"] | None = Field(default=None, description="`stacked-bar` and `heatmap`: what cuts a bar into segments, or makes the columns.")
    unit: Annotated[str, Field(max_length=12)] | None = Field(default=None, description="Shown after the values (`h`, `ms`, `%`).")

    @model_validator(mode="after")
    def _fields_fit_the_type(self) -> ImportChart:
        if (self.type in ("stacked-bar", "heatmap")) != (self.stack is not None):
            raise ValueError("`stack` is set on a `stacked-bar` or a `heatmap`, and only there")
        if (self.type == "scatter") != (self.x is not None):
            raise ValueError("`x` is set on a `scatter`, and only there")
        return self


class ImportRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: Text
    dataset: Text
    metric: Text
    value: float
    direction: Direction
    ci_low: float | None = None
    ci_high: float | None = None
    n: int | None = Field(default=None, ge=0)

    @field_validator("value", "ci_low", "ci_high")
    @classmethod
    def _finite(cls, v: float | None) -> float | None:
        if v is not None and not math.isfinite(v):
            raise ValueError("must be a finite number")
        return v


class ImportFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meta: ImportMeta
    rows: list[ImportRow] = Field(min_length=1, max_length=MAX_ROWS)
    datasets: dict[str, ImportDataset] = Field(
        default_factory=dict, description="Per dataset of `rows`, what it is."
    )
    models: dict[str, ImportModel] = Field(
        default_factory=dict, description="Per model of `rows`, what it is."
    )
    charts: list[ImportChart] = Field(default_factory=list, max_length=12, description="Charts to draw from `rows`, beside the arena's own.")
    leaderboard: ImportLeaderboard | None = Field(
        default=None,
        description="The leaderboard to make over it; a default one when absent.",
    )

    @model_validator(mode="after")
    def _described_only_what_is_scored(self) -> ImportFile:
        scope = self.leaderboard or ImportLeaderboard()
        for key, scored in (
            ("datasets", {r.dataset for r in self.rows}),
            ("models", {r.model for r in self.rows}),
        ):
            for where, named in (
                (key, set(getattr(self, key))),
                (f"leaderboard.{key}", set(getattr(scope, key))),
            ):
                stray = sorted(named - scored)
                if stray:
                    raise ValueError(
                        f"`{where}` names {stray[:3]}, which no row scores: check the spelling against `rows`"
                    )
        # A weight on a name the board never sees would count for nothing, silently.
        scored = {r.dataset for r in self.rows}
        grouped = {d: v.group for d, v in self.datasets.items() if v.group}
        for where, named, known in (
            ("leaderboard.dataset_shares", set(scope.dataset_shares), scored),
            ("leaderboard.group_shares", set(scope.group_shares), {grouped.get(d, d) for d in scored}),
        ):
            stray = sorted(named - known)
            if stray:
                raise ValueError(f"`{where}` names {stray[:3]}, which is none of {sorted(known)[:8]}: check the spelling")
        return self

    @model_validator(mode="after")
    def _charts_read_what_was_imported(self) -> ImportFile:
        metrics = {r.metric for r in self.rows}
        for i, chart in enumerate(self.charts):
            for plotted in filter(None, (chart.metric, chart.x)):
                if plotted not in metrics:
                    raise ValueError(f"charts[{i}] plots {plotted!r}, which no row records: one of {sorted(metrics)[:8]}")
            if chart.stack == "group":
                ungrouped = sorted({r.dataset for r in self.rows} - {d for d, v in self.datasets.items() if v.group})
                if ungrouped:
                    raise ValueError(f"charts[{i}] stacks by group, but {ungrouped[:3]} have no `datasets.<name>.group`")
        return self

    @model_validator(mode="after")
    def _one_value_per_measurement(self) -> ImportFile:
        seen: set[tuple[str, str, str]] = set()
        for i, r in enumerate(self.rows):
            key = (r.model, r.dataset, r.metric)
            if key in seen:
                raise ValueError(
                    f"rows[{i}] repeats model {r.model!r}, dataset {r.dataset!r}, metric {r.metric!r}: "
                    "one value per measurement; name the variants apart (`exact_match@strict-match`)"
                )
            seen.add(key)
        return self

    @model_validator(mode="after")
    def _every_cell_has_a_reward(self) -> ImportFile:
        cells = {(r.model, r.dataset) for r in self.rows}
        rewarded = {(r.model, r.dataset) for r in self.rows if r.metric == "reward"}
        missing = sorted(cells - rewarded)
        if missing:
            model, dataset = missing[0]
            raise ValueError(
                f"{len(missing)} cell(s) have no `reward` row, e.g. model {model!r} on dataset {dataset!r}: "
                "add a row with `metric: reward` holding the cell's primary score"
            )
        return self


def _not_json(constant: str) -> float:
    raise ValueError(
        f"{constant} is not a JSON number: write a finite value or leave the row out"
    )


def parse(raw: bytes) -> ImportFile:
    """A file's bytes as an import, or `ValueError` / `pydantic.ValidationError` saying what to fix."""
    if len(raw) > MAX_BYTES:
        raise ValueError(f"the file is {len(raw)} bytes, over the {MAX_BYTES} limit")
    data = json.loads(raw, parse_constant=_not_json)
    return ImportFile.model_validate(data)


def run_of(imported: ImportFile) -> dict:
    """The import as a run `board()` reads, its id `import:<name>`; a model's `model_id`, when given, is the id it called."""
    prefix = imported.meta.name
    rows = [{**r.model_dump(), "dataset": f"{prefix}/{r.dataset}"} for r in imported.rows]
    model_ids = {name: m.model_id for name, m in imported.models.items() if m.model_id}
    # Each dataset's group, for the board's weights by group and its view by group: prefixed like
    # the datasets, so two imports' `reasoning` stay two groups, and a group never takes the name
    # of an arena dataset.
    groups = {f"{prefix}/{d}": f"{prefix}/{v.group}" for d, v in imported.datasets.items() if v.group}
    return {"id": f"import:{prefix}", "result": {"rows": rows}, "model_ids": model_ids, "groups": groups}


# Each import parsed, and its run, by path, kept while the file is unchanged (mtime, size): the
# leaderboard reads every import on every poll. An entry goes when its file does.
# Handlers run on a thread pool, so every touch of it holds the lock; least recently used goes first
# past `MAX_CACHED`, so the imports of workspaces nobody reads do not stay in memory.
_LOADED: OrderedDict[Path, tuple[tuple[int, int], ImportFile, dict | None]] = OrderedDict()
_LOCK = threading.Lock()
# ponytail: a count, not bytes; an import is at most `MAX_BYTES`, so this bounds memory too.
MAX_CACHED = 32


def _cached(path: Path) -> tuple[tuple[int, int], ImportFile, dict | None]:
    stat = path.stat()
    key = (stat.st_mtime_ns, stat.st_size)
    with _LOCK:
        cached = _LOADED.get(path)
        if cached and cached[0] == key:
            _LOADED.move_to_end(path)
            return cached
    entry = (key, parse(path.read_bytes()), None)  # parsed outside the lock: it can take a while
    with _LOCK:
        _LOADED[path] = entry
        _LOADED.move_to_end(path)
        while len(_LOADED) > MAX_CACHED:
            _LOADED.popitem(last=False)
    return entry


def load(path: str | Path) -> ImportFile:
    """The import stored at `path`, parsed once per version of the file."""
    return _cached(Path(path))[1]


def _run(path: Path) -> dict:
    key, imported, run = _cached(path)
    if run is None:
        run = run_of(imported)
        with _LOCK:
            if path in _LOADED and _LOADED[path][0] == key:
                _LOADED[path] = (key, imported, run)
    return run


def runs_in(folder: str | Path) -> list[dict]:
    """Every `*.json` import of `folder` as a run, by name; none when the folder is missing.

    A file that no longer parses (edited by hand, cut short, or from before a
    stricter schema) is left out and logged: one bad import must not take
    the whole leaderboard down with it.
    """
    folder = Path(folder)
    if not folder.is_dir():
        return []
    present = set(folder.glob("*.json"))
    with _LOCK:
        for gone in [p for p in _LOADED if p.parent == folder and p not in present]:
            del _LOADED[gone]
    runs = []
    for path in sorted(present):
        try:
            runs.append(_run(path))
        except (OSError, ValueError) as e:  # pydantic's ValidationError is a ValueError
            _log.warning("import %s skipped, it no longer reads: %s", path.name, str(e)[:300])
    return runs


class NothingToExport(ValueError):
    """The board's scope holds no scores: an export would have no rows."""


def export(data: dict, board: dict, source: ImportFile | None = None) -> dict:
    """A saved leaderboard as an import file: re-imported, it gives back the same board.

    `data` is `leaderboard.board()`'s output, `board` the saved entry
    (`name`, `title`, `description`, `tags`, `datasets`, `models`, `shares`).
    Every metric of every cell in scope becomes a row, uncertainty and sample
    count kept; `eol_days` is left out, the arena works it out again. When the
    board reads one import only (`source`), its datasets lose the
    `<import>/` prefix and the import's own metadata comes along, so the
    re-import names everything as before.
    """
    # An empty scope is what the board ranks: the arena's own datasets, not an import's.
    datasets = board.get("datasets") or default_datasets(data)
    models = board.get("models") or [e["model"] for e in data["entries"]]
    prefix = f"{source.meta.name}/" if source else ""
    plain = lambda d: d.removeprefix(prefix)  # noqa: E731

    rows = []
    for entry in data["entries"]:
        if entry["model"] not in models:
            continue
        for dataset in datasets:
            cell = entry["scores"].get(dataset)
            for metric, measured in (cell["metrics"] if cell else {}).items():
                if metric == "eol_days":
                    continue
                rows.append({"model": entry["model"], "dataset": plain(dataset), "metric": metric, **measured})

    if not rows:
        raise NothingToExport("no model in this leaderboard's scope has scores")
    exported_datasets = {r["dataset"] for r in rows}
    exported_models = {r["model"] for r in rows}
    # Every dataset's group travels, whoever made it, so the weights by group still apply on re-import.
    groups = {plain(d): plain(g) for d, g in data.get("groups", {}).items() if plain(d) in exported_datasets}
    described_datasets = {d: v.model_dump(exclude_none=True) for d, v in (source.datasets if source else {}).items() if d in exported_datasets}
    for d, g in groups.items():
        described_datasets.setdefault(d, {})["group"] = g
    # A weight key is a group, or a dataset in no group (a group of its own).
    weighable_groups = {groups.get(d, d) for d in exported_datasets}
    # What the import said about its models, and for every model the id it ran as: a registry name
    # (`my-gpt`) re-imported without it would lose its end-of-life date and its vendor.
    described_models = {m: v.model_dump(exclude_none=True) for m, v in (source.models if source else {}).items() if m in exported_models}
    for entry in data["entries"]:
        model_id = entry.get("model_id")
        if entry["model"] in exported_models and model_id and model_id != entry["model"]:
            described_models.setdefault(entry["model"], {}).setdefault("model_id", model_id)
    if source:
        meta = source.meta.model_dump(exclude_none=True)
    else:
        meta = {"name": board["name"], "title": board["title"], "source": "arena", "tags": [t for t in board.get("tags", []) if t != "imported"]}
        if board.get("description"):
            meta["description"] = board["description"]
    document = {
        "meta": meta,
        "rows": rows,
        "datasets": described_datasets,
        "models": described_models,
        # The import's own charts travel with it; a chart on a metric left out of the scope is dropped.
        "charts": [c.model_dump(exclude_none=True) for c in (source.charts if source else []) if all(any(r["metric"] == m for r in rows) for m in filter(None, (c.metric, c.x)))],
        "leaderboard": {
            "name": board["name"],
            "title": board["title"],
            "description": board.get("description"),
            "tags": [t for t in board.get("tags", []) if t != "imported"],
            "datasets": [plain(d) for d in board.get("datasets", []) if plain(d) in exported_datasets],
            "models": [m for m in board.get("models", []) if m in exported_models],
            "shares": board.get("shares", {}),
            "group_shares": {plain(g): v for g, v in (board.get("group_shares") or {}).items() if plain(g) in weighable_groups},
            "dataset_shares": {plain(d): v for d, v in (board.get("dataset_shares") or {}).items() if plain(d) in exported_datasets},
        },
    }
    return ImportFile.model_validate(document).model_dump(exclude_none=True)
