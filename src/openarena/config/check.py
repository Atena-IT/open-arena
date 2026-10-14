# License Apache 2.0: (c) 2026 Atena-Reply

"""Check a registry file before anything reads it: `openarena check [PATH]`.

`check_registry(section, entries, registries=...)` checks a registry held in
memory, `{name: entry}` as the app keeps it, and is what the backend calls;
`check_file(path, section)` reads one `<section>.yaml`, adds what only a
file can get wrong (its YAML, its shape) and the line of each problem.
Both report every problem they find. Each says where it is (`datasets.gsm8k-20.config.path`
and the line), what is wrong, and how to fix the YAML: what to rename, add
or remove, with a snippet when a shape is easier shown than said. An error
is something the engine or the app would refuse; a warning loads, but
likely not as meant. Three layers:

1. The file: valid YAML, no key written twice, a `<section>:` mapping of
   names to entries, names the app accepts.
2. Each entry, against its registry's strict model (`entries.ENTRIES`, the
   same the app's registries validate with: a field it does not know is an
   error), then what a model cannot know alone: a reward's type and kwargs
   against the catalog (`rewards.accepted_kwargs`), a dataset's provider
   settings against the provider.
3. References, against the registries beside the file (the other
   `*.yaml` of its folder): a dataset's reward and MCP servers, an
   environment's MCP servers, an evaluation's models and datasets, the models a reward's judge names. An
   evaluation is then validated whole, as `evaluations run` would. A
   registry missing beside the file is said once, and its references left
   unchecked.

Nothing is fetched or run: a dataset's rows, a model's id or an MCP server
are not reached, only what the file says of them.
"""

from __future__ import annotations

import difflib
import inspect
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import ValidationError

from openarena.config import (
    A_MODEL,
    KIND_BLOCKS,
    MODEL_KWARGS,
    MODEL_LIST_KWARGS,
    Config,
    flatten_dataset,
)
from openarena.config.entries import ENTRIES, NAME_PATTERN
from openarena.config.errors import Invalid
from openarena.config.files import SECTIONS, gather, load_yaml

# The names the app accepts as keys.
NAME = re.compile(NAME_PATTERN)

# The fields each registry's entries may have: its model's.
FIELDS: dict[str, set[str]] = {section: set(model.model_fields) for section, model in ENTRIES.items()}

# A working entry of each registry, shown when a file has the wrong shape.
EXAMPLES: dict[str, str] = {
    "models": "models:\n  qwen:\n    model: ollama/qwen3\n    params:\n      temperature: 0",
    "rewards": "rewards:\n  exact-match:\n    type: exact_match\n    direction: max",
    "datasets": (
        "datasets:\n  gsm8k:\n    type: huggingface\n    config:\n      path: openai/gsm8k\n      name: main\n"
        "      split: test\n      input_template: '{\"question\": {{ question | tojson }}}'\n"
        "      output_template: '{\"answer\": {{ answer | string | tojson }}}'\n    generator: {}\n    reward: exact-match"
    ),
    "mcp_servers": "mcp_servers:\n  sandbox:\n    transport: stdio\n    command: python\n    args: [mcp_servers/python_sandbox.py]",
    "a2a_agents": "a2a_agents:\n  helper:\n    url: http://localhost:9999",
    "environments": "environments:\n  python:\n    docker_image: python:3.12-slim\n    cpus: 2\n    memory_mb: 4096",
    "evaluations": "evaluations:\n  smoke:\n    language_models: [qwen]\n    datasets: [gsm8k]",
    "leaderboards": "leaderboards:\n  overall:\n    title: Overall\n    datasets: []\n    models: []\n    shares: {reward: 100}",
}

# A value to show for a field an entry lacks, in `add` fixes.
PLACEHOLDERS: dict[str, str] = {
    "model": "ollama/qwen3",
    "type": "<type>",
    "title": "My leaderboard",
    "language_models": "[<a models key>]",
    "datasets": "[<a datasets key>]",
    "reward": "<a rewards key>",
    "command": "<executable>",
    "url": "https://<host>/mcp",
    "path": "rows.jsonl",
    "dataset_name": "<dataset name>",
    "table": "<catalog.schema.table>",
}

# Keys of a flat dataset the sweep reads rather than the provider.
_ARENA_KEYS = {"type", *KIND_BLOCKS, "reward", "metrics"}
# Provider settings that are the row's templates and schemas, not where it is read from.
_TEMPLATE_KEYS = {
    "input_data_model",
    "input_schema",
    "input_template",
    "output_data_model",
    "output_schema",
    "output_template",
}


@dataclass
class Problem:
    level: Literal["error", "warning"]
    where: str
    message: str
    fix: str | None = None
    line: int | None = None


@dataclass
class Report:
    path: Path | None
    section: str
    names: list[str] = field(default_factory=list)
    problems: list[Problem] = field(default_factory=list)

    @property
    def errors(self) -> list[Problem]:
        return [p for p in self.problems if p.level == "error"]

    @property
    def warnings(self) -> list[Problem]:
        return [p for p in self.problems if p.level == "warning"]

    def error(self, where: str, message: str, fix: str | None = None, *, line: int | None = None) -> None:
        self.problems.append(Problem("error", where, message, fix, line))

    def warning(self, where: str, message: str, fix: str | None = None, *, line: int | None = None) -> None:
        self.problems.append(Problem("warning", where, message, fix, line))


def _close(word: Any, choices) -> str | None:
    """The one of `choices` `word` is a near miss of, if any."""
    found = difflib.get_close_matches(str(word), [str(c) for c in choices], n=1, cutoff=0.75)
    return found[0] if found else None


def _one_of(choices) -> str:
    return ", ".join(str(c) for c in sorted(choices)) or "(none)"


def _leaf(where: str) -> str:
    """The last key of a location: `c` of `a.b.c`, `b` of `a.b[0]`."""
    return re.sub(r"\[\d+\]$", "", where).rsplit(".", 1)[-1]


def _parent(where: str) -> str:
    return where.rsplit(".", 1)[0] if "." in where else where


def _rename_or_remove(key: str, where: str, choices, what: str) -> str:
    """The fix for a key nothing reads: rename it to its near miss, else remove it."""
    near = _close(key, choices)
    if near:
        return f"rename `{key}:` to `{near}:` under {_parent(where)}"
    return f"remove `{key}:` from {_parent(where)}; {what}: {_one_of(choices)}"


def _replace_value(value: Any, choices, what: str) -> str:
    near = _close(value, choices)
    if near:
        return f"change {value!r} to {near!r}"
    return f"use one of the {what}: {_one_of(choices)}"


def _sentence(message: str) -> str:
    return message.removeprefix("Value error, ").removeprefix("Assertion failed, ")


def _validation_fix(err: dict, where: str) -> str | None:
    """How to fix one pydantic error, from its type."""
    kind, ctx = err["type"], err.get("ctx") or {}
    key = _leaf(where)
    if kind == "missing":
        return f"add `{key}: {PLACEHOLDERS.get(key, '<value>')}` to {_parent(where)}"
    if kind in ("greater_than_equal", "greater_than"):
        return f"set `{key}:` to {ctx.get('ge', ctx.get('gt'))} or more"
    if kind in ("less_than_equal", "less_than"):
        return f"set `{key}:` to {ctx.get('le', ctx.get('lt'))} or less"
    if kind in ("too_short", "string_too_short"):
        return f"give `{key}:` a value; it cannot be empty"
    if kind == "literal_error":
        return f"set `{key}:` to one of {ctx.get('expected')}"
    if kind in ("string_type", "str_type"):
        return f'write `{key}:` as text (quote it if it looks like a number: "123")'
    if kind == "list_type":
        return f"write `{key}:` as a list: `{key}: [a, b]`"
    if kind in ("dict_type", "model_type"):
        return f"write `{key}:` as a mapping of fields (`{key}: {{name: value}}`)"
    if kind in ("int_type", "int_parsing", "int_from_float"):
        return f"write `{key}:` as a whole number"
    if kind in ("float_type", "float_parsing"):
        return f"write `{key}:` as a number"
    if kind in ("bool_type", "bool_parsing"):
        return f"write `{key}:` as true or false"
    message = str(err.get("msg", ""))
    if "mutually exclusive" in message:
        return "keep one of those blocks: a dataset is answered one way (generator, agent, a2a, embedding or tools)"
    if "tool_schemas" in message:
        return (
            "list the functions under `tools:`:\ntools:\n  tool_schemas:\n"
            "    - {name: <function>, description: ..., parameters: {type: object, properties: {}}}"
        )
    if "mcp_servers" in message:
        return "name the servers the agent calls, from mcp_servers.yaml:\nagent:\n  mcp_servers: [<a mcp_servers key>]"
    if "cannot set" in message:
        return "take those keys out of `config:`; `type:` and `direction:` sit beside `config:`, not in it"
    return None


class _Context:
    """The registries beside the file, for references, and the report they are written to."""

    def __init__(self, report: Report, found: dict[str, dict], missing_ok: set[str], missing_fix: str):
        self.report = report
        self.found = found
        self._said: set[str] = set()
        self._missing_ok = missing_ok
        self._missing_fix = missing_fix

    def registry(self, section: str) -> dict | None:
        if section in self.found:
            return self.found[section]
        if section not in self._said and section not in self._missing_ok:
            self._said.add(section)
            self.report.warning(
                section,
                f"no `{section}` registry to check references against; they are not checked",
                self._missing_fix.format(section=section),
            )
        return None

    def ref(self, section: str, key: Any, where: str, *, level: str = "error", note: str = "") -> bool:
        """Whether `key` names an entry of `section`; the problem said at `level` when it does not."""
        entries = self.registry(section)
        if entries is None or key in entries:
            return True
        near = _close(key, entries)
        if near:
            fix = f"change {key!r} to {near!r}"
        elif entries:
            fix = f"use one of the {section} there: {_one_of(entries)}; or add a {key!r} entry to {section}.yaml"
        else:
            fix = f"add a {key!r} entry to {section}.yaml"
        message = f"{key!r} is not in `{section}`" + (f"; {note}" if note else "")
        getattr(self.report, level)(where, message, fix)
        return False

    def validation(self, where: str, exc: ValidationError, section: str, entry: dict) -> None:
        """Each error of an entry's model as a problem, at its place in the entry, with its fix."""
        for err in exc.errors():
            at = where + "".join(f"[{x}]" if isinstance(x, int) else f".{x}" for x in err["loc"])
            cause = (err.get("ctx") or {}).get("error")
            if isinstance(cause, Invalid):
                self.report.error(at, str(cause), cause.fix)
            elif err["type"] == "extra_forbidden":
                self.report.error(at, f"not a field of a `{section}` entry", _extra_fix(section, entry, err["loc"], at))
            elif err["type"] == "missing":
                self.report.error(at, f"`{_leaf(at)}:` is missing", _validation_fix(err, at))
            else:
                self.report.error(at, _sentence(err["msg"]), _validation_fix(err, at))

    def model_refs(self, kwargs: dict[str, Any], where: str) -> None:
        """Every model a reward or metric names: a `models` key of the right type."""
        for key, wanted in MODEL_KWARGS.items():
            if isinstance(kwargs.get(key), str):
                self._model(kwargs[key], wanted, f"{where}.{key}")
        for key, wanted in MODEL_LIST_KWARGS.items():
            if isinstance(kwargs.get(key), list):
                for i, value in enumerate(kwargs[key]):
                    self._model(value, wanted, f"{where}.{key}[{i}]")

    def _model(self, value: Any, wanted: str, where: str) -> None:
        models = self.registry("models")
        if models is None:
            return
        of_type = sorted(n for n, m in models.items() if (m or {}).get("type", "llm") == wanted)
        if value in models:
            got = (models[value] or {}).get("type", "llm")
            if got != wanted:
                fix = (
                    f"name {A_MODEL[wanted]} instead: {_one_of(of_type)}"
                    if of_type
                    else f"add {A_MODEL[wanted]} to models.yaml (`type: {wanted}`) and name it here"
                )
                self.report.error(where, f"{value!r} is {A_MODEL.get(got, got)}; this setting needs {A_MODEL[wanted]}", fix)
            return
        calling = [n for n, m in models.items() if (m or {}).get("model") == value]
        near = _close(value, models)
        if calling:
            fix = f"name the entry that calls it: {calling[0]!r}"
        elif near:
            fix = f"change {value!r} to {near!r}"
        else:
            fix = f"add it to models.yaml, or use one of: {_one_of(of_type)}"
        self.report.error(where, f"{value!r} is not in `models`", fix)


# Call options a model entry keeps under `params:`, recognised when written beside `model:`.
_MODEL_PARAMS = {
    "temperature",
    "max_tokens",
    "max_completion_tokens",
    "top_p",
    "top_k",
    "reasoning_effort",
    "seed",
    "stop",
    "frequency_penalty",
    "presence_penalty",
    "timeout",
    "api_base",
}


def _extra_fix(section: str, entry: dict, loc: tuple, where: str) -> str:
    """How to fix a field the entry's model does not know."""
    key = str(loc[-1])
    if len(loc) == 1 and section == "models":
        if key == "provider":
            return "remove `provider:`; the provider is read off the model id"
        param = _close(key, _MODEL_PARAMS)
        if param:
            spelled = f" (spelled `{param}:`)" if param != key else ""
            return f"move it under `params:`{spelled}:\nparams:\n  {param}: {entry.get(key)!r}"
    if len(loc) == 1:
        return _rename_or_remove(key, where, FIELDS[section], "an entry reads")
    return f"remove `{key}:` from {_parent(where)}, or check its spelling"


def _reward_type(ctx: _Context, type_: str, where: str) -> bool:
    """Whether `type_` is a registered reward."""
    from openarena import rewards

    if type_ not in rewards._REWARD_TYPES:
        ctx.report.error(where, f"unknown reward {type_!r}", _replace_value(type_, rewards._REWARD_TYPES, "rewards"))
        return False
    return True


def _reward_kwargs(ctx: _Context, type_: str, kwargs: dict[str, Any], where: str) -> None:
    """Every key of `kwargs` is one the (registered) reward `type_` takes."""
    from openarena import rewards

    allowed = rewards.accepted_kwargs(type_)
    for key in kwargs:
        if key not in allowed:
            ctx.report.error(
                f"{where}.{key}",
                f"{type_} does not take `{key}`",
                _rename_or_remove(key, f"{where}.{key}", allowed, f"{type_} takes"),
            )


def _validate(ctx: _Context, section: str, entry: dict, where: str):
    """The entry read with its registry's model; None, its errors reported, when it does not read."""
    try:
        return ENTRIES[section].model_validate(entry)
    except ValidationError as e:
        ctx.validation(where, e, section, entry)
        return None


# -- per registry: the model, then what it cannot know alone --------------------


def _models(ctx: _Context, name: str, entry: dict, where: str) -> None:
    _validate(ctx, "models", entry, where)


def _mcp_servers(ctx: _Context, name: str, entry: dict, where: str) -> None:
    _validate(ctx, "mcp_servers", entry, where)


def _a2a_agents(ctx: _Context, name: str, entry: dict, where: str) -> None:
    _validate(ctx, "a2a_agents", entry, where)


def _environments(ctx: _Context, name: str, entry: dict, where: str) -> None:
    _validate(ctx, "environments", entry, where)
    servers = entry.get("mcp_servers")
    for i, server in enumerate(servers if isinstance(servers, list) else []):
        ctx.ref("mcp_servers", server, f"{where}.mcp_servers[{i}]")


def _rewards(ctx: _Context, name: str, entry: dict, where: str) -> None:
    reward = _validate(ctx, "rewards", entry, where)
    if reward is None:
        return
    if _reward_type(ctx, reward.type, f"{where}.type"):
        _reward_kwargs(ctx, reward.type, reward.config, f"{where}.config")
    ctx.model_refs(reward.config, f"{where}.config")


def _provider(ctx: _Context, entry: dict, where: str) -> None:
    """The provider is registered, and `config` gives it the settings it needs and only those it takes."""
    from openarena.datasets import _DATASET_TYPES

    type_ = entry.get("type")
    if type_ not in _DATASET_TYPES:
        ctx.report.error(f"{where}.type", f"unknown provider {type_!r}", _replace_value(type_, _DATASET_TYPES, "providers"))
        return
    try:
        flat = flatten_dataset(entry)
    except Exception:  # noqa: BLE001 — an entry the model refused; it said so
        return
    settings = {k: v for k, v in flat.items() if k not in _ARENA_KEYS}
    at = f"{where}.config"
    params = inspect.signature(_DATASET_TYPES[type_].__init__).parameters
    takes_any = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    named = {k: p for k, p in params.items() if k != "self" and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)}
    offered = sorted(set(named) - _TEMPLATE_KEYS)
    # A required setting misspelled is one problem, said once as the rename.
    renamed = set() if takes_any else {_close(k, offered) for k in settings if k not in named}
    for key, param in named.items():
        if param.default is inspect.Parameter.empty and key not in settings and key not in renamed:
            ctx.report.error(f"{at}.{key}", f"the {type_} provider needs `{key}`", f"add `{key}: {PLACEHOLDERS.get(key, '<value>')}` under {at}")
    if not takes_any:
        for key in settings:
            if key not in named:
                ctx.report.error(
                    f"{at}.{key}",
                    f"not a setting of the {type_} provider",
                    _rename_or_remove(key, f"{at}.{key}", offered, f"the {type_} provider takes"),
                )


def _datasets(ctx: _Context, name: str, entry: dict, where: str) -> None:
    _validate(ctx, "datasets", entry, where)
    if "type" in entry:
        _provider(ctx, entry, where)
    # References, read off the entry as written, so they are checked even when the model refused it.
    reward = entry.get("reward")
    if isinstance(reward, str):
        ctx.ref("rewards", reward, f"{where}.reward")
    for i, metric in enumerate(entry.get("metrics") or []):
        if isinstance(metric, dict):
            ctx.model_refs(metric, f"{where}.metrics[{i}]")
    agent = entry.get("agent")
    if isinstance(agent, dict):
        for i, server in enumerate(agent.get("mcp_servers") or []):
            ctx.ref("mcp_servers", server, f"{where}.agent.mcp_servers[{i}]")
    harbor = entry.get("harbor")
    if isinstance(harbor, dict) and harbor.get("environment"):
        ctx.ref("environments", harbor["environment"], f"{where}.harbor.environment")


def _whole_fix(message: str) -> str | None:
    """How to fix what only shows when an evaluation is validated with everything it names."""
    if "cannot share a run" in message:
        return "split it in two evaluations: one for the datasets of that kind, one for the others"
    if "this sweep needs" in message:
        return "list only models of the type the datasets need in `language_models:`, or move the others to their own evaluation"
    if "has no `reward:`" in message:
        return "set `reward:` on that dataset (a rewards key), or leave it out of this evaluation"
    return None


def _evaluations(ctx: _Context, name: str, entry: dict, where: str) -> None:
    before = len(ctx.report.errors)
    evaluation = _validate(ctx, "evaluations", entry, where)

    def names(key: str) -> list:
        value = entry.get(key)
        return value if isinstance(value, list) else []

    # References are checked whatever the model said, so one pass shows every problem.
    datasets = ctx.registry("datasets")
    for i, ds in enumerate(names("datasets")):
        ctx.ref("datasets", ds, f"{where}.datasets[{i}]")
    # An `a2a` sweep's axis is A2A agents, not `models` keys.
    selected = [ds for ds in names("datasets") if datasets is not None and ds in datasets]
    only_a2a = bool(selected) and all(isinstance((datasets.get(ds) or {}).get("a2a"), dict) for ds in selected)
    for i, model in enumerate(names("language_models")):
        ctx.ref("a2a_agents" if only_a2a else "models", model, f"{where}.language_models[{i}]")
    ctx.model_refs(
        {"language_model": entry.get("default_language_model"), "embedding_model": entry.get("default_embedding_model")},
        where,
    )
    for i, metric in enumerate(names("metrics")):
        if isinstance(metric, dict):
            ctx.model_refs(metric, f"{where}.metrics[{i}]")
    if evaluation is None or len(ctx.report.errors) > before or datasets is None:
        return
    # Whole, as `evaluations run` validates it: the rules across entries (kinds that cannot share a run, model types).
    try:
        Config.model_validate(gather({**ctx.found, "evaluations": {name: entry}}, name))
    except ValidationError as e:
        for err in e.errors():
            # Problems of the entries it names are theirs, reported by their own registry's check.
            if not err["loc"] or err["loc"][0] == "evaluation":
                message = _sentence(err["msg"])
                ctx.report.error(where, message, _whole_fix(message))
            elif not any(p.level == "warning" and p.where == where for p in ctx.report.problems):
                ctx.report.warning(
                    where,
                    f"not validated whole: `{err['loc'][0]}` it names does not read",
                    f"fix that registry first: `openarena check configs/{err['loc'][0]}.yaml`",
                )


def _leaderboards(ctx: _Context, name: str, entry: dict, where: str) -> None:
    _validate(ctx, "leaderboards", entry, where)
    # A leaderboard ranks the scores of runs, which outlive the entries they were made with.
    note = "only scores from runs made before it was removed can be ranked"
    for section in ("datasets", "models"):
        names = entry.get(section)
        for i, key in enumerate(names if isinstance(names, list) else []):
            ctx.ref(section, key, f"{where}.{section}[{i}]", level="warning", note=note)


CHECKS = {
    "models": _models,
    "rewards": _rewards,
    "datasets": _datasets,
    "mcp_servers": _mcp_servers,
    "a2a_agents": _a2a_agents,
    "environments": _environments,
    "evaluations": _evaluations,
    "leaderboards": _leaderboards,
}

# The registries each one's entries may refer to; one missing beside the file is only worth saying when referred to.
_REFERS_TO = {
    "models": set(),
    "rewards": {"models"},
    "datasets": {"rewards", "models", "mcp_servers", "environments"},
    "mcp_servers": set(),
    "a2a_agents": set(),
    "environments": {"mcp_servers"},
    "evaluations": {"models", "datasets", "a2a_agents"},
    "leaderboards": {"models", "datasets"},
}


def _beside(folder: Path, skip: Path, report: Report) -> dict[str, dict]:
    """The other registries of `folder`, each read from its own file, `<section>.yaml`, under its `<section>:` key.

    One that does not read is left out, and said when this registry refers to it.
    """
    found: dict[str, dict] = {}
    for section in SECTIONS:
        path = folder / f"{section}.yaml"
        if not path.is_file() or path.resolve() == skip.resolve():
            continue
        try:
            body = (load_yaml(path) or {}).get(section)
        except (yaml.YAMLError, AttributeError):
            if section in _REFERS_TO[report.section]:
                report.warning(path.name, "does not read; its entries are not used for references", f"fix it first: `openarena check {path}`")
            continue
        if isinstance(body, dict):
            found[section] = {k: v or {} for k, v in body.items()}
    return found


def _yaml_problem(report: Report, e: yaml.YAMLError) -> None:
    """A syntax error, or a key written twice, at the line it is on."""
    mark = getattr(e, "problem_mark", None)
    line = mark.line + 1 if mark is not None else None
    problem = str(getattr(e, "problem", None) or e).strip()
    twice = re.match(r"found (.+) twice", problem)
    if twice:
        first = getattr(e, "context_mark", None)
        in_mapping = f" (in the mapping at line {first.line + 1})" if first is not None else ""
        report.error(
            report.section,
            f"{twice.group(1)} is written twice in one mapping{in_mapping}; YAML would keep only the last",
            f"keep one {twice.group(1)}: delete the other, or rename it if both entries are meant",
            line=line,
        )
        return
    context = str(getattr(e, "context", None) or "").strip()
    report.error(
        report.section,
        f"not valid YAML: {problem}" + (f" ({context})" if context else ""),
        "correct the syntax on that line: indent nested keys by two spaces, close every `[`, `{` and quote, "
        "and quote text that holds `: ` or starts with `{`, `[`, `*`, `&` or `#`",
        line=line,
    )


def _lines(root: yaml.Node | None) -> dict[tuple, int]:
    """The line each key and item starts on, by its path from the top of the document."""
    out: dict[tuple, int] = {}

    def walk(node: yaml.Node, path: tuple) -> None:
        if isinstance(node, yaml.MappingNode):
            for key, value in node.value:
                here = (*path, key.value)
                out[here] = key.start_mark.line + 1
                walk(value, here)
        elif isinstance(node, yaml.SequenceNode):
            for i, item in enumerate(node.value):
                here = (*path, i)
                out[here] = item.start_mark.line + 1
                walk(item, here)

    if root is not None:
        walk(root, ())
    return out


def _line_of(where: str, lines: dict[tuple, int]) -> int | None:
    """The line of `where` (`datasets.gsm8k.config.path`, `a.b[0]`), or of the nearest part of it written in the file.

    Keys may hold dots themselves (`gpt-4.1`), so the path is matched against
    the file's own keys, longest first, rather than split on dots.
    """
    path: tuple = ()
    rest = where
    best = None
    while rest:
        index = re.match(r"\[(\d+)\]", rest)
        if index:
            step: Any = int(index.group(1))
            rest = rest[index.end() :].removeprefix(".")
        else:
            keys = sorted((k[-1] for k in lines if k[:-1] == path and isinstance(k[-1], str)), key=len, reverse=True)
            step = next((k for k in keys if rest == k or rest.startswith(k + ".") or rest.startswith(k + "[")), None)
            if step is None:
                break
            rest = rest[len(step) :].removeprefix(".")
        path = (*path, step)
        if path not in lines:
            break
        best = lines[path]
    return best


def _check_entries(report: Report, section: str, entries: dict, found: dict[str, dict] | None, missing_fix: str) -> None:
    """Each entry of `entries` into `report`; references against `found`, none checked when it is None."""
    missing_ok = set(SECTIONS) if found is None else set(SECTIONS) - _REFERS_TO[section]
    ctx = _Context(report, found or {}, missing_ok=missing_ok, missing_fix=missing_fix)
    for name, entry in entries.items():
        where = f"{section}.{name}"
        report.names.append(str(name))
        if not isinstance(name, str) or not NAME.match(name):
            fixed = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(name)).strip("-_.") or "entry"
            report.error(
                where,
                "a name starts with a letter or a digit, then letters, digits, `_`, `.` or `-`",
                f"rename it, e.g. `{fixed}:`, and update what refers to it",
            )
        if entry is None:
            entry = {}
        if not isinstance(entry, dict):
            report.error(where, "an entry is a mapping of its fields", f"write its fields under the name, e.g.\n{EXAMPLES[section]}")
            continue
        CHECKS[section](ctx, str(name), entry, where)


def check_registry(section: str, entries: dict[str, Any], *, registries: dict[str, dict] | None = None) -> Report:
    """Every problem of the `section` registry `entries` (`{name: entry}`), held in memory.

    `registries` are the other registries by section (`{"models": {name:
    entry}, ...}`), the ones references are checked against; `entries` stands
    for its own section whatever `registries` holds. None checks no reference.
    A section `registries` lacks that `section` refers to is said once, as a
    warning. The report's `path` is None and its problems carry no line.
    """
    if section not in CHECKS:
        raise ValueError(f"no registry {section!r}; one of: {', '.join(SECTIONS)}")
    report = Report(None, section)
    found = None if registries is None else {k: dict(v or {}) for k, v in registries.items() if k in SECTIONS}
    if found is not None:
        found[section] = {k: v or {} for k, v in entries.items()}
    _check_entries(report, section, entries, found, "give the {section} registry along with this one")
    return report


def section_of(path: Path) -> str | None:
    """The registry a file holds, read off its name: `datasets.yaml` holds `datasets`; None for any other name."""
    path = Path(path)
    return path.stem if path.suffix == ".yaml" and path.stem in SECTIONS else None


def _misnamed(path: Path) -> Report:
    """A file whose name is no registry's: the name to give it."""
    report = Report(path, "registry")
    near = _close(path.stem.replace("-", "_"), SECTIONS)
    if near:
        fix = f"rename it to `{near}.yaml`"
    else:
        fix = f"rename it after the registry it holds: {', '.join(f'{s}.yaml' for s in SECTIONS)}"
    report.error(path.name, "not a registry file: a registry is read only from `<registry>.yaml`", fix)
    return report


def check_file(path: Path, *, beside: Path | None = None) -> Report:
    """Every problem of the registry file `path`, `<registry>.yaml`, references checked against the registries in `beside` (its folder by default)."""
    path = Path(path)
    section = section_of(path)
    if section is None:
        if not path.is_file():
            report = Report(path, "registry")
            report.error(path.name, f"no file at {path}", "pass the path of a registry file, or a configs folder")
            return report
        return _misnamed(path)
    report = Report(path, section)
    if not path.is_file():
        report.error(
            section,
            f"no file at {path}",
            f"pass the path of the {section} file, or write a working configs/ folder with `openarena init`",
        )
        return report
    try:
        raw = load_yaml(path)
    except yaml.YAMLError as e:
        _yaml_problem(report, e)
        return report
    raw = {} if raw is None else raw
    if not isinstance(raw, dict) or section not in raw:
        held = ", ".join(f"`{k}:`" for k in raw) if isinstance(raw, dict) and raw else "nothing"
        report.error(
            section,
            f"no `{section}:` at the top of the file (it holds {held})",
            f"put the entries under a top-level `{section}:` key, e.g.\n{EXAMPLES[section]}",
            line=1,
        )
        return report
    lines = _lines(yaml.compose(path.read_text(encoding="utf-8")))
    for key in raw:
        if key != section:
            fix = f"move `{key}:` and its entries to {key}.yaml" if key in SECTIONS else f"remove `{key}:` from the top of the file"
            report.error(str(key), f"not read from {path.name}: it holds `{section}:` only", fix)
    body = raw[section]
    if body is None:
        body = {}
    if not isinstance(body, dict):
        report.error(
            section,
            f"`{section}:` maps each name to its entry",
            f"write one `name:` per entry under `{section}:`, e.g.\n{EXAMPLES[section]}",
        )
        body = {}

    found = _beside(Path(beside) if beside is not None else path.parent, path, report)
    found[section] = {k: v or {} for k, v in body.items()}
    _check_entries(
        report,
        section,
        body,
        found,
        "put {section}.yaml in the same folder, or point at the folder that has it: "
        f"`openarena check {path} --configs DIR`",
    )
    for problem in report.problems:
        if problem.line is None:
            problem.line = _line_of(problem.where, lines)
    return report


def check_folder(configs: Path) -> list[Report]:
    """`check_file` for each `<registry>.yaml` of `configs`, and a problem for any other YAML file there, which nothing reads."""
    configs = Path(configs)
    if not configs.is_dir():
        raise FileNotFoundError(f"no configs folder at {configs}; write one with `openarena init`")
    reports = [check_file(configs / f"{s}.yaml") for s in SECTIONS if (configs / f"{s}.yaml").is_file()]
    stray = sorted(p for p in configs.iterdir() if p.is_file() and p.suffix in (".yaml", ".yml") and section_of(p) is None)
    return reports + [_misnamed(p) for p in stray]
