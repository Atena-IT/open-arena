# License Apache 2.0: (c) 2026 Atena-Reply

"""A `configs/` folder: one registry per file, `{section: {name: entry}}`, the shape an arena workspace keeps.

`registries` reads every file of a folder, merged by section;
`evaluation_config` gathers one evaluation and what it names into the
config `run_sweep` reads. `load_yaml` refuses a key written twice, which
plain YAML would let the second silently win. `expand` replaces each
`${NAME}` with the environment variable, when a run loads its config.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

# Every registry a `configs/` folder may hold, each in `<section>.yaml`.
SECTIONS = ("models", "rewards", "datasets", "mcp_servers", "a2a_agents", "environments", "evaluations", "leaderboards")


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _mapping(loader: yaml.SafeLoader, node: yaml.MappingNode, deep: bool = False) -> dict:
    seen: set = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            raise yaml.constructor.ConstructorError(
                "while reading a mapping", node.start_mark, f"found {key!r} twice", key_node.start_mark
            )
        seen.add(key)
    return loader.construct_mapping(node, deep=deep)


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


# `${NAME}` in a value is the environment variable NAME, read when a run loads its config;
# `$${NAME}` writes `${NAME}` as it is. Jinja's `{{ ... }}` never matches.
_REFERENCE = re.compile(r"\$(\$?)\{([A-Za-z_][A-Za-z0-9_]*)\}")


def references(value: Any, where: str = "") -> list[tuple[str, str]]:
    """Every `${NAME}` in `value` (a config, an entry, a string), as `(where, NAME)`: `where` the dotted path to its string."""
    if isinstance(value, str):
        return [(where, m.group(2)) for m in _REFERENCE.finditer(value) if not m.group(1)]
    if isinstance(value, dict):
        return [ref for k, v in value.items() for ref in references(v, f"{where}.{k}" if where else str(k))]
    if isinstance(value, list):
        return [ref for i, v in enumerate(value) for ref in references(v, f"{where}[{i}]")]
    return []


def expand(value: Any, environ: Mapping[str, str] | None = None, *, keep_unset: bool = False) -> Any:
    """`value` with every `${NAME}` replaced by the variable's value; a `ValueError` naming each one that is not set.

    A config's secrets (an A2A agent's or an MCP server's `headers`, a
    server's or an environment's `env`) can then stay in `.env` and out of
    `configs/`; the run's `config.yaml` keeps the references, not the values.
    `keep_unset` leaves a reference to an unset variable as it is, for
    `check`, which warns about it instead.
    """
    environ = os.environ if environ is None else environ
    unset = sorted({(where, name) for where, name in references(value) if name not in environ})
    if unset and not keep_unset:
        said = "; ".join(f"{name} (at {where})" for where, name in unset)
        raise ValueError(f"not set in the environment: {said}. Add each to the workspace's .env, or export it in your shell.")

    def one(m: re.Match) -> str:
        if m.group(1) or m.group(2) not in environ:
            return "${" + m.group(2) + "}" if m.group(1) else m.group(0)
        return environ[m.group(2)]

    def sub(v):
        if isinstance(v, str):
            return _REFERENCE.sub(one, v)
        if isinstance(v, dict):
            return {k: sub(x) for k, x in v.items()}
        if isinstance(v, list):
            return [sub(x) for x in v]
        return v

    return sub(value)


def load_yaml(path: Path):
    """The YAML document at `path`; `yaml.YAMLError` on a syntax error or a key written twice in one mapping."""
    # From the open file, so an error's position names it.
    with open(path, encoding="utf-8") as stream:
        return yaml.load(stream, Loader=_UniqueKeyLoader)  # noqa: S506 — a SafeLoader subclass


def registries(configs: Path) -> dict[str, dict]:
    """Each registry of `configs`, `{section: {name: entry}}`, read from its own file: `<section>.yaml`, under its `<section>:` key."""
    if not configs.is_dir():
        raise FileNotFoundError(f"no configs folder at {configs}; write one with `openarena init`")
    found: dict[str, dict] = {}
    for section in SECTIONS:
        path = configs / f"{section}.yaml"
        if path.is_file():
            body = (load_yaml(path) or {}).get(section)
            found[section] = body if isinstance(body, dict) else {}
    return found


def evaluation_config(configs: Path, name: str) -> dict:
    """The config `run_sweep` reads for the evaluation saved as `name` in `configs`; see `gather`."""
    return gather(registries(configs), name, where=configs)


def gather(found: dict[str, dict], name: str, *, where: object = "configs") -> dict:
    """The evaluation `name` of `found` (registries by section), with the entries it names.

    The datasets it selects and the rewards and MCP servers those name, the
    A2A agents an `a2a` sweep's axis names; every model is kept, since rewards and metrics name theirs by key
    too. Raises `KeyError` for an unknown evaluation.
    """
    evaluations = found.get("evaluations", {})
    if name not in evaluations:
        raise KeyError(f"no evaluation {name!r} in {where}; choose one of: {', '.join(sorted(evaluations)) or '(none)'}")
    evaluation = {"name": name, **(evaluations[name] or {})}
    all_datasets = found.get("datasets", {})
    datasets = {d: all_datasets[d] for d in evaluation.get("datasets") or [] if d in all_datasets}

    def named(section: str, refs: list) -> dict:
        entries = found.get(section, {})
        return {r: entries[r] for r in refs if isinstance(r, str) and r in entries}

    harbor_environments = named("environments", [(d.get("harbor") or {}).get("environment") for d in datasets.values()])
    raw = {
        "evaluation": evaluation,
        "models": found.get("models", {}),
        "rewards": named("rewards", [d.get("reward") for d in datasets.values()]),
        "datasets": datasets,
        "mcp_servers": named(
            "mcp_servers",
            [s for d in datasets.values() for s in (d.get("agent") or {}).get("mcp_servers") or []]
            + [s for e in harbor_environments.values() for s in (e or {}).get("mcp_servers") or []],
        ),
        "environments": harbor_environments,
        "a2a_agents": named("a2a_agents", list(evaluation.get("language_models") or []))
        if datasets and all(isinstance(d.get("a2a"), dict) for d in datasets.values())
        else {},
    }
    return {k: v for k, v in raw.items() if v or k in ("evaluation", "datasets")}
