# License Apache 2.0: (c) 2026 Atena-Reply

"""A `configs/` folder: one registry per file, `{section: {name: entry}}`, the shape an arena workspace keeps.

`registries` reads every file of a folder, merged by section;
`evaluation_config` gathers one evaluation and what it names into the
config `run_sweep` reads. `load_yaml` refuses a key written twice, which
plain YAML would let the second silently win.
"""

from __future__ import annotations

from pathlib import Path

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
