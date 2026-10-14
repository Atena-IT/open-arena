# License Apache 2.0: (c) 2026 Atena-Reply

"""The documentation site (`docs/`, built by Zensical from `mkdocs.yml`): its YAML is what the engine accepts, its CLI reference is the CLI."""

import re
from pathlib import Path

import click
import jinja2
import pytest
import yaml

from openarena.cli.main import cli
from openarena.config.check import SECTIONS, check_registry

ROOT = Path(__file__).parent.parent
DOCS = ROOT / "docs"
PAGES = sorted(DOCS.rglob("*.md"))
YAML_BLOCK = re.compile(r"```yaml\n(.*?)```", re.S)


def _registry_examples():
    for page in PAGES:
        for block in YAML_BLOCK.findall(page.read_text()):
            if "# ..." in block:
                continue
            doc = yaml.safe_load(block)
            if isinstance(doc, dict) and doc and set(doc) <= set(SECTIONS):
                for section, entries in doc.items():
                    yield pytest.param(section, entries, id=f"{page.relative_to(DOCS)}:{section}")


def test_every_page_is_in_the_nav():
    """A page left out of `nav` builds but cannot be reached."""
    nav = (ROOT / "mkdocs.yml").read_text()
    for page in PAGES:
        assert str(page.relative_to(DOCS)) in nav, page


@pytest.mark.parametrize(("section", "entries"), list(_registry_examples()))
def test_every_example_is_a_valid_registry(section, entries):
    report = check_registry(section, entries)
    assert [p for p in report.problems if p.level == "error"] == []
    if section == "datasets":
        for entry in entries.values():
            for key, value in (entry.get("config") or {}).items():
                if key.endswith("_template"):
                    jinja2.Environment().parse(value)


def _commands(group: click.Group, path=()):
    for name, command in group.commands.items():
        if isinstance(command, click.Group):
            yield from _commands(command, (*path, name))
        else:
            yield (*path, name), command


@pytest.mark.parametrize(("path", "command"), list(_commands(cli)), ids=lambda v: " ".join(v) if isinstance(v, tuple) else "")
def test_the_cli_reference_has_every_command_and_option(path, command):
    page = (DOCS / "cli.md").read_text()
    assert f"openarena {' '.join(path)}" in page
    for param in command.params:
        if isinstance(param, click.Option):
            assert max(param.opts, key=len) in page, f"{' '.join(path)} {param.opts}"
