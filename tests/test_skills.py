# License Apache 2.0: (c) 2026 Atena-Reply

"""The Agent Skills shipped in `openarena/skills/`: well-formed, and every registry they show is one the engine accepts."""

import re
from pathlib import Path

import jinja2
import pytest
import yaml

import openarena
from openarena.config.check import SECTIONS, check_registry

SKILLS = Path(openarena.__file__).parent / "skills"
FOLDERS = sorted(p for p in SKILLS.iterdir() if (p / "SKILL.md").is_file())
YAML_BLOCK = re.compile(r"```yaml\n(.*?)```", re.S)


def _frontmatter(text: str) -> dict:
    head, _, _ = text.removeprefix("---\n").partition("\n---\n")
    return yaml.safe_load(head)


def _registry_examples():
    """`(skill, section, entries)` for every YAML block of a skill that is a whole registry document; a block eliding with `# ...` is left out."""
    for folder in FOLDERS:
        for block in YAML_BLOCK.findall((folder / "SKILL.md").read_text()):
            if "# ..." in block:
                continue
            doc = yaml.safe_load(block)
            if isinstance(doc, dict) and doc and set(doc) <= set(SECTIONS):
                for section, entries in doc.items():
                    yield pytest.param(section, entries, id=f"{folder.name}:{section}")


def test_there_is_a_skill_per_action():
    assert len(FOLDERS) >= 12


@pytest.mark.parametrize("folder", FOLDERS, ids=lambda p: p.name)
def test_each_skill_says_what_it_is_for(folder):
    meta = _frontmatter((folder / "SKILL.md").read_text())
    assert meta["name"] == folder.name
    assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", meta["name"]) and len(meta["name"]) <= 64
    assert meta["description"].startswith("Use ") and len(meta["description"]) <= 1024


@pytest.mark.parametrize(("section", "entries"), list(_registry_examples()))
def test_every_example_is_a_valid_registry(section, entries):
    report = check_registry(section, entries)
    assert [p for p in report.problems if p.level == "error"] == []


@pytest.mark.parametrize(("section", "entries"), [p for p in _registry_examples() if p.values[0] == "datasets"])
def test_every_template_parses(section, entries):
    env = jinja2.Environment()
    for name, entry in entries.items():
        for key, value in (entry.get("config") or {}).items():
            if key.endswith("_template"):
                env.parse(value)


def test_every_skill_named_exists():
    """A skill pointing at another (`openarena-add-reward`) points at one that is there."""
    names = {p.name for p in FOLDERS}
    for folder in FOLDERS:
        for named in set(re.findall(r"\b(openarena-[a-z0-9-]+[a-z0-9])\b", (folder / "SKILL.md").read_text())):
            assert named in names, f"{folder.name} names {named}"
