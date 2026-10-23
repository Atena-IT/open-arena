# License Apache 2.0: (c) 2026 Atena-Reply

"""Every dataset example shown to a user reads its rows: the docs, the skills, the README, the init template.

Checking an example against the schema is not enough: a template rendering
`{"question": ...}` with no `input_schema` is a valid entry whose rows fail when
a run reads them as chat messages. Each example is rebuilt here as a `local`
dataset over a row of the shape its source has, and its first batch read.
"""

import json
import re
from pathlib import Path

import pytest
import yaml

from openarena.datasets import load_dataset_from_config

ROOT = Path(__file__).parent.parent
SOURCES = [
    *sorted((ROOT / "docs").rglob("*.md")),
    *sorted((ROOT / "src" / "openarena" / "skills").glob("*/SKILL.md")),
    ROOT / "README.md",
    ROOT / "src" / "openarena" / "cli" / "templates" / "default" / "configs" / "datasets.yaml",
]
KINDS = ("generator", "agent", "a2a", "embedding", "tools", "harbor")
SHAPE = ("input_template", "output_template", "input_schema", "output_schema", "tools_template")

_GSM8K = {"question": "What is 6 x 7?", "answer": "6 x 7 = 42 #### 42"}
_ASK = {"question": "What is the capital of France?"}
_TOOL_ROW = {
    "messages": [{"role": "user", "content": "Weather in Paris?"}],
    "tools": [{"type": "function", "function": {"name": "get_weather", "description": "The weather in a city.", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}],
    "calls": [{"name": "get_weather", "arguments": {"city": "Paris"}}],
}
# A row of each example's source, by the example's name.
ROWS = {
    "gsm8k-20": _GSM8K,
    "gsm8k-terminal": _GSM8K,
    "gsm8k-tools": _GSM8K,
    "mmlu-math": {"question": "2 + 2 = ?", "choices": [3, 4, 5, 6], "answer": 1},
    "passages": {"query": "capital of France", "passages": ["Paris is the capital.", "Rome is in Italy."], "gold": ["Paris is the capital."]},
    "weather-calls": {"question": "Weather in Paris?", "calls": [{"name": "get_weather", "arguments": {"city": "Paris"}}]},
    # a tool-calling row: a Chat Completions request, messages and tools, and the gold calls
    "bfcl-simple": _TOOL_ROW,
    "row-calls": _TOOL_ROW,
    "web-questions": {"question": "What is on example.com?", "answer": "An example page."},
    "support-questions": {"question": "How do I reset my password?", "expected": "From the sign-in page."},
    "ask-agents": _ASK,
    "support-open": {"question": "How do I reset my password?"},
    "gsm8k-clean": _GSM8K,
    "gsm8k-pinned": _GSM8K,
    "gsm8k-smoke": _GSM8K,
    "gsm8k-full": _GSM8K,
    "ticket-routing": {"subject": "Charged twice", "body": "My card was billed twice this month.", "product": "Pro plan", "expected_team": " Billing "},
    "chat-replies": {"prompt": "Say hello.", "reply": "Hello!"},
    "qa-local": {"question": "How many legs does a spider have?", "answer": 8},
    # a folder row: the file's own keys plus `_filename`, `_stem`, `_path`
    "cases": {"question": "Is the contract valid?", "answer": "Yes.", "_filename": "case-01.json", "_stem": "case-01", "_path": "case-01.json"},
    # platform rows, as each provider hands them over
    "support-golden": {"input": {"question": "How do I reset my password?"}, "expected_output": {"answer": "From the sign-in page."}, "metadata": {"tier": "pro"}},
    "lf-golden": {"input": {"question": "How do I reset my password?"}, "expected_output": {"answer": "From the sign-in page."}, "metadata": {}},
    "ls-golden": {"inputs": {"question": "How do I reset my password?"}, "outputs": {"answer": "From the sign-in page."}, "metadata": {}},
    "opik-regressions": {"input": {"question": "How do I reset my password?"}, "expected_output": {"answer": "From the sign-in page."}, "metadata": {}},
    "phoenix-golden": {"input": {"question": "How do I reset my password?"}, "output": {"answer": "From the sign-in page."}, "metadata": {}},
    "bt-golden": {"input": {"question": "How do I reset my password?"}, "expected": {"answer": "From the sign-in page."}, "metadata": {}},
    "mlflow-golden": {"inputs": {"question": "How do I reset my password?"}, "expectations": {"answer": "From the sign-in page."}},
    "uc-questions": {"question": "How do I reset my password?", "reference_answer": "From the sign-in page."},
}


def _uncommented(text: str) -> str:
    """The template with its commented-out entries as entries: `  # name:` becomes `  name:`."""
    return "\n".join(re.sub(r"^  # ", "  ", line) if re.match(r"^  # ( |[A-Za-z0-9])", line) else line for line in text.splitlines())


def _examples():
    seen = set()
    for source in SOURCES:
        text = source.read_text()
        blocks = YAML_BLOCKS(text) if source.suffix == ".md" else [text, _uncommented(text)]
        for block in blocks:
            if "# ..." in block:
                continue  # an abridged snippet, not a whole entry
            try:
                doc = yaml.safe_load(block)
            except yaml.YAMLError:
                continue
            datasets = doc.get("datasets") if isinstance(doc, dict) else None
            for name, entry in (datasets or {}).items():
                if not isinstance(entry, dict) or entry.get("type") == "harbor_tasks":
                    continue  # task folders: no templates, the folders are the rows
                key = (source.name, name, json.dumps(entry, sort_keys=True, default=str))
                if key in seen:
                    continue
                seen.add(key)
                yield pytest.param(name, entry, id=f"{source.relative_to(ROOT)}:{name}")


def YAML_BLOCKS(text):
    return re.findall(r"```yaml\n(.*?)```", text, re.S)


@pytest.mark.parametrize(("name", "entry"), list(_examples()))
def test_every_dataset_example_reads_its_rows(name, entry, tmp_path):
    assert name in ROWS, f"no sample row for the example {name!r}: add one to ROWS"
    rows = tmp_path / "rows.jsonl"
    rows.write_text(json.dumps(ROWS[name]) + "\n")
    config = entry.get("config") or {}
    flat = {"type": "local", "path": str(rows), **{k: config[k] for k in SHAPE if k in config}}
    flat.update({k: entry[k] for k in KINDS if entry.get(k) is not None})
    dataset = load_dataset_from_config({"datasets": {name: flat}}, name)
    batch = next(iter(dataset()))
    assert len(batch[0]) == 1
