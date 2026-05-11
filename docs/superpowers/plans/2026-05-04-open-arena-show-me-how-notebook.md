# Open Arena Show-Me-How Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a separate demo folder containing a fast 10–15 minute English notebook that explains Open Arena end-to-end through a business QA example, a local-style Langfuse walkthrough, and a light SME evaluation section.

**Architecture:** Keep the artifact self-contained under `demo/show_me_how_open_arena/` with one notebook, one small CSV dataset, and one compact YAML config. Ground every technical explanation in the real repository files so the notebook is presentation-friendly while still mapping to the actual CLI flow and schema.

**Tech Stack:** Jupyter notebook (`.ipynb`), CSV, YAML, Python stdlib/json, Open Arena config schema, local file references.

---

### Task 1: Create the demo dataset and runnable config

**Files:**
- Create: `demo/show_me_how_open_arena/data/business_qa_demo.csv`
- Create: `demo/show_me_how_open_arena/configs/business_qa_demo.yaml`
- Check: `.env.example:1-16`
- Check: `config.example.yaml:83-100`
- Check: `src/config/types.py:8-181`

- [ ] **Step 1: Create the business QA dataset**

```csv
scenario_id,business_area,difficulty,sme_owner,question,expected_answer
support_refund_001,Customer Support,medium,Support Operations,"A customer on the Growth plan asks for a refund 10 days after renewal. What should the assistant say?","Explain that refunds are allowed within 14 days of renewal for the Growth plan, confirm eligibility, and direct the customer to the billing workflow."
ops_incident_001,Operations,high,Incident Management,"Checkout latency in the EU region is above the internal threshold. What should the assistant recommend first?","Recommend treating the issue as a production incident, confirm the affected region and metric, and escalate through the incident workflow before proposing optimizations."
finance_billing_001,Finance,medium,Billing Operations,"A finance analyst sees duplicate charges for the same customer invoice. What is the safest first response?","Advise verifying the invoice IDs and payment events, avoid issuing an immediate refund before validation, and route the case to billing operations for confirmation."
policy_retention_001,Compliance,high,Compliance Lead,"A user asks how long support transcripts are retained. How should the assistant answer?","State the approved retention policy only if it is documented in the knowledge source; otherwise say the answer must be confirmed with the compliance owner."
ops_sla_001,Operations,low,Service Management,"A prospect asks whether the standard SLA guarantees a 15-minute response time on weekends. What should the assistant do?","Do not promise an unsupported SLA. Answer with the documented standard SLA terms and suggest confirming enterprise exceptions with sales or support leadership."
cs_escalation_001,Customer Support,medium,Support Operations,"A customer asks for an exception to a closed refund window because of a product outage. What is the best response?","Acknowledge the context, avoid approving the exception directly, and explain that refund exceptions require review through the documented escalation path."
```

- [ ] **Step 2: Create the compact local demo config**

```yaml
dataset:
  name: "Business QA Demo"
  description: "Small business-style QA dataset for a show-me-how walkthrough"
  source:
    provider: "local"
    path: "demo/show_me_how_open_arena/data/business_qa_demo.csv"
    format: "csv"
  input: |
    Business area: {{ business_area }}
    SME owner: {{ sme_owner }}
    Difficulty: {{ difficulty }}

    Question:
    {{ question }}
  expected_output: "{{ expected_answer }}"
  limit: 6

system_prompt: |
  You are an enterprise support and operations assistant.
  Answer clearly, avoid unsupported promises, and stay consistent with documented policy.

experiments:
  - name: "llama3.2-local"
    litellm:
      model: "ollama_chat/llama3.2:latest"
      api_base: "http://localhost:11434"
      temperature: 0.2
      max_tokens: 256
  - name: "qwen3-0.6b-local"
    litellm:
      model: "ollama_chat/qwen3:0.6b"
      api_base: "http://localhost:11434"
      temperature: 0.2
      max_tokens: 256

evaluation:
  method: "llm_as_judge"
  litellm:
    model: "ollama_chat/llama3.2:latest"
    api_base: "http://localhost:11434"
    temperature: 0.0
    max_tokens: 1024
  score_name: "business_quality_score"
  max_concurrency: 5
  system_prompt: >
    You are an expert evaluator for enterprise question-answering.
    Given a JSON object with the fields "input", "output", and "expected_output",
    score how well the output matches the expected answer for correctness, policy safety,
    and practical usefulness.
    Return ONLY a JSON object with this format:
    {"thinking": "Short reasoning.", "score": 4}
    The score must be an integer from 1 to 5.
  system_prompt_no_reference: >
    You are an expert evaluator for enterprise question-answering.
    Given a JSON object with the fields "input" and "output",
    score the answer for correctness, policy safety, and practical usefulness.
    Return ONLY a JSON object with this format:
    {"thinking": "Short reasoning.", "score": 4}
    The score must be an integer from 1 to 5.
```

- [ ] **Step 3: Validate the config against the real schema**

```bash
python - <<'PY'
from pathlib import Path
from src.config.types import ExperimentsFile

config_path = Path("demo/show_me_how_open_arena/configs/business_qa_demo.yaml")
config = ExperimentsFile.from_yaml(config_path)
print(config.dataset.name)
print(len(config.experiments))
print(config.evaluation.method)
PY
```

Expected output:
- `Business QA Demo`
- `2`
- `llm_as_judge`

### Task 2: Create the notebook

**Files:**
- Create: `demo/show_me_how_open_arena/open_arena_show_me_how.ipynb`
- Read: `README.md:5-15`
- Read: `README.md:54-75`
- Read: `src/main_cli.py:205-238`
- Read: `src/config/types.py:8-181`
- Read: `resources/data/QA.xlsx`
- Read: `resources/data/ToolsExample.xlsx`

- [ ] **Step 1: Write the notebook sections in order**

```text
1. Title and goals
2. Why this matters for business teams
3. Open Arena mental model: dataset -> experiments -> evaluation -> Langfuse
4. Demo folder discovery and repo grounding
5. Business QA dataset preview from CSV
6. Demo config walkthrough
7. Schema validation against src/config/types.py
8. Runtime walkthrough based on src/main_cli.py
9. Langfuse local demo checklist based on .env.example
10. Existing repo dataset examples: QA.xlsx and ToolsExample.xlsx
11. Light SME evaluation extension with verifier criteria example
12. 10-15 minute presenter talk track and closing summary
```

- [ ] **Step 2: Include executable code cells for safe local inspection**

```python
from pathlib import Path
import json
import yaml
import polars as pl


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "src").exists():
            return candidate
    raise FileNotFoundError("Could not locate the repository root.")

REPO_ROOT = find_repo_root(Path.cwd())
DEMO_DIR = REPO_ROOT / "demo" / "show_me_how_open_arena"
DATASET_PATH = DEMO_DIR / "data" / "business_qa_demo.csv"
CONFIG_PATH = DEMO_DIR / "configs" / "business_qa_demo.yaml"

pl.read_csv(DATASET_PATH)
```

- [ ] **Step 3: Add guided, non-deceptive runtime cells**

```python
import os

required_env = [
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_HOST",
]
{key: bool(os.getenv(key)) for key in required_env}
```

```python
from textwrap import dedent

print(dedent(f"""
Run when Langfuse and your model backend are ready:
  arena --config {CONFIG_PATH.relative_to(REPO_ROOT)}

Optional:
  arena --config {CONFIG_PATH.relative_to(REPO_ROOT)} --skip-upload
"""))
```

- [ ] **Step 4: Add a presenter talk track instead of extra implementation detail**

```text
- 0:00-1:30 business framing
- 1:30-3:00 mental model
- 3:00-5:00 dataset walkthrough
- 5:00-7:30 config walkthrough
- 7:30-9:30 execution flow
- 9:30-11:30 Langfuse walkthrough
- 11:30-13:00 SME evaluation angle
- 13:00-15:00 closing and next steps
```

### Task 3: Verify the notebook and references

**Files:**
- Check: `demo/show_me_how_open_arena/open_arena_show_me_how.ipynb`
- Check: `demo/show_me_how_open_arena/data/business_qa_demo.csv`
- Check: `demo/show_me_how_open_arena/configs/business_qa_demo.yaml`

- [ ] **Step 1: Validate notebook JSON and file existence**

```bash
python - <<'PY'
from pathlib import Path
import json

repo = Path('.')
notebook_path = repo / 'demo/show_me_how_open_arena/open_arena_show_me_how.ipynb'
config_path = repo / 'demo/show_me_how_open_arena/configs/business_qa_demo.yaml'
dataset_path = repo / 'demo/show_me_how_open_arena/data/business_qa_demo.csv'

for path in [notebook_path, config_path, dataset_path]:
    assert path.exists(), f"Missing file: {path}"

with notebook_path.open() as f:
    notebook = json.load(f)

assert notebook['nbformat'] == 4
assert len(notebook['cells']) >= 10
print('Notebook and demo files look structurally valid.')
PY
```

Expected output:
- `Notebook and demo files look structurally valid.`

- [ ] **Step 2: Verify the notebook mentions the real repo touchpoints**

```bash
python - <<'PY'
from pathlib import Path
import json

notebook_path = Path('demo/show_me_how_open_arena/open_arena_show_me_how.ipynb')
text = notebook_path.read_text()
required_strings = [
    'README.md',
    'src/main_cli.py',
    'src/config/types.py',
    'resources/data/QA.xlsx',
    'resources/data/ToolsExample.xlsx',
    'LANGFUSE_HOST',
]
for item in required_strings:
    assert item in text, f'Missing notebook reference: {item}'
print('Notebook references the expected repository files and Langfuse variables.')
PY
```

Expected output:
- `Notebook references the expected repository files and Langfuse variables.`

- [ ] **Step 3: Do not commit**

```text
Leave the branch uncommitted exactly as requested by the user.
```
