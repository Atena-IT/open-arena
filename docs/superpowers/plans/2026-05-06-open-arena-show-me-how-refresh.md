# Open Arena Show-Me-How Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the generic business-QA demo with a 200+ mission-based research dataset, align the local demo config and notebook with that agent behavior, and create a polished speaker-script `.docx` in `docs/` for the full 25-minute session.

**Architecture:** Keep runnable demo assets under `demo/show_me_how_open_arena/`, with the CSV and YAML describing research missions for a constrained deep-research agent. Keep the notebook focused on the 10-minute live demo, while the Word document in `docs/` carries the full intro/demo/takeaway/Q&A choreography and speaker script.

**Tech Stack:** CSV, YAML, Jupyter notebook JSON, Python stdlib, existing Open Arena config schema, OfficeCLI for `.docx` creation and validation.

---

### Task 1: Replace the demo dataset and prompt template

**Files:**
- Modify: `demo/show_me_how_open_arena/data/business_qa_demo.csv`
- Modify: `demo/show_me_how_open_arena/configs/business_qa_demo.yaml`
- Check: `demo/show_me_how_open_arena/open_arena_show_me_how.ipynb`

- [ ] **Step 1: Write the failing structural check for the dataset/config pair**

```bash
python - <<'PY'
from pathlib import Path
import csv
import yaml

repo = Path('.')
dataset_path = repo / 'demo/show_me_how_open_arena/data/business_qa_demo.csv'
config_path = repo / 'demo/show_me_how_open_arena/configs/business_qa_demo.yaml'

required_columns = {
    'scenario_id',
    'mission_title',
    'research_domain',
    'topic_cluster',
    'difficulty',
    'sme_owner',
    'timeframe_start',
    'timeframe_end',
    'allowed_domains',
    'focus_semantics',
    'output_type',
    'audience',
    'question',
    'expected_answer',
}

with dataset_path.open(newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

with config_path.open(encoding='utf-8') as f:
    config_text = f.read()

missing = required_columns - set(rows[0].keys())
assert not missing, f'Missing columns: {sorted(missing)}'
assert len(rows) >= 200, f'Expected at least 200 rows, found {len(rows)}'
assert 'Mission title:' in config_text
assert '{{ allowed_domains }}' in config_text
assert '{{ timeframe_start }}' in config_text
assert '{{ timeframe_end }}' in config_text
print('Dataset/config mission structure looks correct.')
PY
```

Expected: FAIL because the current CSV is too small and the current config still describes business QA rows.

- [ ] **Step 2: Run the check and verify it fails**

Run: `python - <<'PY' ... PY`
Expected: assertion failure about missing columns and/or row count.

- [ ] **Step 3: Rewrite the CSV as a mission-based dataset with at least 200 rows**

```python
from pathlib import Path
import csv

DATASET_PATH = Path('demo/show_me_how_open_arena/data/business_qa_demo.csv')

clusters = {
    'generative_3d_world_models': {
        'domains': ['arxiv.org', 'blogs.nvidia.com', 'technologyreview.com'],
        'focuses': ['world generation', 'spatial intelligence', 'simulation workflows'],
    },
    'synthetic_tabular_data': {
        'domains': ['arxiv.org', 'github.com', 'openreview.net'],
        'focuses': ['privacy-utility tradeoff', 'evaluation benchmarks', 'enterprise adoption'],
    },
    'spatial_semantics_scene_graphs': {
        'domains': ['arxiv.org', 'omniverse.nvidia.com', 'github.com'],
        'focuses': ['scene graph pipelines', 'USD semantics', 'semantic querying'],
    },
    'robotics_simulation': {
        'domains': ['blogs.nvidia.com', 'arxiv.org', 'developer.nvidia.com'],
        'focuses': ['sim-to-real', 'synthetic data', 'robotics evaluation'],
    },
    'industrial_ai_scouting': {
        'domains': ['technologyreview.com', 'arxiv.org', 'openai.com'],
        'focuses': ['industrial copilots', 'simulation stacks', 'innovation signals'],
    },
    'foundation_model_benchmarking': {
        'domains': ['arxiv.org', 'openai.com', 'anthropic.com'],
        'focuses': ['evaluation design', 'benchmark quality', 'decision criteria'],
    },
    'enterprise_agent_tooling': {
        'domains': ['langfuse.com', 'langchain.com', 'github.com'],
        'focuses': ['tracing', 'tool-enabled agents', 'orchestrated evaluation'],
    },
    'domain_constrained_scouting': {
        'domains': ['arxiv.org', 'blogs.nvidia.com', 'technologyreview.com'],
        'focuses': ['mission-native retrieval', 'domain filtering', 'time-bounded scouting'],
    },
}

time_windows = [
    ('2026-01-01', '2026-03-31'),
    ('2025-10-01', '2026-03-31'),
    ('2025-01-01', '2026-04-30'),
]
output_types = ['executive memo', 'benchmark snapshot', 'deep-dive report', 'technology radar']
audiences = ['innovation lead', 'product manager', 'R&D director', 'strategy team']
difficulties = ['medium', 'high']

rows = []
for cluster_name, cluster in clusters.items():
    for idx in range(27):
        start, end = time_windows[idx % len(time_windows)]
        focus = cluster['focuses'][idx % len(cluster['focuses'])]
        output_type = output_types[idx % len(output_types)]
        audience = audiences[idx % len(audiences)]
        domain_label = cluster_name.replace('_', ' ').title()
        question_domain = cluster_name.replace('_', ' ')
        rows.append({
            'scenario_id': f'{cluster_name}_{idx + 1:03d}',
            'mission_title': f"{domain_label} mission {idx + 1}",
            'research_domain': domain_label,
            'topic_cluster': cluster_name,
            'difficulty': difficulties[idx % len(difficulties)],
            'sme_owner': 'Vanguard Research Lead',
            'timeframe_start': start,
            'timeframe_end': end,
            'allowed_domains': ', '.join(cluster['domains']),
            'focus_semantics': focus,
            'output_type': output_type,
            'audience': audience,
            'question': (
                f"Prepare a {output_type} on {focus} within {question_domain}. "
                f"Use only the allowed domains, treat the {start} to {end} window as a hard constraint, "
                f"prioritize mission-native retrieval, and finish with a structured report plus cited sources for the {audience}."
            ),
            'expected_answer': (
                'The answer should stay inside the requested topic and timeframe, use allowed domains only, '
                'deduplicate overlapping sources, synthesize findings into a structured report, and end with cited sources plus concrete decision-oriented takeaways.'
            ),
        })

assert len(rows) == 216
with DATASET_PATH.open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
```

- [ ] **Step 4: Update the YAML prompt template to render a mission brief**

```yaml
dataset:
  name: "Vanguard Research Missions Demo"
  description: "Mission-based dataset for constrained deep-research agent evaluation"
  source:
    provider: "local"
    path: "demo/show_me_how_open_arena/data/business_qa_demo.csv"
    format: "csv"
  input: |
    Mission title: {{ mission_title }}
    Research domain: {{ research_domain }}
    Topic cluster: {{ topic_cluster }}
    Time window: {{ timeframe_start }} to {{ timeframe_end }}
    Allowed domains: {{ allowed_domains }}
    Focus semantics: {{ focus_semantics }}
    Output type: {{ output_type }}
    Audience: {{ audience }}
    Difficulty: {{ difficulty }}
    SME owner: {{ sme_owner }}

    Mission:
    {{ question }}
  expected_output: "{{ expected_answer }}"
```

- [ ] **Step 5: Run the structural check again**

Run: `python - <<'PY' ... PY`
Expected: `Dataset/config mission structure looks correct.`

### Task 2: Rewrite the notebook around the mission-based demo

**Files:**
- Modify: `demo/show_me_how_open_arena/open_arena_show_me_how.ipynb`
- Check: `demo/show_me_how_open_arena/data/business_qa_demo.csv`
- Check: `demo/show_me_how_open_arena/configs/business_qa_demo.yaml`

- [ ] **Step 1: Write the failing notebook content check**

```bash
python - <<'PY'
from pathlib import Path
import json

notebook_path = Path('demo/show_me_how_open_arena/open_arena_show_me_how.ipynb')
text = notebook_path.read_text(encoding='utf-8')
required_strings = [
    'research missions',
    'mission-native retrieval',
    'allowed domains',
    'structured report with sources',
    'Vanguard Research',
]
for item in required_strings:
    assert item in text, f'Missing notebook phrase: {item}'
assert 'customer support assistants' not in text
print('Notebook reflects the mission-based research demo.')
PY
```

Expected: FAIL because the current notebook still describes a business QA assistant.

- [ ] **Step 2: Run the notebook content check and confirm failure**

Run: `python - <<'PY' ... PY`
Expected: assertion failure on missing mission-based phrases.

- [ ] **Step 3: Replace the markdown arc so the notebook supports the live demo**

```text
1. Title: Open Arena as evaluation backbone for a constrained research agent
2. Why realistic mission-based evaluation matters
3. Mental model: dataset -> experiments -> evaluation -> Langfuse
4. What makes the Vanguard-style agent different
5. Mission dataset preview
6. Config walkthrough as mission brief
7. Runtime story and Langfuse visibility
8. Hero mission walkthrough
9. What the final report is meant to look like
10. Suggested demo narration for the presenter
11. Final takeaway
```

- [ ] **Step 4: Update the executable cells to preview the new mission fields**

```python
with DATASET_PATH.open(newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

hero_fields = [
    'mission_title',
    'research_domain',
    'timeframe_start',
    'timeframe_end',
    'allowed_domains',
    'focus_semantics',
    'question',
]

print(f'Mission rows in demo dataset: {len(rows)}')
{k: rows[0][k] for k in hero_fields}
```

```python
hero_mission = rows[0]
print({
    'mission_title': hero_mission['mission_title'],
    'focus_semantics': hero_mission['focus_semantics'],
    'allowed_domains': hero_mission['allowed_domains'],
    'time_window': f"{hero_mission['timeframe_start']} -> {hero_mission['timeframe_end']}",
})
```

- [ ] **Step 5: Rerun the notebook content check**

Run: `python - <<'PY' ... PY`
Expected: `Notebook reflects the mission-based research demo.`

### Task 3: Create the speaker-script Word document

**Files:**
- Create: `docs/Open Arena Show Me How - Speaker Script.docx`
- Check: `docs/presentation/Xchange deck Show Me How Open Arena.pptx`
- Check: `demo/show_me_how_open_arena/open_arena_show_me_how.ipynb`

- [ ] **Step 1: Inspect the slide order and lock the script structure**

```text
Slides 1-13 = intro story
Slide 14 = demo handoff
Slides 15-17 = takeaways + close
Slide 18 = Q&A
```

- [ ] **Step 2: Create the document skeleton with explicit sections**

```text
Title page
Session objective
Minute-by-minute agenda
Slide-by-slide script for intro
10-minute demo choreography
Fallback plan
Takeaways
Q&A handoff
```

- [ ] **Step 3: Write the core document content in English**

```markdown
# Open Arena Show Me How — Speaker Script

## Session objective
Show how Open Arena turns model and agent evaluation into a reproducible operational workflow.

## Agenda
- 0:00-6:30 Introduction
- 6:30-16:30 Demo
- 16:30-19:30 Takeaways
- 19:30-25:00 Q&A

## Demo choreography
1. Introduce the hero mission.
2. Show how the mission is encoded in the CSV.
3. Show the YAML config and explain the mission brief template.
4. Switch to the notebook and preview the hero mission.
5. Explain what will appear in Langfuse.
6. Describe the expected final report structure and decision value.
```

- [ ] **Step 4: Build the `.docx` and apply basic formatting with OfficeCLI**

```bash
FILE="docs/Open Arena Show Me How - Speaker Script.docx"
officecli create "$FILE"
officecli open "$FILE"
officecli add "$FILE" /body --type paragraph --prop text="Open Arena Show Me How — Speaker Script" --prop style=Heading1 --prop size=22pt --prop bold=true --prop spaceAfter=12pt
officecli add "$FILE" /body --type paragraph --prop text="Session objective" --prop style=Heading2 --prop size=14pt --prop bold=true --prop spaceBefore=12pt --prop spaceAfter=6pt
officecli add "$FILE" /body --type paragraph --prop text="Show how Open Arena turns model and agent evaluation into a reproducible operational workflow." --prop size=11pt
officecli add "$FILE" / --type footer --prop type=default --prop align=center --prop size=9pt --prop text="Page " --prop field=page
officecli close "$FILE"
```

- [ ] **Step 5: Validate the Word file**

Run: `officecli validate "docs/Open Arena Show Me How - Speaker Script.docx"`
Expected: `no errors found`

### Task 4: Verify the refreshed assets together

**Files:**
- Check: `demo/show_me_how_open_arena/data/business_qa_demo.csv`
- Check: `demo/show_me_how_open_arena/configs/business_qa_demo.yaml`
- Check: `demo/show_me_how_open_arena/open_arena_show_me_how.ipynb`
- Check: `docs/Open Arena Show Me How - Speaker Script.docx`

- [ ] **Step 1: Verify dataset size and schema**

```bash
python - <<'PY'
from pathlib import Path
import csv

path = Path('demo/show_me_how_open_arena/data/business_qa_demo.csv')
with path.open(newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
print(len(rows))
print(rows[0]['mission_title'])
print(rows[-1]['topic_cluster'])
PY
```

Expected output:
- `216`
- a non-empty mission title
- a non-empty topic cluster

- [ ] **Step 2: Verify notebook/config references**

```bash
python - <<'PY'
from pathlib import Path

notebook_text = Path('demo/show_me_how_open_arena/open_arena_show_me_how.ipynb').read_text(encoding='utf-8')
config_text = Path('demo/show_me_how_open_arena/configs/business_qa_demo.yaml').read_text(encoding='utf-8')
for item in ['mission-native retrieval', 'allowed_domains', 'timeframe_start', 'Vanguard Research']:
    assert item in notebook_text or item in config_text, f'Missing reference: {item}'
print('Notebook and config reference the mission-based demo correctly.')
PY
```

Expected output:
- `Notebook and config reference the mission-based demo correctly.`

- [ ] **Step 3: Validate and visually inspect the docx**

Run:
- `officecli validate "docs/Open Arena Show Me How - Speaker Script.docx"`
- `officecli view "docs/Open Arena Show Me How - Speaker Script.docx" html`

Expected:
- validation passes
- rendered HTML shows the agenda, intro script, demo choreography, takeaways, and Q&A sections

- [ ] **Step 4: Do not commit**

```text
Leave the branch uncommitted unless the user explicitly asks for a commit.
```
