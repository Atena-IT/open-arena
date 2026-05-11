# Open Arena Show-Me-How Refresh Design

## Goal
Refresh the Open Arena show-me-how assets so the presentation supports a 25-minute English session with a realistic live-demo core: 6–7 minutes of introduction, 10 minutes of demo, 3 minutes of takeaways, and 5 minutes of live Q&A. The final deliverable must include a polished `.docx` speaker script inside `docs/` and a much more realistic CSV dataset with at least 200 interactions.

## Constraints
- Keep the overall message aligned with the existing deck: Open Arena as a lightweight, reproducible evaluation workflow.
- Shift the demo semantics away from generic business QA and toward a constrained deep-research agent similar to the Vanguard Research agent.
- Keep materials in English to match the existing deck and notebook.
- Assume Langfuse runs locally and is connected to the notebook/demo flow.
- Do not require slide edits unless a mismatch is discovered that blocks the script.

## Existing Inputs
- Deck: `docs/presentation/Xchange deck Show Me How Open Arena.pptx`
- Notebook: `demo/show_me_how_open_arena/open_arena_show_me_how.ipynb`
- Current dataset: `demo/show_me_how_open_arena/data/business_qa_demo.csv`
- Current config: `demo/show_me_how_open_arena/configs/business_qa_demo.yaml`
- Example research-report style reference: `Generative 3D World Models & Spatial Intelligence.md`

## Design Direction
Use a hybrid framing:
1. Preserve the deck’s AI Ops / evaluation narrative.
2. Replace the current FAQ-style demo data with mission-style research requests.
3. Center the live demo on one hero mission that shows mission definition, constrained retrieval intent, evaluation setup, and report output.
4. Use the `.docx` to connect the deck, notebook, and live operator steps into one presenter-friendly runbook.

## Artifact Changes
### 1. Dataset refresh
Update `demo/show_me_how_open_arena/data/business_qa_demo.csv` into a mission-based dataset with at least 200 rows.

Each row should represent one realistic interaction for a constrained deep-research agent. The dataset should cover multiple topic clusters and request types while keeping the same overall use case: turning a mission into a structured report with cited sources.

The CSV will use richer columns so the missions feel operational instead of synthetic:
- `scenario_id`
- `mission_title`
- `research_domain`
- `topic_cluster`
- `difficulty`
- `sme_owner`
- `timeframe_start`
- `timeframe_end`
- `allowed_domains`
- `focus_semantics`
- `output_type`
- `audience`
- `question`
- `expected_answer`

`question` will be a realistic mission prompt. `expected_answer` will not be a full report; it will describe the expected report behavior and structure, such as scope adherence, domain-native retrieval, source citation, deduplication, and final synthesis quality.

The rows should span realistic themes such as:
- generative 3D world models
- synthetic tabular data
- spatial semantics and scene graphs
- robotics simulation
- industrial AI scouting
- foundation-model benchmarking
- enterprise agent tooling
- domain-constrained innovation scouting

### 2. Config refresh
Update `demo/show_me_how_open_arena/configs/business_qa_demo.yaml` so the prompt template matches the new mission-style columns.

The config will still support a compact local demo, but the rendered input should read like a mission brief, including:
- target domain/topic
- time window
- allowed sources/domains
- semantic focus
- output expectation

The expected output field should continue to point at `expected_answer` so the evaluation story remains simple and demo-friendly.

### 3. Notebook refresh
Update `demo/show_me_how_open_arena/open_arena_show_me_how.ipynb` so its narrative matches the new demo semantics.

The notebook should:
- explain Open Arena as the evaluation backbone
- explain the dataset as a collection of research missions rather than support questions
- keep the Langfuse local checklist
- highlight one hero mission for the live walkthrough
- connect the mission to the idea of a structured final report with sources
- keep the demo portion practical and presenter-friendly rather than implementation-heavy

The notebook remains a demo asset, not the full 25-minute script.

### 4. Speaker-script document
Create a polished Word document at `docs/Open Arena Show Me How - Speaker Script.docx`.

The document should contain:
- session objective and audience framing
- minute-by-minute structure for the full 25-minute session
- slide-by-slide speaker script aligned with the current deck order
- a dedicated 10-minute live demo choreography based on the notebook
- logical operator steps to follow during the demo
- fallback guidance if Langfuse, local models, or notebook execution do not cooperate live
- concise takeaways aligned with the deck’s final message
- a final Q&A section that explicitly notes it is live and unscripted

## Talk Structure
### Intro (6–7 minutes)
Use slides 1–13 to frame the problem, the AI Ops loop, and why reproducible evaluation matters.

### Demo (10 minutes)
Use the notebook and one hero mission to show:
1. what the mission looks like
2. how Open Arena encodes it in config + dataset
3. what a run looks like
4. what Langfuse makes visible
5. what kind of report output the setup is meant to evaluate

### Takeaways (3 minutes)
Use the existing takeaway slides to reinforce:
- evaluate on realistic workload, not intuition
- compare variants under the same conditions
- treat traces, scores, and decision rationale as operational evidence

### Q&A (5 minutes)
Leave this unscripted in practice, but mention likely question themes and transition language in the docx.

## Verification
The refresh is complete when:
- the CSV contains at least 200 non-empty mission rows
- the YAML config references the new mission-style columns correctly
- the notebook text and demo cells no longer describe a generic business QA assistant
- the Word file exists under `docs/` and includes the timed flow, script, and demo steps
- the Word file passes document validation and a rendered visual check

## Scope Boundaries
- No deck redesign unless the existing slide order creates a direct mismatch with the script.
- No attempt to make the notebook a full production workflow.
- No fabricated “live results” that imply a run happened when it did not.
- No generic FAQ/support dataset patterns unless they are explicitly reframed as mission examples.
