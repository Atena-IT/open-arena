# Open Arena TODO

Outstanding gaps from [`feature-matrix.md`](./feature-matrix.md) /
[`notes.md`](./notes.md), turned into actionable items. Anything not
listed here is already covered by the current code.

> **Note on the matrix.** Several rows in `feature-matrix.md` are
> marked **Working** based on a sibling/earlier version of the project
> (MCP execution, Langfuse persistence, `llm_as_verifier`,
> `ExecutionResult`, `AgentCaller`). None of those exist in the
> current `src/` tree. They are listed below under
> [Wiki ↔ code mismatches](#wiki--code-mismatches) and need either
> implementation or matrix correction.

> **Note on keras-tuner.** Several "platform" capabilities described
> in `notes.md` are already provided by the keras-tuner layer the
> sweep runs on top of, and do not need to be reinvented. See
> [What keras-tuner already covers](#what-keras-tuner-already-covers).

> **Note on synthetic data.** `notes.md §6` proposes a synthetic-data
> module inside the platform. That's out of scope — synthetic
> generation, selection, and dataset preparation are **per-project**
> concerns implemented by users in `prepare_data.py` (the documented
> hook: "anything useful to prepare the raw data"). Users can plug
> in any external generation engine they want there — e.g. NVIDIA
> Data Designer for structured synthetic-data workflows — and emit
> rows into `raw_data/` for the existing `local` / `folder` / `huggingface`
> adapters to ingest. The platform's job is to ingest whatever rows
> that script produces, not to ship a generation engine.

## High priority

### 1. Dataset & benchmark registry
- [ ] Promote the canonical `(input, expected_output, metadata)` row
      into a benchmark registry: dataset identity, version, source
      provenance, task type, schema shape, output compatibility.
- [ ] Unify provider-specific version hints (HF revision, Phoenix
      version, Braintrust/Weave version) behind one identity+version
      schema every run can reference.
- [ ] Add a second canonical dataset shape for **scenario-based** agent
      tasks (environment assumptions, tools, trajectory expectations,
      verifier attachments) alongside the prompt/reference shape.

### 2. Execution arena
- [ ] Make the arena **step-aware**: promote turns, tool calls, and
      intermediate states to first-class run structures rather than
      optional attachments. (The trial → metric model from keras-tuner
      handles run-level identity and persistence; what's missing is
      sub-trial / per-step structure.)

### 3. Evaluation & verifier subsystem
- [ ] Turn the current reward layer into a plugin-style verification
      framework with a common contract (pointwise / pairwise / panel /
      reference-based / reference-free / trajectory-aware).
- [ ] Step-level evaluators with a dedicated step-score schema,
      separate from final-answer scoring (tool correctness, recovery,
      planning quality, policy adherence).

### 4. Observability & benchmark outputs
- [ ] Cross-run aggregation: today each `arena` invocation writes its
      own `.kt/` dir + `last_run.tsv`. Define a way to assemble
      historical runs into one leaderboard (keyed by benchmark +
      dataset version + experiment config + eval method + date) on
      top of the per-trial artifacts keras-tuner already produces.
- [ ] Optional export of the per-trial keras-tuner artifact to a
      separate observability backend (Langfuse / MLflow / W&B …) for
      teams that want shared inspection. Keep `.kt/` as the source of
      truth.

### 5. Agent evaluation
- [ ] Dedicated agent-evaluation module — not a special case of
      tool-enabled execution. Remote endpoint abstraction, normalized
      trajectory capture, response-envelope contract.
- [ ] End-to-end verification across deployed agents (final outcome +
      path).

## Medium priority

### Evaluation
- [ ] DeepEval (or equivalent) wrapper: normalize external semantic
      metric engines into the common scoring model.
- [ ] MultiJudge Panel methodology — multi-grader aggregation
      abstraction (the existing `JudgePanel` reward is a starting
      point; promote it to a generic verifier pattern).

### Agent evaluation
- [ ] A2A / ACP protocol-aware trajectory contracts.
- [ ] REST API response model for agent runs (depends on §7).

### 7. API & integration layer
- [ ] Programmatic API for runs, datasets, leaderboards (sits beside
      the CLI, shares internal modules).
- [ ] Integration surface for remote orchestration on top of the API.

## Low priority

- [ ] Sandbox support for agent evaluation (env setup, side effects,
      cleanup, isolation). Defer until scenario execution is mature.
- [ ] Deeper orchestration integrations — only after the domain model
      stabilizes.

## What keras-tuner already covers

Things `notes.md` describes as future platform work that the sweep
already gets for free from `keras_tuner` — don't reinvent these:

- **Run identity & hyperparameter resolution.** `trial.hyperparameters`
  is the run identity (`language_model`, `dataset`, future axes).
  Configuration resolution = the HP space.
- **Durable, resumable run artifact.** Every trial is serialized to
  `.kt/<project>/trial_<id>/` with HP + metrics + status. `--no-cache`
  discards it; otherwise resume is automatic. This is the portable
  run record — no separate artifact format needed.
- **Per-trial metric registry.** `trial.metrics.register(alias,
  direction=…)` + `get_best_value(alias)` already gives a typed,
  multi-metric scoring model with min/max semantics. Each
  experiment-level reward is one alias.
- **Best-trial / leaderboard query.** `oracle.get_best_trials()` is the
  in-memory leaderboard. The CLI uses it to render the matrices and
  `analyze.py` reads `last_run.tsv` for the long-format view.
- **Status tracking.** `trial.status` (`COMPLETED` / `INVALID` /
  `FAILED`) — failed cells render as the status string in the matrix.
- **Project-level isolation.** Different `project_name` ⇒ different
  `.kt/` subdir, so multiple sweeps coexist without colliding.

The actual missing platform piece is **cross-run** aggregation
(leaderboard across many sweeps over time), not per-run.

## Wiki ↔ code mismatches

These rows are marked **Working** in `feature-matrix.md` but are not
present in the current `src/`. Either implement them in this repo or
correct the matrix — pick one per row.

- [ ] **MCP-backed agentic execution / `AgentCaller`.** No MCP code in
      `src/`. Matrix row "Model + MCP servers" claims this works.
- [ ] **Langfuse run / score persistence.** Only the dataset *reader*
      exists (`src/datasets/langfuse_dataset.py`); no trace upload or
      score writeback. Matrix row "Langfuse tracing and score
      persistence" claims this works.
- [ ] **`llm_as_verifier` (pairwise + logprob extraction).** Not in
      `src/` or in the synalinks built-ins this repo wraps (only
      `lm_as_judge`, `exact_match`, `cosine_similarity` are exported
      via `synalinks.rewards`). Matrix row "LLM as verifier" claims
      this works.
- [ ] **`ExecutionResult` run artifact.** Referenced in
      `notes.md §2`; no such type exists in the codebase. The current
      sweep returns synalinks `program.evaluate()` dicts directly.
- [ ] **Trajectory capture in the execution layer.** Trajectories are
      mentioned only inside reward modules (RLM agent's internal log).
      The arena itself doesn't model or persist trajectories.
