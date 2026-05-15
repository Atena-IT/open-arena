# Open Arena TODO

Outstanding gaps from [`feature-matrix.md`](./feature-matrix.md) /
[`notes.md`](./notes.md), turned into actionable items. Anything not
listed here is already covered by the current code.

> **Note on the matrix.** Some rows in `feature-matrix.md` are marked
> **Working** based on a sibling/earlier version of the project
> (`ExecutionResult`, Langfuse persistence). They do not exist in the
> current `src/` tree — listed below under
> [Wiki ↔ code mismatches](#wiki--code-mismatches) and need either
> implementation or matrix correction. MCP-backed agentic execution,
> DeepEval, `lm_as_verifier`, and trajectory capture were previously
> listed there: the first two are implemented as eval rewards;
> `lm_as_verifier` exists as a deliberately-separate verifier
> subsystem at `src/verifiers/` (group-relative scoring — not a
> YAML-resolvable reward); and the agent's `FunctionCallingAgent`
> emits its full trajectory into `y_pred` by default via
> `return_inputs_with_trajectory`, so trajectory capture already
> flows through the eval pipeline.

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

## Recently completed

These items appeared as gaps in earlier passes; they have since shipped
and are listed here only so a reader of this file can tell what's
already in place.

- **Per-dataset reward direction (`max` / `min`).** Per-dataset `reward:`
  blocks accept `direction:`; top-level `metrics:` and per-dataset
  `metrics:` entries do too. Honored by the per-dataset tuner (used as
  the `Objective.direction`) and persisted into the `direction` column
  of `.open-arena/last_run.tsv` for downstream agents.
- **Function-calling agents with MCP tools.** `synalinks.FunctionCallingAgent`
  + `synalinks.MultiServerMCPClient` wired through `src/program.py:build_agent`
  + a top-level `mcp_servers:` registry referenced by each agentic
  dataset's `agent.mcp_servers: [name1, name2]` list. The agent emits
  its full tool-call / observation trajectory into `y_pred` by default
  (`return_inputs_with_trajectory=True`), so trajectories already flow
  through to rewards / metrics on every agentic row.
- **DeepEval reward wrapper.** `src/rewards/deep_eval.py` exposes any
  `deepeval.metrics` class via the standard YAML reward dict, with
  per-slot mask routing for `LLMTestCase` inputs.
- **Verifier subsystem (`src/verifiers/`).** Pairwise / group-relative
  scoring primitives live here — currently `LMAsVerifier`. Kept
  separate from `src/rewards/` on purpose: verifier scores are
  meaningful only relative to a batch composition, so they don't fit
  `program.evaluate()`-style sweeps where an absolute,
  dataset-aggregable score is expected. Useful for GRPO-style
  rollout-advantage training signals downstream of the arena.
- **Operational + classification metric registry from synalinks.**
  Top-level `metrics:` is the always-on observability list (LM /
  embedding / program-wide counters, token throughput, cache hit
  rates). Per-dataset `metrics:` adds task-shaped classification
  metrics (`categorical_*`, `binary_*`, token-level `accuracy /
  precision / recall / f1_score`) — declarative, ride along the
  primary `evaluate()` pass.
- **Synalinks default LM / embedder + seed from YAML.** Top-level
  `default_language_model:`, `default_embedding_model:`, `seed:`
  applied before any rewards / programs are built.
- **Async-aware trial dispatch.** `OpenArenaTuner.run_trial` calls
  `synalinks.utils.run_maybe_nested` (not `asyncio.run`), so the
  sweep can be driven from inside a live event loop (Jupyter,
  FastAPI handler, pytest-asyncio).
- **Keras stub removed.** `src/evaluate.py` calls
  `synalinks.disable_keras_backend()` directly; the local
  `src/keras_stub.py` shim is gone.
- **Scripts moved under `src/`.** `evaluate.py` and `program.py` now
  live under `src/`; the `arena` console script resolves
  `src.evaluate:main`. Invoke the sweep with `uv run arena`.
- **`config.py` extracted.** Pydantic-validated YAML parsing lives in
  `src/config.py` (`Config.load(path)`); `src/evaluate.py` only carries
  sweep and program-creation logic.
- **`analyze.py` removed.** Post-sweep scoring used to be a fixed
  script (`top1` / `pairwise` / Spearman columns). The TSV is now
  the contract: the autoresearch agent reads `.open-arena/last_run.tsv`
  directly and computes whatever statistic fits the iteration.

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
> Rewards are already pluggable (snake_case-keyed registry, project-local
> classes auto-discovered, DeepEval wrapper, `MultiJudgePanel`,
> `RecursiveLMAsJudge`). Operational + classification metrics ride along
> the primary `evaluate()` pass via the top-level `metrics:` block plus
> per-dataset `metrics:` lists. The remaining gaps are about *contracts*
> across reward shapes, not about whether rewards exist.

- [ ] Formalize the verifier contract — today pointwise / pairwise /
      panel / reference-based / reference-free / trajectory-aware
      verifiers all subclass synalinks `Reward`, but there's no
      typed shape declaring which family a reward belongs to (so
      a sweep can't reason about "this column is a panel score" vs
      "this column is exact match").
- [ ] Step-level evaluators with a dedicated step-score schema,
      separate from final-answer scoring (tool correctness, recovery,
      planning quality, policy adherence). Trajectories are already
      emitted by the agent (`return_inputs_with_trajectory=True`);
      what's missing is a typed schema for them (see §5) so step
      evaluators can bind to named slots instead of string-indexing
      a generic message list.

### 4. Observability & benchmark outputs
- [ ] Cross-run aggregation: today each `arena` invocation writes its
      own `.open-arena/` dir + `last_run.tsv`. Define a way to assemble
      historical runs into one leaderboard (keyed by benchmark +
      dataset version + experiment config + eval method + date) on
      top of the per-trial artifacts keras-tuner already produces.
- [ ] Optional export of the per-trial keras-tuner artifact to a
      separate observability backend (Langfuse / MLflow / W&B …) for
      teams that want shared inspection. Keep `.open-arena/` as the source of
      truth.

### 5. Agent evaluation
> Local agents (function-calling + MCP tools) are wired through
> `program.build_agent` since the v2 agent commit (`e173a28`). The
> agent emits its full tool-call / observation trajectory into `y_pred`
> by default (`FunctionCallingAgent.return_inputs_with_trajectory=True`),
> so trajectories already flow through the eval pipeline and are
> persisted per row in the trial's `program.evaluate()` output. What's
> missing is structure *around* that trajectory, not the trajectory itself.

- [ ] Typed trajectory schema — today the trajectory is a generic
      `ChatMessages` list; rewards / metrics have to know by convention
      that "this field is the agent path." A typed shape (with tool
      calls, observations, intermediate states as named slots) would
      let downstream aggregation surface trajectory-level columns in the
      sweep TSV, and let step-evaluators (§3) bind to slots instead of
      strings.
- [ ] Remote-agent endpoint abstraction — talk to a deployed agent
      over HTTP / A2A / ACP instead of constructing it locally from a
      synalinks Program.
- [ ] End-to-end verification across deployed agents (final outcome +
      path), built on the typed trajectory schema above.

## Medium priority

### Evaluation
- [ ] Generalize `MultiJudgePanel` into a configurable multi-grader
      aggregation abstraction. The existing reward already provides
      concurrent panel + spread-based escalation; what's missing is
      pluggable aggregators (majority vote, weighted mean, Bradley-Terry
      ratings, etc.) so users don't have to subclass to swap policies.
- [ ] YAML key harmonization — final pass. `class:` is now the
      canonical class-identifier key everywhere it appears in the
      example file (`config.example.yaml`); the registry-side change
      that reads `class:` (and warns on the deprecated `name:`) for
      rewards + DeepEval inner spec is still pending, and
      `config.yaml` still uses `name:` for reward entries.

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

## Repo hygiene

Small items that don't fit a priority bucket but are worth tracking.

- [ ] Upstream synalinks: `FBetaScore` family default `.name` doesn't
      match its registry identifier (`"fbeta_score"` vs `"f_beta_score"`,
      and same for `BinaryFBetaScore` / `CategoricalFBetaScore`). F1
      variants are consistent. Normalize the class defaults to match
      `to_snake_case(cls.__name__)`.

## What keras-tuner already covers

Things `notes.md` describes as future platform work that the sweep
already gets for free from `keras_tuner` — don't reinvent these:

- **Run identity & hyperparameter resolution.** `trial.hyperparameters`
  is the run identity (`language_model`, `dataset`, future axes).
  Configuration resolution = the HP space.
- **Durable, resumable run artifact.** Every trial is serialized to
  `.open-arena/<dataset>/trial_<id>/` with HP + metrics + status. `--no-cache`
  discards it; otherwise resume is automatic. This is the portable
  run record — no separate artifact format needed.
- **Per-trial metric registry.** `trial.metrics.register(alias,
  direction=…)` + `get_best_value(alias)` already gives a typed,
  multi-metric scoring model with min/max semantics. Each candidate
  reward under `metrics:` is one alias.
- **Best-trial / leaderboard query.** `oracle.get_best_trials()` is the
  in-memory leaderboard; the long-format `.open-arena/last_run.tsv` is
  the contract the CLI exposes to downstream agents (plus
  `.open-arena/frontier.tsv` for multi-objective datasets).
- **Status tracking.** `trial.status` (`COMPLETED` / `INVALID` /
  `FAILED`) — failed cells are written into the `status` field of the
  per-row JSON output and omitted from the TSV.
- **Project-level isolation.** Different `project_name` ⇒ different
  `.open-arena/` subdir, so multiple sweeps coexist without colliding.

The actual missing platform piece is **cross-run** aggregation
(leaderboard across many sweeps over time), not per-run.

## Wiki ↔ code mismatches

These rows are marked **Working** in `feature-matrix.md` but are not
present in the current `src/`. Either implement them in this repo or
correct the matrix — pick one per row.

- [ ] **Langfuse run / score persistence.** Only the dataset *reader*
      exists (`src/datasets/langfuse_dataset.py`); no trace upload or
      score writeback. Matrix row "Langfuse tracing and score
      persistence" claims this works.
- [ ] **`ExecutionResult` run artifact.** Referenced in
      `notes.md §2`; no such type exists in the codebase. The current
      sweep returns synalinks `program.evaluate()` dicts directly.
