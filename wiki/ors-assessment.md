# Preliminary assessment — the Open Reward Standard (ORS) vs. how Open Arena defines environments and yields reward

> **Status:** preliminary / exploratory. Written 2026-06-23.
> **Scope:** compares the [Open Reward Standard](https://openrewardstandard.io/)
> (ORS) against three ways Open Arena defines an *environment* and produces a
> *reward*: (a) the YAML world on `main` as it stood before today, (b) the
> Prime Intellect `verifiers` route, and (c) the persistent REST API wave now
> on `feat/api-backend` (PR #46 / #22, epic #33). Ends with what adopting
> something ORS-shaped would unlock and a phased recommendation.
> **Nature of the claim:** a design note, not a decision. No code is changed by
> this document.

---

## 0. TL;DR

ORS, Prime `verifiers`, and our API wave are **not three competing answers to
the same question** — they live on three different planes:

| Plane | What it owns | Embodied by |
|---|---|---|
| **Data plane** (runtime wire protocol) | How an agent and an environment *talk* during an episode — actions, observations, reward, termination | **ORS** (HTTP/SSE), MCP (no reward) |
| **Implementation plane** (env + reward code) | What the environment *is* and how it *computes* a score | **Prime `verifiers`** (Python lib), our `src/rewards` + `src/verifiers` |
| **Control plane** (registry / governance) | How environments and verifiers are *named, versioned, sourced, sandboxed, scheduled, cached* | **This wave** (`/v1/environments`, `/v1/verifiers`, `/v1/runs`) |

The headline finding: **ORS is complementary, not redundant.** It standardises
the one plane we have *not* built — the interactive runtime contract between an
agent and an environment that emits reward *per step* and signals *when an
episode is finished*. Everything Open Arena does today (both on `main` and on
the API branch) computes reward **once, terminally, over a batch**. Adopting an
ORS-shaped boundary is the cleanest path from "eval-only" (which the API spec
[explicitly declares itself to be](#33-the-api-spec-is-deliberately-eval-only))
to the *RL leg* the README already advertises ("between post-training evals and
the next round of fine-tuning / RL").

And the pleasant surprise: **our `openapi.yaml` is already shaped to receive
it** — see [§8](#8-how-ors-slots-into-the-current-api-model). `prime_environment_hub`
is already an `EnvironmentSourceKind`; an `ors_server` kind is the natural next
sibling.

---

## 1. The question, restated

The prompt frames it precisely: *"in the env definition, and the way the env
yields rewards, if we use prime verifiers we good."* True — for the
**implementation plane**. If we source a Prime `verifiers` environment, the env
definition (dataset + harness + tools) and the reward emission (a rubric of
weighted reward functions) come bundled and solved. The branch even ships a
`PrimeEnvHubBackend` and lists `prime_environment_hub` in `EnvironmentSourceKind`
(`openapi.yaml:913-919`), so we can pull one as a first-class environment.

But "the way the env yields rewards" has *two* readings:

1. **Who computes the score** (a function over a finished transcript) — Prime
   `verifiers` answers this well, and so do our `src/rewards/*`.
2. **How the score is delivered over time** (a scalar at the end vs. a signal
   each turn, plus an explicit "episode is over" flag) — this is the part
   neither `main` nor the API wave addresses, and the part **ORS is built
   around**.

This assessment is mostly about reading (2), because that is where ORS adds
something we do not have.

---

## 2. The landscape — four ways to define an environment and yield reward

### 2.1 Prior to today — the YAML world on `main`

On `main` there is **no first-class "environment" object**. Its role is played
by a `datasets.<name>:` block that bundles a dataset source + a program graph +
a reward, crossed at sweep time with `experiments.language_models:`. So an
"environment" today ≈ **(dataset block) × (model)**, scored by that dataset's
reward.

- **Definition:** a YAML entry — `type:` selects a provider
  (`huggingface`/`local`/`folder`/`langfuse`/`langsmith`/`opik`/`phoenix`/`braintrust`,
  `src/datasets/__init__.py:15-24`), Jinja `input_template`/`output_template`
  render rows to JSON, and a per-dataset `generator:` **xor** `agent:` block
  (`src/config.py:140-146`) picks single-shot `synalinks.Generator` vs.
  multi-step `synalinks.FunctionCallingAgent` over MCP tools
  (`src/program.py:22-129`).
- **Reward:** a required per-dataset `reward:` (`src/config.py:137`) resolved by
  snake_case name from a registry (`src/rewards/__init__.py:71-113`) into a
  `synalinks.rewards.Reward` whose contract is
  `call(y_true, y_pred) -> float in [0,1]`. Built-ins: `exact_match`,
  `cosine_similarity`, `lm_as_judge`; project-local: `multi_judge_panel`,
  `recursive_lm_as_judge`, `rubrics_as_judge`, `deep_eval`, `deep_as_judge`,
  `agent_as_judge`, `metric_reward`.
- **How reward is yielded:** **once, terminally.** The sweep runs one
  `keras-tuner` `GridSearch` per dataset and dispatches to
  `program.evaluate(x=ds)` (`src/evaluate.py:435-447`); the reward is computed
  over the batch of finished outputs and reduced to a per-dataset scalar. Even
  in `agent:` mode the multi-step loop happens *inside* the program, and reward
  is scored on the final answer — there is **no per-step reward and no
  environment-driven episode loop**.
- **Group-relative scoring exists but is walled off:** `src/verifiers/`
  (`LMAsVerifier`, a `synalinks.BatchReward`) does pairwise/round-robin scoring,
  deliberately *not* in the YAML reward registry because its per-sample score is
  only meaningful relative to the batch (`src/verifiers/__init__.py:1-17`) — a
  GRPO-shaped signal with nowhere to flow on `main`.

### 2.2 The Prime Intellect `verifiers` route

[`PrimeIntellect-ai/verifiers`](https://github.com/PrimeIntellect-ai/verifiers)
is a **Python library** where an *environment* bundles a dataset + a harness
(tools, sandboxes) + a **rubric** (reward).

- **Definition:** subclass `MultiTurnEnv` (or `SingleTurnEnv`/`ToolEnv`); a
  module exposes `load_environment(...) -> vf.Environment`. The env owns the
  rollout loop (`async rollout(...) -> State`), turn logic
  (`env_response(...)`), and composable termination via `@vf.stop` conditions
  (`max_turns_reached`, `no_tools_called`, …).
- **Reward:** a `Rubric(funcs=[...], weights=[...])` of reward functions, each
  `f(prompt, completion, answer, state, …) -> float | list[float] | dict`,
  aggregated as a **weighted sum**. Crucially it supports **group reward
  functions** that return `list[float]` and set
  `state["advantage"] = reward_i − group_mean` — the exact GRPO signal — and
  back-fills per-turn `reward`/`advantage` onto each `TrajectoryStep`. So
  verifiers natively produces **per-step *and* group-relative** reward.
- **Distribution:** each environment is a **pip-installable wheel** published to
  the **Environments Hub** via `prime env push`; `prime eval run <env>` runs it.
- **How it relates to us:** this is the "if we use prime verifiers we good"
  path. It solves the implementation plane comprehensively and is **already a
  supported source** (`prime_environment_hub` + `PrimeEnvHubBackend`). It is
  also the only one of our four that already does per-step / group reward.

The catch: verifiers is **Python, in-process**. The rollout loop, rubric, and
even the GRPO trainer all run in your process; the env is a Python object, not a
network service. Distribution is coupled to the Prime Hub and the `prime` CLI.
There is no language-agnostic wire boundary — which is exactly what ORS is.

### 2.3 This wave — the persistent REST control plane

PR #46/#22 (branch `feat/api-backend`, epic #33) promotes environments and
verifiers from inline YAML fragments into **persisted, versioned,
provenance-bearing REST resources**, served by a FastAPI app over a
SQLAlchemy/SQLite store, with a ports→adapters design (Postgres, Gitea, Unity
Catalog, MLflow, E2B, Keycloak per epic #33).

- **`Environment`** (`openapi.yaml:1513-1532`) = a UUID + a versioned
  **`EnvironmentSource`** (`openapi.yaml:1425-1452`): `kind ∈
  {huggingface_hub, github_repo, prime_environment_hub, inline}`, plus `uri`,
  `git_ref`, `external_id`, `commit_sha`/`content_hash` (WS3). An inline
  environment (`InlineEnvironmentDefinition`, `:1454-1476`) bundles a
  `DatasetBinding` + a `VerifierSuiteBinding` + `runtime` + `SandboxPolicy` +
  `ResetPolicy`.
- **`VerifierSuite`** (`/v1/verifiers`, `openapi.yaml:1356-1383`) = a named,
  reusable ensemble of weighted `MetricDefinition`s with an `aggregation`
  (default `weighted_mean`). Each metric carries a `backbone`, `objective`/
  `direction`, and **`context_bindings`** mapping slots to sources
  `input|reference|trajectory|output|environment` via a `FieldSelector`
  (`in_mask`/`out_mask`/`extract` — the same masking vocabulary as the YAML
  rewards).
- **`SandboxPolicy`** (`:1216-1233`, `isolation_mode ∈ none|container|vm|remote`)
  and **`ResetPolicy`** (`:1235-1248`, `strategy ∈
  per_task|per_run|manual|external`, with a `reset_endpoint` URI).
- **`Run`** (`/v1/runs`) is `mode: generator | agent`, selects leaderboard
  slices or `direct_pairs`, and is cache-backed by a deterministic SHA-256
  `run_fingerprint` over `ReusePolicy.key_fields`.
- **How reward is yielded:** still **terminally**. A run materialises
  `SubjectResult` rows (per model × environment) with `metrics`,
  `cache_status`, and a `trajectory_summary`. The verifier suite scores finished
  outputs/trajectories; there is no live reward stream.

#### 3.3 The API spec is deliberately *eval-only*

This is the single most load-bearing fact for the ORS comparison. The spec says
so in its own header (`openapi.yaml:30-33`):

> *"`Run` is scoped to evaluation submissions plus their cache-backed result
> materialization. Job-control primitives (cancel/retry/progress/worker leases)
> and training or optimization jobs are intentionally out of scope for this
> version of the spec."*

So the wave builds the **registry/governance/eval-orchestration** layer
beautifully, and explicitly stops short of the RL/training loop. That boundary
is precisely where an ORS-shaped runtime would live.

### 2.4 ORS — the Open Reward Standard

ORS (by **General Reasoning / gr.inc**, the protocol under their *OpenReward*
platform; reference SDK `github.com/openrewardstandard/python-sdk`, Apache-2.0)
is an **HTTP/SSE wire protocol** for connecting an agent to an RL environment.
Its design principle is *"actions are tools"* — it is **MCP plus the RL
primitives MCP lacks**. See [§3](#3-what-ors-actually-is) for the spec digest.

The one-line version: an environment is a **server**; the agent opens a
**session** bound to a **task** (addressed by `(split, index)` or an inline
`task_spec`), calls **tools**, and every tool result is a `ToolOutput` carrying
`blocks` (the observation), an **optional scalar `reward`**, and a **`finished`**
flag. The episode ends when `finished == true`.

---

## 3. What ORS actually is (spec digest)

Quoted from the reference SDK source (`ors/types.py`, `ors/server.py`,
`ors/environment.py`). *Caveat: the prose spec pages at openrewardstandard.io /
docs.openreward.ai return 403 to automated fetches; field names below are from
the SDK, which is the canonical implementation.*

**The reward-bearing return type** — every tool/action returns:

```python
class ToolOutput:
    blocks:   Blocks                       # required: the observation (text/image blocks)
    metadata: Optional[JSONObject] = None
    reward:   Optional[float]      = None  # OPTIONAL scalar reward, can be emitted any step
    finished: bool                 = False # episode-termination flag
```

**Concepts and verbatim field names:**

- **Task** = an arbitrary JSON object (`task_spec`) describing a setup + goal.
- **Split** = `{name: str, type: "train"|"validation"|"test"}`; tasks are
  addressed by `(split, index)`, which *is* the reproducibility/seed mechanism
  (no separate numeric seed field found).
- **Session / Episode** = a stateful agent↔task binding keyed by an
  `X-Session-ID` header; runs until a `ToolOutput.finished == true`.
- **Prompt** = `get_prompt() -> Blocks`.
- **Action** = a `@tool`-decorated method taking one Pydantic model, returning
  `ToolOutput`. Declared as MCP-aligned `ToolSpec{name, description,
  input_schema}`.

**Environment server interface** (abridged):

```python
class Environment(ABC):
    def setup(self); def teardown(self)
    def get_prompt(self) -> Blocks
    @classmethod def list_splits(cls) -> Sequence[Split | str]
    @classmethod def list_tasks(cls, split) -> Sequence[JSONObject]
    @classmethod def get_task(cls, split, index) -> JSONObject   # + num_tasks / get_task_range
    @tool def <action>(self, params: Model) -> ToolOutput
```

**Transport:** FastAPI/Uvicorn over HTTP; tool calls stream over **SSE**.
Notable endpoints: `POST /create` (bind session to task), `POST /{env}/call`
(SSE stream of `RunToolOutput`), `GET /{env}/prompt`, `GET /{env}/tools`,
`POST /{env}/tasks|task|num_tasks`, `GET /{env}/splits`. Sessions expire after
~900 s idle (clients `POST /ping`).

**Reward model:** a **single optional `float` per tool call** — per-step or
terminal, the env's choice. There is **no multi-objective / typed / rubric
vector on the wire**; richer scoring is left to the host above ORS. (This is the
inverse of our `VerifierSuite`, which is explicitly multi-metric.)

**Relationship to MCP:** same tool-listing/tool-calling and content-block
observations; ORS *adds* `reward`, `finished`, tasks, and splits — the four
things gr.inc notes MCP omits for agentic RL.

---

## 4. Head-to-head comparison

| Dimension | `main` (YAML) | Prime `verifiers` | This wave (REST API) | **ORS** |
|---|---|---|---|---|
| **Primary plane** | Implementation (inline) | Implementation (library) | Control (registry/governance) | **Data (wire protocol)** |
| **Env definition** | `datasets.<name>` YAML block | `load_environment()` Python | `Environment` resource + `EnvironmentSource` | ORS `Environment` server (any language) |
| **Identity / versioning** | YAML key only | pip wheel version on Hub | UUID + versioned source + `commit_sha`/`content_hash` | env name + `(split,index)` task addressing |
| **Tools / actions** | MCP via `FunctionCallingAgent` | OpenAI-style + optional `MCPEnv` | `agent` run mode (tool/remote) | **tools = actions** (MCP-aligned, native) |
| **Reward shape** | 1 scalar `[0,1]` (terminal) | weighted-sum rubric, **per-step + group/advantage** | multi-metric `VerifierSuite` (terminal) | **1 optional scalar per tool call** |
| **Episode loop** | none (batch `evaluate`) | in-process `rollout()` loop | none (batch run) | **explicit, `finished` flag, over the wire** |
| **Termination signal** | n/a | `@vf.stop` conditions | n/a | **`ToolOutput.finished`** |
| **Reproducibility** | tuner cache | dataset slicing | SHA-256 `run_fingerprint` + `ReusePolicy` | `(split, index)` deterministic addressing |
| **Transport / coupling** | in-process Python | in-process Python | REST (our server) | **HTTP/SSE, language-agnostic, runs anywhere** |
| **RL/training** | no | **yes** (`prime-rl`, GRPO) | explicitly **out of scope** | **yes** (the protocol *is* RL-native) |
| **Multi-objective** | via `metrics:` columns | rubric (many funcs) | `VerifierSuite` (many metrics) | **no** (single float on wire) |
| **Governance / provenance** | none | Prime Hub | strong (Gitea/UC/MLflow/Keycloak) | minimal (it's a wire spec) |

---

## 5. The key insight — three planes, not three competitors

Read the matrix column by column and the architecture writes itself:

- **ORS owns the data plane** we have never built: the *live* agent↔env contract
  with per-step reward and explicit termination.
- **`verifiers` owns the implementation plane**: it is the richest reward
  model of the four (weighted multi-func rubric + GRPO advantage), and it is
  *already sourceable* via `prime_environment_hub`.
- **The API wave owns the control plane**: identity, versioning, provenance,
  sandbox/reset policy, caching, leaderboards, multi-tenant governance.

They compose. A concrete, plausible end-state:

```
/v1/environments  →  EnvironmentSource{kind: ors_server | prime_environment_hub | ...}
                         │
   control plane         │ resolves to a running env
   (our API)             ▼
                     ORS server  ◄──HTTP/SSE──►  Open Arena runtime adapter
   data plane            │  (tools=actions, reward, finished, splits)
                         ▼
                     trajectory + per-step reward
                         │
   implementation        ▼
   plane             VerifierSuite (optional) re-scores the trajectory
                     OR trust env-native reward
```

The env-native reward (ORS `ToolOutput.reward`) and our `VerifierSuite` are not
mutually exclusive: ORS gives us the *cheap, online* signal during a rollout;
the `VerifierSuite` gives us the *rich, multi-objective, auditable* score after.
`MetricDefinition.context_bindings` already accepts `trajectory` and
`environment` as slot sources (`openapi.yaml`), so it can score an ORS episode
without modification.

---

## 6. "If we use prime verifiers, we're good" — is that enough?

Mostly yes, for *one* use case (sourcing a Python env with a bundled rubric, for
eval and for GRPO via `prime-rl`). But the verifiers route alone leaves four
things on the table that an ORS-shaped boundary would give us:

1. **Language lock-in.** A `verifiers` env is a Python object in our process.
   Teams with environments in Rust/Go/TS, or behind a service boundary (a real
   app, a browser, a terminal sandbox on another host), cannot expose them as
   `verifiers` envs without a Python rewrite. An ORS server is a network
   service in *any* language.
2. **Distribution coupling.** `verifiers` distribution is the Prime Hub + the
   `prime` CLI + pip wheels. ORS environments "run anywhere" with no dependency
   on OpenReward infra — a better fit for our ports→adapters, per-org-isolation
   posture (epic #33) and for Gitea/object-storage sourcing.
3. **No standard interaction wire.** Two `verifiers` envs share a Python API,
   not a protocol; you cannot point an arbitrary external trainer/harness at one
   over the network. ORS is consumable by *any* ORS client (incl. the gr.inc
   ecosystem: `firehorse`, `inspect-openreward`, …) — interop in both
   directions.
4. **Eval↔train symmetry over the wire.** With `verifiers`, train and eval are
   both in-process Python. With ORS, the *same* env server serves our eval runs
   *and* an external RL trainer, with identical reward semantics — which is
   exactly the eval→RL handoff the README promises and the API spec defers.

So: **adopt the verifiers route for what it's great at (Python envs + rubrics +
GRPO), and add an ORS boundary for everything that needs to cross a process,
language, or org line.** They are not either/or.

---

## 7. What ORS would unlock — "what else could we achieve?"

Beyond parity, an ORS-shaped runtime opens capabilities none of the three
current paths offer:

1. **True online / interactive rollouts.** Per-step reward + `finished` turns
   Open Arena from a *batch grader* into an *episode runner*. This is the exact
   item the roadmap already flags: `wiki/notes.md` — *"the arena needs to model
   turns, tool invocations, intermediate states … as first-class run structure
   rather than optional attachments"*; `wiki/TODO.md §3` — *"step-level
   evaluators with a dedicated step-score schema."* ORS's `ToolOutput` is that
   step-score schema, standardised.
2. **The RL leg the product promises.** The README positions Open Arena
   "between post-training evals and the next round of fine-tuning / RL," yet the
   API is eval-only by design. An ORS env is RL-native (reward + finished +
   train/validation/test splits); wiring it in is the minimal bridge to actually
   *produce training signal*, not just rank models. Our `LMAsVerifier`'s
   group-relative scores (today walled off in `src/verifiers/`) finally have a
   place to flow.
3. **Polyglot, networked, app-grade environments.** Web/browser tasks, terminal
   sandboxes, real SaaS apps, multi-service simulators — anything that can speak
   HTTP/SSE becomes an environment, decoupled from our Python runtime and
   runnable under `SandboxPolicy.isolation_mode: remote`.
4. **Two-way ecosystem interop.** *Consume:* any ORS env (the OpenReward catalog
   — SWE, terminal, math/Lean corpora from NVIDIA/Nebius/Eigent) becomes an
   `EnvironmentSource`. *Expose:* wrap our inline/HF/Gitea environments behind an
   ORS server so external trainers (and the gr.inc harnesses) can train against
   *our* governed, versioned environments. The control plane stays ours; the
   interaction surface becomes standard.
5. **Standardised, reproducible task splits across the leaderboard.** ORS
   `Split` + `(split, index)` addressing gives a portable train/val/test
   contract that maps onto leaderboard perimeters and the `run_fingerprint`
   (env version + task index → deterministic cache key).
6. **A cleaner home for trajectory-aware verifiers.** With episodes as
   first-class, `VerifierSuite` metrics bound to the `trajectory`/`environment`
   slots can score *how* a task was solved (tool efficiency, safety, recovery),
   not just the final answer — the step-aware verification `wiki/TODO.md`
   already asks for.

---

## 8. How ORS slots into the current API model

The encouraging part: the wave's schema was drawn with a remote, resettable,
reward-emitting environment already in mind. ORS lands with **little schema
churn**:

| ORS concept | Existing Open Arena hook | Gap to close |
|---|---|---|
| Env served over HTTP/SSE | `SandboxPolicy.isolation_mode: remote`; `agent` run mode = *"tool-enabled or remote-agent execution with trajectory capture"* (`openapi.yaml:14-16`) | Add a runtime adapter that drives the `/create` → `/{env}/call` → `finished` loop |
| Episode reset / lifecycle | `ResetPolicy.strategy: external` + `reset_endpoint` (`openapi.yaml:1235-1248`) | Map `reset_endpoint`/`per_task` onto ORS `POST /create` |
| Source registry | `EnvironmentSourceKind` already has `prime_environment_hub` (`:913-919`) | **Add `ors_server`** kind; `uri` = base URL, `external_id` = env name |
| Tasks / splits | `DatasetBinding` (inline or referenced) | Allow a binding to resolve to ORS `list_splits`/`list_tasks` rather than a static dataset |
| Per-step reward | `SubjectResult.trajectory_summary`; `MetricDefinition.context_bindings: trajectory` | Capture `ToolOutput.reward` stream into the trajectory; let it feed (or bypass) the `VerifierSuite` |
| Tools = actions | `FunctionCallingAgent` + MCP already in the runtime | ORS tools convert to provider tool schemas (the ORS client does this) |
| Reproducibility | `run_fingerprint` over `key_fields` incl. `environment_version` | Fold `(split, index)` into the fingerprint |

Net: ORS is roughly *"a new `EnvironmentSourceKind` (`ors_server`) + a runtime
adapter that speaks the ORS rollout loop + capturing `reward`/`finished` into
the existing trajectory/result model."* The control-plane resources
(`Environment`, `VerifierSuite`, `Run`, leaderboards, caching) are reused as-is.

---

## 9. Risks, gaps, and open questions

- **ORS reward is a single scalar; our `VerifierSuite` is multi-metric.** On the
  wire we'd lose the per-objective breakdown unless we (a) keep scoring
  trajectories with `VerifierSuite` after the episode, or (b) pack structure
  into `ToolOutput.metadata` (non-standard). Recommend (a).
- **ORS does not standardise the verifier/rubric** — the env *owns* its reward.
  Consuming a third-party ORS env means *trusting its reward* or re-scoring its
  trajectory ourselves. Our differentiator (auditable, reusable verifier suites)
  stays relevant precisely because ORS leaves this open.
- **Maturity & governance.** ORS is young and gr.inc-led; the spec prose pages
  are bot-blocked (we read the SDK source instead). The `openreward` PyPI
  package shows a **license inconsistency** (MIT in metadata vs. Apache-2.0 in
  headers) — verify before depending on it. The reference `ors-sdk` is
  Apache-2.0.
- **No explicit numeric seed.** Reproducibility is `(split, index)` addressing
  only; stochastic envs need their own seed-in-`task_spec` convention.
- **Overlap to de-conflict.** ORS `setup`/`teardown`/sessions overlap our
  `SandboxPolicy.bootstrap/teardown` and `ResetPolicy`. Decide which layer owns
  process lifecycle (recommend: ORS server owns *episode* state; our
  `SandboxPolicy` owns *infra* isolation).
- **Statefulness vs. our cache.** ORS sessions are stateful and TTL'd;
  `run_fingerprint` reuse assumes deterministic replay. Interactive episodes are
  only cacheable at the trajectory level, not the step level.
- **It does not replace the wave.** ORS gives us none of versioning, provenance,
  multi-tenancy, or leaderboards. It is additive to epic #33, not a substitute.

---

## 10. Preliminary recommendation

1. **Treat the three planes as complementary** and say so in the architecture
   docs: ORS = data plane, `verifiers`/`rewards` = implementation plane, the API
   wave = control plane. This resolves the "ORS vs. what we're building"
   framing — it's "ORS *and* what we're building."
2. **Keep the verifiers route** for Python envs + rubrics + GRPO; it is the
   best implementation-plane answer and is already sourceable.
3. **Spike an `ors_server` `EnvironmentSourceKind`** plus a thin ORS **client
   adapter** (drive `create` → `call` → `finished`, capture the
   `reward`/`finished` stream into the existing trajectory/`SubjectResult`
   model). Lowest-risk, highest-signal experiment; reuses all control-plane
   resources. Target one read-only ORS env (e.g. the GSM8K example server) end
   to end through `/v1/runs`.
4. **Keep `VerifierSuite` as the scoring authority** for consumed ORS envs —
   re-score the captured trajectory rather than blindly trusting env-native
   reward; bind metrics to the `trajectory`/`environment` slots.
5. **Sequence it after epic #33's M1–M2** (store + env versioning). ORS is the
   natural **M-RL** follow-on: it is the concrete mechanism that turns the
   eval-only spec into the "next round of fine-tuning / RL" the README promises.
6. **Later, expose our environments *as* ORS servers** so external trainers can
   train against Open Arena's governed, versioned environments — the control
   plane stays ours, the interaction surface becomes an open standard.

**Bottom line:** "if we use prime verifiers we're good" is true for the
implementation plane. ORS is good for the plane we skipped — the live
agent↔environment runtime — and our `openapi.yaml` is already shaped to receive
it. The two are complementary, and together they're the bridge from *ranking
models* to *training them*.

---

## Appendix A — ORS field ↔ Open Arena concept map

| ORS | Open Arena (`main` / API wave) |
|---|---|
| `Environment` (server) | `datasets.<name>` block / `Environment` resource |
| `task_spec` (JSON) | a dataset row / `DatasetBinding` item |
| `Split{name,type}` + `(split,index)` | dataset `split:` + tuner trial / `RunSelection` |
| `get_prompt() -> Blocks` | `input_template` → input data model |
| `@tool` action | MCP tool on a `FunctionCallingAgent` |
| `ToolOutput.blocks` | observation / tool result message |
| `ToolOutput.reward` | a reward's `[0,1]` scalar — but per step |
| `ToolOutput.finished` | (no equivalent; new) episode-termination flag |
| session (`X-Session-ID`) | (no equivalent; new) stateful episode |
| `setup`/`teardown` | `SandboxPolicy.bootstrap/teardown`, `ResetPolicy` |
| reward as weighted vector | `VerifierSuite` of weighted `MetricDefinition`s |

## Appendix B — sources

**Codebase (this repo):** `openapi.yaml` (info `:6-33`, `EnvironmentSourceKind`
`:913-919`, `SandboxPolicy` `:1216-1233`, `ResetPolicy` `:1235-1248`,
`VerifierSuite` `:1356-1383`, `EnvironmentSource` `:1425-1452`,
`InlineEnvironmentDefinition` `:1454-1476`, `Environment` `:1513-1532`);
`src/rewards/__init__.py:71-113`; `src/verifiers/__init__.py:1-17`;
`src/program.py:22-129`; `src/config.py:61-177`; `src/evaluate.py:318-486`;
`REWARDS_BUILDING.md`; `wiki/notes.md`; `wiki/TODO.md`; `wiki/feature-matrix.md`;
GitHub epic #33, PRs #46 / #22.

**ORS (primary — SDK source):**
- Reference SDK: <https://github.com/openrewardstandard/python-sdk>
  (`ors/types.py`, `ors/server.py`, `ors/environment.py`, `examples/gsm8k_server.py`)
- `openreward` PyPI: <https://pypi.org/project/openreward/>
- Spec / platform (prose pages, 403 to automated fetch — used search snippets):
  <https://openrewardstandard.io/>, <https://docs.openreward.ai/>,
  <https://www.gr.inc/releases/introducing-openreward>

**Prime Intellect `verifiers` (primary — source):**
- <https://github.com/PrimeIntellect-ai/verifiers> (README,
  `verifiers/types.py`, `envs/*`, `rubrics/*`, `parsers/*`),
  `docs/environments.md`
- Environments Hub / CLI: <https://github.com/PrimeIntellect-ai/prime>,
  <https://github.com/PrimeIntellect-ai/prime-rl>,
  <https://app.primeintellect.ai/dashboard/environments>

> **Unverified flags:** `openreward` license (MIT vs Apache-2.0 conflict in
> package metadata); absence of an explicit numeric seed field in ORS (only
> `(split,index)` addressing observed); verbatim ORS spec prose (sites block
> automated access — field names taken from the canonical SDK source).
