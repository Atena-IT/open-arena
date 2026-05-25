-- Open Arena leaderboard-oriented persistence model (PostgreSQL)
-- Requires pgcrypto (for gen_random_uuid()) to be enabled by an admin before
-- applying this schema in managed Postgres environments.

create table if not exists leaderboard (
    id uuid primary key default gen_random_uuid(),
    name text not null,
    description text,
    visibility text not null check (visibility in ('private', 'organization', 'public')),
    ranking_policy jsonb not null default '{}'::jsonb,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists model_definition (
    id uuid primary key default gen_random_uuid(),
    name text not null,
    display_name text,
    family text,
    provider text not null,
    model_name text not null,
    model_version text not null,
    -- Caller-provided canonical fingerprint of the normalized runtime payload.
    -- This is the model-definition dedupe key under
    -- unique (provider, model_name, model_version, runtime_fingerprint), so
    -- callers must compute it deterministically from the provider/model
    -- identity plus endpoint, auth/env wiring, and any execution-affecting
    -- runtime overrides.
    runtime_fingerprint text not null,
    runtime jsonb not null,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    unique (provider, model_name, model_version, runtime_fingerprint)
);

create table if not exists leaderboard_model (
    leaderboard_id uuid not null references leaderboard(id) on delete cascade,
    model_id uuid not null references model_definition(id) on delete restrict,
    ordinal integer not null default 0,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    primary key (leaderboard_id, model_id)
);

create table if not exists verifier_suite (
    id uuid primary key default gen_random_uuid(),
    name text not null,
    version text,
    aggregation text not null default 'weighted_mean' check (aggregation in ('mean', 'weighted_mean', 'pairwise', 'custom')),
    definition jsonb not null,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists environment_definition (
    id uuid primary key default gen_random_uuid(),
    source_kind text not null check (source_kind in ('huggingface_hub', 'github_repo', 'prime_environment_hub', 'inline')),
    canonical_name text not null,
    environment_version text not null,
    source_ref text,
    source_uri text,
    object_storage_layout jsonb not null default '{}'::jsonb,
    dataset_binding jsonb,
    runtime_policy jsonb,
    sandbox_policy jsonb,
    reset_policy jsonb,
    verifier_suite_id uuid references verifier_suite(id) on delete restrict,
    inline_verifier jsonb,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    -- Exactly one verifier source must be present: either a reusable suite ref or an inline verifier definition.
    check ((verifier_suite_id is null) <> (inline_verifier is null)),
    unique (source_kind, canonical_name, environment_version)
);

create table if not exists leaderboard_environment (
    leaderboard_id uuid not null references leaderboard(id) on delete cascade,
    environment_id uuid not null references environment_definition(id) on delete restrict,
    ordinal integer not null default 0,
    overrides jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    primary key (leaderboard_id, environment_id)
);

create table if not exists run_request (
    id uuid primary key default gen_random_uuid(),
    leaderboard_id uuid references leaderboard(id) on delete set null,
    mode text not null check (mode in ('generator', 'agent')),
    publish_to_public_leaderboard boolean not null default false,
    selection jsonb not null,
    execution_config jsonb not null default '{}'::jsonb,
    reuse_policy jsonb not null default '{}'::jsonb,
    cache_status text not null check (cache_status in ('pending', 'miss', 'partial_hit', 'hit', 'bypassed')) default 'pending',
    status text not null check (status in ('queued', 'running', 'succeeded', 'failed', 'cancelled')) default 'queued',
    labels jsonb not null default '{}'::jsonb,
    idempotency_key text,
    error jsonb,
    created_at timestamptz not null default now(),
    started_at timestamptz,
    completed_at timestamptz
);

create table if not exists run_subject (
    id uuid primary key default gen_random_uuid(),
    run_id uuid not null references run_request(id) on delete cascade,
    model_id uuid not null references model_definition(id) on delete restrict,
    environment_id uuid not null references environment_definition(id) on delete restrict,
    model_version text not null,
    environment_version text not null,
    -- Caller-provided deterministic cache key for a resolved run_subject
    -- execution. Unlike runtime_fingerprint, this spans the model/environment
    -- pairing plus the active reuse_policy.key_fields so cached results can be
    -- reused across multiple runs. By default this includes model_version,
    -- environment_version, mode, temperature, and max_tokens.
    execution_fingerprint text not null,
    cache_status text not null check (cache_status in ('pending', 'miss', 'partial_hit', 'hit', 'bypassed')),
    source_run_id uuid references run_request(id) on delete set null,
    trajectory_summary jsonb,
    created_at timestamptz not null default now(),
    unique (run_id, model_id, environment_id)
);

create table if not exists metric_result (
    id uuid primary key default gen_random_uuid(),
    run_subject_id uuid not null references run_subject(id) on delete cascade,
    metric_name text not null,
    metric_kind text,
    direction text not null check (direction in ('max', 'min')),
    weight numeric not null default 1 check (weight >= 0),
    metric_value double precision not null,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    unique (run_subject_id, metric_name)
);

create table if not exists leaderboard_entry (
    leaderboard_id uuid not null references leaderboard(id) on delete cascade,
    model_id uuid not null references model_definition(id) on delete restrict,
    environment_id uuid not null references environment_definition(id) on delete restrict,
    last_run_subject_id uuid references run_subject(id) on delete set null,
    score double precision not null,
    rank integer not null check (rank >= 1),
    breakdown jsonb not null default '{}'::jsonb,
    updated_at timestamptz not null default now(),
    primary key (leaderboard_id, model_id, environment_id)
);

create table if not exists public_leaderboard_entry (
    environment_id uuid not null references environment_definition(id) on delete restrict,
    model_id uuid not null references model_definition(id) on delete restrict,
    last_run_subject_id uuid references run_subject(id) on delete set null,
    score double precision not null,
    breakdown jsonb not null default '{}'::jsonb,
    updated_at timestamptz not null default now(),
    primary key (environment_id, model_id)
);

create index if not exists idx_run_subject_fingerprint
    on run_subject (execution_fingerprint);

create index if not exists idx_run_request_leaderboard_status
    on run_request (leaderboard_id, status, created_at desc);

create index if not exists idx_leaderboard_entry_rank
    on leaderboard_entry (leaderboard_id, rank nulls last, score desc);

create index if not exists idx_public_leaderboard_entry_score
    on public_leaderboard_entry (environment_id, score desc);
