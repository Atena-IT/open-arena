# Minimal eval example

A self-contained run-request that exercises the inline-environment path of the
Open Arena API.  No git repo, no external sandbox, no LLM API key required when
the engine boundary is mocked (e.g. in tests).

## What it contains

| File | Purpose |
|---|---|
| `run.json` | Run-request payload — inline dataset + exact-match verifier, no sandbox |

## Submitting via the CLI

Start the API server first:

```bash
arena serve                  # binds to 127.0.0.1:8000 by default
```

Then submit the run in a second terminal:

```bash
arena run submit --local --file examples/minimal_eval/run.json
```

`--local` tells the CLI to call the in-process service directly rather than
posting to a remote server.  The command prints the resulting `RunResult` JSON
to stdout.

## Environment definition

The run uses an `inline_definition` environment — no git clone, no registry
lookup.  The dataset provider is `local` pointing at `data.jsonl` (not
shipped here; replace with your own JSONL file where each line is
`{"input": "...", "expected_output": "..."}`).  The verifier is an inline
`exact_match` metric.

## Single-tenancy note

Open Arena is single-tenant by design: each ModelFactory org-node runs its
own instance.  Authentication is optional (`OPEN_ARENA_API_TOKEN`; defaults
to the built-in dev token for local testing).
