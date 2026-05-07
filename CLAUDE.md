# Notes for AI coding agents

## Autoresearch trigger

When the user asks to **start the research loop** (or "begin
autoresearch", "run autoresearch", "kick off the autoresearch loop", or
any obvious paraphrase), read `AUTORESEARCH.md` end-to-end and follow
it:

1. Walk through the **Setup** section with the user (agree on a run
   tag, create the `autoresearch/<tag>` branch, read the in-scope
   files, smoke-test datasets, initialize `results.tsv`).
2. Wait for explicit confirmation from the user before kicking off
   experiments.
3. Once confirmed, enter the **experiment loop** and run autonomously
   — the "NEVER STOP" rule in `AUTORESEARCH.md` applies; do not pause
   to ask if you should continue.
