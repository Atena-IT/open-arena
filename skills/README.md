# Open Arena Agent Skills

Six Claude Code skills that package the operational knowledge for Open Arena. Each skill is a `SKILL.md` with a YAML frontmatter trigger and step-by-step instructions.

> Note: `.claude/` is gitignored in this repo, so skills live under `skills/` at the repo root.

## Skills index

| Skill | Directory | When to use |
|---|---|---|
| **arena-configure** | `skills/arena-configure/` | Author or edit `config.yaml`: datasets, rewards, metrics, experiments blocks |
| **arena-run-sweep** | `skills/arena-run-sweep/` | Run the evaluation sweep and interpret `last_run.tsv` / the leaderboard matrix |
| **arena-prepare-data** | `skills/arena-prepare-data/` | Write `prepare_data.py`, wire loaders in `config.yaml`, smoke-test rows load |
| **arena-build-reward** | `skills/arena-build-reward/` | Add or iterate on a custom reward in `src/rewards/` and wire it into the sweep |
| **arena-autoresearch** | `skills/arena-autoresearch/` | Start and operate the autonomous reward-R&D experiment loop |
| **arena-api** | `skills/arena-api/` | Start the REST API server and make authenticated requests |

## Source docs these skills reference

- `README.md` — install, CLI flags, config schema, dataset providers, rewards, agent/MCP
- `AUTORESEARCH.md` — autonomous experiment loop protocol + `results.tsv` schema
- `PREPARE_DATA.md` — dataset preparation pipeline, all loader providers, pitfalls
- `REWARDS_BUILDING.md` — custom reward authoring, base classes, masking, 7-step walkthrough
- `config.example.yaml` — canonical reference for every YAML key
- `src/evaluate.py` — `arena` CLI: sweep command + `arena serve` + `arena request`
- `src/api/app.py` + `openapi.yaml` — REST API routes and schemas
