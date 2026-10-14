"""Open arena: reads an evaluation's YAML and runs it.

The engine the arena was ported from: `config` validates the sweep's YAML,
`evaluate.run_sweep` runs every `models x datasets` cell over the framework
programs `programs/` builds, reading rows through the dataset providers in
`datasets/` (each row rendered through the dataset's Jinja templates) and
scoring them with the framework's rewards and the arena's own in `rewards/`.
"""
