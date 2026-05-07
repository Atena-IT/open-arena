# License Apache 2.0: (c) 2026 Athena-Reply

"""Prepare datasets for the open-arena evaluation sweep.

Companion script to `evaluate.py`. The platform itself does not ship a
synthetic-data engine — generation, selection, deduplication, and any
other dataset-preparation logic lives here, per project.

Anything useful to prepare the raw data is fair game here; this script
is intentionally free-form.
"""