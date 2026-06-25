# License Apache 2.0: (c) 2026 Athena-Reply
"""initial: create all Open Arena tables

Revision ID: 0001
Revises:
Create Date: 2026-06-23

Creates the six tables that back the Open Arena API Store port:

* ``verifiers``     — VerifierSuite resources
* ``environments``  — Environment resources
* ``leaderboards``  — Leaderboard resources
* ``runs``          — Run resources (with idempotency-key UNIQUE constraint)
* ``run_results``   — RunResult documents keyed by run_id
* ``subject_cache`` — Per-subject result cache keyed by fingerprint

All ``doc`` columns are ``JSONB`` on PostgreSQL (binary JSON with GIN index
support) and plain ``JSON`` (``TEXT``) on every other dialect via SQLAlchemy
type variants.

Apply with::

    alembic upgrade head
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import JSON

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _json_col(name: str) -> sa.Column:
    """JSONB on Postgres, plain JSON elsewhere."""
    return sa.Column(name, JSON().with_variant(JSONB, "postgresql"), nullable=False)


def upgrade() -> None:
    op.create_table(
        "verifiers",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("name", sa.String, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        _json_col("doc"),
    )
    op.create_index("ix_verifiers_created_at", "verifiers", ["created_at"])

    op.create_table(
        "environments",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("name", sa.String, nullable=False),
        sa.Column("source_kind", sa.String, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        _json_col("doc"),
    )
    op.create_index("ix_environments_created_at", "environments", ["created_at"])

    op.create_table(
        "leaderboards",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("name", sa.String, nullable=False),
        sa.Column("visibility", sa.String, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        _json_col("doc"),
    )
    op.create_index("ix_leaderboards_created_at", "leaderboards", ["created_at"])

    op.create_table(
        "runs",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("mode", sa.String, nullable=False),
        sa.Column("status", sa.String, nullable=False),
        sa.Column("cache_status", sa.String, nullable=False),
        sa.Column("leaderboard_id", sa.String, nullable=True),
        sa.Column("idempotency_key", sa.String, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        _json_col("doc"),
        sa.UniqueConstraint("idempotency_key", name="uq_runs_idempotency_key"),
    )
    op.create_index("ix_runs_leaderboard_id", "runs", ["leaderboard_id"])
    op.create_index("ix_runs_created_at", "runs", ["created_at"])

    op.create_table(
        "run_results",
        sa.Column("run_id", sa.String, primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        _json_col("doc"),
    )
    op.create_index("ix_run_results_created_at", "run_results", ["created_at"])

    op.create_table(
        "subject_cache",
        sa.Column("fingerprint", sa.String, primary_key=True),
        sa.Column("run_id", sa.String, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        _json_col("doc"),
    )


def downgrade() -> None:
    op.drop_table("subject_cache")
    op.drop_index("ix_run_results_created_at", table_name="run_results")
    op.drop_table("run_results")
    op.drop_index("ix_runs_created_at", table_name="runs")
    op.drop_index("ix_runs_leaderboard_id", table_name="runs")
    op.drop_table("runs")
    op.drop_index("ix_leaderboards_created_at", table_name="leaderboards")
    op.drop_table("leaderboards")
    op.drop_index("ix_environments_created_at", table_name="environments")
    op.drop_table("environments")
    op.drop_index("ix_verifiers_created_at", table_name="verifiers")
    op.drop_table("verifiers")