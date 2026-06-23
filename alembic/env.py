# License Apache 2.0: (c) 2026 Athena-Reply
"""Alembic environment configuration for Open Arena.

Reads the database URL from the ``DATABASE_URL`` environment variable
(defaulting to ``postgresql+psycopg://localhost/open-arena``) and runs
migrations in either offline or online mode.

Usage
-----
Apply all pending migrations::

    alembic upgrade head

Generate a new autogenerate migration::

    alembic revision --autogenerate -m "describe change"

Downgrade one step::

    alembic downgrade -1
"""
from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# Import the shared metadata from the store module so Alembic can
# inspect the table definitions for autogenerate support.
from src.api.stores.sqlalchemy_store import metadata as target_metadata

# ---------------------------------------------------------------------------
# Alembic Config object
# ---------------------------------------------------------------------------

config = context.config

# Override sqlalchemy.url from the environment variable if set.
_dsn = os.environ.get("DATABASE_URL", "postgresql+psycopg://localhost/open-arena")
config.set_main_option("sqlalchemy.url", _dsn)

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)


# ---------------------------------------------------------------------------
# Offline migrations
# ---------------------------------------------------------------------------

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL and not an Engine, though
    an Engine is acceptable here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


# ---------------------------------------------------------------------------
# Online migrations
# ---------------------------------------------------------------------------

def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine and associate a connection
    with the context.
    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()