# License Apache 2.0: (c) 2026 Athena-Reply
"""``src.api.manifest`` -- back-compat re-export.

The EvalEnvironment manifest loader now lives in the thin
:mod:`open_arena_core.manifest` package so the standalone CLI
(``open-arena-cli``) can parse ``eval.yaml`` manifests without depending
on the full engine (WS12). This module re-exports the public API so
existing ``from src.api.manifest import load_manifest`` imports keep
working.
"""
from __future__ import annotations

from open_arena_core.manifest import (  # noqa: F401
    load_manifest,
)

__all__ = ["load_manifest"]
