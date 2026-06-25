"""open-arena-core: Pydantic models, API client, and constants for Open Arena.

This package is installable without the heavy evaluation engine.
Provides:
- ``open_arena_core.models`` — Pydantic models generated from openapi.yaml
- ``open_arena_core.client`` — ArenaAPIClient for making authenticated HTTP requests
- ``open_arena_core.constants`` — shared constants (default token, etc.)
"""
from open_arena_core import client, constants, models

__all__ = ["client", "constants", "models"]
