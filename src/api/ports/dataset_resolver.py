# License Apache 2.0: (c) 2026 Athena-Reply
"""Port 3 — DatasetResolver

Translates an :class:`~src.api.models.DatasetBinding` into the flat
dictionary entry that the Arena YAML runner understands.

The default adapter is :class:`LegacyDatasetResolver` which reproduces
the original ``_dataset_entry`` / ``_DATASET_TYPES`` logic verbatim and
dispatches by ``binding.provider`` — including WS4 ``unity_catalog``
bindings, which map ``source_ref`` to the ``table`` kwarg (see
``PROVIDER_SOURCE_FIELDS``). No provider-specific resolver subclass is
required.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from open_arena_core import models as api


PROVIDER_SOURCE_FIELDS: dict[str, str] = {
    "braintrust": "dataset_name",
    "folder": "path",
    "huggingface": "path",
    "langfuse": "dataset_name",
    "langsmith": "dataset_name",
    "local": "path",
    "opik": "dataset_name",
    "phoenix": "dataset_name",
    # WS4: UnityCatalogDataset expects the fully-qualified table name
    # (``catalog.schema.table``) under the ``table`` kwarg.
    "unity_catalog": "table",
}


class DatasetResolver(ABC):
    """Port for translating a dataset binding into a runner YAML entry.

    A resolver receives a :class:`~src.api.models.DatasetBinding` and
    returns the ``dict`` that will be embedded in the ``datasets:`` section
    of the YAML config consumed by ``run_sweep``.
    """

    @abstractmethod
    def resolve(self, binding: api.DatasetBinding) -> dict[str, Any]:
        """Translate *binding* into a runner dataset entry dict.

        Args:
            binding: The dataset binding from an
                :class:`~src.api.models.InlineEnvironmentDefinition`.

        Returns:
            A ``dict`` compatible with the ``datasets:`` YAML section
            expected by ``run_sweep``.

        Raises:
            :class:`~src.api.service.ApiError`: When *binding.provider* is
                unknown or the binding is otherwise invalid.
        """


class LegacyDatasetResolver(DatasetResolver):
    """Default adapter — delegates to ``_DATASET_TYPES`` / ``load_dataset_from_yaml``.

    Reproduces the original ``ArenaAPIService._dataset_entry`` logic
    exactly so no existing behavior changes.  Provider dispatch (including
    WS4 ``unity_catalog``) is data-driven via ``PROVIDER_SOURCE_FIELDS`` +
    ``_DATASET_TYPES`` — no per-provider subclass is needed.
    """

    def resolve(self, binding: api.DatasetBinding) -> dict[str, Any]:  # noqa: D102
        from src.api.service import ApiError  # local import avoids circular dep
        from src.datasets import _DATASET_TYPES

        supported = set(_DATASET_TYPES)
        if binding.provider not in supported:
            raise ApiError(
                "unknown_dataset_provider",
                f"Unsupported dataset provider {binding.provider!r}.",
                details={"supported": sorted(supported)},
            )

        entry: dict[str, Any] = {"type": binding.provider}
        if binding.input_template is not None:
            entry["input_template"] = binding.input_template
        if binding.output_template is not None:
            entry["output_template"] = binding.output_template

        selector = dict(binding.selector or {})
        field = PROVIDER_SOURCE_FIELDS.get(binding.provider, "source_ref")
        if binding.source_ref is not None:
            selector.setdefault(field, binding.source_ref)
        if binding.version is not None:
            if binding.provider == "huggingface":
                selector.setdefault("revision", binding.version)
            else:
                selector.setdefault("version", binding.version)
        for key, value in (binding.metadata or {}).items():
            selector.setdefault(key, value)
        entry.update(selector)
        return entry
