# License Apache 2.0: (c) 2026 Athena-Reply
"""Port 2 — EnvironmentBackend

Resolves an :class:`~src.api.models.Environment` (or the source embedded
in a pending subject) into the concrete configuration that the execution
layer needs to launch a run.

The default adapter is :class:`InlineEnvironmentBackend` which handles
``kind=inline`` environments exactly as the original ``_config_for_pending``
logic did — no git cloning, no remote fetching.

WS2 (Gitea/GitHub) will implement a ``GitEnvironmentBackend`` that clones
the repo at *commit_sha* and returns the resolved ``InlineEnvironmentDefinition``
from the checked-out tree.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from src.api import models as api


@dataclass
class ResolvedEnvironment:
    """Carrier returned by :meth:`EnvironmentBackend.resolve`.

    Attributes:
        definition: The inline definition that the execution layer will
            consume.  Always set for ``kind=inline``; set by WS2 adapters
            after fetching the remote source.
        commit_sha: The resolved VCS commit SHA, if applicable.
            ``None`` for inline environments.
        content_hash: A stable hash of the environment content, if
            applicable.  ``None`` for inline environments.
        local_path: Filesystem path to the checked-out environment tree,
            if applicable.  ``None`` for inline environments.
    """

    definition: api.InlineEnvironmentDefinition
    commit_sha: str | None = field(default=None)
    content_hash: str | None = field(default=None)
    local_path: str | None = field(default=None)


class EnvironmentBackend(ABC):
    """Port for resolving an :class:`~src.api.models.Environment` source.

    Implementations receive the full :class:`~src.api.models.Environment`
    object and return a :class:`ResolvedEnvironment` that bundles the
    concrete ``InlineEnvironmentDefinition`` together with optional VCS
    provenance metadata.
    """

    @abstractmethod
    def resolve(self, environment: api.Environment) -> ResolvedEnvironment:
        """Resolve *environment* into a :class:`ResolvedEnvironment`.

        Args:
            environment: The environment to resolve.

        Returns:
            A :class:`ResolvedEnvironment` with at least ``definition`` set.

        Raises:
            NotImplementedError: When the environment ``source.kind`` is not
                supported by this adapter.
            :class:`~src.api.service.ApiError`: When the environment cannot
                be executed (e.g. ``kind=inline`` with no ``inline_definition``).
        """


class InlineEnvironmentBackend(EnvironmentBackend):
    """Default adapter — handles ``kind=inline`` environments as-is.

    For ``kind=inline`` the ``inline_definition`` stored on the environment
    object is used verbatim; ``commit_sha`` and ``content_hash`` are ``None``
    because there is no VCS source.

    For all other ``source.kind`` values a :exc:`NotImplementedError` is
    raised with a TODO comment indicating which workstream will implement it.
    """

    def resolve(self, environment: api.Environment) -> ResolvedEnvironment:  # noqa: D102
        from src.api.service import ApiError  # local import avoids circular dep

        if environment.source.kind == api.EnvironmentSourceKind.inline:
            if environment.inline_definition is None:
                raise ApiError(
                    "non_executable_environment",
                    f"Environment {environment.id} has no inline definition to execute.",
                )
            return ResolvedEnvironment(
                definition=environment.inline_definition,
                commit_sha=None,
                content_hash=None,
                local_path=None,
            )

        # WS2 (Gitea/GitHub) will implement git-backed environment resolution.
        raise NotImplementedError(
            f"EnvironmentBackend for source.kind={environment.source.kind!r} is not yet "
            "implemented.  WS2 will add a GitEnvironmentBackend that clones the repo at "
            "commit_sha and returns the resolved InlineEnvironmentDefinition."
        )
