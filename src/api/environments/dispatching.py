# License Apache 2.0: (c) 2026 Athena-Reply
"""``src.api.environments.dispatching`` -- DispatchingEnvironmentBackend (P2-1).

Routes :meth:`resolve` calls to the appropriate sub-backend by
``environment.source.kind``:

* ``inline``               -> :class:`~src.api.ports.environment_backend.InlineEnvironmentBackend`
* ``github_repo``          -> :class:`~src.api.environments.git_backend.GitEnvironmentBackend`
* ``prime_environment_hub``-> :class:`~src.api.environments.prime_hub_backend.PrimeEnvHubBackend`
* ``huggingface_hub``      -> :class:`~src.api.ports.environment_backend.InlineEnvironmentBackend`
  (falls through to inline -- HF hub resolution is not yet implemented and
  delegates gracefully to the existing inline path if an inline_definition is
  present, or raises :exc:`NotImplementedError` otherwise)

This allows a single deployment to serve mixed-kind environments without
requiring a global ``OPEN_ARENA_ENV_BACKEND`` choice that locks in one backend
for all requests.

Usage
-----
The :class:`DispatchingEnvironmentBackend` is the **default** backend
(selected when ``OPEN_ARENA_ENV_BACKEND`` is ``"dispatch"`` or unset after the
P2-1 settings change).  Individual single-backend values (``"inline"``,
``"git"``, ``"prime_hub"``) remain as explicit overrides for deployments that
only serve one kind of environment.
"""
from __future__ import annotations

from open_arena_core import models as api
from src.api.ports.environment_backend import (
    EnvironmentBackend,
    InlineEnvironmentBackend,
    ResolvedEnvironment,
)


class DispatchingEnvironmentBackend(EnvironmentBackend):
    """Routes ``resolve()`` by ``source.kind`` to the matching sub-backend.

    Sub-backends are lazily imported to avoid pulling in heavy optional
    dependencies (``httpx``, ``gitpython``, etc.) unless a request that
    needs them actually arrives.

    Args:
        inline_backend: Override the inline sub-backend (default: a fresh
            :class:`InlineEnvironmentBackend`).  Useful for testing.
        git_backend: Override the git sub-backend.  When ``None`` (default),
            a :class:`~src.api.environments.git_backend.GitEnvironmentBackend`
            is instantiated on first use.
        prime_hub_backend: Override the Prime Hub sub-backend.  When ``None``
            (default), a :class:`~src.api.environments.prime_hub_backend.PrimeEnvHubBackend`
            is instantiated on first use.
    """

    def __init__(
        self,
        inline_backend: EnvironmentBackend | None = None,
        git_backend: EnvironmentBackend | None = None,
        prime_hub_backend: EnvironmentBackend | None = None,
    ) -> None:
        self._inline = inline_backend or InlineEnvironmentBackend()
        self._git: EnvironmentBackend | None = git_backend
        self._prime_hub: EnvironmentBackend | None = prime_hub_backend

    # ------------------------------------------------------------------
    # Lazy accessors (avoid importing optional dependencies at startup)
    # ------------------------------------------------------------------

    @property
    def _git_backend(self) -> EnvironmentBackend:
        if self._git is None:
            from src.api.environments.git_backend import GitEnvironmentBackend
            self._git = GitEnvironmentBackend()
        return self._git

    @property
    def _prime_hub_backend(self) -> EnvironmentBackend:
        if self._prime_hub is None:
            from src.api.environments.prime_hub_backend import PrimeEnvHubBackend
            self._prime_hub = PrimeEnvHubBackend()
        return self._prime_hub

    # ------------------------------------------------------------------
    # EnvironmentBackend protocol
    # ------------------------------------------------------------------

    def resolve(self, environment: api.Environment) -> ResolvedEnvironment:
        """Dispatch to the appropriate sub-backend based on ``source.kind``.

        Args:
            environment: The environment to resolve.

        Returns:
            A :class:`ResolvedEnvironment` from the matching sub-backend.

        Raises:
            NotImplementedError: When ``source.kind`` is not recognised.
        """
        kind = environment.source.kind

        if kind == api.EnvironmentSourceKind.inline:
            return self._inline.resolve(environment)

        if kind == api.EnvironmentSourceKind.github_repo:
            return self._git_backend.resolve(environment)

        if kind == api.EnvironmentSourceKind.prime_environment_hub:
            return self._prime_hub_backend.resolve(environment)

        if kind == api.EnvironmentSourceKind.huggingface_hub:
            # HF hub resolution is not yet implemented; fall through to inline
            # when an inline_definition is present so existing payloads still work.
            if environment.inline_definition is not None:
                return self._inline.resolve(environment)
            raise NotImplementedError(
                f"DispatchingEnvironmentBackend: source.kind={kind!r} with no "
                "inline_definition is not yet supported.  Contribute a "
                "HuggingFaceEnvironmentBackend to handle this case."
            )

        raise NotImplementedError(
            f"DispatchingEnvironmentBackend: unrecognised source.kind={kind!r}. "
            "Supported values: 'inline', 'github_repo', 'prime_environment_hub', "
            "'huggingface_hub' (inline fallback)."
        )