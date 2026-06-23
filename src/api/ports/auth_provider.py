# License Apache 2.0: (c) 2026 Athena-Reply
"""Port 6 — AuthProvider

Authenticates an incoming HTTP ``Authorization`` header and returns a
:class:`Principal` describing the caller.

The default adapter is :class:`StaticBearerAuthProvider` which compares
the header value against the ``OPEN_ARENA_API_TOKEN`` environment variable
exactly as ``require_bearer`` did before.

WS7: Keycloak — add a ``KeycloakAuthProvider`` that validates a JWT bearer
token against a Keycloak realm and populates ``Principal.org`` from the
token's ``azp`` / ``organization`` claims.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from open_arena_core.constants import DEFAULT_API_TOKEN


@dataclass
class Principal:
    """Authenticated caller identity.

    Attributes:
        subject: The authenticated subject identifier (e.g. token name,
            user ID, service account name).
        org: Optional organization / tenant identifier.  ``None`` for
            single-tenant deployments.
    """

    subject: str
    org: str | None = field(default=None)


class AuthProvider(ABC):
    """Port for authenticating incoming requests.

    Implementations receive the raw ``Authorization`` header value and
    either return a :class:`Principal` or raise
    :class:`~src.api.service.ApiError` with ``status_code=401``.

    WS7: Keycloak — implement ``KeycloakAuthProvider`` that introspects
    the bearer JWT against ``OPEN_ARENA_KEYCLOAK_REALM`` and returns a
    ``Principal`` with ``org`` populated from the token claims.
    """

    @abstractmethod
    def authenticate(self, authorization: str | None) -> Principal:
        """Validate *authorization* and return the caller :class:`Principal`.

        Args:
            authorization: The raw value of the HTTP ``Authorization``
                header, or ``None`` if absent.

        Returns:
            A :class:`Principal` describing the authenticated caller.

        Raises:
            :class:`~src.api.service.ApiError`: With ``status_code=401``
                when authentication fails.
        """


class StaticBearerAuthProvider(AuthProvider):
    """Default adapter — compares against a static ``OPEN_ARENA_API_TOKEN``.

    Reproduces the original ``require_bearer`` behavior verbatim: the
    expected token is read from the ``OPEN_ARENA_API_TOKEN`` environment
    variable, falling back to :data:`~src.api.constants.DEFAULT_API_TOKEN`.
    The ``Principal.subject`` is set to ``"static"`` and ``org`` is
    ``None``.

    WS7: Keycloak — replace this provider by setting
    ``OPEN_ARENA_AUTH=keycloak`` and registering ``KeycloakAuthProvider``
    in the adapter registry.
    """

    def authenticate(self, authorization: str | None) -> Principal:  # noqa: D102
        from src.api.service import ApiError  # local import avoids circular dep

        token = os.getenv("OPEN_ARENA_API_TOKEN", DEFAULT_API_TOKEN)
        expected = "Bearer " + token
        if authorization != expected:
            raise ApiError("unauthorized", "Missing or invalid bearer token.", status_code=401)
        return Principal(subject="static", org=None)
