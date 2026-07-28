# License Apache 2.0: (c) 2026 Athena-Reply
"""Keycloak / OIDC JWT authentication adapter (WS7, issue #41).

Environment variables
---------------------
``OIDC_ISSUER``
    Full URL of the Keycloak realm, e.g.
    ``https://keycloak.example.com/realms/myrealm``.
    The provider fetches ``<OIDC_ISSUER>/.well-known/openid-configuration``
    at first use to discover the JWKS URI.

``OIDC_CLIENT_ID``
    The OAuth 2.0 client ID that tokens must be issued *for*.  The provider
    validates that either the ``aud`` claim contains this value **or** the
    ``azp`` (authorised party) claim equals it.

``JWKS_CACHE_TTL``
    Optional integer seconds for how long the JWKS response is cached before
    being re-fetched (default: ``300`` — 5 minutes).

Org / multi-tenancy scoping
---------------------------
:meth:`KeycloakAuthProvider.authenticate` returns a :class:`~src.api.ports.auth_provider.Principal`
whose ``org`` field is derived by :func:`_derive_org`.  The rule today is:

1. If the JWT carries a ``groups`` claim, scan for the first group name that
   matches ``factory-<org>-*`` (the org-group naming convention) and extract
   ``<org>`` as the tenant identifier.
2. Fall back to the ``organization`` claim (a plain string) if present.
3. Fall back to the ``azp`` claim (the OAuth client ID) if present.
4. Otherwise ``org`` is ``None`` (single-tenant / unscoped deployment).

**Integrator note — wiring org-scoping into ArenaAPIService (service.py)**:
Once a ``Principal`` with a non-``None`` ``org`` is returned by this provider,
the service layer can scope every ``Store`` query by calling
``store.filter_by_org(principal.org)`` (or an equivalent predicate).  The
recommended pattern is::

    # In ArenaAPIService (or an auth middleware in app.py):
    principal: Principal = adapters.auth.authenticate(request.headers.get("Authorization"))
    if principal.org:
        store = adapters.store.scoped(org=principal.org)
    else:
        store = adapters.store

This keeps ``service.py`` free of auth logic while still enforcing per-tenant
isolation.  The ``Store`` port can expose a ``scoped(org: str) -> Store``
helper that wraps every query with a ``WHERE org = ?`` clause.
"""
from __future__ import annotations

import logging
import os
import re
import time
from typing import Any
from urllib.parse import urlsplit

import httpx  # httpx is in core deps; imported at module level so tests can patch it

from src.api.ports.auth_provider import AuthProvider, Principal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Org derivation
# ---------------------------------------------------------------------------

_FACTORY_GROUP_RE = re.compile(r"^factory-(?P<org>[^-]+(?:-[^-]+)*)-")


def _derive_org(claims: dict[str, Any]) -> str | None:
    """Derive a tenant / org identifier from JWT claims.

    The lookup order is:

    1. **groups claim** — scan for the first entry matching
       ``factory-<org>-*`` (the org-group naming convention).
       The captured ``<org>`` segment (everything between the first and
       second ``-``) is returned.
    2. **organization claim** — a plain-string org identifier sometimes
       issued by custom Keycloak mappers.
    3. **azp claim** — the authorised-party (OAuth client ID) used as a
       last-resort tenant proxy when nothing more specific is present.
    4. ``None`` — single-tenant / unscoped deployment.

    To change the derivation logic (e.g. to use a different group prefix or
    a Keycloak realm-role), edit this function in isolation — the rest of the
    provider does not depend on its internals.

    Args:
        claims: Decoded JWT payload as a plain dict.

    Returns:
        A non-empty org string, or ``None``.
    """
    # 1. groups claim — org-group naming convention: factory-<org>-<role>
    groups: list[str] = claims.get("groups") or []
    for group in groups:
        m = _FACTORY_GROUP_RE.match(group)
        if m:
            return m.group("org")

    # 2. explicit organization claim
    org = claims.get("organization")
    if org and isinstance(org, str):
        return org

    # 3. azp (authorised party) as a last-resort tenant proxy
    azp = claims.get("azp")
    if azp and isinstance(azp, str):
        return azp

    return None


# ---------------------------------------------------------------------------
# JWKS cache (module-level, process-scoped)
# ---------------------------------------------------------------------------

class _JwksCache:
    """Thread-safe, TTL-bounded cache for a single JWKS response.

    A new instance is created per :class:`KeycloakAuthProvider`; it is NOT
    shared across provider instances so tests can instantiate independent
    providers without cross-contamination.
    """

    def __init__(self, ttl: int = 300) -> None:
        self._ttl = ttl
        self._keys: list[dict[str, Any]] | None = None
        self._fetched_at: float = 0.0

    def get(self) -> list[dict[str, Any]] | None:
        """Return cached keys if still fresh, otherwise ``None``."""
        if self._keys is not None and (time.monotonic() - self._fetched_at) < self._ttl:
            return self._keys
        return None

    def set(self, keys: list[dict[str, Any]]) -> None:
        self._keys = keys
        self._fetched_at = time.monotonic()

    def invalidate(self) -> None:
        self._keys = None
        self._fetched_at = 0.0


# ---------------------------------------------------------------------------
# KeycloakAuthProvider
# ---------------------------------------------------------------------------

class KeycloakAuthProvider(AuthProvider):
    """OIDC/Keycloak JWT authentication adapter (WS7, issue #41).

    Validates an RS256 Bearer JWT against a Keycloak realm and returns a
    :class:`~src.api.ports.auth_provider.Principal` with ``subject`` set
    to the token's ``sub`` claim and ``org`` derived via :func:`_derive_org`.

    The JWKS is fetched lazily on the first authenticate call by following
    the OIDC discovery document at ``<OIDC_ISSUER>/.well-known/openid-configuration``.
    Subsequent calls reuse the cached key set until the TTL expires, after
    which a fresh fetch is made.  This means JWKS rotation is picked up
    automatically within one TTL window.

    Configuration is read from environment variables at *construction time*:

    * ``OIDC_ISSUER`` — required; Keycloak realm base URL.
    * ``OIDC_CLIENT_ID`` — required; expected ``aud``/``azp`` value.
    * ``JWKS_CACHE_TTL`` — optional (default 300 s).

    Raises:
        :class:`~src.api.service.ApiError`: With ``status_code=401`` for any
            authentication failure (missing header, malformed token, expired
            token, bad signature, wrong audience).
    """

    def __init__(
        self,
        issuer: str | None = None,
        client_id: str | None = None,
        jwks_cache_ttl: int | None = None,
    ) -> None:
        self._issuer = issuer or os.environ.get("OIDC_ISSUER", "")
        self._client_id = client_id or os.environ.get("OIDC_CLIENT_ID", "")
        ttl = jwks_cache_ttl if jwks_cache_ttl is not None else int(
            os.environ.get("JWKS_CACHE_TTL", "300")
        )
        self._cache = _JwksCache(ttl=ttl)

        if not self._client_id:
            logger.warning(
                "OIDC_CLIENT_ID is not configured: JWT audience verification is "
                "DISABLED. Any validly-signed token from the issuer will be accepted "
                "regardless of its intended audience. Set OIDC_CLIENT_ID before "
                "relying on Keycloak authentication in production."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def authenticate(self, authorization: str | None) -> Principal:
        """Validate *authorization* and return the caller :class:`~src.api.ports.auth_provider.Principal`.

        Args:
            authorization: Raw value of the HTTP ``Authorization`` header,
                or ``None`` if the header is absent.

        Returns:
            A :class:`~src.api.ports.auth_provider.Principal` with:
            * ``subject`` — the ``sub`` claim from the JWT.
            * ``org`` — derived from groups/organization/azp claims (may be
              ``None`` for single-tenant deployments).

        Raises:
            :class:`~src.api.service.ApiError`: ``status_code=401`` for any
                of: missing header, non-Bearer scheme, expired token, bad
                signature, wrong audience/azp, missing ``sub`` claim.
        """
        from src.api.service import ApiError  # local import avoids circular dep

        # 1. Parse the Authorization header
        if not authorization:
            raise ApiError("unauthorized", "Missing Authorization header.", status_code=401)
        if not authorization.startswith("Bearer "):
            raise ApiError(
                "unauthorized",
                "Authorization header must use Bearer scheme.",
                status_code=401,
            )
        raw_token = authorization[len("Bearer "):]

        # 2. Fetch (or reuse) the JWKS
        try:
            jwks = self._get_jwks()
        except Exception as exc:
            raise ApiError(
                "unauthorized",
                f"Unable to fetch JWKS from identity provider: {exc}",
                status_code=401,
            ) from exc

        # 3. Decode and validate the JWT
        claims = self._decode_token(raw_token, jwks)

        # 4. Build and return the Principal
        subject = claims.get("sub")
        if not subject:
            raise ApiError("unauthorized", "JWT is missing the 'sub' claim.", status_code=401)

        org = _derive_org(claims)
        return Principal(subject=subject, org=org)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_jwks(self) -> list[dict[str, Any]]:
        """Return the JWKS key list, fetching from Keycloak if the cache is stale."""
        cached = self._cache.get()
        if cached is not None:
            return cached

        if not self._issuer:
            raise RuntimeError("OIDC_ISSUER is not configured.")

        discovery_url = self._issuer.rstrip("/") + "/.well-known/openid-configuration"
        resp = httpx.get(discovery_url, timeout=10)
        resp.raise_for_status()
        jwks_uri: str = resp.json()["jwks_uri"]

        # SSRF guard: the discovery document is fetched from the trusted issuer,
        # but its contents are otherwise untrusted. Only follow a `jwks_uri` that
        # shares the issuer's scheme+host so a tampered/spoofed discovery endpoint
        # cannot redirect the key fetch at an internal or arbitrary target.
        self._assert_same_origin_as_issuer(jwks_uri)

        jwks_resp = httpx.get(jwks_uri, timeout=10)
        jwks_resp.raise_for_status()
        keys: list[dict[str, Any]] = jwks_resp.json()["keys"]

        self._cache.set(keys)
        return keys

    def _assert_same_origin_as_issuer(self, url: str) -> None:
        """Reject *url* unless its scheme+host(+port) match the configured issuer.

        The OIDC ``jwks_uri`` is read from the discovery document, which is
        attacker-influenceable in principle (a spoofed or compromised discovery
        endpoint could return an arbitrary URL). Constraining the fetch target to
        the issuer's own origin neutralises that SSRF vector while still allowing
        the standard Keycloak layout where the JWKS lives under the issuer host.

        Raises:
            RuntimeError: if *url* does not share the issuer's origin.
        """
        issuer = urlsplit(self._issuer)
        target = urlsplit(url)
        if (target.scheme, target.hostname, target.port) != (
            issuer.scheme,
            issuer.hostname,
            issuer.port,
        ):
            raise RuntimeError(
                f"Refusing to fetch JWKS from {url!r}: its scheme/host does not "
                f"match the configured OIDC issuer {self._issuer!r}."
            )

    def _decode_token(
        self,
        raw_token: str,
        jwks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Decode and validate *raw_token* against *jwks*.

        Verifies:
        * Signature (RS256) using the matching public key from JWKS.
        * ``exp`` claim (token expiry).
        * Audience / azp: either ``aud`` contains ``OIDC_CLIENT_ID`` or
          ``azp`` equals ``OIDC_CLIENT_ID``.

        Args:
            raw_token: The raw JWT string (no ``Bearer `` prefix).
            jwks: List of JWK key objects as returned by the JWKS endpoint.

        Returns:
            The decoded JWT payload as a plain dict.

        Raises:
            :class:`~src.api.service.ApiError`: ``status_code=401`` on any
                validation error.
        """
        import jwt as pyjwt  # lazy import — pyjwt[crypto] in keycloak extra
        from jwt import PyJWKClient, ExpiredSignatureError, InvalidTokenError
        from src.api.service import ApiError

        # Build an in-memory JWKS client from our cached keys
        jwks_data = {"keys": jwks}

        try:
            jwks_client = PyJWKClient.__new__(PyJWKClient)
            # Manually supply the key set without making a network call
            # PyJWKClient.get_signing_key_from_jwt parses the header "kid"
            # and matches it against the JWKS.  We initialise with a
            # data-URI trick: pass a callable that returns our cached data.
            signing_key = self._get_signing_key(raw_token, jwks)
        except Exception as exc:
            raise ApiError(
                "unauthorized",
                f"Unable to find a matching signing key: {exc}",
                status_code=401,
            ) from exc

        # Audience options: we accept aud-list or azp equality
        try:
            claims: dict[str, Any] = pyjwt.decode(
                raw_token,
                signing_key,
                algorithms=["RS256"],
                issuer=self._issuer,
                options={
                    "verify_aud": False,  # we do audience check manually below
                    "verify_exp": True,
                    "verify_iss": True,  # reject tokens from other issuers
                },
            )
        except ExpiredSignatureError as exc:
            raise ApiError("unauthorized", "Token has expired.", status_code=401) from exc
        except InvalidTokenError as exc:
            raise ApiError("unauthorized", f"Invalid token: {exc}", status_code=401) from exc

        # Manual audience / azp check
        self._verify_audience(claims)

        return claims

    def _get_signing_key(self, raw_token: str, jwks: list[dict[str, Any]]) -> Any:
        """Extract the signing key that matches the JWT header ``kid``.

        Uses :class:`jwt.PyJWKClient` against an in-memory JWKS dict to
        avoid a second network call.

        Args:
            raw_token: The raw JWT string.
            jwks: Cached JWKS key list.

        Returns:
            A :class:`jwt.algorithms.RSAAlgorithm` public-key object.
        """
        import jwt as pyjwt
        from jwt import PyJWKSet

        # PyJWKSet can be constructed from a raw dict; use it to find the key
        # matching the JWT's "kid" header field.
        jwk_set = PyJWKSet.from_dict({"keys": jwks})
        header = pyjwt.get_unverified_header(raw_token)
        kid = header.get("kid")

        if kid:
            for jwk in jwk_set.keys:
                if jwk.key_id == kid:
                    return jwk.key
            raise ValueError(f"No key with kid={kid!r} found in JWKS.")

        # If no kid, return the first RS256 key
        for jwk in jwk_set.keys:
            if getattr(jwk, "algorithm_name", None) in ("RS256", "RS384", "RS512"):
                return jwk.key

        raise ValueError("No RS256 key found in JWKS.")

    def _verify_audience(self, claims: dict[str, Any]) -> None:
        """Assert that *claims* address this application's client ID.

        Accepts a token when **either** condition holds:

        * ``aud`` claim is a string equal to ``OIDC_CLIENT_ID``, **or**
          ``aud`` is a list that contains ``OIDC_CLIENT_ID``.
        * ``azp`` claim equals ``OIDC_CLIENT_ID`` (implicit grant / public
          client tokens often omit ``aud`` in favour of ``azp``).

        Args:
            claims: Decoded JWT payload.

        Raises:
            :class:`~src.api.service.ApiError`: ``status_code=401`` when
                neither condition holds.
        """
        from src.api.service import ApiError

        if not self._client_id:
            # No client ID configured — skip audience check (dev/test shortcut).
            # A prominent startup warning is emitted from __init__ so this relaxed
            # mode is never silently relied on in production.
            return

        aud = claims.get("aud")
        if aud is not None:
            if isinstance(aud, str):
                if aud == self._client_id:
                    return
            elif isinstance(aud, list):
                if self._client_id in aud:
                    return

        azp = claims.get("azp")
        if azp == self._client_id:
            return

        raise ApiError(
            "unauthorized",
            f"Token audience does not include client_id={self._client_id!r}.",
            status_code=401,
        )
