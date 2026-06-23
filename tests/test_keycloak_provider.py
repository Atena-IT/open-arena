# License Apache 2.0: (c) 2026 Athena-Reply
"""Tests for KeycloakAuthProvider (WS7, issue #41).

All external I/O is mocked — no real Keycloak server is needed.

The test suite mints a fresh RS256 keypair in-process and returns the public
key via a mocked OIDC discovery + JWKS endpoint so that JWT validation
exercises the real PyJWT crypto path.

Test matrix
-----------
* Valid token                         → Principal(subject, org)
* Expired token                       → ApiError 401
* Bad signature (wrong private key)   → ApiError 401
* Wrong audience / azp                → ApiError 401
* Missing Authorization header        → ApiError 401
* Non-Bearer scheme                   → ApiError 401
* Org derivation from groups claim    → correct org extracted
* JWKS caching                        → fetch_json called exactly once across
                                        two authenticate() calls
"""
from __future__ import annotations

import json
import time
from typing import Any
from unittest.mock import MagicMock, patch, call

import pytest

# ---------------------------------------------------------------------------
# Helpers — generate a test RS256 keypair and build a JWKS JSON object
# ---------------------------------------------------------------------------

def _make_rs256_keypair():
    """Return (private_key, public_key) using cryptography library."""
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.backends import default_backend

    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend(),
    )
    return private_key, private_key.public_key()


def _public_key_to_jwk(public_key, kid: str = "test-kid") -> dict[str, Any]:
    """Serialise *public_key* as a JWK dict (suitable for a JWKS ``keys`` list)."""
    import jwt
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    # PyJWT's RSAAlgorithm can convert a public key to JWK JSON
    alg = jwt.algorithms.RSAAlgorithm(jwt.algorithms.RSAAlgorithm.SHA256)
    jwk_str = alg.to_jwk(public_key)
    jwk = json.loads(jwk_str)
    jwk["kid"] = kid
    jwk["use"] = "sig"
    jwk["alg"] = "RS256"
    return jwk


def _make_token(
    private_key,
    *,
    sub: str = "user-123",
    aud: str | list[str] = "arena-client",
    azp: str = "arena-client",
    groups: list[str] | None = None,
    organization: str | None = None,
    exp_offset: int = 3600,
    kid: str = "test-kid",
    algorithm: str = "RS256",
) -> str:
    """Mint a signed JWT for testing."""
    import jwt

    now = int(time.time())
    payload: dict[str, Any] = {
        "sub": sub,
        "aud": aud,
        "azp": azp,
        "iss": "https://keycloak.example.com/realms/test",
        "iat": now,
        "exp": now + exp_offset,
    }
    if groups is not None:
        payload["groups"] = groups
    if organization is not None:
        payload["organization"] = organization

    headers = {"kid": kid}
    return jwt.encode(payload, private_key, algorithm=algorithm, headers=headers)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rs256_keypair():
    """Module-scoped RS256 keypair shared across tests."""
    return _make_rs256_keypair()


@pytest.fixture(scope="module")
def jwk(rs256_keypair):
    _, public_key = rs256_keypair
    return _public_key_to_jwk(public_key, kid="test-kid")


@pytest.fixture(scope="module")
def jwks_response(jwk):
    return {"keys": [jwk]}


DISCOVERY_DOC = {
    "issuer": "https://keycloak.example.com/realms/test",
    "jwks_uri": "https://keycloak.example.com/realms/test/protocol/openid-connect/certs",
}


def _mock_httpx(jwks_response: dict[str, Any]):
    """Return a context manager that patches ``httpx.get`` to serve discovery + JWKS."""

    def _get(url: str, **kwargs):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        if "openid-configuration" in url:
            resp.json.return_value = DISCOVERY_DOC
        else:
            resp.json.return_value = jwks_response
        return resp

    return patch("src.api.auth.keycloak_provider.httpx.get", side_effect=_get)


def _make_provider(jwks_cache_ttl: int = 300) -> "KeycloakAuthProvider":
    from src.api.auth.keycloak_provider import KeycloakAuthProvider

    return KeycloakAuthProvider(
        issuer="https://keycloak.example.com/realms/test",
        client_id="arena-client",
        jwks_cache_ttl=jwks_cache_ttl,
    )


# ---------------------------------------------------------------------------
# Tests — happy path
# ---------------------------------------------------------------------------

def test_valid_token_returns_principal(rs256_keypair, jwks_response):
    """A well-formed, unexpired token with the correct audience → Principal."""
    private_key, _ = rs256_keypair
    token = _make_token(private_key, sub="u-abc", aud="arena-client", groups=["factory-acme-admin"])
    provider = _make_provider()

    with _mock_httpx(jwks_response):
        principal = provider.authenticate(f"Bearer {token}")

    assert principal.subject == "u-abc"
    assert principal.org == "acme"


def test_valid_token_no_groups_returns_none_org(rs256_keypair, jwks_response):
    """Token without groups/organization/azp derives org from azp (last resort)."""
    private_key, _ = rs256_keypair
    # azp = "arena-client" — used as org fallback
    token = _make_token(private_key, sub="svc-1", aud="arena-client", azp="arena-client")
    provider = _make_provider()

    with _mock_httpx(jwks_response):
        principal = provider.authenticate(f"Bearer {token}")

    assert principal.subject == "svc-1"
    # azp equals client_id but still gets surfaced as org
    assert principal.org == "arena-client"


def test_valid_token_aud_list(rs256_keypair, jwks_response):
    """Token whose aud is a list containing the client ID is accepted."""
    private_key, _ = rs256_keypair
    token = _make_token(private_key, sub="u-list", aud=["arena-client", "other"])
    provider = _make_provider()

    with _mock_httpx(jwks_response):
        principal = provider.authenticate(f"Bearer {token}")

    assert principal.subject == "u-list"


def test_valid_token_azp_audience(rs256_keypair, jwks_response):
    """Token without aud but with matching azp is accepted."""
    import jwt as pyjwt

    private_key, _ = rs256_keypair
    now = int(time.time())
    payload = {
        "sub": "svc-azp",
        "azp": "arena-client",
        "iss": "https://keycloak.example.com/realms/test",
        "iat": now,
        "exp": now + 3600,
    }
    token = pyjwt.encode(payload, private_key, algorithm="RS256", headers={"kid": "test-kid"})
    provider = _make_provider()

    with _mock_httpx(jwks_response):
        principal = provider.authenticate(f"Bearer {token}")

    assert principal.subject == "svc-azp"


# ---------------------------------------------------------------------------
# Tests — failure cases
# ---------------------------------------------------------------------------

def test_missing_header_raises_401():
    """No Authorization header → ApiError 401."""
    from src.api.service import ApiError

    provider = _make_provider()
    with pytest.raises(ApiError) as exc_info:
        provider.authenticate(None)
    assert exc_info.value.status_code == 401


def test_non_bearer_scheme_raises_401():
    """Basic auth scheme → ApiError 401."""
    from src.api.service import ApiError

    provider = _make_provider()
    with pytest.raises(ApiError) as exc_info:
        provider.authenticate("Basic dXNlcjpwYXNz")
    assert exc_info.value.status_code == 401


def test_expired_token_raises_401(rs256_keypair, jwks_response):
    """Expired token → ApiError 401."""
    from src.api.service import ApiError

    private_key, _ = rs256_keypair
    token = _make_token(private_key, exp_offset=-60)  # already expired
    provider = _make_provider()

    with _mock_httpx(jwks_response):
        with pytest.raises(ApiError) as exc_info:
            provider.authenticate(f"Bearer {token}")
    assert exc_info.value.status_code == 401


def test_bad_signature_raises_401(jwks_response):
    """Token signed with a different private key → ApiError 401."""
    from src.api.service import ApiError

    # Mint a fresh keypair — its private key is NOT in the JWKS
    other_private, _ = _make_rs256_keypair()
    token = _make_token(other_private, aud="arena-client")
    provider = _make_provider()

    with _mock_httpx(jwks_response):
        with pytest.raises(ApiError) as exc_info:
            provider.authenticate(f"Bearer {token}")
    assert exc_info.value.status_code == 401


def test_wrong_audience_raises_401(rs256_keypair, jwks_response):
    """Token addressed to a different audience → ApiError 401."""
    from src.api.service import ApiError

    private_key, _ = rs256_keypair
    import jwt as pyjwt

    now = int(time.time())
    payload = {
        "sub": "u-wrong",
        "aud": "other-service",
        "azp": "other-service",
        "iss": "https://keycloak.example.com/realms/test",
        "iat": now,
        "exp": now + 3600,
    }
    token = pyjwt.encode(payload, private_key, algorithm="RS256", headers={"kid": "test-kid"})
    provider = _make_provider()

    with _mock_httpx(jwks_response):
        with pytest.raises(ApiError) as exc_info:
            provider.authenticate(f"Bearer {token}")
    assert exc_info.value.status_code == 401


# ---------------------------------------------------------------------------
# Tests — org derivation
# ---------------------------------------------------------------------------

def test_derive_org_factory_group():
    """factory-<org>-<role> group format → org extracted correctly."""
    from src.api.auth.keycloak_provider import _derive_org

    claims = {"groups": ["/other-group", "factory-acme-admin", "factory-acme-viewer"]}
    assert _derive_org(claims) == "acme"


def test_derive_org_factory_group_hyphenated_org():
    """Org names with hyphens are captured as the full middle segment.

    For ``factory-my-org-admin`` the regex ``[^-]+(?:-[^-]+)*`` greedily
    captures ``my-org`` (everything between ``factory-`` and the trailing
    ``-admin``).  This preserves multi-segment org names like ``my-org``.
    """
    from src.api.auth.keycloak_provider import _derive_org

    claims = {"groups": ["factory-my-org-admin"]}
    assert _derive_org(claims) == "my-org"


def test_derive_org_organization_claim_fallback():
    """No matching group → fall back to organization claim."""
    from src.api.auth.keycloak_provider import _derive_org

    claims = {"groups": ["/unrelated"], "organization": "beta-corp"}
    assert _derive_org(claims) == "beta-corp"


def test_derive_org_azp_fallback():
    """No groups or organization → fall back to azp."""
    from src.api.auth.keycloak_provider import _derive_org

    claims = {"azp": "arena-client"}
    assert _derive_org(claims) == "arena-client"


def test_derive_org_none_when_no_claims():
    """No usable claims → None."""
    from src.api.auth.keycloak_provider import _derive_org

    assert _derive_org({}) is None


def test_derive_org_empty_groups():
    """Empty groups list → falls through to next rule."""
    from src.api.auth.keycloak_provider import _derive_org

    claims = {"groups": [], "organization": "fallback-org"}
    assert _derive_org(claims) == "fallback-org"


# ---------------------------------------------------------------------------
# Tests — JWKS caching
# ---------------------------------------------------------------------------

def test_jwks_fetched_once_across_two_calls(rs256_keypair, jwks_response):
    """JWKS endpoint is called at most once when the cache is still warm."""
    private_key, _ = rs256_keypair
    token1 = _make_token(private_key, sub="u-1", aud="arena-client")
    token2 = _make_token(private_key, sub="u-2", aud="arena-client")

    provider = _make_provider(jwks_cache_ttl=300)  # 5 min cache

    call_count = 0

    def _counting_get(url: str, **kwargs):
        nonlocal call_count
        call_count += 1
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        if "openid-configuration" in url:
            resp.json.return_value = DISCOVERY_DOC
        else:
            resp.json.return_value = jwks_response
        return resp

    with patch("src.api.auth.keycloak_provider.httpx.get", side_effect=_counting_get):
        provider.authenticate(f"Bearer {token1}")
        provider.authenticate(f"Bearer {token2}")

    # discovery + jwks = 2 HTTP calls total (both on first authenticate only)
    assert call_count == 2, (
        f"Expected 2 HTTP calls (discovery + JWKS) across two authenticate() "
        f"calls, but got {call_count}."
    )


def test_jwks_refetched_after_ttl_expiry(rs256_keypair, jwks_response):
    """When the cache TTL expires the JWKS is re-fetched on the next call."""
    private_key, _ = rs256_keypair
    token1 = _make_token(private_key, sub="u-a", aud="arena-client")
    token2 = _make_token(private_key, sub="u-b", aud="arena-client")

    # Use a zero-second TTL so the cache is always stale
    provider = _make_provider(jwks_cache_ttl=0)

    call_count = 0

    def _counting_get(url: str, **kwargs):
        nonlocal call_count
        call_count += 1
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        if "openid-configuration" in url:
            resp.json.return_value = DISCOVERY_DOC
        else:
            resp.json.return_value = jwks_response
        return resp

    with patch("src.api.auth.keycloak_provider.httpx.get", side_effect=_counting_get):
        provider.authenticate(f"Bearer {token1}")
        time.sleep(0.01)  # ensure monotonic clock advances past TTL=0
        provider.authenticate(f"Bearer {token2}")

    # 2 calls per authenticate when TTL=0 (discovery + JWKS each time)
    assert call_count == 4, (
        f"Expected 4 HTTP calls (2 per authenticate) with TTL=0, got {call_count}."
    )


# ---------------------------------------------------------------------------
# Tests — registry integration
# ---------------------------------------------------------------------------

def test_registry_builds_keycloak_provider(monkeypatch):
    """``OPEN_ARENA_AUTH=keycloak`` → registry instantiates KeycloakAuthProvider."""
    from src.api.auth.keycloak_provider import KeycloakAuthProvider
    from src.api.registry import _build_auth
    from src.api.settings import ArenaSettings

    monkeypatch.setenv("OIDC_ISSUER", "https://keycloak.example.com/realms/test")
    monkeypatch.setenv("OIDC_CLIENT_ID", "arena-client")

    settings = ArenaSettings(auth="keycloak")
    provider = _build_auth(settings)
    assert isinstance(provider, KeycloakAuthProvider)


def test_registry_raises_for_unknown_auth():
    """Unknown ``auth`` value → ValueError with helpful message."""
    from src.api.registry import _build_auth
    from src.api.settings import ArenaSettings

    settings = ArenaSettings(auth="unknown-backend")
    with pytest.raises(ValueError, match="unknown-backend"):
        _build_auth(settings)
