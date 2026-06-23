# License Apache 2.0: (c) 2026 Athena-Reply
"""Port-2 adapter — PrimeEnvHubBackend (P2-3: Prime Intellect Environment Hub, issue #65).

Resolves :class:`~src.api.models.Environment` objects whose
``source.kind == EnvironmentSourceKind.prime_environment_hub`` by fetching
the artifact from the Prime Intellect Hub REST API, downloading it to a
content-addressable local cache, and returning a fully-populated
:class:`~src.api.ports.environment_backend.ResolvedEnvironment`.

This adapter handles **resolution and pinning only** — it does not execute
the environment package.  Execution is handled by the sandbox layer (P2-4,
issue #66).

Source URI convention
---------------------
The adapter derives ``owner``, ``slug``, and ``version`` from the
:class:`~src.api.models.EnvironmentSource` fields in the following priority
order:

* ``source.uri`` — parsed as ``https://hub.primeintellect.ai/{owner}/{slug}``
  (any authority component is ignored; only the path is used).
* ``source.name`` — interpreted as ``{owner}/{slug}`` when it contains a
  slash; otherwise treated as the bare ``slug`` with ``owner`` left blank.
* ``source.external_id`` — raw ``{owner}/{slug}`` string.

The resolved version is taken from ``source.version`` (default ``"latest"``).

Prime Intellect Hub REST API
-----------------------------
Base URL: ``https://hub.primeintellect.ai``

``GET /{owner}/{name}/versions``
    Returns a JSON list of available version strings (or objects with a
    ``version`` key).  Used to resolve the ``"latest"`` alias to a concrete
    immutable version id.

``GET /{owner}/{name}/@{version}``
    Returns environment metadata including a ``download_url`` field pointing
    to the pip-installable artifact (a ``verifiers`` package: ``env.py`` +
    ``pyproject.toml``).

Authentication
--------------
Bearer token read from ``PI_API_KEY`` environment variable.  A clear
:exc:`ValueError` is raised when the variable is absent.

Cache layout
------------
``~/.oa-cache/envs/prime_intellect/{owner}/{slug}/{resolved_version}/``

The artifact is stored as ``artifact.tar.gz`` (or the extension inferred
from the download URL) inside the version directory.  The content hash is
computed as the SHA-256 of the raw artifact bytes.

Pinning discipline
------------------
When ``version == "latest"``, the adapter fetches ``/versions`` and
pins to the first (most-recent) item in the list.  The pinned version id
is stored in ``ResolvedEnvironment.commit_sha`` for reproducibility.

Error handling
--------------
* Missing ``PI_API_KEY`` → :exc:`ValueError` with a clear message.
* 404 from any Hub endpoint → :exc:`LookupError` with the owner/slug/version.
* Other HTTP errors → propagated as :exc:`httpx.HTTPStatusError`.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from urllib.parse import urlparse

import httpx

from open_arena_core import models as api
from src.api.ports.environment_backend import (
    EnvironmentBackend,
    InlineEnvironmentBackend,
    ResolvedEnvironment,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HUB_BASE_URL = "https://hub.primeintellect.ai"
_DEFAULT_VERSION = "latest"
_CACHE_ROOT = Path.home() / ".oa-cache" / "envs" / "prime_intellect"


# ---------------------------------------------------------------------------
# Helpers — URI / source parsing
# ---------------------------------------------------------------------------

def _parse_owner_slug(source: api.EnvironmentSource) -> tuple[str, str]:
    """Return ``(owner, slug)`` extracted from *source*.

    Priority:
    1. ``source.uri`` path component (``/{owner}/{slug}`` after stripping the authority).
    2. ``source.external_id`` (``{owner}/{slug}``).
    3. ``source.name`` (``{owner}/{slug}`` when it contains a slash; bare slug otherwise).

    Raises:
        ValueError: When neither owner nor slug can be determined.
    """
    # 1. URI
    if source.uri is not None:
        path = urlparse(str(source.uri)).path.strip("/")
        parts = path.split("/")
        if len(parts) >= 2 and parts[0] and parts[1]:
            return parts[0], parts[1]

    # 2. external_id
    if source.external_id:
        raw = source.external_id.strip("/")
        parts = raw.split("/")
        if len(parts) >= 2 and parts[0] and parts[1]:
            return parts[0], parts[1]

    # 3. source.name
    if source.name:
        raw = source.name.strip("/")
        parts = raw.split("/")
        if len(parts) >= 2 and parts[0] and parts[1]:
            return parts[0], parts[1]
        # bare slug — owner is unknown
        return "", parts[0]

    raise ValueError(
        f"Cannot determine owner/slug from EnvironmentSource(name={source.name!r}, "
        f"uri={source.uri!r}, external_id={source.external_id!r}).  "
        "Set source.uri to 'https://hub.primeintellect.ai/{owner}/{slug}'."
    )


def _resolved_version(source: api.EnvironmentSource) -> str:
    """Return the version string from *source*, defaulting to ``'latest'``."""
    v = (source.version or "").strip()
    return v if v else _DEFAULT_VERSION


# ---------------------------------------------------------------------------
# Helpers — Hub REST client
# ---------------------------------------------------------------------------

def _require_api_key() -> str:
    """Return ``PI_API_KEY`` or raise :exc:`ValueError`."""
    key = os.getenv("PI_API_KEY", "")
    if not key:
        raise ValueError(
            "PI_API_KEY environment variable is not set.  "
            "Obtain an API key from https://hub.primeintellect.ai and export it as PI_API_KEY."
        )
    return key


def _hub_client(api_key: str) -> httpx.Client:
    """Return a configured :class:`httpx.Client` for the Prime Intellect Hub."""
    return httpx.Client(
        base_url=_HUB_BASE_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        },
        timeout=60,
        follow_redirects=True,
    )


def _resolve_latest_version(
    client: httpx.Client,
    owner: str,
    slug: str,
) -> str:
    """Resolve ``'latest'`` to a concrete version id via ``GET /{owner}/{slug}/versions``.

    Returns the first element in the versions list (assumed most-recent).

    Raises:
        LookupError: When the environment is not found (HTTP 404).
        httpx.HTTPStatusError: On other HTTP errors.
    """
    url = f"/{owner}/{slug}/versions"
    response = client.get(url)
    if response.status_code == 404:
        raise LookupError(
            f"Prime Intellect Hub environment not found: {owner}/{slug}.  "
            f"Verify the owner and slug at {_HUB_BASE_URL}."
        )
    response.raise_for_status()

    data = response.json()
    # data may be a list of strings or a list of dicts with a "version" key
    if not data:
        raise LookupError(
            f"Prime Intellect Hub returned an empty versions list for {owner}/{slug}."
        )

    first = data[0]
    if isinstance(first, str):
        return first
    if isinstance(first, dict):
        return str(first.get("version") or first.get("id") or first.get("name") or first)
    return str(first)


def _fetch_metadata(
    client: httpx.Client,
    owner: str,
    slug: str,
    version: str,
) -> dict:
    """Fetch environment metadata from ``GET /{owner}/{slug}/@{version}``.

    Returns the parsed JSON response dict.

    Raises:
        LookupError: On HTTP 404.
        httpx.HTTPStatusError: On other HTTP errors.
    """
    url = f"/{owner}/{slug}/@{version}"
    response = client.get(url)
    if response.status_code == 404:
        raise LookupError(
            f"Prime Intellect Hub environment version not found: {owner}/{slug}@{version}.  "
            "Check that the version exists at "
            f"{_HUB_BASE_URL}/{owner}/{slug}/versions."
        )
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# Helpers — cache + download
# ---------------------------------------------------------------------------

def _cache_path(owner: str, slug: str, version: str) -> Path:
    """Return the content-addressable cache directory for the given coordinates."""
    # owner may be empty for bare-slug sources
    if owner:
        return _CACHE_ROOT / owner / slug / version
    return _CACHE_ROOT / "_" / slug / version


def _infer_artifact_filename(download_url: str) -> str:
    """Derive the artifact filename from the download URL."""
    path = urlparse(download_url).path
    name = path.split("/")[-1]
    # Ensure we have a sensible fallback
    return name if name else "artifact.tar.gz"


def _download_artifact(download_url: str, dest_dir: Path, api_key: str) -> Path:
    """Download the artifact at *download_url* to *dest_dir*.

    Uses a streaming GET with the bearer token.  The downloaded file is
    placed at ``dest_dir / <filename>``.

    Returns the :class:`Path` of the downloaded file.
    """
    filename = _infer_artifact_filename(download_url)
    dest_file = dest_dir / filename

    with httpx.Client(
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=300,
        follow_redirects=True,
    ) as dl_client:
        with dl_client.stream("GET", download_url) as response:
            response.raise_for_status()
            dest_dir.mkdir(parents=True, exist_ok=True)
            with dest_file.open("wb") as fh:
                for chunk in response.iter_bytes(chunk_size=65536):
                    fh.write(chunk)

    return dest_file


def _compute_sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of the file at *path*."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Inline definition loading
# ---------------------------------------------------------------------------

def _synthesize_inline_definition(
    source: api.EnvironmentSource,
    owner: str,
    slug: str,
    version: str,
    local_path: Path,
) -> api.InlineEnvironmentDefinition:
    """Synthesise a minimal :class:`InlineEnvironmentDefinition` from Hub metadata.

    The execution layer (P2-4, issue #66) is responsible for actually running
    the package.  Here we create a placeholder that carries the provenance so
    the rest of the pipeline can proceed.
    """
    return api.InlineEnvironmentDefinition(
        name=source.name or f"{owner}/{slug}",
        version=version,
        description=(
            f"Prime Intellect Environment Hub: {owner}/{slug}@{version}.  "
            "Artifact cached at: " + str(local_path)
        ),
        dataset=api.DatasetBinding(
            provider="local",
            source_ref=str(local_path / "data"),
        ),
        verifier=api.VerifierSuiteBinding(
            root=api.VerifierSuiteInline(
                binding_type="inline",
                name=f"{slug}-verifier",
                metrics=[
                    api.MetricDefinition(
                        name="accuracy",
                        metric_kind="exact_match",
                        weight=1.0,
                    )
                ],
            )
        ),
        runtime=api.EnvironmentRuntimePolicy(),
    )


# ---------------------------------------------------------------------------
# Main adapter
# ---------------------------------------------------------------------------

class PrimeEnvHubBackend(EnvironmentBackend):
    """Adapter that resolves Prime Intellect Environment Hub sources.

    Fetches environment metadata and artifacts from the Hub REST API,
    pins mutable version aliases (``"latest"``) to concrete version ids,
    caches artifacts content-addressably under ``~/.oa-cache/envs/prime_intellect/``,
    and populates :class:`~src.api.ports.environment_backend.ResolvedEnvironment`
    with ``commit_sha`` (the pinned version id) + ``content_hash`` (SHA-256 of
    the artifact) + ``local_path``.

    This adapter also composes
    :class:`~src.api.ports.environment_backend.InlineEnvironmentBackend` so it
    is a safe drop-in superset of the default adapter.

    Args:
        cache_root: Override the cache root directory.  Defaults to
            ``~/.oa-cache/envs/prime_intellect/``.
        hub_base_url: Override the Hub base URL.  Defaults to
            ``https://hub.primeintellect.ai``.
    """

    def __init__(
        self,
        cache_root: Path | None = None,
        hub_base_url: str | None = None,
    ) -> None:
        self._inline = InlineEnvironmentBackend()
        self._cache_root = cache_root if cache_root is not None else _CACHE_ROOT
        self._hub_base_url = (hub_base_url or _HUB_BASE_URL).rstrip("/")

    # ------------------------------------------------------------------
    # EnvironmentBackend protocol
    # ------------------------------------------------------------------

    def resolve(self, environment: api.Environment) -> ResolvedEnvironment:  # noqa: D102
        kind = environment.source.kind

        if kind == api.EnvironmentSourceKind.inline:
            return self._inline.resolve(environment)

        if kind == api.EnvironmentSourceKind.prime_environment_hub:
            return self._resolve_hub(environment)

        raise NotImplementedError(
            f"PrimeEnvHubBackend does not support source.kind={kind!r}.  "
            "Supported kinds: 'inline', 'prime_environment_hub'."
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_hub(self, environment: api.Environment) -> ResolvedEnvironment:
        source = environment.source
        api_key = _require_api_key()

        owner, slug = _parse_owner_slug(source)
        requested_version = _resolved_version(source)

        with httpx.Client(
            base_url=self._hub_base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
            },
            timeout=60,
            follow_redirects=True,
        ) as client:
            # Pin mutable "latest" alias to a concrete version id
            if requested_version == _DEFAULT_VERSION:
                pinned_version = _resolve_latest_version(client, owner, slug)
            else:
                pinned_version = requested_version

            # Fetch metadata + download URL
            metadata = _fetch_metadata(client, owner, slug, pinned_version)

        # Determine cache path
        cache_dir = self._cache_path(owner, slug, pinned_version)

        # Determine artifact filename/path
        download_url: str = metadata.get("download_url") or metadata.get("artifact_url") or ""
        if download_url:
            artifact_filename = _infer_artifact_filename(download_url)
        else:
            artifact_filename = "artifact.tar.gz"
        artifact_path = cache_dir / artifact_filename

        # Download artifact if not already cached
        if not artifact_path.exists():
            if not download_url:
                raise LookupError(
                    f"Prime Intellect Hub metadata for {owner}/{slug}@{pinned_version} "
                    "does not include a 'download_url'.  Cannot cache the artifact."
                )
            self._download_artifact(download_url, cache_dir, api_key)

        # Compute content hash
        content_hash = _compute_sha256(artifact_path)

        # Synthesise inline definition (P2-4 will replace this with real execution)
        definition = _synthesize_inline_definition(
            source=source,
            owner=owner,
            slug=slug,
            version=pinned_version,
            local_path=cache_dir,
        )

        return ResolvedEnvironment(
            definition=definition,
            commit_sha=pinned_version,  # version id acts as the reproducible pin
            content_hash=content_hash,
            local_path=str(cache_dir),
        )

    def _cache_path(self, owner: str, slug: str, version: str) -> Path:
        """Return the cache directory, honouring the instance's ``_cache_root``."""
        if owner:
            return self._cache_root / owner / slug / version
        return self._cache_root / "_" / slug / version

    @staticmethod
    def _download_artifact(download_url: str, dest_dir: Path, api_key: str) -> None:
        """Download the artifact (thin wrapper; extracted for easier mocking in tests)."""
        _download_artifact(download_url, dest_dir, api_key)
