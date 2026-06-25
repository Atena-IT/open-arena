# License Apache 2.0: (c) 2026 Athena-Reply
"""Port-2 adapter — GitEnvironmentBackend (WS2: Gitea/GitHub, issue #36).

Resolves git-backed :class:`~src.api.models.Environment` objects by
shallow-cloning the source repository at the requested ``git_ref`` and
returning a fully-populated :class:`~src.api.ports.environment_backend.ResolvedEnvironment`.

Supported ``source.kind`` values
---------------------------------
* ``github_repo`` — clones ``source.uri`` directly; injects ``GITHUB_TOKEN``
  as OAuth2 credentials when the env-var is set.
* Gitea repos addressed via ``github_repo`` kind — same code-path, but the
  URL is built from ``GITEA_BASE_URL`` + ``GITEA_ORG`` + the repo name
  extracted from ``source.uri`` when those env-vars are set.  Auth uses
  ``GITEA_TOKEN``.
* ``inline`` — delegated to :class:`InlineEnvironmentBackend` so this
  adapter is a safe superset of the default.

Environment variables
---------------------
``GITEA_BASE_URL``
    Base URL of the Gitea instance, e.g. ``https://gitea.example.com``.
    When set, any ``github_repo`` source whose URI matches this host (or
    whose ``external_id`` starts with ``gitea:``) is treated as a Gitea
    repo and authenticated with ``GITEA_TOKEN``.

``GITEA_ORG``
    Default organisation / namespace on Gitea.  Used when the URI path
    contains only a repo name (no org prefix).

``GITEA_TOKEN``
    Personal-access or OAuth2 token for Gitea authentication.

``GITHUB_TOKEN``
    Optional bearer token for GitHub API / rate-limit avoidance.

``OPEN_ARENA_ENV_CACHE_DIR``
    Directory used for the shallow-clone cache.  Defaults to
    ``.open-arena/env-cache/`` relative to the current working directory.

Clone-URL construction
----------------------
Gitea (detected by ``GITEA_BASE_URL`` matching source host, or by the
``gitea:`` prefix in ``source.external_id``)::

    https://oauth2:<GITEA_TOKEN>@<host>/<org>/<repo>.git

GitHub (all other ``github_repo`` sources)::

    https://oauth2:<GITHUB_TOKEN>@github.com/<owner>/<repo>.git   # with token
    https://github.com/<owner>/<repo>.git                         # without token

Content-hash
------------
The ``content_hash`` is computed as the SHA-256 of the sorted file-path /
SHA-blob pairs from ``git ls-tree -r HEAD``.  This is stable across
identical tree contents regardless of how the repo was cloned and is
cheaper than hashing every file byte-by-byte.

snapshot_inline
---------------
:meth:`GitEnvironmentBackend.snapshot_inline` promotes an inline environment
definition into a new Gitea repo so that every published leaderboard entry
is pinned to a reproducible commit.  It:

1. Creates a new repo via the Gitea API (``POST /api/v1/orgs/<org>/repos``).
2. Generates a minimal ``env.py`` + ``pyproject.toml`` from the inline
   definition.
3. Commits both files via the Gitea Contents API
   (``POST /api/v1/repos/<org>/<repo>/contents/<path>``).
4. Returns ``(repo_url, commit_sha)`` so the caller can update the
   ``EnvironmentSource`` to point at the pinned commit.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from open_arena_core import models as api
from src.api.ports.environment_backend import (
    EnvironmentBackend,
    InlineEnvironmentBackend,
    ResolvedEnvironment,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_CACHE_DIR = Path(".open-arena") / "env-cache"


def _cache_dir() -> Path:
    raw = os.getenv("OPEN_ARENA_ENV_CACHE_DIR")
    return Path(raw) if raw else _DEFAULT_CACHE_DIR


def _is_gitea_source(source: api.EnvironmentSource) -> bool:
    """Return ``True`` when *source* should be treated as a Gitea repo."""
    # Explicit marker via external_id
    if source.external_id and source.external_id.startswith("gitea:"):
        return True
    # Detect by comparing URI host against GITEA_BASE_URL
    gitea_base = os.getenv("GITEA_BASE_URL", "").rstrip("/")
    if gitea_base and source.uri:
        uri_str = str(source.uri)
        parsed_gitea = urlparse(gitea_base)
        parsed_source = urlparse(uri_str)
        if parsed_gitea.hostname and parsed_source.hostname:
            return parsed_gitea.hostname == parsed_source.hostname
    return False


def _build_clone_url(source: api.EnvironmentSource) -> str:
    """Return the authenticated clone URL for *source*.

    Raises:
        ValueError: When the clone URL cannot be constructed (e.g. missing
            ``uri`` on a ``github_repo`` source).
    """
    if _is_gitea_source(source):
        return _build_gitea_clone_url(source)
    return _build_github_clone_url(source)


def _build_gitea_clone_url(source: api.EnvironmentSource) -> str:
    """Build an authenticated Gitea clone URL."""
    gitea_base = os.getenv("GITEA_BASE_URL", "").rstrip("/")
    gitea_org = os.getenv("GITEA_ORG", "")
    gitea_token = os.getenv("GITEA_TOKEN", "")

    if not gitea_base:
        raise ValueError(
            "GITEA_BASE_URL must be set to clone a Gitea environment source."
        )

    # Derive repo name from URI or source name
    repo_name = _extract_repo_name(source)

    parsed = urlparse(gitea_base)
    host = parsed.netloc  # host[:port]

    # Build path: /org/repo.git
    if gitea_org:
        path = f"{gitea_org}/{repo_name}"
    else:
        path = repo_name

    if gitea_token:
        return f"{parsed.scheme}://oauth2:{gitea_token}@{host}/{path}.git"
    return f"{parsed.scheme}://{host}/{path}.git"


def _build_github_clone_url(source: api.EnvironmentSource) -> str:
    """Build an authenticated (or anonymous) GitHub clone URL."""
    if source.uri is None:
        raise ValueError(
            f"EnvironmentSource for kind={source.kind!r} has no URI; "
            "cannot construct a clone URL."
        )
    uri_str = str(source.uri)
    # Normalise to end with .git
    if not uri_str.endswith(".git"):
        uri_str = uri_str.rstrip("/") + ".git"

    github_token = os.getenv("GITHUB_TOKEN", "")
    if github_token:
        parsed = urlparse(uri_str)
        return f"{parsed.scheme}://oauth2:{github_token}@{parsed.netloc}{parsed.path}"
    return uri_str


def _extract_repo_name(source: api.EnvironmentSource) -> str:
    """Extract the bare repo name (without org/owner) from *source*."""
    # Try external_id first: "gitea:my-repo" → "my-repo"
    if source.external_id:
        raw = source.external_id
        if ":" in raw:
            raw = raw.split(":", 1)[1]
        return raw.rstrip(".git")

    # Try URI path
    if source.uri:
        path = urlparse(str(source.uri)).path.rstrip("/")
        name = path.split("/")[-1]
        return name.removesuffix(".git")

    # Fall back to the canonical source name
    return source.name


def _shallow_clone(
    clone_url: str,
    git_ref: str,
    dest: Path,
) -> None:
    """Shallow-clone *clone_url* at *git_ref* into *dest*.

    Uses ``git clone --depth 1 --branch <ref>`` for lightweight tags/branches.
    Falls back to a two-step fetch for detached SHAs.

    The directory *dest* must not already exist.
    """
    try:
        subprocess.run(
            [
                "git", "clone",
                "--depth", "1",
                "--branch", git_ref,
                "--",
                clone_url,
                str(dest),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        # git_ref might be a commit SHA — try the two-step approach
        dest.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "init", str(dest)],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "-C", str(dest), "remote", "add", "origin", clone_url],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "-C", str(dest), "fetch", "--depth", "1", "origin", git_ref],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "-C", str(dest), "checkout", "FETCH_HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )


def _resolve_commit_sha(repo_path: Path) -> str:
    """Return the resolved commit SHA (``git rev-parse HEAD``)."""
    result = subprocess.run(
        ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _compute_content_hash(repo_path: Path) -> str:
    """Compute a stable content hash for the checked-out tree.

    Uses ``git ls-tree -r HEAD`` to list all blobs with their SHA-1 object
    IDs, sorts the entries, then takes SHA-256 of the concatenated string.
    This is deterministic for identical tree contents regardless of clone
    details.
    """
    result = subprocess.run(
        ["git", "-C", str(repo_path), "ls-tree", "-r", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = sorted(result.stdout.splitlines())
    digest = hashlib.sha256("\n".join(lines).encode()).hexdigest()
    return digest


def _load_inline_definition_from_repo(
    repo_path: Path,
    source: api.EnvironmentSource,
) -> api.InlineEnvironmentDefinition:
    """Load the ``InlineEnvironmentDefinition`` from the cloned repo.

    Expects one of:
    * ``environment.json`` — a JSON-serialised ``InlineEnvironmentDefinition``
    * ``env.py`` + ``pyproject.toml`` — a Prime-Intellect *verifiers* package
      that exposes ``load_environment()`` returning a dict compatible with
      ``InlineEnvironmentDefinition``.

    If neither is present, a minimal placeholder definition is synthesised
    from the source metadata so the rest of the pipeline can proceed.
    """
    # Preferred: explicit JSON manifest
    manifest_path = repo_path / "environment.json"
    if manifest_path.exists():
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        return api.InlineEnvironmentDefinition(**data)

    # Prime-Intellect verifiers package convention
    env_py = repo_path / "env.py"
    pyproject = repo_path / "pyproject.toml"
    # SECURITY: importing env.py runs arbitrary code from the resolved repo
    # *in the API process*. Until execution is confined to a per-task sandbox
    # (P2-4), this best-effort loader is OPT-IN and OFF by default; the safe
    # placeholder below is synthesised instead. Enable only for trusted
    # sources via OPEN_ARENA_ALLOW_INPROCESS_ENV_EXEC=1.
    _allow_exec = os.getenv(
        "OPEN_ARENA_ALLOW_INPROCESS_ENV_EXEC", ""
    ).strip().lower() in ("1", "true", "yes")
    if env_py.exists() and pyproject.exists() and _allow_exec:
        # Attempt to import load_environment() dynamically.
        # This is an optional best-effort path; errors are swallowed and we
        # fall through to the placeholder.
        try:
            import importlib.util
            import sys

            spec = importlib.util.spec_from_file_location(
                "_arena_env_loader", env_py
            )
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                sys.modules["_arena_env_loader"] = mod
                spec.loader.exec_module(mod)  # type: ignore[union-attr]
                if hasattr(mod, "load_environment"):
                    env_data: dict[str, Any] = mod.load_environment()
                    return api.InlineEnvironmentDefinition(**env_data)
        except Exception:  # noqa: BLE001
            pass

    # Fallback: synthesise a placeholder from source metadata
    from pydantic import AnyUrl

    return api.InlineEnvironmentDefinition(
        name=source.name,
        version=source.version,
        description=f"Loaded from git source: {source.uri}",
        dataset=api.DatasetBinding(
            provider="local",
            source_ref=str(repo_path / "data"),
        ),
        verifier=api.VerifierSuiteBinding(
            root=api.VerifierSuiteInline(
                binding_type="inline",
                name=f"{source.name}-verifier",
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
# Gitea REST helpers  (used by snapshot_inline)
# ---------------------------------------------------------------------------

def _gitea_client(base_url: str, token: str) -> httpx.Client:
    return httpx.Client(
        base_url=base_url.rstrip("/"),
        headers={
            "Authorization": f"token {token}",
            "Content-Type": "application/json",
        },
        timeout=30,
    )


def _gitea_create_repo(
    client: httpx.Client,
    org: str,
    repo_name: str,
    description: str = "",
    private: bool = True,
    auto_init: bool = True,
) -> dict[str, Any]:
    """Create a new repo under *org* via the Gitea API."""
    response = client.post(
        f"/api/v1/orgs/{org}/repos",
        json={
            "name": repo_name,
            "description": description,
            "private": private,
            "auto_init": auto_init,
            "default_branch": "main",
        },
    )
    response.raise_for_status()
    return response.json()


def _gitea_put_file(
    client: httpx.Client,
    org: str,
    repo: str,
    path: str,
    content: str,
    message: str,
    branch: str = "main",
) -> dict[str, Any]:
    """Create or update a file in a Gitea repo via the Contents API."""
    import base64

    encoded = base64.b64encode(content.encode()).decode()
    # Check if file exists to decide between create and update
    existing = client.get(f"/api/v1/repos/{org}/{repo}/contents/{path}")
    payload: dict[str, Any] = {
        "message": message,
        "content": encoded,
        "branch": branch,
    }
    if existing.status_code == 200:
        payload["sha"] = existing.json()["sha"]
        response = client.put(
            f"/api/v1/repos/{org}/{repo}/contents/{path}", json=payload
        )
    else:
        response = client.post(
            f"/api/v1/repos/{org}/{repo}/contents/{path}", json=payload
        )
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# Scaffold helpers for snapshot_inline
# ---------------------------------------------------------------------------

def _render_env_py(definition: api.InlineEnvironmentDefinition) -> str:
    """Generate a minimal ``env.py`` for a Prime-Intellect verifiers package."""
    definition_json = definition.model_dump_json(indent=4)
    return f"""\
# Auto-generated by Open Arena snapshot_inline
# Do not edit manually — regenerate via the Arena API.
import json

_DEFINITION = {definition_json!r}

def load_environment():
    \"\"\"Return the InlineEnvironmentDefinition as a plain dict.\"\"\"
    return json.loads(_DEFINITION)
"""


def _render_pyproject_toml(definition: api.InlineEnvironmentDefinition) -> str:
    """Generate a minimal ``pyproject.toml`` for the snapshotted environment."""
    return f"""\
[project]
name = "{definition.name}"
version = "{definition.version}"
description = "Open Arena environment snapshot — {definition.name} v{definition.version}"
requires-python = ">=3.12"

[project.entry-points."open_arena.environments"]
environment = "env:load_environment"
"""


# ---------------------------------------------------------------------------
# Main adapter
# ---------------------------------------------------------------------------


class GitEnvironmentBackend(EnvironmentBackend):
    """Adapter that resolves git-backed environment sources by cloning them.

    Composes :class:`~src.api.ports.environment_backend.InlineEnvironmentBackend`
    for ``kind=inline`` so this class is a safe drop-in superset of the
    default adapter.

    Args:
        cache_dir: Override the cache directory.  Defaults to
            ``OPEN_ARENA_ENV_CACHE_DIR`` or ``.open-arena/env-cache/``.
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._inline = InlineEnvironmentBackend()
        self._cache_dir = cache_dir if cache_dir is not None else _cache_dir()

    # ------------------------------------------------------------------
    # EnvironmentBackend protocol
    # ------------------------------------------------------------------

    def resolve(self, environment: api.Environment) -> ResolvedEnvironment:  # noqa: D102
        kind = environment.source.kind

        if kind == api.EnvironmentSourceKind.inline:
            return self._inline.resolve(environment)

        if kind in (
            api.EnvironmentSourceKind.github_repo,
            api.EnvironmentSourceKind.prime_environment_hub,
        ):
            return self._resolve_git(environment)

        raise NotImplementedError(
            f"GitEnvironmentBackend does not support source.kind={kind!r}. "
            "Supported kinds: 'inline', 'github_repo', 'prime_environment_hub'."
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_git(self, environment: api.Environment) -> ResolvedEnvironment:
        source = environment.source
        git_ref = source.git_ref or "main"
        clone_url = _build_clone_url(source)

        # Deterministic cache key: <source-name>__<version>__<git_ref>
        safe_name = source.name.replace("/", "_").replace(":", "_")
        cache_key = f"{safe_name}__{source.version}__{git_ref}"
        dest = self._cache_dir / cache_key

        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp_dest = dest.parent / f"{cache_key}.tmp"
            try:
                _shallow_clone(clone_url, git_ref, tmp_dest)
                # _shallow_clone (or the git binary) creates tmp_dest.
                # Ensure it exists in case the implementation was mocked.
                tmp_dest.mkdir(parents=True, exist_ok=True)
                if dest.exists():
                    # Another process may have populated dest concurrently.
                    shutil.rmtree(tmp_dest, ignore_errors=True)
                else:
                    tmp_dest.rename(dest)
            except Exception:
                shutil.rmtree(tmp_dest, ignore_errors=True)
                raise

        commit_sha = _resolve_commit_sha(dest)
        content_hash = _compute_content_hash(dest)
        definition = _load_inline_definition_from_repo(dest, source)

        return ResolvedEnvironment(
            definition=definition,
            commit_sha=commit_sha,
            content_hash=content_hash,
            local_path=str(dest),
        )

    # ------------------------------------------------------------------
    # snapshot_inline
    # ------------------------------------------------------------------

    def snapshot_inline(
        self,
        definition: api.InlineEnvironmentDefinition,
        repo_name: str | None = None,
        *,
        gitea_base_url: str | None = None,
        gitea_org: str | None = None,
        gitea_token: str | None = None,
        branch: str = "main",
        private: bool = True,
    ) -> tuple[str, str]:
        """Promote an inline environment definition into a new Gitea repo.

        Creates a new repo, commits ``env.py`` + ``pyproject.toml``, and
        returns ``(repo_url, commit_sha)``.

        Args:
            definition: The inline environment definition to snapshot.
            repo_name: Override the generated repo name.  Defaults to
                ``arena-env-<name>-<version>``.
            gitea_base_url: Override ``GITEA_BASE_URL``.
            gitea_org: Override ``GITEA_ORG``.
            gitea_token: Override ``GITEA_TOKEN``.
            branch: Target branch (default ``"main"``).
            private: Whether the created repo should be private.

        Returns:
            ``(repo_url, commit_sha)`` — the HTTPS URL of the new repo and
            the SHA of the commit that added the generated files.

        Raises:
            ValueError: When required Gitea credentials are missing.
            httpx.HTTPStatusError: On API failures.
        """
        base_url = gitea_base_url or os.getenv("GITEA_BASE_URL", "")
        org = gitea_org or os.getenv("GITEA_ORG", "")
        token = gitea_token or os.getenv("GITEA_TOKEN", "")

        if not base_url:
            raise ValueError(
                "GITEA_BASE_URL (or gitea_base_url) is required for snapshot_inline."
            )
        if not org:
            raise ValueError(
                "GITEA_ORG (or gitea_org) is required for snapshot_inline."
            )
        if not token:
            raise ValueError(
                "GITEA_TOKEN (or gitea_token) is required for snapshot_inline."
            )

        safe_ver = definition.version.replace(".", "-")
        effective_repo_name = repo_name or f"arena-env-{definition.name}-{safe_ver}"

        env_py_content = _render_env_py(definition)
        pyproject_content = _render_pyproject_toml(definition)

        with _gitea_client(base_url, token) as client:
            repo_data = _gitea_create_repo(
                client,
                org=org,
                repo_name=effective_repo_name,
                description=(
                    f"Open Arena environment snapshot: "
                    f"{definition.name} v{definition.version}"
                ),
                private=private,
                auto_init=True,
            )
            repo_url: str = repo_data["clone_url"]

            _gitea_put_file(
                client,
                org=org,
                repo=effective_repo_name,
                path="env.py",
                content=env_py_content,
                message=f"feat: add env.py for {definition.name} v{definition.version}",
                branch=branch,
            )
            pyproject_result = _gitea_put_file(
                client,
                org=org,
                repo=effective_repo_name,
                path="pyproject.toml",
                content=pyproject_content,
                message=(
                    f"feat: add pyproject.toml for "
                    f"{definition.name} v{definition.version}"
                ),
                branch=branch,
            )

        # The commit SHA of the *last* write is the tree HEAD — pinning to it
        # yields a tree containing both env.py and pyproject.toml.
        commit_sha: str = pyproject_result["commit"]["sha"]
        return repo_url, commit_sha
