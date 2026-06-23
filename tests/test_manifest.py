# License Apache 2.0: (c) 2026 Athena-Reply
"""Tests for the EvalEnvironment manifest loader (P2-1, issue #63).

Covers:
* load_manifest from a FILE path
* load_manifest from a DIRECTORY (auto-detects eval.yaml)
* eval_env_ref -> stub EnvironmentInlineMembership with metadata
* SandboxPolicy fields (image, per_task_sandbox) round-trip through manifest
* non-inline dataset.source -> _manifest_source_kind metadata annotation
* FileNotFoundError on missing manifest
* ValueError on wrong apiVersion/kind
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from open_arena_core import models as api


# ---------------------------------------------------------------------------
# Fixtures: write minimal eval.yaml files
# ---------------------------------------------------------------------------

_MINIMAL_MANIFEST = textwrap.dedent("""\
    apiVersion: open-arena/v1
    kind: EvalEnvironment
    metadata:
      name: my-eval
      version: 1.2.3
    dataset:
      source: inline
      ref: my-tasks
      path: tasks/
    verifier:
      type: exact_match
      entry: tests/run.sh
      timeout_sec: 120
      pass_threshold: 0.9
    agent:
      timeout_sec: 600
""")

_SANDBOX_MANIFEST = textwrap.dedent("""\
    apiVersion: open-arena/v1
    kind: EvalEnvironment
    metadata:
      name: sandbox-eval
      version: 0.2.0
    environment:
      image: ghcr.io/org/eval-env@sha256:deadbeef
      per_task_sandbox: true
      allow_internet: false
      resources:
        cpu: "2"
        memory_gb: 4
    dataset:
      source: inline
    verifier:
      type: script
""")

_GIT_MANIFEST = textwrap.dedent("""\
    apiVersion: open-arena/v1
    kind: EvalEnvironment
    metadata:
      name: git-eval
      version: 2.0.0
    dataset:
      source: gitea
      ref: "org/my-repo@abc123"
      path: data/
    verifier:
      type: script
      entry: run.sh
""")

_EVAL_ENV_REF_MANIFEST = textwrap.dedent("""\
    apiVersion: open-arena/v1
    kind: EvalEnvironment
    metadata:
      name: ref-eval
      version: 1.0.0
    eval_env_ref: "acme/my-benchmark@2.5.0"
    dataset:
      source: inline
    verifier:
      type: exact_match
""")

_EVAL_ENV_REF_NO_OWNER_MANIFEST = textwrap.dedent("""\
    apiVersion: open-arena/v1
    kind: EvalEnvironment
    metadata:
      name: ref-eval2
      version: 1.0.0
    eval_env_ref: "my-benchmark@3.0.0"
    dataset:
      source: inline
    verifier:
      type: exact_match
""")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_manifest(tmp_path: Path, content: str, filename: str = "eval.yaml") -> Path:
    p = tmp_path / filename
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Tests: load from FILE
# ---------------------------------------------------------------------------

def test_load_manifest_from_file(tmp_path):
    """load_manifest accepts an explicit file path."""
    p = _write_manifest(tmp_path, _MINIMAL_MANIFEST)
    from src.api.manifest import load_manifest
    result = load_manifest(p)
    assert isinstance(result, api.GeneratorRunCreate)
    assert result.mode == "generator"
    assert result.labels == {"manifest_name": "my-eval", "manifest_version": "1.2.3"}


def test_load_manifest_from_directory(tmp_path):
    """load_manifest accepts a directory and auto-detects eval.yaml inside."""
    _write_manifest(tmp_path, _MINIMAL_MANIFEST)
    from src.api.manifest import load_manifest
    result = load_manifest(tmp_path)
    assert isinstance(result, api.GeneratorRunCreate)
    assert result.labels["manifest_name"] == "my-eval"


def test_load_manifest_inline_env_structure(tmp_path):
    """Minimal manifest produces an EnvironmentInlineMembership with correct fields."""
    p = _write_manifest(tmp_path, _MINIMAL_MANIFEST)
    from src.api.manifest import load_manifest
    result = load_manifest(p)
    pair = result.selection.root.direct_pairs[0]
    env = pair.environment
    assert isinstance(env, api.EnvironmentInlineMembership)
    defn = env.inline_definition
    assert defn.name == "my-eval"
    assert defn.version == "1.2.3"
    # Dataset mapping
    assert defn.dataset.provider == "local"
    assert defn.dataset.source_ref == "my-tasks"
    assert defn.dataset.selector == {"path": "tasks/"}
    # Runtime
    assert defn.runtime.metadata["agent_timeout_sec"] == 600


def test_load_manifest_inline_env_verifier(tmp_path):
    """Verifier section maps to an inline VerifierSuiteBinding."""
    p = _write_manifest(tmp_path, _MINIMAL_MANIFEST)
    from src.api.manifest import load_manifest
    result = load_manifest(p)
    pair = result.selection.root.direct_pairs[0]
    defn = pair.environment.inline_definition
    suite = defn.verifier.root
    assert isinstance(suite, api.VerifierSuiteInline)
    assert suite.metrics[0].metric_kind == "exact_match"
    assert suite.metrics[0].config["entry"] == "tests/run.sh"
    assert suite.metrics[0].config["timeout_sec"] == 120
    assert suite.metrics[0].config["pass_threshold"] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Tests: SandboxPolicy round-trip
# ---------------------------------------------------------------------------

def test_sandbox_policy_fields_in_manifest(tmp_path):
    """SandboxPolicy image + per_task_sandbox fields are populated from manifest."""
    p = _write_manifest(tmp_path, _SANDBOX_MANIFEST)
    from src.api.manifest import load_manifest
    result = load_manifest(p)
    pair = result.selection.root.direct_pairs[0]
    defn = pair.environment.inline_definition
    sandbox = defn.sandbox
    assert sandbox is not None
    assert sandbox.image == "ghcr.io/org/eval-env@sha256:deadbeef"
    assert sandbox.per_task_sandbox is True
    assert sandbox.enabled is True
    assert sandbox.limits is not None
    assert sandbox.limits["allow_internet"] is False
    assert sandbox.limits["cpu"] == "2"
    assert sandbox.limits["memory_gb"] == 4


def test_sandbox_policy_round_trip_serialization(tmp_path):
    """SandboxPolicy new fields survive model_dump/model_validate round-trip."""
    p = _write_manifest(tmp_path, _SANDBOX_MANIFEST)
    from src.api.manifest import load_manifest
    result = load_manifest(p)
    data = result.model_dump(mode="json")
    # Reconstruct
    result2 = api.GeneratorRunCreate.model_validate(data)
    pair = result2.selection.root.direct_pairs[0]
    sandbox = pair.environment.inline_definition.sandbox
    assert sandbox.image == "ghcr.io/org/eval-env@sha256:deadbeef"
    assert sandbox.per_task_sandbox is True


# ---------------------------------------------------------------------------
# Tests: non-inline dataset source
# ---------------------------------------------------------------------------

def test_git_source_annotates_metadata(tmp_path):
    """Git-source dataset annotates _manifest_source_kind in DatasetBinding.metadata."""
    p = _write_manifest(tmp_path, _GIT_MANIFEST)
    from src.api.manifest import load_manifest
    result = load_manifest(p)
    pair = result.selection.root.direct_pairs[0]
    defn = pair.environment.inline_definition
    assert defn.dataset.metadata is not None
    assert defn.dataset.metadata["_manifest_source_kind"] == "github_repo"
    # "org/my-repo@abc123" -> git_ref="abc123", uri_part="org/my-repo" (not a full URL,
    # so _manifest_uri is NOT stored; only _manifest_git_ref is.)
    assert defn.dataset.metadata["_manifest_git_ref"] == "abc123"


# ---------------------------------------------------------------------------
# Tests: eval_env_ref
# ---------------------------------------------------------------------------

def test_eval_env_ref_with_owner(tmp_path):
    """eval_env_ref: 'owner/name@version' creates stub with correct name+version."""
    p = _write_manifest(tmp_path, _EVAL_ENV_REF_MANIFEST)
    from src.api.manifest import load_manifest
    result = load_manifest(p)
    pair = result.selection.root.direct_pairs[0]
    env = pair.environment
    assert isinstance(env, api.EnvironmentInlineMembership)
    defn = env.inline_definition
    assert defn.name == "acme/my-benchmark"
    assert defn.version == "2.5.0"
    assert defn.metadata is not None
    assert defn.metadata["_eval_env_ref"] == "acme/my-benchmark@2.5.0"


def test_eval_env_ref_without_owner(tmp_path):
    """eval_env_ref: 'name@version' (no owner) produces correct name."""
    p = _write_manifest(tmp_path, _EVAL_ENV_REF_NO_OWNER_MANIFEST)
    from src.api.manifest import load_manifest
    result = load_manifest(p)
    pair = result.selection.root.direct_pairs[0]
    defn = pair.environment.inline_definition
    assert defn.name == "my-benchmark"
    assert defn.version == "3.0.0"


# ---------------------------------------------------------------------------
# Tests: error cases
# ---------------------------------------------------------------------------

def test_load_manifest_file_not_found(tmp_path):
    from src.api.manifest import load_manifest
    with pytest.raises(FileNotFoundError):
        load_manifest(tmp_path / "nonexistent.yaml")


def test_load_manifest_wrong_api_version(tmp_path):
    bad = _MINIMAL_MANIFEST.replace("apiVersion: open-arena/v1", "apiVersion: open-arena/v99")
    p = _write_manifest(tmp_path, bad)
    from src.api.manifest import load_manifest
    with pytest.raises(ValueError, match="apiVersion"):
        load_manifest(p)


def test_load_manifest_wrong_kind(tmp_path):
    bad = _MINIMAL_MANIFEST.replace("kind: EvalEnvironment", "kind: SomethingElse")
    p = _write_manifest(tmp_path, bad)
    from src.api.manifest import load_manifest
    with pytest.raises(ValueError, match="kind"):
        load_manifest(p)


def test_load_manifest_dir_missing_eval_yaml(tmp_path):
    """Directory without eval.yaml raises FileNotFoundError."""
    from src.api.manifest import load_manifest
    empty_dir = tmp_path / "no-yaml"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        load_manifest(empty_dir)


# ---------------------------------------------------------------------------
# Tests: SandboxPolicy standalone (new fields round-trip)
# ---------------------------------------------------------------------------

def test_sandbox_policy_new_fields_pydantic_round_trip():
    """SandboxPolicy image + per_task_sandbox survive Pydantic round-trip."""
    sp = api.SandboxPolicy(
        image="ghcr.io/org/env@sha256:abc123",
        per_task_sandbox=True,
        enabled=True,
    )
    data = sp.model_dump()
    sp2 = api.SandboxPolicy.model_validate(data)
    assert sp2.image == "ghcr.io/org/env@sha256:abc123"
    assert sp2.per_task_sandbox is True
    assert sp2.enabled is True


def test_sandbox_policy_defaults_are_none_and_false():
    """SandboxPolicy new fields default to None and False."""
    sp = api.SandboxPolicy()
    assert sp.image is None
    assert sp.per_task_sandbox is False