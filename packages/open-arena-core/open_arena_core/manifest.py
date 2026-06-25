# License Apache 2.0: (c) 2026 Athena-Reply
"""``open_arena_core.manifest`` -- EvalEnvironment manifest loader.

Parses an ``eval.yaml`` (``apiVersion: open-arena/v1``, ``kind:
EvalEnvironment``) from a file path or a directory containing one, and
returns a :class:`~open_arena_core.models.GeneratorRunCreate` payload ready
for ``POST /v1/runs`` / :meth:`~src.api.service.ArenaAPIService.create_run`.

Manifest schema (YAML)
-----------------------
::

    apiVersion: open-arena/v1
    kind: EvalEnvironment
    metadata:
      name: my-eval
      version: 1.2.3
    environment:
      image: ghcr.io/org/eval-env@sha256:...
      build_context: ./environment
      allow_internet: false
      resources: {cpu: "2", memory_gb: 4}
      per_task_sandbox: true
    dataset:
      source: gitea|prime_intellect|huggingface|unity_catalog|inline
      ref: "org/repo@sha"
      path: tasks/
    verifier:
      type: script|llm_judge|oracle
      entry: tests/run.sh
      timeout_sec: 300
      pass_threshold: 1.0
    agent:
      timeout_sec: 3600

Alternatively, carry ``eval_env_ref: "owner/name@version"`` (or
``"name@version"``) at the top level to reference a registered environment by
name+version instead of embedding a full inline definition.

Mapping to RunCreate
---------------------
The manifest is translated to a :class:`GeneratorRunCreate` with a single
:class:`DirectRunPair`:

* ``environment.*`` maps to :class:`SandboxPolicy` (``image``, ``per_task_sandbox``)
* ``dataset.*``     maps to :class:`DatasetBinding`
* ``verifier.*``    maps to inline :class:`VerifierSuiteInline`
* A placeholder :class:`ModelDefinitionCreate` is included (caller may replace).

Public API
----------
::

    from open_arena_core.manifest import load_manifest
    run_create = load_manifest("path/to/eval.yaml")
    run_create = load_manifest("path/to/my-eval/")  # dir -> eval.yaml inside

Raises :exc:`FileNotFoundError` when the manifest does not exist,
:exc:`ValueError` on schema errors.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from open_arena_core import models as api

_MANIFEST_FILENAME = "eval.yaml"
_API_VERSION = "open-arena/v1"
_KIND = "EvalEnvironment"


def _dataset_from_manifest(dataset: dict[str, Any]) -> api.DatasetBinding:
    source = dataset.get("source", "local")
    ref = dataset.get("ref")
    path = dataset.get("path")
    _PROVIDER_MAP = {
        "gitea": "local",
        "prime_intellect": "local",
        "huggingface": "huggingface",
        "huggingface_hub": "huggingface",
        "unity_catalog": "unity_catalog",
        "inline": "local",
        "local": "local",
        "folder": "folder",
    }
    provider = _PROVIDER_MAP.get(source, source)
    selector: dict[str, Any] = {}
    if path:
        selector["path"] = path
    return api.DatasetBinding(
        provider=provider,
        source_ref=ref,
        selector=selector or None,
    )


def _source_kind_from_dataset(dataset: dict[str, Any]) -> api.EnvironmentSourceKind:
    source = dataset.get("source", "inline")
    _KIND_MAP = {
        "gitea": api.EnvironmentSourceKind.github_repo,
        "github": api.EnvironmentSourceKind.github_repo,
        "github_repo": api.EnvironmentSourceKind.github_repo,
        "prime_intellect": api.EnvironmentSourceKind.prime_environment_hub,
        "prime_environment_hub": api.EnvironmentSourceKind.prime_environment_hub,
        "huggingface": api.EnvironmentSourceKind.huggingface_hub,
        "huggingface_hub": api.EnvironmentSourceKind.huggingface_hub,
        "unity_catalog": api.EnvironmentSourceKind.inline,
        "inline": api.EnvironmentSourceKind.inline,
        "local": api.EnvironmentSourceKind.inline,
        "folder": api.EnvironmentSourceKind.inline,
    }
    return _KIND_MAP.get(source, api.EnvironmentSourceKind.inline)


def _verifier_from_manifest(
    verifier: dict[str, Any], name: str
) -> api.VerifierSuiteBinding:
    verifier_type = verifier.get("type", "script")
    entry = verifier.get("entry")
    timeout_sec = verifier.get("timeout_sec", 300)
    pass_threshold = verifier.get("pass_threshold", 1.0)
    config: dict[str, Any] = {}
    if entry:
        config["entry"] = entry
    if timeout_sec is not None:
        config["timeout_sec"] = timeout_sec
    if pass_threshold is not None:
        config["pass_threshold"] = pass_threshold
    metric = api.MetricDefinition(
        name=f"{name}-{verifier_type}",
        metric_kind=verifier_type,
        weight=1.0,
        config=config or None,
    )
    suite = api.VerifierSuiteInline(
        binding_type="inline",
        name=f"{name}-verifier",
        metrics=[metric],
    )
    return api.VerifierSuiteBinding(root=suite)


def _sandbox_from_manifest(environment: dict[str, Any]) -> api.SandboxPolicy | None:
    image = environment.get("image")
    per_task = environment.get("per_task_sandbox", False)
    allow_internet = environment.get("allow_internet")
    resources = environment.get("resources")
    limits: dict[str, Any] = {}
    if allow_internet is not None:
        limits["allow_internet"] = allow_internet
    if resources:
        limits.update(resources)
    if not (image or per_task or limits):
        return None
    return api.SandboxPolicy(
        enabled=True,
        image=image,
        per_task_sandbox=per_task,
        limits=limits or None,
    )


def _env_source_from_manifest(
    manifest_name: str,
    manifest_version: str,
    dataset: dict[str, Any],
    source_kind: api.EnvironmentSourceKind,
) -> api.EnvironmentSource:
    ref = dataset.get("ref")
    git_ref: str | None = None
    uri_str: str | None = None
    if ref and "@" in ref:
        uri_part, git_ref = ref.rsplit("@", 1)
        # Only set uri when it looks like a full URL (has a scheme)
        if "://" in uri_part or uri_part.startswith("https://") or uri_part.startswith("http://"):
            uri_str = uri_part
    elif ref and ("://" in ref or ref.startswith("https://") or ref.startswith("http://")):
        uri_str = ref
    return api.EnvironmentSource(
        kind=source_kind,
        name=manifest_name,
        version=manifest_version,
        git_ref=git_ref,
        uri=uri_str,
    )


def _inline_env_from_manifest(
    manifest: dict[str, Any],
) -> api.InlineEnvironmentDefinition:
    metadata = manifest.get("metadata", {})
    name: str = metadata.get("name", "unnamed-eval")
    version: str = str(metadata.get("version", "0.1.0"))
    environment_sec = manifest.get("environment", {})
    dataset_sec = manifest.get("dataset", {})
    verifier_sec = manifest.get("verifier", {})
    agent_sec = manifest.get("agent", {})
    dataset = _dataset_from_manifest(dataset_sec)
    verifier = _verifier_from_manifest(verifier_sec, name)
    sandbox = _sandbox_from_manifest(environment_sec)
    agent_timeout = agent_sec.get("timeout_sec")
    runtime_meta: dict[str, Any] = {}
    if agent_timeout:
        runtime_meta["agent_timeout_sec"] = agent_timeout
    runtime = api.EnvironmentRuntimePolicy(metadata=runtime_meta or None)
    return api.InlineEnvironmentDefinition(
        name=name,
        version=version,
        dataset=dataset,
        verifier=verifier,
        runtime=runtime,
        sandbox=sandbox,
    )


def _parse_eval_env_ref(
    eval_env_ref: str,
) -> tuple[str, str | None, str | None]:
    """Parse "owner/name@version" or "name@version" into (name, owner|None, version|None)."""
    version: str | None = None
    if "@" in eval_env_ref:
        ref_part, version = eval_env_ref.rsplit("@", 1)
    else:
        ref_part = eval_env_ref
    if "/" in ref_part:
        owner, name = ref_part.split("/", 1)
    else:
        owner = None
        name = ref_part
    return name, owner, version


def load_manifest(path: str | Path) -> api.GeneratorRunCreate:
    """Load an ``eval.yaml`` manifest and return a :class:`GeneratorRunCreate`.

    Args:
        path: Path to an ``eval.yaml`` file, or a directory containing one.

    Returns:
        A :class:`GeneratorRunCreate` with ``mode="generator"`` and a
        single :class:`DirectRunPair` in ``selection.direct_pairs``.

    Raises:
        FileNotFoundError: When the manifest file does not exist.
        ValueError: On apiVersion / kind mismatches or missing required fields.
    """
    p = Path(path)
    if p.is_dir():
        p = p / _MANIFEST_FILENAME
    if not p.exists():
        raise FileNotFoundError(f"Manifest file not found: {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    api_version = raw.get("apiVersion", "")
    kind = raw.get("kind", "")
    if api_version != _API_VERSION:
        raise ValueError(
            f"Unsupported apiVersion {api_version!r}; expected {_API_VERSION!r}"
        )
    if kind != _KIND:
        raise ValueError(f"Unsupported kind {kind!r}; expected {_KIND!r}")
    metadata = raw.get("metadata", {})
    name: str = metadata.get("name", "unnamed-eval")
    version: str = str(metadata.get("version", "0.1.0"))
    dataset_sec = raw.get("dataset", {})
    eval_env_ref: str | None = raw.get("eval_env_ref") or raw.get("job", {}).get("eval_env_ref")
    if eval_env_ref:
        # eval_env_ref is a registry pointer "owner/name@version" or "name@version".
        # EnvironmentRef requires a UUID which we don't have at manifest load time, so
        # we build a stub EnvironmentInlineMembership that carries the ref info in
        # metadata.  The execution layer (or a lookup step) resolves the real UUID.
        env_name, owner, env_version = _parse_eval_env_ref(eval_env_ref)
        full_name = f"{owner}/{env_name}" if owner else env_name
        env_version_resolved = env_version or version
        stub_inline = api.InlineEnvironmentDefinition(
            name=full_name,
            version=env_version_resolved,
            dataset=api.DatasetBinding(provider="local"),
            verifier=api.VerifierSuiteBinding(
                root=api.VerifierSuiteInline(
                    binding_type="inline",
                    name=f"{full_name}-verifier",
                    metrics=[
                        api.MetricDefinition(
                            name="pass",
                            metric_kind="script",
                            weight=1.0,
                        )
                    ],
                )
            ),
            runtime=api.EnvironmentRuntimePolicy(),
            metadata={"_eval_env_ref": eval_env_ref},
        )
        environment_member: api.EnvironmentRef | api.EnvironmentInlineMembership = (
            api.EnvironmentInlineMembership(inline_definition=stub_inline)
        )
    else:
        inline_def = _inline_env_from_manifest(raw)
        source_kind = _source_kind_from_dataset(dataset_sec)
        if source_kind != api.EnvironmentSourceKind.inline:
            env_source = _env_source_from_manifest(name, version, dataset_sec, source_kind)
            if inline_def.dataset.metadata is None:
                inline_def.dataset.metadata = {}
            inline_def.dataset.metadata["_manifest_source_kind"] = source_kind.value
            if env_source.git_ref:
                inline_def.dataset.metadata["_manifest_git_ref"] = env_source.git_ref
            if env_source.uri:
                inline_def.dataset.metadata["_manifest_uri"] = str(env_source.uri)
        environment_member = api.EnvironmentInlineMembership(inline_definition=inline_def)
    placeholder_model = api.ModelDefinitionCreate(
        name=f"{name}-agent",
        runtime=api.ModelExecutionConfig(
            provider="manifest",
            model_name=name,
            model_version=version,
        ),
    )
    pair = api.DirectRunPair(
        model=placeholder_model,
        environment=environment_member,
    )
    selection = api.RunSelection(root=api.RunSelection2(direct_pairs=[pair]))
    return api.GeneratorRunCreate(
        mode="generator",
        selection=selection,
        labels={"manifest_name": name, "manifest_version": version},
    )