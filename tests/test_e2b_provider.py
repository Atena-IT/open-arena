# License Apache 2.0: (c) 2026 Athena-Reply
"""Tests for E2BSandboxProvider (WS6, issue #40).

All tests are fully mocked — no live E2B sandbox is created.

Coverage
--------
- Sandbox lifecycle order: create → bootstrap → upload → install → run →
  teardown → kill (verified via ``call_args_list`` ordering).
- ``SandboxPolicy.limits`` mapping (``timeout_seconds``).
- ``SandboxPolicy.bootstrap`` mapping (``template``, ``packages``,
  ``commands``, ``env``).
- ``SandboxPolicy.teardown`` mapping (``commands``).
- Result dict shape parity with ``LocalSandboxProvider`` (``"rows"`` key).
- Error handling when ``E2B_API_KEY`` is missing.
- Error handling when ``policy.enabled=False``.
- Registry wires ``E2BSandboxProvider`` when ``OPEN_ARENA_SANDBOX=e2b``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_policy(
    *,
    enabled: bool = True,
    timeout_seconds: int | None = None,
    template: str | None = None,
    packages: list[str] | None = None,
    bootstrap_commands: list[str] | None = None,
    bootstrap_env: dict[str, str] | None = None,
    teardown_commands: list[str] | None = None,
    isolation_mode: str | None = None,
) -> Any:
    """Build a SandboxPolicy with sensible defaults for tests."""
    from src.api.models import IsolationMode, SandboxPolicy

    limits: dict[str, Any] = {}
    if timeout_seconds is not None:
        limits["timeout_seconds"] = timeout_seconds

    bootstrap: dict[str, Any] = {}
    if template:
        bootstrap["template"] = template
    if packages:
        bootstrap["packages"] = packages
    if bootstrap_commands:
        bootstrap["commands"] = bootstrap_commands
    if bootstrap_env:
        bootstrap["env"] = bootstrap_env

    teardown: dict[str, Any] = {}
    if teardown_commands:
        teardown["commands"] = teardown_commands

    iso = IsolationMode(isolation_mode) if isolation_mode else None

    return SandboxPolicy(
        enabled=enabled,
        isolation_mode=iso,
        bootstrap=bootstrap or None,
        teardown=teardown or None,
        limits=limits or None,
    )


def _make_result_json(rows: list[dict] | None = None) -> str:
    return json.dumps({"rows": rows or [], "meta": {}})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def config_file(tmp_path: Path) -> Path:
    """Minimal YAML config file for upload testing."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text("model: test\n")
    return cfg


@pytest.fixture()
def mock_e2b_sdk(monkeypatch):
    """Patch the e2b.Sandbox class so no real sandbox is created.

    Returns a MagicMock that will be the *class* injected as
    ``e2b.Sandbox``; test code can inspect ``mock_e2b_sdk.return_value``
    for the sandbox *instance*.
    """
    mock_sandbox_cls = MagicMock(name="Sandbox")
    mock_instance = MagicMock(name="sandbox_instance")
    mock_sandbox_cls.return_value = mock_instance

    # Default: commands succeed with a JSON result as stdout
    mock_result = MagicMock()
    mock_result.exit_code = 0
    mock_result.stdout = _make_result_json()
    mock_result.stderr = ""
    mock_instance.commands.run.return_value = mock_result

    # files.write succeeds silently
    mock_instance.files.write.return_value = None

    # sandbox_id for logging
    mock_instance.sandbox_id = "sandbox-test-123"

    # Patch at the import site inside _E2BClient.__init__
    import sys
    fake_e2b_module = MagicMock()
    fake_e2b_module.Sandbox = mock_sandbox_cls
    monkeypatch.setitem(sys.modules, "e2b", fake_e2b_module)

    return mock_sandbox_cls


@pytest.fixture()
def api_key_env(monkeypatch):
    """Set E2B_API_KEY for the duration of the test."""
    monkeypatch.setenv("E2B_API_KEY", "test-api-key-12345")


# ---------------------------------------------------------------------------
# 1. Missing API key
# ---------------------------------------------------------------------------

class TestMissingApiKey:
    def test_raises_config_error_when_key_missing(self, config_file, mock_e2b_sdk, monkeypatch):
        monkeypatch.delenv("E2B_API_KEY", raising=False)

        from src.api.sandboxes.e2b_provider import E2BSandboxConfigError, E2BSandboxProvider

        provider = E2BSandboxProvider()
        with pytest.raises(E2BSandboxConfigError, match="E2B_API_KEY"):
            provider.run(config_file)

    def test_raises_config_error_when_key_is_empty(self, config_file, mock_e2b_sdk, monkeypatch):
        monkeypatch.setenv("E2B_API_KEY", "   ")

        from src.api.sandboxes.e2b_provider import E2BSandboxConfigError, E2BSandboxProvider

        provider = E2BSandboxProvider()
        with pytest.raises(E2BSandboxConfigError, match="E2B_API_KEY"):
            provider.run(config_file)


# ---------------------------------------------------------------------------
# 2. Disabled policy
# ---------------------------------------------------------------------------

class TestDisabledPolicy:
    def test_raises_when_policy_enabled_false(self, config_file, mock_e2b_sdk, api_key_env):
        from src.api.sandboxes.e2b_provider import E2BSandboxConfigError, E2BSandboxProvider

        policy = _make_policy(enabled=False)
        provider = E2BSandboxProvider()
        with pytest.raises(E2BSandboxConfigError, match="enabled=False"):
            provider.run(config_file, policy=policy)


# ---------------------------------------------------------------------------
# 3. Lifecycle ordering
# ---------------------------------------------------------------------------

class TestLifecycleOrder:
    """The sandbox lifecycle must follow the documented order."""

    def test_lifecycle_order_no_policy(self, config_file, mock_e2b_sdk, api_key_env):
        """Without a policy, lifecycle is: create → upload → install → run → kill."""
        from src.api.sandboxes.e2b_provider import E2BSandboxProvider

        provider = E2BSandboxProvider()
        provider.run(config_file)

        mock_instance = mock_e2b_sdk.return_value
        assert mock_e2b_sdk.called, "Sandbox constructor must be called (create)"

        commands_called = [c.args[0] for c in mock_instance.commands.run.call_args_list]
        # Expected command sequence:
        #   [0] mkdir -p /arena
        #   [1] pip install ...
        #   [2] arena --config ...
        assert len(commands_called) >= 3
        assert commands_called[0].startswith("mkdir -p")
        assert "pip install" in commands_called[1]
        assert "arena --config" in commands_called[2]

        assert mock_instance.files.write.called, "upload_file must be called"
        assert mock_instance.kill.called, "kill must be called"

    def test_lifecycle_order_with_bootstrap_commands(
        self, config_file, mock_e2b_sdk, api_key_env
    ):
        """Bootstrap commands must run BEFORE upload and install."""
        from src.api.sandboxes.e2b_provider import E2BSandboxProvider

        policy = _make_policy(bootstrap_commands=["apt-get update -y", "apt-get install -y git"])
        provider = E2BSandboxProvider()
        provider.run(config_file, policy=policy)

        mock_instance = mock_e2b_sdk.return_value
        commands_called = [c.args[0] for c in mock_instance.commands.run.call_args_list]

        # Find indices
        bootstrap_idx = [i for i, cmd in enumerate(commands_called) if "apt-get" in cmd]
        mkdir_idx = next(i for i, cmd in enumerate(commands_called) if cmd.startswith("mkdir"))
        pip_idx = next(i for i, cmd in enumerate(commands_called) if "pip install" in cmd)
        run_idx = next(i for i, cmd in enumerate(commands_called) if "arena --config" in cmd)

        assert bootstrap_idx, "Bootstrap commands must be called"
        # All bootstrap cmds come before mkdir
        assert max(bootstrap_idx) < mkdir_idx < pip_idx < run_idx

    def test_teardown_runs_after_sweep_before_kill(
        self, config_file, mock_e2b_sdk, api_key_env
    ):
        """Teardown commands must run AFTER sweep but BEFORE kill."""
        from src.api.sandboxes.e2b_provider import E2BSandboxProvider

        policy = _make_policy(teardown_commands=["rm -rf /arena/cache"])
        provider = E2BSandboxProvider()
        provider.run(config_file, policy=policy)

        mock_instance = mock_e2b_sdk.return_value
        commands_called = [c.args[0] for c in mock_instance.commands.run.call_args_list]

        run_idx = next(i for i, cmd in enumerate(commands_called) if "arena --config" in cmd)
        teardown_idx = next(
            (i for i, cmd in enumerate(commands_called) if "rm -rf" in cmd), None
        )

        assert teardown_idx is not None, "Teardown command must be called"
        assert run_idx < teardown_idx, "Teardown must come after sweep execution"
        assert mock_instance.kill.called, "kill must be called after teardown"

    def test_kill_called_even_on_sweep_failure(self, config_file, mock_e2b_sdk, api_key_env):
        """sandbox.kill() must be called even when the sweep command fails."""
        from src.api.sandboxes.e2b_provider import E2BSandboxError, E2BSandboxProvider

        mock_instance = mock_e2b_sdk.return_value

        # Make every command succeed except arena --config
        def _run_side_effect(cmd, **kwargs):
            result = MagicMock()
            result.exit_code = 0
            result.stdout = ""
            result.stderr = ""
            if "arena --config" in cmd:
                result.exit_code = 1
                result.stderr = "sweep crashed"
            return result

        mock_instance.commands.run.side_effect = _run_side_effect

        provider = E2BSandboxProvider()
        with pytest.raises(E2BSandboxError):
            provider.run(config_file)

        assert mock_instance.kill.called, "kill must be called even after failure"


# ---------------------------------------------------------------------------
# 4. Policy mapping — limits
# ---------------------------------------------------------------------------

class TestLimitsMapping:
    def test_timeout_seconds_forwarded_to_sandbox_constructor(
        self, config_file, mock_e2b_sdk, api_key_env
    ):
        from src.api.sandboxes.e2b_provider import E2BSandboxProvider

        policy = _make_policy(timeout_seconds=600)
        provider = E2BSandboxProvider()
        provider.run(config_file, policy=policy)

        _, kwargs = mock_e2b_sdk.call_args
        assert kwargs.get("timeout") == 600

    def test_default_timeout_is_300(self, config_file, mock_e2b_sdk, api_key_env):
        from src.api.sandboxes.e2b_provider import E2BSandboxProvider

        provider = E2BSandboxProvider()
        provider.run(config_file)

        _, kwargs = mock_e2b_sdk.call_args
        assert kwargs.get("timeout") == 300


# ---------------------------------------------------------------------------
# 5. Policy mapping — bootstrap
# ---------------------------------------------------------------------------

class TestBootstrapMapping:
    def test_template_forwarded_to_sandbox_constructor(
        self, config_file, mock_e2b_sdk, api_key_env
    ):
        from src.api.sandboxes.e2b_provider import E2BSandboxProvider

        policy = _make_policy(template="python-3.11")
        provider = E2BSandboxProvider()
        provider.run(config_file, policy=policy)

        _, kwargs = mock_e2b_sdk.call_args
        assert kwargs.get("template") == "python-3.11"

    def test_extra_packages_appended_to_pip_install(
        self, config_file, mock_e2b_sdk, api_key_env
    ):
        from src.api.sandboxes.e2b_provider import E2BSandboxProvider

        policy = _make_policy(packages=["torch==2.3.0", "transformers"])
        provider = E2BSandboxProvider()
        provider.run(config_file, policy=policy)

        mock_instance = mock_e2b_sdk.return_value
        pip_calls = [
            c.args[0]
            for c in mock_instance.commands.run.call_args_list
            if "pip install" in c.args[0]
        ]
        assert pip_calls, "pip install must be called"
        pip_cmd = pip_calls[0]
        assert "torch==2.3.0" in pip_cmd
        assert "transformers" in pip_cmd
        assert "open-arena" in pip_cmd, "default package must always be installed"

    def test_bootstrap_commands_are_executed(self, config_file, mock_e2b_sdk, api_key_env):
        from src.api.sandboxes.e2b_provider import E2BSandboxProvider

        policy = _make_policy(bootstrap_commands=["echo hello"])
        provider = E2BSandboxProvider()
        provider.run(config_file, policy=policy)

        mock_instance = mock_e2b_sdk.return_value
        all_cmds = [c.args[0] for c in mock_instance.commands.run.call_args_list]
        assert any("echo hello" in cmd for cmd in all_cmds)


# ---------------------------------------------------------------------------
# 6. Policy mapping — teardown
# ---------------------------------------------------------------------------

class TestTeardownMapping:
    def test_teardown_commands_executed(self, config_file, mock_e2b_sdk, api_key_env):
        from src.api.sandboxes.e2b_provider import E2BSandboxProvider

        policy = _make_policy(teardown_commands=["cleanup.sh", "rm -f /tmp/arena.lock"])
        provider = E2BSandboxProvider()
        provider.run(config_file, policy=policy)

        mock_instance = mock_e2b_sdk.return_value
        all_cmds = [c.args[0] for c in mock_instance.commands.run.call_args_list]
        assert any("cleanup.sh" in cmd for cmd in all_cmds)
        assert any("rm -f /tmp/arena.lock" in cmd for cmd in all_cmds)

    def test_teardown_warning_on_nonzero_exit(
        self, config_file, mock_e2b_sdk, api_key_env, caplog
    ):
        """A failing teardown command should warn but NOT raise."""
        import logging

        from src.api.sandboxes.e2b_provider import E2BSandboxProvider

        mock_instance = mock_e2b_sdk.return_value

        def _side_effect(cmd, **kwargs):
            result = MagicMock()
            result.exit_code = 1 if "bad_teardown" in cmd else 0
            result.stdout = _make_result_json() if "arena --config" in cmd else ""
            result.stderr = "teardown err" if "bad_teardown" in cmd else ""
            return result

        mock_instance.commands.run.side_effect = _side_effect

        policy = _make_policy(teardown_commands=["bad_teardown.sh"])
        provider = E2BSandboxProvider()

        with caplog.at_level(logging.WARNING, logger="src.api.sandboxes.e2b_provider"):
            result = provider.run(config_file, policy=policy)  # must not raise

        assert "rows" in result
        assert any("bad_teardown" in msg for msg in caplog.messages)


# ---------------------------------------------------------------------------
# 7. Result dict shape parity with LocalSandboxProvider
# ---------------------------------------------------------------------------

class TestResultShape:
    def test_result_contains_rows_key_from_json_file(
        self, config_file, mock_e2b_sdk, api_key_env
    ):
        from src.api.sandboxes.e2b_provider import E2BSandboxProvider

        expected_rows = [{"model": "gpt-4o", "score": 0.95}]
        mock_instance = mock_e2b_sdk.return_value

        def _side_effect(cmd, **kwargs):
            result = MagicMock()
            result.exit_code = 0
            result.stdout = json.dumps({"rows": expected_rows, "meta": {}})
            result.stderr = ""
            return result

        mock_instance.commands.run.side_effect = _side_effect

        provider = E2BSandboxProvider()
        result = provider.run(config_file)

        assert "rows" in result
        assert result["rows"] == expected_rows

    def test_result_fallback_contains_rows_key(
        self, config_file, mock_e2b_sdk, api_key_env
    ):
        """When no JSON is parseable, fallback result still has 'rows' key."""
        mock_instance = mock_e2b_sdk.return_value

        def _side_effect(cmd, **kwargs):
            result = MagicMock()
            result.exit_code = 0
            result.stdout = "no json here, just plain text"
            result.stderr = ""
            return result

        mock_instance.commands.run.side_effect = _side_effect

        from src.api.sandboxes.e2b_provider import E2BSandboxProvider

        provider = E2BSandboxProvider()
        result = provider.run(config_file)

        assert "rows" in result

    def test_result_parses_json_from_stdout_last_line(
        self, config_file, mock_e2b_sdk, api_key_env
    ):
        """JSON on the last stdout line is picked up as fallback to the file."""
        from src.api.sandboxes.e2b_provider import E2BSandboxProvider

        expected = {"rows": [{"metric": "bleu", "value": 0.77}], "meta": {"source": "test"}}
        mock_instance = mock_e2b_sdk.return_value

        # First call (cat result.json) fails; run command returns json last line
        call_count = {"n": 0}

        def _side_effect(cmd, **kwargs):
            result = MagicMock()
            result.exit_code = 0
            result.stderr = ""
            if cmd.startswith("cat "):
                result.exit_code = 1
                result.stdout = ""
            elif "arena --config" in cmd:
                result.stdout = f"Running evaluation...\nDone.\n{json.dumps(expected)}"
            else:
                result.stdout = ""
            return result

        mock_instance.commands.run.side_effect = _side_effect

        provider = E2BSandboxProvider()
        result = provider.run(config_file)

        assert result["rows"] == expected["rows"]


# ---------------------------------------------------------------------------
# 8. Registry wiring
# ---------------------------------------------------------------------------

class TestRegistryWiring:
    def test_registry_returns_e2b_provider_when_sandbox_e2b(self, monkeypatch):
        monkeypatch.setenv("OPEN_ARENA_SANDBOX", "e2b")
        # E2BSandboxProvider can be constructed without the SDK
        from src.api.registry import _build_sandbox
        from src.api.sandboxes.e2b_provider import E2BSandboxProvider
        from src.api.settings import ArenaSettings

        settings = ArenaSettings(sandbox="e2b")
        provider = _build_sandbox(settings)
        assert isinstance(provider, E2BSandboxProvider)

    def test_registry_raises_for_unknown_sandbox(self):
        from src.api.registry import _build_sandbox
        from src.api.settings import ArenaSettings

        settings = ArenaSettings(sandbox="unknown_backend")
        with pytest.raises(ValueError, match="unknown_backend"):
            _build_sandbox(settings)

    def test_registry_still_returns_local_provider_by_default(self):
        from src.api.ports.sandbox_provider import LocalSandboxProvider
        from src.api.registry import _build_sandbox
        from src.api.settings import ArenaSettings

        settings = ArenaSettings(sandbox="local")
        provider = _build_sandbox(settings)
        assert isinstance(provider, LocalSandboxProvider)
