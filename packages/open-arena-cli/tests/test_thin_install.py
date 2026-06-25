# License Apache 2.0: (c) 2026 Athena-Reply
"""Test that thin CLI install does NOT pull in the heavy engine.

Asserts that importing the CLI module and running a remote command (with
mocked httpx/ArenaAPIClient) does NOT import synalinks, proving the CLI
is engine-free for remote operations.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch


def test_importing_cli_does_not_import_synalinks():
    """Importing open_arena_cli must not cause synalinks to be loaded."""
    for key in list(sys.modules.keys()):
        if "synalinks" in key:
            del sys.modules[key]

    import open_arena_cli  # noqa: F401
    import open_arena_cli.main  # noqa: F401

    assert "synalinks" not in sys.modules, (
        "Importing open_arena_cli pulled in synalinks -- the thin CLI is not engine-free!"
    )


def test_importing_groups_does_not_import_synalinks():
    """Importing open_arena_cli.groups must not cause synalinks to be loaded."""
    for key in list(sys.modules.keys()):
        if "synalinks" in key:
            del sys.modules[key]

    import open_arena_cli.groups  # noqa: F401

    assert "synalinks" not in sys.modules, (
        "Importing open_arena_cli.groups pulled in synalinks -- the thin CLI is not engine-free!"
    )


def test_remote_command_does_not_import_synalinks():
    """Running a remote command (mock httpx) does NOT import synalinks."""
    for key in list(sys.modules.keys()):
        if "synalinks" in key:
            del sys.modules[key]

    from click.testing import CliRunner
    from open_arena_cli.main import main

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.content = b"[]"
    mock_response.json.return_value = {"items": [], "next_cursor": None}

    mock_client_instance = MagicMock()
    mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
    mock_client_instance.__exit__ = MagicMock(return_value=False)
    mock_client_instance.head = MagicMock()
    mock_client_instance.request = MagicMock(return_value=mock_response)

    with patch("httpx.Client", return_value=mock_client_instance):
        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(
            main,
            ["env", "list", "--server", "http://fake-server:8000"],
            catch_exceptions=False,
        )

    assert "synalinks" not in sys.modules, (
        "Running arena env list (remote path) pulled in synalinks! "
        "CLI output: " + result.output
    )


def test_request_command_does_not_import_synalinks():
    """The arena request command (thin HTTP) does NOT import synalinks."""
    for key in list(sys.modules.keys()):
        if "synalinks" in key:
            del sys.modules[key]

    from click.testing import CliRunner
    from open_arena_cli.main import main

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.content = b'{"items": []}'
    mock_response.json.return_value = {"items": []}

    mock_client_instance = MagicMock()
    mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
    mock_client_instance.__exit__ = MagicMock(return_value=False)
    mock_client_instance.request = MagicMock(return_value=mock_response)

    with patch("httpx.Client", return_value=mock_client_instance):
        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(
            main,
            ["request", "GET", "/v1/metric-kinds", "--server", "http://fake-server:8000"],
            catch_exceptions=False,
        )

    assert "synalinks" not in sys.modules, (
        "Running arena request pulled in synalinks! CLI output: " + result.output
    )


def test_open_arena_core_does_not_import_synalinks():
    """Importing open_arena_core must not cause synalinks to be loaded."""
    for key in list(sys.modules.keys()):
        if "synalinks" in key:
            del sys.modules[key]

    import open_arena_core  # noqa: F401
    import open_arena_core.models  # noqa: F401
    import open_arena_core.client  # noqa: F401
    import open_arena_core.constants  # noqa: F401

    assert "synalinks" not in sys.modules, (
        "Importing open_arena_core pulled in synalinks -- the core package is not engine-free!"
    )
