# License Apache 2.0: (c) 2026 Athena-Reply
"""Tests for the arena CLI sub-groups added in WS9 (issue #43).

Coverage:
- Command tree wiring (every sub-group and sub-command resolves under `main`)
- Remote path: ArenaAPIClient is mocked, commands delegate to it correctly
- Local path: commands run against an in-process ArenaAPIService backed by a
  temp SQLite DB (OPEN_ARENA_DB_PATH points at a tmp file)
- Dispatch logic: --local flag forces in-process; server-unreachable auto-falls
  back to in-process
"""
from __future__ import annotations

import json
import os
import tempfile
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from src.evaluate import main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _invoke(args: list[str], env: dict[str, str] | None = None) -> "Result":
    runner = CliRunner(mix_stderr=False)
    return runner.invoke(main, args, env=env, catch_exceptions=False)


def _tmp_db(tmp_path: Path) -> dict[str, str]:
    """Return env overrides that point the local backend at a fresh SQLite DB."""
    return {"OPEN_ARENA_DB_PATH": str(tmp_path / "arena_test.db")}


# ---------------------------------------------------------------------------
# Command tree: wiring
# ---------------------------------------------------------------------------


class TestCommandTree:
    """Verify every sub-group and sub-command is accessible under `main`."""

    def test_top_level_help(self):
        result = _invoke(["--help"])
        assert result.exit_code == 0, result.output
        for name in ("env", "verifier", "leaderboard", "run", "discover", "serve", "request"):
            assert name in result.output, f"'{name}' not found in --help output"

    @pytest.mark.parametrize("group,sub", [
        ("env", "list"),
        ("env", "get"),
        ("env", "create"),
        ("env", "delete"),
        ("verifier", "list"),
        ("verifier", "get"),
        ("verifier", "create"),
        ("verifier", "delete"),
        ("leaderboard", "list"),
        ("leaderboard", "get"),
        ("leaderboard", "create"),
        ("leaderboard", "delete"),
        ("leaderboard", "models"),
        ("leaderboard", "environments"),
        ("leaderboard", "entries"),
        ("run", "submit"),
        ("run", "get"),
        ("run", "results"),
        ("run", "list"),
        ("discover", "metric-kinds"),
        ("discover", "aggregations"),
        ("discover", "model-providers"),
        ("discover", "dataset-providers"),
    ])
    def test_subcommand_help(self, group, sub):
        result = _invoke([group, sub, "--help"])
        assert result.exit_code == 0, f"{group} {sub} --help failed:\n{result.output}"
        assert "--help" in result.output

    def test_run_group_help_mentions_local(self):
        result = _invoke(["run", "--help"])
        assert result.exit_code == 0
        assert "--local" in result.output or "local" in result.output.lower()


# ---------------------------------------------------------------------------
# Remote path: ArenaAPIClient mocked
# ---------------------------------------------------------------------------


class TestRemotePath:
    """Commands delegate to ArenaAPIClient when a live server is reachable."""

    def _patch_reachable(self):
        """Patch httpx so the connectivity check always succeeds."""
        import httpx

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        return patch.object(httpx.Client, "head", return_value=mock_resp)

    def _patch_client_request(self, return_value):
        """Patch ArenaAPIClient.request to return *return_value*."""
        return patch("src.api.client.ArenaAPIClient.request", return_value=return_value)

    def test_discover_metric_kinds_remote(self):
        payload = {"items": [{"id": "exact_match", "display_name": "exact match"}]}
        with self._patch_reachable(), self._patch_client_request(payload) as mock_req:
            result = _invoke(["discover", "metric-kinds", "--server", "http://fake:9999"])
        assert result.exit_code == 0, result.output
        assert "exact_match" in result.output
        mock_req.assert_called_once()
        call_args = mock_req.call_args
        assert call_args[0][0].upper() == "GET"
        assert "/v1/metric-kinds" in call_args[0][1]

    def test_discover_aggregations_remote(self):
        payload = {"items": [{"id": "weighted_mean"}]}
        with self._patch_reachable(), self._patch_client_request(payload):
            result = _invoke(["discover", "aggregations", "--server", "http://fake:9999"])
        assert result.exit_code == 0
        assert "weighted_mean" in result.output

    def test_env_list_remote(self):
        payload = {"items": [], "next_cursor": None}
        with self._patch_reachable(), self._patch_client_request(payload) as mock_req:
            result = _invoke(["env", "list", "--server", "http://fake:9999"])
        assert result.exit_code == 0
        mock_req.assert_called_once()
        assert "/v1/environments" in mock_req.call_args[0][1]

    def test_verifier_list_remote(self):
        payload = {"items": []}
        with self._patch_reachable(), self._patch_client_request(payload):
            result = _invoke(["verifier", "list", "--server", "http://fake:9999"])
        assert result.exit_code == 0

    def test_leaderboard_list_remote(self):
        payload = {"items": []}
        with self._patch_reachable(), self._patch_client_request(payload):
            result = _invoke(["leaderboard", "list", "--server", "http://fake:9999"])
        assert result.exit_code == 0

    def test_run_list_remote(self):
        payload = {"items": []}
        with self._patch_reachable(), self._patch_client_request(payload):
            result = _invoke(["run", "list", "--server", "http://fake:9999"])
        assert result.exit_code == 0

    def test_env_get_remote(self):
        eid = str(uuid.uuid4())
        payload = {"id": eid, "source": {"kind": "inline", "name": "env", "version": "1"}, "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-01T00:00:00Z"}
        with self._patch_reachable(), self._patch_client_request(payload) as mock_req:
            result = _invoke(["env", "get", eid, "--server", "http://fake:9999"])
        assert result.exit_code == 0
        assert eid in result.output
        assert f"/v1/environments/{eid}" in mock_req.call_args[0][1]

    def test_run_get_remote(self):
        rid = str(uuid.uuid4())
        payload = {
            "id": rid,
            "mode": "generator",
            "selection": {"leaderboard_id": str(uuid.uuid4())},
            "status": "succeeded",
            "cache_status": "miss",
            "created_at": "2026-01-01T00:00:00Z",
        }
        with self._patch_reachable(), self._patch_client_request(payload) as mock_req:
            result = _invoke(["run", "get", rid, "--server", "http://fake:9999"])
        assert result.exit_code == 0
        assert f"/v1/runs/{rid}" in mock_req.call_args[0][1]


# ---------------------------------------------------------------------------
# Local path: in-process ArenaAPIService with temp SQLite
# ---------------------------------------------------------------------------


@pytest.fixture()
def local_env(tmp_path):
    """Env vars that route the local backend to a fresh temp DB."""
    return {
        "OPEN_ARENA_DB_PATH": str(tmp_path / "arena_test.db"),
        # Ensure static auth token is set so service won't complain
        "OPEN_ARENA_API_TOKEN": "test-token",
    }


class TestLocalPath:
    """Commands run against an in-process ArenaAPIService (--local flag)."""

    def test_discover_metric_kinds_local(self, local_env):
        result = _invoke(["discover", "metric-kinds", "--local"], env=local_env)
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert "items" in data
        ids = [item["id"] for item in data["items"]]
        # The service always returns at least the built-in aggregation types
        # and reward identifiers; just check the list is non-empty.
        assert len(ids) > 0

    def test_discover_aggregations_local(self, local_env):
        result = _invoke(["discover", "aggregations", "--local"], env=local_env)
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        ids = [item["id"] for item in data["items"]]
        assert "weighted_mean" in ids

    def test_discover_model_providers_local(self, local_env):
        result = _invoke(["discover", "model-providers", "--local"], env=local_env)
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert "items" in data
        ids = [item["id"] for item in data["items"]]
        assert "openai" in ids or "anthropic" in ids

    def test_discover_dataset_providers_local(self, local_env):
        result = _invoke(["discover", "dataset-providers", "--local"], env=local_env)
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert "items" in data

    def test_env_list_local_empty(self, local_env):
        result = _invoke(["env", "list", "--local"], env=local_env)
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data["items"] == []

    def test_verifier_list_local_empty(self, local_env):
        result = _invoke(["verifier", "list", "--local"], env=local_env)
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data["items"] == []

    def test_leaderboard_list_local_empty(self, local_env):
        result = _invoke(["leaderboard", "list", "--local"], env=local_env)
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data["items"] == []

    def test_run_list_local_empty(self, local_env):
        result = _invoke(["run", "list", "--local"], env=local_env)
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data["items"] == []

    def test_verifier_create_and_get_local(self, local_env, tmp_path):
        payload = {
            "name": "test-verifier",
            "description": "A test verifier",
            "aggregation": "weighted_mean",
            "metrics": [
                {
                    "name": "accuracy",
                    "metric_kind": "exact_match",
                    "weight": 1.0,
                    "direction": "max",
                }
            ],
        }
        payload_file = tmp_path / "verifier.json"
        payload_file.write_text(json.dumps(payload))

        create_result = _invoke(
            ["verifier", "create", "--local", "--file", str(payload_file)],
            env=local_env,
        )
        assert create_result.exit_code == 0, create_result.output
        created = json.loads(create_result.output)
        assert created["name"] == "test-verifier"
        verifier_id = created["id"]

        # Get it back
        get_result = _invoke(["verifier", "get", verifier_id, "--local"], env=local_env)
        assert get_result.exit_code == 0, get_result.output
        got = json.loads(get_result.output)
        assert got["id"] == verifier_id
        assert got["name"] == "test-verifier"

    def test_leaderboard_create_and_get_local(self, local_env, tmp_path):
        payload = {
            "name": "test-leaderboard",
            "visibility": "private",
            "ranking": {"primary_metric": "reward", "aggregation": "weighted_mean"},
        }
        payload_file = tmp_path / "lb.json"
        payload_file.write_text(json.dumps(payload))

        create_result = _invoke(
            ["leaderboard", "create", "--local", "--file", str(payload_file)],
            env=local_env,
        )
        assert create_result.exit_code == 0, create_result.output
        created = json.loads(create_result.output)
        assert created["name"] == "test-leaderboard"
        lb_id = created["id"]

        # Get it back
        get_result = _invoke(["leaderboard", "get", lb_id, "--local"], env=local_env)
        assert get_result.exit_code == 0, get_result.output
        got = json.loads(get_result.output)
        assert got["id"] == lb_id

    def test_leaderboard_delete_local(self, local_env, tmp_path):
        payload = {
            "name": "to-delete",
            "visibility": "private",
            "ranking": {"primary_metric": "reward", "aggregation": "weighted_mean"},
        }
        payload_file = tmp_path / "lb2.json"
        payload_file.write_text(json.dumps(payload))

        create_result = _invoke(
            ["leaderboard", "create", "--local", "--file", str(payload_file)],
            env=local_env,
        )
        lb_id = json.loads(create_result.output)["id"]

        delete_result = _invoke(["leaderboard", "delete", lb_id, "--local"], env=local_env)
        assert delete_result.exit_code == 0, delete_result.output
        assert lb_id in delete_result.output or "deleted" in delete_result.output

    def test_verifier_create_requires_file(self, local_env):
        result = _invoke(["verifier", "create", "--local"], env=local_env)
        assert result.exit_code != 0

    def test_run_submit_requires_file(self, local_env):
        result = _invoke(["run", "submit", "--local"], env=local_env)
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Dispatch: server-unreachable auto-fallback
# ---------------------------------------------------------------------------


class TestAutoFallback:
    """When --local is not given but server is unreachable, falls back in-process."""

    def test_unreachable_server_falls_back(self, local_env):
        """Connectivity probe fails → _LocalBackend is used transparently."""
        import httpx

        with patch.object(httpx.Client, "head", side_effect=httpx.ConnectError("refused")):
            result = _invoke(
                ["discover", "metric-kinds", "--server", "http://localhost:19999"],
                env=local_env,
            )
        assert result.exit_code == 0, result.output
        # Fallback message on stderr
        data = json.loads(result.output)
        assert "items" in data

    def test_local_flag_skips_connectivity_probe(self, local_env):
        """--local never tries to connect, even if httpx would raise."""
        import httpx

        with patch.object(httpx.Client, "head", side_effect=AssertionError("should not be called")):
            result = _invoke(
                ["discover", "aggregations", "--local", "--server", "http://localhost:19999"],
                env=local_env,
            )
        assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# JSON output format
# ---------------------------------------------------------------------------


class TestOutputFormat:
    """Commands print valid, pretty-printed JSON."""

    def test_output_is_valid_json(self, local_env):
        result = _invoke(["discover", "aggregations", "--local"], env=local_env)
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert isinstance(parsed, dict)

    def test_list_output_has_items_key(self, local_env):
        result = _invoke(["env", "list", "--local"], env=local_env)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "items" in data
        assert isinstance(data["items"], list)
