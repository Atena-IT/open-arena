# License Apache 2.0: (c) 2026 Athena-Reply
"""Tests for the ``arena eval submit`` CLI command (P2-1, issue #63).

Uses Click's CliRunner to exercise the command without a running server.
The ``--local`` flag exercises the in-process path (ArenaAPIService),
which is patched to avoid actual execution.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest
from click.testing import CliRunner

from open_arena_cli.main import main


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_MINIMAL_MANIFEST = textwrap.dedent("""\
    apiVersion: open-arena/v1
    kind: EvalEnvironment
    metadata:
      name: cli-test-eval
      version: 0.1.0
    dataset:
      source: inline
      ref: test-tasks
    verifier:
      type: exact_match
""")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_eval_submit_requires_file_or_dir():
    """arena eval submit without --file or --dir prints a usage error."""
    runner = CliRunner()
    result = runner.invoke(main, ["eval", "submit", "--local"])
    assert result.exit_code != 0
    assert "Either --file or --dir" in result.output or "UsageError" in result.output or result.exception is not None


def test_eval_submit_rejects_both_file_and_dir(tmp_path):
    """arena eval submit with both --file and --dir is rejected."""
    p = tmp_path / "eval.yaml"
    p.write_text(_MINIMAL_MANIFEST, encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(main, [
        "eval", "submit", "--local",
        "--file", str(p),
        "--dir", str(tmp_path),
    ])
    assert result.exit_code != 0


def test_eval_submit_local_via_file(tmp_path, monkeypatch):
    """arena eval submit --local --file eval.yaml calls svc.create_run with a RunCreate."""
    from unittest.mock import MagicMock, patch
    from open_arena_core import models as api

    p = tmp_path / "eval.yaml"
    p.write_text(_MINIMAL_MANIFEST, encoding="utf-8")

    # Patch the DB so we use a tmp path
    monkeypatch.setenv("OPEN_ARENA_DB_PATH", str(tmp_path / "test.db"))

    captured = {}

    def fake_create_run(self, run_create):
        captured["run_create"] = run_create
        from datetime import datetime, timezone
        from uuid import uuid4
        now = datetime.now(timezone.utc)
        return api.Run(
            id=uuid4(),
            mode=api.RunMode.generator,
            selection=run_create.selection,
            status=api.RunStatus.queued,
            cache_status=api.CacheStatus.pending,
            created_at=now,
        )

    runner = CliRunner()
    with patch("src.api.service.ArenaAPIService.create_run", fake_create_run):
        result = runner.invoke(main, [
            "eval", "submit", "--local",
            "--file", str(p),
        ])

    assert result.exit_code == 0, f"exit_code={result.exit_code}\noutput={result.output}"
    assert "run_create" in captured
    rc = captured["run_create"]
    assert isinstance(rc, api.GeneratorRunCreate)
    assert rc.mode == "generator"
    assert rc.labels["manifest_name"] == "cli-test-eval"


def test_eval_submit_local_via_dir(tmp_path, monkeypatch):
    """arena eval submit --local --dir ./my-eval picks up eval.yaml inside."""
    from unittest.mock import patch
    from open_arena_core import models as api

    eval_dir = tmp_path / "my-eval"
    eval_dir.mkdir()
    (eval_dir / "eval.yaml").write_text(_MINIMAL_MANIFEST, encoding="utf-8")

    monkeypatch.setenv("OPEN_ARENA_DB_PATH", str(tmp_path / "test.db"))

    captured = {}

    def fake_create_run(self, run_create):
        captured["run_create"] = run_create
        from datetime import datetime, timezone
        from uuid import uuid4
        now = datetime.now(timezone.utc)
        return api.Run(
            id=uuid4(),
            mode=api.RunMode.generator,
            selection=run_create.selection,
            status=api.RunStatus.queued,
            cache_status=api.CacheStatus.pending,
            created_at=now,
        )

    runner = CliRunner()
    with patch("src.api.service.ArenaAPIService.create_run", fake_create_run):
        result = runner.invoke(main, [
            "eval", "submit", "--local",
            "--dir", str(eval_dir),
        ])

    assert result.exit_code == 0, f"exit_code={result.exit_code}\noutput={result.output}"
    assert "run_create" in captured
    assert captured["run_create"].labels["manifest_name"] == "cli-test-eval"