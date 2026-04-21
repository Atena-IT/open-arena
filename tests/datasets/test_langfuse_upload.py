from __future__ import annotations

from unittest.mock import patch

from src.datasets.langfuse_upload import _warn_on_reserved_keys


def test_warn_on_reserved_keys_scans_all_rows():
    rows = [
        ("q1", "a1", {"topic": "safe"}),
        ("q2", "a2", {"lf_item_id": "item-2"}),
    ]

    with patch("src.datasets.langfuse_upload._logger.warning") as warning:
        _warn_on_reserved_keys(rows)

    warning.assert_called_once()
    assert "lf_item_id" in warning.call_args.args[0]
