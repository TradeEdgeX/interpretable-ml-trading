"""Preflight contract for exported monitor parquet."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.monitoring.validate_monitor_window import validate_monitor_parquet


def test_validate_monitor_parquet_ok(tmp_path: Path) -> None:
    pq = tmp_path / "window.parquet"
    pd.DataFrame(
        {
            "ema_1200_position": [0.1, 0.2],
            "adx_50": [25.0, 30.0],
            "tpc_semantic_chop": [0.2, 0.25],
        }
    ).to_parquet(pq, index=False)
    report = validate_monitor_parquet(pq)
    assert report["ok"] is True


def test_validate_monitor_parquet_psi_override_skips_defaults(tmp_path: Path) -> None:
    """Explicit psi_features replaces TPC defaults (no retired vol_* required)."""
    from scripts.monitoring.validate_monitor_window import required_monitor_columns

    assert required_monitor_columns(psi_features=["ema_1200_position", "adx_50"]) == [
        "ema_1200_position",
        "adx_50",
    ]
    pq = tmp_path / "window.parquet"
    pd.DataFrame({"ema_1200_position": [0.1], "adx_50": [20.0]}).to_parquet(
        pq, index=False
    )
    report = validate_monitor_parquet(
        pq,
        required_columns=required_monitor_columns(
            psi_features=["ema_1200_position", "adx_50"]
        ),
    )
    assert report["ok"] is True


def test_validate_monitor_parquet_missing_column(tmp_path: Path) -> None:
    pq = tmp_path / "window.parquet"
    pd.DataFrame({"ema_1200_position": [0.1]}).to_parquet(pq, index=False)
    with pytest.raises(ValueError, match="MISSING_FEATURE"):
        validate_monitor_parquet(pq)
