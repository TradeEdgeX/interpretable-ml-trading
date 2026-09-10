"""Unit tests for labeled parquet catalog."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.monitoring.catalog_labeled_parquets import (
    discover_parquets,
    format_table,
    summarize_parquet,
)


def test_discover_parquets_filters_strategy(tmp_path: Path):
    tpc_dir = tmp_path / "train_final" / "tpc" / "run1" / "tpc"
    bpc_dir = tmp_path / "train_final" / "bpc" / "run1" / "bpc"
    tpc_dir.mkdir(parents=True)
    bpc_dir.mkdir(parents=True)
    (tpc_dir / "features_labeled.parquet").write_bytes(b"x")
    (bpc_dir / "features_labeled.parquet").write_bytes(b"y")

    all_paths = discover_parquets(tmp_path, limit=0)
    assert len(all_paths) == 2

    tpc_only = discover_parquets(tmp_path, strategy="tpc", limit=0)
    assert len(tpc_only) == 1
    assert "tpc" in tpc_only[0].as_posix()


def test_summarize_parquet_metadata(tmp_path: Path):
    path = tmp_path / "features_labeled.parquet"
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-10-01", periods=5, freq="2h", tz="UTC"),
            "symbol": ["BTCUSDT", "ETHUSDT", "BTCUSDT", "ETHUSDT", "BTCUSDT"],
            "forward_rr": [0.1, -0.2, 0.3, 0.0, 0.5],
            "ema_1200_position": [-0.1, 0.2, 0.0, -0.3, 0.1],
        }
    )
    df.to_parquet(path, index=False)

    meta = summarize_parquet(path)
    assert meta["rows"] == 5
    assert meta["has_forward_rr"] is True
    assert meta["has_ema_1200_position"] is True
    assert meta["n_symbols"] == 2
    assert "BTCUSDT" in meta["symbols"]
    assert meta["time_start"] is not None


def test_format_table_empty():
    assert "No features_labeled" in format_table([])
