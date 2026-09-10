"""Unit tests for Vision bookDepth downloader."""

from __future__ import annotations

import io
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd

from src.data_tools.download_book_depth_vision import (
    BookDepthVisionDownloader,
    _aggregate_book_depth_csv,
    _normalize_symbol,
)


def test_normalize_symbol():
    assert _normalize_symbol("btc") == "BTCUSDT"


def test_aggregate_book_depth_csv_one_timestamp():
    df = pd.DataFrame(
        {
            "timestamp": ["2025-06-01 10:00:00"] * 10,
            "percentage": [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5],
            "depth": [1.0] * 10,
            "notional": [1e6, 2e6, 3e6, 4e6, 5e6, 1e6, 2e6, 3e6, 4e6, 6e6],
        }
    )
    out = _aggregate_book_depth_csv(df, symbol="BTCUSDT")
    assert len(out) == 1
    assert out["wall_bid_notional_usd_max"].iloc[0] == 5e6
    assert out["wall_ask_notional_usd_max"].iloc[0] == 6e6
    assert out["wall_bid_pct_band"].iloc[0] == -1
    assert out["wall_ask_pct_band"].iloc[0] == 5


def test_zip_to_parquet_roundtrip(tmp_path):
    csv = (
        "timestamp,percentage,depth,notional\n"
        "2025-06-01 10:00:00,-1,100,5000000\n"
        "2025-06-01 10:00:00,1,50,3000000\n"
        "2025-06-01 10:00:31,-2,200,8000000\n"
        "2025-06-01 10:00:31,2,80,2000000\n"
    )
    zpath = tmp_path / "BTCUSDT-bookDepth-2025-06-01.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("BTCUSDT-bookDepth-2025-06-01.csv", csv)

    dl = BookDepthVisionDownloader(
        data_dir=tmp_path / "zip",
        parquet_dir=tmp_path / "parquet",
    )
    out = dl._zip_to_parquet(zpath, symbol="BTCUSDT")
    assert len(out) == 2
    assert out["wall_bid_notional_usd_max"].max() == 8_000_000.0
