"""Minimal ashare daily helpers (court path, no aux Lab)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_tools.ashare_downloader import (
    is_ashare_daily_path,
    load_ashare_daily_bars,
    normalize_ashare_symbol,
)


def test_normalize_ashare_symbol():
    assert normalize_ashare_symbol("000300.SH") == "000300.SH"
    assert normalize_ashare_symbol("sh000300") == "000300.SH"
    assert normalize_ashare_symbol("600519") == "600519"


def test_load_ashare_daily_bars_roundtrip(tmp_path: Path):
    raw = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "symbol": ["000300.SH", "000300.SH"],
            "open": [1.0, 1.1],
            "high": [1.2, 1.3],
            "low": [0.9, 1.0],
            "close": [1.1, 1.25],
            "volume": [100, 110],
        }
    )
    out = tmp_path / "000300.SH.parquet"
    raw.to_parquet(out, index=False)
    assert is_ashare_daily_path(tmp_path, "000300.SH")
    bars = load_ashare_daily_bars(tmp_path, "000300.SH")
    assert len(bars) == 2
    assert bars["close"].tolist() == [1.1, 1.25]
    assert bars.index.tz is not None
