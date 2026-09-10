"""Unit tests for Vision OI downloader helpers."""

from __future__ import annotations

import io
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd

from src.data_tools.download_open_interest_vision import (
    OpenInterestVisionDownloader,
    _date_list,
    _normalize_symbol,
)


def test_normalize_symbol():
    assert _normalize_symbol("hype") == "HYPEUSDT"
    assert _normalize_symbol("ETHUSDT") == "ETHUSDT"


def test_date_list_inclusive():
    days = _date_list(date(2025, 5, 31), date(2025, 6, 2))
    assert days == [date(2025, 5, 31), date(2025, 6, 1), date(2025, 6, 2)]


def test_zip_to_oi_df_parses_vision_csv(tmp_path):
    csv = (
        "create_time,symbol,sum_open_interest,sum_open_interest_value,"
        "count_toptrader_long_short_ratio,sum_toptrader_long_short_ratio,"
        "count_long_short_ratio,sum_taker_long_short_vol_ratio\n"
        "2025-06-01 00:05:00,HYPEUSDT,100.0,2000000.0,1,1,1,1\n"
        "2025-06-01 00:10:00,HYPEUSDT,101.0,2020000.0,1,1,1,1\n"
    )
    zpath = tmp_path / "HYPEUSDT-metrics-2025-06-01.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("HYPEUSDT-metrics-2025-06-01.csv", csv)

    dl = OpenInterestVisionDownloader(
        data_dir=tmp_path / "zip",
        parquet_dir=tmp_path / "parquet",
    )
    df = dl._zip_to_oi_df(zpath, symbol="HYPEUSDT")
    assert len(df) == 2
    assert list(df.columns) == ["_symbol", "oi_contracts", "oi_usd"]
    assert df["oi_usd"].iloc[0] == 2_000_000.0
    assert str(df.index[0]) == "2025-06-01 00:05:00+00:00"


def test_merge_monthly_parquet_dedupes(tmp_path):
    dl = OpenInterestVisionDownloader(
        data_dir=tmp_path / "zip",
        parquet_dir=tmp_path / "parquet",
    )
    idx = pd.date_range("2025-06-01 00:05", periods=2, freq="5min", tz="UTC")
    old = pd.DataFrame(
        {"_symbol": "HYPEUSDT", "oi_contracts": [1.0, 2.0], "oi_usd": [100.0, 200.0]},
        index=idx,
    )
    ppath = dl._parquet_path("HYPEUSDT", 2025, 6)
    ppath.parent.mkdir(parents=True, exist_ok=True)
    old.to_parquet(ppath)

    new_idx = pd.date_range("2025-06-01 00:05", periods=2, freq="5min", tz="UTC")
    vision_df = pd.DataFrame(
        {"_symbol": "HYPEUSDT", "oi_contracts": [9.0, 3.0], "oi_usd": [900.0, 300.0]},
        index=new_idx,
    )
    merged = dl._merge_monthly_parquet("HYPEUSDT", 2025, 6, vision_df)
    assert len(merged) == 2
    assert merged["oi_usd"].iloc[-1] == 300.0
