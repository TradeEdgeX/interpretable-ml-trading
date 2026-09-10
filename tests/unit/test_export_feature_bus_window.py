"""Unit tests for feature-bus window export."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from scripts.monitoring.export_feature_bus_window import export_feature_bus_window


def _write_bus_symbol(
    bus_root: Path,
    *,
    symbol: str,
    timeframe: str = "120T",
    n_recent: int = 5,
    n_old: int = 3,
) -> None:
    now = datetime.now(timezone.utc)
    rows = []
    for i in range(n_old):
        rows.append(
            {
                "timestamp": now - timedelta(days=10 + i, hours=1),
                "ema_1200_position": 0.1,
                "feat_a": float(i),
            }
        )
    for i in range(n_recent):
        rows.append(
            {
                "timestamp": now - timedelta(days=i, hours=2),
                "ema_1200_position": 0.2,
                "feat_a": 10.0 + i,
            }
        )
    df = pd.DataFrame(rows)
    out = bus_root / "features" / timeframe / f"{symbol}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)


def test_export_filters_lookback_and_concat_symbols(tmp_path):
    bus = tmp_path / "bus"
    _write_bus_symbol(bus, symbol="BTCUSDT", timeframe="120T")
    _write_bus_symbol(bus, symbol="ETHUSDT", timeframe="120T")
    out = tmp_path / "window.parquet"

    path = export_feature_bus_window(
        bus_root=bus,
        timeframe="120T",
        lookback_days=7,
        output=out,
        symbols="BTCUSDT,ETHUSDT",
    )

    df = pd.read_parquet(path)
    assert path == out
    assert set(df["symbol"].unique()) == {"BTCUSDT", "ETHUSDT"}
    assert len(df) == 10
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=7)
    assert (pd.to_datetime(df["timestamp"], utc=True) >= cutoff).all()


def test_export_lookback_zero_exports_all_rows(tmp_path):
    bus = tmp_path / "bus"
    _write_bus_symbol(bus, symbol="BTCUSDT", timeframe="120T", n_recent=2, n_old=4)
    out = tmp_path / "all.parquet"
    export_feature_bus_window(
        bus_root=bus,
        timeframe="120T",
        lookback_days=0,
        output=out,
        symbols="BTCUSDT",
    )
    df = pd.read_parquet(out)
    assert len(df) == 6


def test_export_raises_when_no_rows_in_window(tmp_path):
    bus = tmp_path / "bus"
    now = datetime.now(timezone.utc)
    df = pd.DataFrame(
        {
            "timestamp": [now - timedelta(days=30)],
            "ema_1200_position": [0.1],
        }
    )
    p = bus / "features" / "120T" / "BTCUSDT.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)

    with pytest.raises(FileNotFoundError, match="no bus rows"):
        export_feature_bus_window(
            bus_root=bus,
            timeframe="120T",
            lookback_days=7,
            output=tmp_path / "out.parquet",
            symbols="BTCUSDT",
        )


def test_export_lists_bus_symbols_when_explicit_empty(tmp_path):
    bus = tmp_path / "bus"
    _write_bus_symbol(bus, symbol="SOLUSDT", timeframe="120T", n_recent=4, n_old=0)
    out = tmp_path / "out.parquet"
    export_feature_bus_window(
        bus_root=bus,
        timeframe="120T",
        lookback_days=7,
        output=out,
        symbols=None,
    )
    df = pd.read_parquet(out)
    assert list(df["symbol"].unique()) == ["SOLUSDT"]
