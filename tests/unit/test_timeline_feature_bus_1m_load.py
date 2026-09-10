"""Regression: feature-bus 1m load must keep all symbols per minute."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.backtest_multileg_timeline import _load_1m_from_feature_bus


def test_load_1m_from_feature_bus_keeps_all_symbols_per_timestamp(
    tmp_path: Path,
) -> None:
    root = tmp_path / "shared_feature_bus"
    bars = root / "bars_1min"
    bars.mkdir(parents=True)
    idx = pd.date_range("2026-07-22", periods=3, freq="1min", tz="UTC")
    for i, sym in enumerate(("BTCUSDT", "ETHUSDT", "SOLUSDT")):
        df = pd.DataFrame(
            {
                "timestamp": idx,
                "open": 100.0 + i,
                "high": 101.0 + i,
                "low": 99.0 + i,
                "close": 100.5 + i,
                "volume": 1.0,
            }
        )
        df.to_parquet(bars / f"{sym}.parquet", index=False)

    out = _load_1m_from_feature_bus(
        root,
        ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        start=pd.Timestamp("2026-07-22", tz="UTC"),
        end=pd.Timestamp("2026-07-23", tz="UTC"),
    )
    assert len(out) == 9
    assert sorted(out["symbol"].unique()) == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    assert int(out.groupby(out.index).size().max()) == 3
