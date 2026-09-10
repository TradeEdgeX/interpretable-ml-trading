import tempfile
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def test_footprint_monthly_cache_hits_avoid_reloading_ticks(monkeypatch):
    """
    Footprint is tick-heavy. When using ticks_loader_json, we should cache per-month outputs on disk
    so repeated runs (multi-seed / feature-group-search steps) don't reload ticks again.
    """
    tmp_root = Path(tempfile.mkdtemp(prefix="fp_monthly_cache_"))
    try:
        ticks_dir = tmp_root / "ticks"
        cache_dir = tmp_root / "cache"
        ticks_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)

        symbol = "BTCUSDT"

        def _write_month(month_str: str, start: str, periods: int):
            ts = pd.date_range(start=start, periods=periods, freq="1min")
            df = pd.DataFrame(
                {
                    "timestamp": ts,
                    "price": 100.0 + np.linspace(0, 1, len(ts)),
                    "volume": np.ones(len(ts)),
                    "side": np.where(np.arange(len(ts)) % 2 == 0, 1, -1),
                }
            )
            df.to_parquet(ticks_dir / f"{symbol}_{month_str}.parquet", index=False)

        _write_month("2024-01", start="2024-01-31 20:00:00", periods=300)
        _write_month("2024-02", start="2024-02-01 00:00:00", periods=300)

        # Bars span both months; footprint will be computed for each bar by slicing ticks.
        bar_index = pd.date_range("2024-01-31 22:00:00", periods=12, freq="1H")
        ohlcv = pd.DataFrame(
            {
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 1000.0,
            },
            index=bar_index,
        )

        from src.data_tools import tick_loader
        from src.data_tools.tick_loader import serialize_tick_loader_params
        from src.features.loader import feature_wrappers as fw

        payload = serialize_tick_loader_params(
            {
                "symbol": symbol,
                "tick_files": [
                    str(ticks_dir / f"{symbol}_2024-01.parquet"),
                    str(ticks_dir / f"{symbol}_2024-02.parquet"),
                ],
                "start_ts": "2024-01-31 22:00:00",
                "end_ts": "2024-02-01 10:00:00",
                "lookback_minutes": 60,
                "ticks_dir": str(ticks_dir),
            }
        )

        # Count tick loads (patch the canonical loader).
        real_load = tick_loader.load_tick_data
        calls = {"n": 0}

        def _counting_load(*args, **kwargs):
            calls["n"] += 1
            return real_load(*args, **kwargs)

        monkeypatch.setattr(tick_loader, "load_tick_data", _counting_load)

        # First run: should load ticks for the involved months and write cache.
        calls["n"] = 0
        out1 = fw.compute_footprint_features(
            ohlcv,
            ticks=None,
            ticks_loader_json=payload,
            monthly_cache_dir=str(cache_dir),
            persist_monthly=True,
        )
        assert "fp_poc" in out1.columns
        # Must be unitless ATR-distance, not raw price level (~100)
        assert pd.to_numeric(out1["fp_poc"], errors="coerce").abs().max() < 20
        assert calls["n"] >= 1

        # Second run: should hit cache and NOT call load_tick_data again.
        calls["n"] = 0
        out2 = fw.compute_footprint_features(
            ohlcv,
            ticks=None,
            ticks_loader_json=payload,
            monthly_cache_dir=str(cache_dir),
            persist_monthly=True,
        )
        assert "fp_poc" in out2.columns
        assert pd.to_numeric(out2["fp_poc"], errors="coerce").abs().max() < 20
        assert calls["n"] == 0
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
