"""Tests for depth wall aggregation and wall features (T5α Phase 1B)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_tools.depth_wall_aggregate import aggregate_walls_from_depth
from src.features.time_series.wall_features import compute_wall_features_from_df


def _synthetic_book(*, mid: float = 65_000.0, wall_qty: float = 800.0) -> tuple:
    """Bid wall ~738 BTC cluster below mid; thin asks above."""
    bids = [
        [str(mid - 500), str(wall_qty)],
        [str(mid - 1000), str(wall_qty * 0.5)],
        [str(mid - 50), "10"],
    ]
    asks = [
        [str(mid + 50), "10"],
        [str(mid + 500), "20"],
        [str(mid + 1000), "15"],
    ]
    return bids, asks


class TestDepthWallAggregate:
    def test_finds_bid_wall_notional(self):
        bids, asks = _synthetic_book()
        wall = aggregate_walls_from_depth(bids, asks, bucket_pct=0.005)
        assert wall.wall_bid_notional_usd_max > 50_000_000
        assert wall.wall_bid_price < wall.mid
        assert wall.wall_ask_notional_usd_max > 0
        assert wall.spread_bps > 0

    def test_empty_book_raises(self):
        with pytest.raises(ValueError):
            aggregate_walls_from_depth([], [["1", "1"]])


class TestWallFeaturesCausal:
    def _write_snapshots(self, path: Path, times_notional: list) -> None:
        rows = []
        for ts, bid_n in times_notional:
            rows.append(
                {
                    "datetime": ts,
                    "_symbol": "BTCUSDT",
                    "mid": 65000.0,
                    "spread_bps": 1.0,
                    "bucket_pct": 0.005,
                    "wall_bid_notional_usd_max": bid_n,
                    "wall_ask_notional_usd_max": bid_n * 0.5,
                    "wall_bid_price": 64500.0,
                    "wall_ask_price": 65500.0,
                    "best_bid": 64999.0,
                    "best_ask": 65001.0,
                    "depth_limit": 1000,
                }
            )
        df = pd.DataFrame(rows).set_index("datetime")
        df.index = pd.to_datetime(df.index, utc=True)
        df.to_parquet(path)

    def test_merge_asof_only_uses_past_snapshot(self, tmp_path):
        snap_path = tmp_path / "BTCUSDT_2026-06-15_depth_snap.parquet"
        self._write_snapshots(
            snap_path,
            [
                ("2026-06-15 08:00:00+00:00", 50e6),
                ("2026-06-15 12:00:00+00:00", 80e6),
            ],
        )

        bar_ts = pd.DatetimeIndex(["2026-06-15 10:00:00"], tz="UTC")
        bars = pd.DataFrame(
            {"close": [65000.0], "atr": [500.0], "_symbol": ["BTCUSDT"]},
            index=bar_ts,
        )
        out = compute_wall_features_from_df(bars, depth_dir=str(tmp_path))
        assert np.isclose(out["wall_bid_notional_usd_max"].iloc[0], 50e6)

        # Mutate future snapshot only — bar at 10:00 unchanged.
        self._write_snapshots(
            snap_path,
            [
                ("2026-06-15 08:00:00+00:00", 50e6),
                ("2026-06-15 12:00:00+00:00", 999e6),
            ],
        )
        out2 = compute_wall_features_from_df(bars, depth_dir=str(tmp_path))
        assert np.isclose(out2["wall_bid_notional_usd_max"].iloc[0], 50e6)

    def test_appending_future_bars_no_look_ahead(self, tmp_path):
        snap_path = tmp_path / "BTCUSDT_2026-06-15_depth_snap.parquet"
        self._write_snapshots(
            snap_path,
            [("2026-06-15 06:00:00+00:00", 40e6), ("2026-06-15 14:00:00+00:00", 90e6)],
        )
        idx = pd.date_range("2026-06-15 08:00", periods=8, freq="2h", tz="UTC")
        bars = pd.DataFrame(
            {
                "close": 65000.0 + np.arange(8) * 10,
                "atr": [500.0] * 8,
                "_symbol": "BTCUSDT",
            },
            index=idx,
        )
        full = compute_wall_features_from_df(bars, depth_dir=str(tmp_path))
        prefix = compute_wall_features_from_df(bars.iloc[:4], depth_dir=str(tmp_path))
        for col in ["wall_bid_notional_usd_max", "wall_nearest_dist_atr"]:
            pd.testing.assert_series_equal(
                prefix[col], full[col].iloc[:4], check_names=False
            )

    def test_ws_columns_nan_in_rest_phase(self, tmp_path):
        snap_path = tmp_path / "BTCUSDT_2026-06-15_depth_snap.parquet"
        self._write_snapshots(snap_path, [("2026-06-15 08:00:00+00:00", 50e6)])
        bars = pd.DataFrame(
            {
                "close": [65000.0],
                "atr": [500.0],
                "_symbol": ["BTCUSDT"],
            },
            index=pd.DatetimeIndex(["2026-06-15 10:00:00"], tz="UTC"),
        )
        out = compute_wall_features_from_df(bars, depth_dir=str(tmp_path))
        assert pd.isna(out["wall_persist_sec"].iloc[0])
        assert pd.isna(out["wall_cancel_rate_5m"].iloc[0])
        assert pd.isna(out["wall_eaten_ratio_1h"].iloc[0])

    def test_vision_pct_band_dist_atr(self, tmp_path):
        rows = [
            {
                "datetime": "2026-06-15 08:00:00+00:00",
                "_symbol": "BTCUSDT",
                "wall_bid_notional_usd_max": 50e6,
                "wall_ask_notional_usd_max": 30e6,
                "wall_bid_pct_band": -1.0,
                "wall_ask_pct_band": 2.0,
                "wall_bid_price": float("nan"),
                "wall_ask_price": float("nan"),
                "mid": float("nan"),
                "source": "vision_book_depth",
            }
        ]
        snap_path = tmp_path / "BTCUSDT_2026-06-15_book_depth.parquet"
        vdf = pd.DataFrame(rows).set_index("datetime")
        vdf.index = pd.to_datetime(vdf.index, utc=True)
        vdf.to_parquet(snap_path)

        bars = pd.DataFrame(
            {"close": [65000.0], "atr": [500.0], "_symbol": ["BTCUSDT"]},
            index=pd.DatetimeIndex(["2026-06-15 10:00:00"], tz="UTC"),
        )
        out = compute_wall_features_from_df(bars, depth_dir=str(tmp_path))
        # nearest wall at -1% → 650 price units → 650/500 = 1.3 ATR
        assert np.isclose(out["wall_nearest_dist_atr"].iloc[0], 1.3)
        assert np.isclose(out["wall_bid_price"].iloc[0], 65000.0 * 0.99)
