"""Tests for feature-bus → FeatureStore monthly merge."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.feature_store.bus_to_store_merge import (
    load_bus_feature_frame,
    merge_bus_to_store,
    normalize_bus_frame_to_bars,
)
from src.feature_store.feature_store import FeatureStore, FeatureStoreSpec


def test_normalize_bus_frame_collapses_sub_bar_ticks():
    start = pd.Timestamp("2026-07-01 00:00:00", tz="UTC")
    raw = pd.DataFrame(
        {
            "timestamp": [start + pd.Timedelta(minutes=15 * i) for i in range(8)],
            "srb_l3_breakout_age_decay": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9],
        }
    )
    out = normalize_bus_frame_to_bars(raw)
    assert len(out) == 1
    assert float(out["srb_l3_breakout_age_decay"].iloc[-1]) == pytest.approx(0.9)


def test_merge_bus_to_store_writes_monthly_partitions(tmp_path):
    bus = tmp_path / "bus"
    feat = bus / "features" / "120T"
    feat.mkdir(parents=True)
    start = pd.Timestamp("2026-07-10 00:00:00", tz="UTC")
    pd.DataFrame(
        {
            "timestamp": [start + pd.Timedelta(hours=2 * i) for i in range(4)],
            "srb_l3_breakout_age_decay": [0.1, 0.2, 0.3, 0.4],
            "ema_1200_position": [-0.05, -0.04, -0.03, -0.02],
        }
    ).to_parquet(feat / "BTCUSDT.parquet", index=False)

    store_root = tmp_path / "feature_store"
    summary = merge_bus_to_store(
        bus_root=bus,
        store_root=store_root,
        symbols=["BTCUSDT"],
        layer="features_srb_120T_test",
        lookback_days=0,
    )
    res = summary["results"][0]
    assert res["skipped"] is False
    assert res["months"] == 1
    assert res["rows"] == 4

    store = FeatureStore(str(store_root))
    spec = FeatureStoreSpec(
        layer="features_srb_120T_test", symbol="BTCUSDT", timeframe="120T"
    )
    month = store.read_month(spec, "2026-07")
    assert "srb_l3_breakout_age_decay" in month.columns
    assert len(month) >= 4


def test_merge_bus_merges_into_existing_month(tmp_path):
    bus = tmp_path / "bus"
    feat = bus / "features" / "120T"
    feat.mkdir(parents=True)
    layer = "features_srb_120T_test"
    fs_sym = tmp_path / "feature_store" / layer / "BTCUSDT" / "120T"
    fs_sym.mkdir(parents=True)
    old = pd.DataFrame(
        {"srb_l3_breakout_age_decay": [0.01]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-07-01 00:00:00")]),
    )
    old.to_parquet(fs_sym / "2026-07.parquet")

    pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-07-10 02:00:00", tz="UTC")],
            "srb_l3_breakout_age_decay": [0.88],
        }
    ).to_parquet(feat / "BTCUSDT.parquet", index=False)

    merge_bus_to_store(
        bus_root=bus,
        store_root=tmp_path / "feature_store",
        symbols=["BTCUSDT"],
        layer=layer,
    )
    store = FeatureStore(str(tmp_path / "feature_store"))
    spec = FeatureStoreSpec(layer=layer, symbol="BTCUSDT", timeframe="120T")
    month = store.read_month(spec, "2026-07")
    assert float(
        month.loc[pd.Timestamp("2026-07-01"), "srb_l3_breakout_age_decay"]
    ) == pytest.approx(0.01)
    assert float(
        month.loc[pd.Timestamp("2026-07-10 02:00:00"), "srb_l3_breakout_age_decay"]
    ) == pytest.approx(0.88)


def test_load_bus_returns_empty_when_missing(tmp_path):
    df = load_bus_feature_frame(tmp_path / "missing", "BTCUSDT")
    assert df.empty
