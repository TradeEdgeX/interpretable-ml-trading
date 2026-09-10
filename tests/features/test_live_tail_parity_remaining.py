"""Parity: live bounded-tail vs full-history for remaining hot paths.

Covers percentile helpers, volume participation, box_structure, and orderflow
bar-tail slicing. Live only consumes the last row of each snapshot.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.live_tail_context import live_feature_tail
from src.features.time_series.baseline_features import (
    compute_atr_percentile_from_series,
    compute_percentile_rank_from_series,
)
from src.features.time_series.box_structure_features import (
    compute_box_structure_from_series,
)
from src.features.time_series.utils_interaction_features import (
    compute_volume_participation_score_from_series,
)
from src.features.time_series.utils_order_flow_features import (
    extract_order_flow_features,
)


def _make_bars(n: int = 1800, seed: int = 21) -> pd.DataFrame:
    np.random.seed(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="2h")
    rets = np.random.randn(n) * 0.004
    close = 100 * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(np.random.randn(n) * 0.002))
    low = close * (1 - np.abs(np.random.randn(n) * 0.002))
    return pd.DataFrame(
        {
            "open": close * (1 + np.random.randn(n) * 0.001),
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.uniform(1000, 50000, n),
        },
        index=idx,
    )


def _make_ticks(n: int = 20000, seed: int = 23) -> pd.DataFrame:
    np.random.seed(seed)
    # ~14 days of 1-min ticks so VPIN adaptive window has room
    idx = pd.date_range("2024-06-01", periods=n, freq="1min")
    side = np.where(np.random.rand(n) > 0.48, 1, -1)
    price = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.0002))
    return pd.DataFrame(
        {
            "price": price,
            "volume": np.random.uniform(0.1, 5.0, n),
            "side": side,
        },
        index=idx,
    )


def test_atr_percentile_tail_matches():
    """ATR is IIR (TA-Lib) → must be computed full-history then sliced. Assert a
    range of trailing rows (not just last) so any rank flip surfaces."""
    df = _make_bars(1800)
    full = compute_atr_percentile_from_series(
        high=df["high"], low=df["low"], close=df["close"], window=540
    )
    with live_feature_tail(300):
        bounded = compute_atr_percentile_from_series(
            high=df["high"], low=df["low"], close=df["close"], window=540
        )
    assert list(bounded.index) == list(full.index)
    # last `tail` rows must be bit-identical
    a = bounded["atr_percentile"].iloc[-250:].to_numpy(dtype=float)
    b = full["atr_percentile"].iloc[-250:].to_numpy(dtype=float)
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-12, equal_nan=True)


def test_percentile_rank_tail_matches():
    df = _make_bars(1800)
    series = df["volume"]
    full = compute_percentile_rank_from_series(
        series=series, window=540, output_name="pct"
    )
    with live_feature_tail(300):
        bounded = compute_percentile_rank_from_series(
            series=series, window=540, output_name="pct"
        )
    a = bounded["pct"].iloc[-250:].to_numpy(dtype=float)
    b = full["pct"].iloc[-250:].to_numpy(dtype=float)
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-12, equal_nan=True)


def test_volume_participation_tail_matches():
    df = _make_bars(1800)
    full = compute_volume_participation_score_from_series(volume=df["volume"])
    with live_feature_tail(300):
        bounded = compute_volume_participation_score_from_series(volume=df["volume"])
    for col in [
        "volume_participation_score",
        "volume_activity_pct",
        "volume_velocity_pct",
        "volume_stability",
    ]:
        a = bounded[col].iloc[-250:].to_numpy(dtype=float)
        b = full[col].iloc[-250:].to_numpy(dtype=float)
        # velocity uses EWM(span=5); fully converged over the tail → tight atol
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-9, equal_nan=True, err_msg=col)


def test_box_structure_tail_matches():
    """Live chop_grid consumes box_pos_60; long windows may be skipped under tail."""
    df = _make_bars(1800)
    full = compute_box_structure_from_series(
        close=df["close"], high=df["high"], low=df["low"]
    )
    with live_feature_tail(300):
        bounded = compute_box_structure_from_series(
            close=df["close"], high=df["high"], low=df["low"]
        )
    assert list(bounded.index) == list(full.index)
    # windows <= tail(300) must be exact on the trailing region live consumes
    for col in [
        "box_pos_60",
        "box_pos_120",
        "box_pos_240",
        "box_width_pct_120",
        "box_compression_score",
        "box_stability_120",
    ]:
        a = bounded[col].iloc[-120:].to_numpy(dtype=float)
        b = full[col].iloc[-120:].to_numpy(dtype=float)
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-9, equal_nan=True, err_msg=col)
    # regime label last row matches
    assert bounded["box_regime_label"].iloc[-1] == full["box_regime_label"].iloc[-1]
    # Contract columns for skipped long windows still present
    assert "box_pos_1200" in bounded.columns
    assert "box_pos_480" in bounded.columns


def test_orderflow_bar_tail_last_row_vpin_matches():
    """Bar-tail only: ticks stay full (VPIN buckets are path-dependent)."""
    bars = _make_bars(400)
    ticks = _make_ticks(12000)
    bars = bars.copy()
    bars.index = pd.date_range(
        ticks.index[-1] - pd.Timedelta(hours=2 * 399), periods=400, freq="2h"
    )

    full = extract_order_flow_features(
        bars,
        ticks=ticks,
        freq="2h",
        include_trade_clustering=True,
        compute_vpin_derived=True,
    )
    with live_feature_tail(120):
        bounded = extract_order_flow_features(
            bars,
            ticks=ticks,
            freq="2h",
            include_trade_clustering=True,
            compute_vpin_derived=True,
        )
    assert list(bounded.index) == list(full.index)
    for col in ["vpin", "vpin_signed_imbalance"]:
        if col not in full.columns:
            continue
        a = float(bounded[col].iloc[-1])
        b = float(full[col].iloc[-1])
        assert np.isclose(
            a, b, rtol=0, atol=1e-9, equal_nan=True
        ), f"{col}: bounded={a} full={b}"
