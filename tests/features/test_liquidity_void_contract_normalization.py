import numpy as np
import pandas as pd

from src.features.time_series.utils_liquidity_features import (
    compute_liquidity_void_features_from_series,
)


def test_liquidity_void_outputs_are_finite_and_reasonably_bounded():
    n = 240
    idx = pd.date_range("2024-01-01", periods=n, freq="H")
    close = pd.Series(
        100 + np.cumsum(np.random.default_rng(1).normal(0, 0.2, size=n)), index=idx
    )
    volume = pd.Series(
        1000 + np.random.default_rng(2).normal(0, 50, size=n), index=idx
    ).clip(lower=1.0)
    atr = pd.Series(1.0, index=idx)

    out = compute_liquidity_void_features_from_series(
        close=close, volume=volume, atr=atr, lookback_window=20
    )
    for c in [
        "liquidity_void_detected",
        "liquidity_void_speed",
        "liquidity_void_volume_ratio",
        "liquidity_void_price_impact",
        "liquidity_void_retracement",
        "liquidity_void_false_breakout_risk",
    ]:
        assert c in out.columns
        vals = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
        assert np.isfinite(vals.to_numpy()).all()

    det = out["liquidity_void_detected"].fillna(0.0)
    assert ((det >= 0.0) & (det <= 1.0)).all()

    risk = out["liquidity_void_false_breakout_risk"].fillna(0.0)
    assert ((risk >= 0.0) & (risk <= 1.0)).all()
