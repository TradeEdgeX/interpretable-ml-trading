import numpy as np
import pandas as pd

from src.features.time_series.utils_interaction_features import (
    compute_vpin_scene_semantic_scores_from_series,
)
from src.features.time_series.utils_order_flow_features import (
    compute_vpin_signed_zscore_features_from_series,
    compute_vpin_zscore_features_from_series,
)


def _toy_ohlcv(n: int = 200) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="H")
    close = pd.Series(
        100 + np.cumsum(np.random.default_rng(7).normal(0, 0.2, size=n)), index=idx
    )
    open_ = close.shift(1).fillna(close.iloc[0])
    high = pd.concat([open_, close], axis=1).max(axis=1) + 0.1
    low = pd.concat([open_, close], axis=1).min(axis=1) - 0.1
    volume = pd.Series(
        1000 + np.random.default_rng(8).normal(0, 50, size=n), index=idx
    ).clip(lower=1.0)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )


def test_vpin_zscores_are_finite():
    df = _toy_ohlcv(260)
    # Make a synthetic "vpin" proxy that is already scale-free (0..1-ish)
    vpin = (
        (df["close"].pct_change().abs().rolling(10, min_periods=1).mean())
        .clip(0, 1)
        .fillna(0.0)
    )
    out = compute_vpin_zscore_features_from_series(vpin=vpin)
    assert set(out.columns) == {"vpin_zscore_20", "vpin_zscore_50"}
    assert np.isfinite(out.fillna(0.0).to_numpy()).all()


def test_vpin_signed_zscores_are_finite():
    df = _toy_ohlcv(260)
    signed = df["close"].pct_change().fillna(0.0)
    out = compute_vpin_signed_zscore_features_from_series(vpin_signed_imbalance=signed)
    assert set(out.columns) == {
        "vpin_signed_imbalance_zscore_20",
        "vpin_signed_imbalance_zscore_50",
    }
    assert np.isfinite(out.fillna(0.0).to_numpy()).all()


def test_vpin_scene_semantic_scores_are_bounded_0_1():
    df = _toy_ohlcv(260)
    # Provide z-score-like inputs
    vpin_z50 = pd.Series(
        np.random.default_rng(9).normal(0, 1, size=len(df)), index=df.index
    )
    vpin_signed_z50 = pd.Series(
        np.random.default_rng(10).normal(0, 1, size=len(df)), index=df.index
    )

    # Context features already normalized by contract in this repo
    atr = pd.Series(1.0, index=df.index)
    compression_score = pd.Series(0.5, index=df.index)
    dist_to_nearest_sr = pd.Series(0.002, index=df.index)  # pct distance
    volume_anomaly = pd.Series(0.0, index=df.index)
    trend_r2_20 = pd.Series(0.5, index=df.index)

    out = compute_vpin_scene_semantic_scores_from_series(
        vpin_zscore_50=vpin_z50,
        vpin_signed_imbalance_zscore_50=vpin_signed_z50,
        open=df["open"],
        close=df["close"],
        high=df["high"],
        low=df["low"],
        atr=atr,
        compression_score=compression_score,
        dist_to_nearest_sr=dist_to_nearest_sr,
        volume_anomaly=volume_anomaly,
        trend_r2_20=trend_r2_20,
        clip_z=5.0,
        disp_atr_threshold=0.5,
        sr_prox_atr=1.5,
    )

    expected = {
        "vpin_compression_score",
        "vpin_ignition_score",
        "vpin_absorption_score",
        "vpin_exhaustion_scene_score",
    }
    assert set(out.columns) == expected
    vals = out.fillna(0.0).to_numpy()
    assert np.isfinite(vals).all()
    assert (vals >= 0.0 - 1e-9).all()
    assert (vals <= 1.0 + 1e-9).all()
