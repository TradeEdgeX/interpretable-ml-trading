import numpy as np
import pandas as pd

from src.features.time_series.baseline_features import (
    compute_compression_duration_from_series,
    compute_compression_to_breakout_prob_from_series,
)


def test_compression_duration_is_bounded_and_non_degenerate():
    n = 400
    idx = pd.date_range("2024-01-01", periods=n, freq="H")
    rng = np.random.default_rng(0)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 0.2, size=n)), index=idx)
    high = close + np.abs(rng.normal(0.2, 0.05, size=n))
    low = close - np.abs(rng.normal(0.2, 0.05, size=n))

    out = compute_compression_duration_from_series(
        high=high, low=low, close=close, percentile_window=120
    )
    s = out.fillna(0.0)
    assert (s >= -1e-9).all()
    assert (s <= 1.0 + 1e-9).all()
    assert s.nunique() > 1


def test_compression_to_breakout_prob_is_bounded_0_1():
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="H")
    cd = pd.Series(np.linspace(0.0, 1.0, n), index=idx)
    roc = pd.Series(np.random.default_rng(1).normal(0, 1, size=n), index=idx)
    out = compute_compression_to_breakout_prob_from_series(
        compression_duration=cd, roc_5=roc
    )
    s = out["compression_to_breakout_prob"].fillna(0.0)
    assert (s >= -1e-9).all()
    assert (s <= 1.0 + 1e-9).all()
