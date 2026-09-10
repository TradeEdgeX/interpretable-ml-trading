import numpy as np
import pandas as pd
import pytest

from src.features.time_series.baseline_features import (
    compute_path_curvature_from_series,
    compute_volatility_cone_position_from_series,
)


def _make_price_from_returns(returns: pd.Series, start_price: float) -> pd.Series:
    returns = returns.fillna(0.0).astype(float)
    price = (1.0 + returns).cumprod() * float(start_price)
    return price


def test_volatility_cone_position_bounded_and_causal_truncate():
    # Synthetic returns with regime changes
    n = 1200
    idx = pd.RangeIndex(n)
    r = pd.Series(0.0, index=idx, dtype=float)
    r.iloc[200:400] = 0.01 * np.random.default_rng(0).standard_normal(200)
    r.iloc[600:800] = 0.03 * np.random.default_rng(1).standard_normal(200)
    close = _make_price_from_returns(r, start_price=100.0)

    out_full = compute_volatility_cone_position_from_series(
        close=close, window=20, lookback=252, min_periods=252
    )["volatility_cone_position"]

    # Bounded [0,1] (allow tiny numerical slop)
    assert out_full.dropna().between(-1e-9, 1.0 + 1e-9).all()

    # Causality sanity: computing on truncated history should match at the last available index
    t = 900
    out_trunc = compute_volatility_cone_position_from_series(
        close=close.iloc[: t + 1], window=20, lookback=252, min_periods=252
    )["volatility_cone_position"]
    assert np.isfinite(out_full.iloc[t])
    assert float(out_full.iloc[t]) == pytest.approx(float(out_trunc.iloc[t]), abs=1e-12)


def test_volatility_cone_position_timeframe_aware_lookback_days_infers_from_datetimeindex():
    # 4H bars: 6 bars/day. lookback_days=10 => 60 bars.
    n = 400
    idx = pd.date_range("2024-01-01", periods=n, freq="4H")
    rng = np.random.default_rng(123)
    r = pd.Series(0.01 * rng.standard_normal(n), index=idx, dtype=float)
    close = _make_price_from_returns(r, start_price=100.0)

    out = compute_volatility_cone_position_from_series(
        close=close, window=20, lookback=None, lookback_days=10, min_periods=60
    )["volatility_cone_position"]

    # Should be bounded and finite after warmup.
    tail = out.iloc[-100:]
    assert tail.dropna().between(0.0, 1.0).all()


def test_path_curvature_cross_asset_comparable_and_causal_truncate():
    n = 1200
    idx = pd.RangeIndex(n)
    rng = np.random.default_rng(42)
    r = pd.Series(0.01 * rng.standard_normal(n), index=idx, dtype=float)
    close_a = _make_price_from_returns(r, start_price=100.0)
    close_b = _make_price_from_returns(
        r, start_price=50000.0
    )  # same returns, different price scale

    a = compute_path_curvature_from_series(close=close_a)["path_curvature"]
    b = compute_path_curvature_from_series(close=close_b)["path_curvature"]

    # Should be very close because returns are identical (unitless definition)
    # Ignore early warmup where rolling stats stabilize.
    start = 400
    diff = (a.iloc[start:] - b.iloc[start:]).abs()
    assert float(diff.max()) < 1e-6

    # Causality sanity (truncate)
    t = 900
    a_full_t = float(a.iloc[t])
    a_trunc = compute_path_curvature_from_series(close=close_a.iloc[: t + 1])[
        "path_curvature"
    ]
    assert a_full_t == pytest.approx(float(a_trunc.iloc[t]), abs=1e-12)
