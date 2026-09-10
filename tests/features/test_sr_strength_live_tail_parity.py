"""Parity test: live bounded-tail SR strength == full-history (last rows).

``compute_sr_strength_max_from_series`` is a strictly-causal per-bar-loop feature
(internal window=60). Live only consumes the last row, so under an active
``live_feature_tail`` context it computes only the last ``tail`` bars. This must
be **bit-identical** to full-history compute for the last output rows (which is
all any downstream consumer / the live snapshot reads).

If this parity ever breaks, the live SR-based signals (SRB / DTW / FER) would
silently diverge from backtest — hence exact equality is asserted.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.time_series.baseline_features import (
    compute_poc_hal_features_from_series,
    compute_sr_strength_max_from_series,
)
from src.features.live_tail_context import live_feature_tail, get_live_feature_tail


def _make_ohlc(n: int = 800, seed: int = 7) -> pd.DataFrame:
    np.random.seed(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="2h")
    returns = np.random.randn(n) * 0.005
    prices = 100 * np.exp(np.cumsum(returns))
    # inject range/mean-reversion so POC/HAL boundaries are meaningful
    for i in range(n // 3, n, n // 3):
        if i < n:
            prices[i:] += (prices[:i].mean() - prices[i]) * 0.3
    high = prices + np.abs(np.random.randn(n) * 0.003 * prices) + 0.001 * prices
    low = prices - np.abs(np.random.randn(n) * 0.003 * prices) - 0.001 * prices
    df = pd.DataFrame(
        {
            "open": prices * (1 + np.random.randn(n) * 0.001),
            "high": high,
            "low": low,
            "close": prices,
            "volume": np.random.uniform(1000, 10000, n),
        },
        index=dates,
    )
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr"] = tr.rolling(window=14, min_periods=1).mean().clip(lower=1e-6)
    return df


def _sr_inputs(df: pd.DataFrame) -> dict:
    poc_hal = compute_poc_hal_features_from_series(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        volume=df["volume"],
        poc_window=160,
    )
    return dict(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        atr=df["atr"],
        poc=poc_hal["poc"],
        hal_high=poc_hal["hal_high"],
        hal_low=poc_hal["hal_low"],
    )


def test_live_tail_matches_full_history_last_rows():
    df = _make_ohlc(800)
    kwargs = _sr_inputs(df)

    full = compute_sr_strength_max_from_series(**kwargs)
    assert get_live_feature_tail() is None  # context restored / not leaking

    tail = 300
    with live_feature_tail(tail):
        assert get_live_feature_tail() == tail
        bounded = compute_sr_strength_max_from_series(**kwargs)
    assert get_live_feature_tail() is None  # restored on exit

    # same shape / index as full history (older rows just zero-filled)
    assert list(bounded.index) == list(full.index)
    assert list(bounded.columns) == list(full.columns)

    # last (tail - internal_window - lookback margin) rows must be EXACT.
    # internal window=60, price-action lookbacks<=20, output shift(1) → margin ~90.
    n_exact = tail - 100  # 200 rows, comfortably beyond any downstream lookback
    for col in ["sr_strength_max", "dist_to_nearest_sr", "direction_to_nearest_sr"]:
        a = full[col].iloc[-n_exact:].to_numpy()
        b = bounded[col].iloc[-n_exact:].to_numpy()
        np.testing.assert_array_equal(
            b, a, err_msg=f"tail-compute diverged from full history on {col}"
        )


def test_live_tail_disabled_is_full_history():
    """tail=None / <=0 must be a no-op (identical to full history everywhere)."""
    df = _make_ohlc(500)
    kwargs = _sr_inputs(df)

    full = compute_sr_strength_max_from_series(**kwargs)
    with live_feature_tail(None):
        none_ctx = compute_sr_strength_max_from_series(**kwargs)
    with live_feature_tail(0):
        zero_ctx = compute_sr_strength_max_from_series(**kwargs)

    pd.testing.assert_frame_equal(none_ctx, full)
    pd.testing.assert_frame_equal(zero_ctx, full)


def test_live_tail_shorter_than_history_noop_when_series_short():
    """When history <= tail, all rows are computed (no slicing) → full parity."""
    df = _make_ohlc(200)
    kwargs = _sr_inputs(df)
    full = compute_sr_strength_max_from_series(**kwargs)
    with live_feature_tail(300):  # tail > len → no slice
        bounded = compute_sr_strength_max_from_series(**kwargs)
    pd.testing.assert_frame_equal(bounded, full)
