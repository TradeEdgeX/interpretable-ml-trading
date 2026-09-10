"""Parity: live bounded-tail vs full-history for WPT / POC-HAL / OFCI tick path."""

import numpy as np
import pandas as pd
import pytest

from src.features.live_tail_context import live_feature_tail
from src.features.time_series.baseline_features import (
    compute_poc_hal_features_from_series,
    compute_sr_strength_max_from_series,
)
from src.features.time_series.utils_wpt_features import extract_wpt_features
from src.time_series_model.live.incremental_feature_computer import (
    IncrementalFeatureComputer,
)


def _make_bars(n: int = 800, seed: int = 11) -> pd.DataFrame:
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


def _make_ticks(n: int = 12000, seed: int = 13) -> pd.DataFrame:
    np.random.seed(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="1min")
    side = np.where(np.random.rand(n) > 0.48, 1, -1)
    return pd.DataFrame(
        {"side": side, "volume": np.random.uniform(0.1, 5.0, n)},
        index=idx,
    )


def test_wpt_tail_matches_full_history_last_rows():
    """Tail WPT matches full-history on the live-read window (last row + margin).

    WPT uses scipy wavelet ops; tail vs full may differ by ~1e-10 on float64 last
    row due to array layout — well below any strategy threshold.
    """
    df = _make_bars(800)
    full = extract_wpt_features(df)
    with live_feature_tail(300):
        bounded = extract_wpt_features(df)
    assert list(bounded.index) == list(full.index)
    tail = 300
    n_exact = tail - 101  # window=100 + shift(1)
    for col in ["wpt_price_reconstructed", "wpt_price_trend", "wpt_price_fluctuation"]:
        a = bounded[col].iloc[-n_exact:].to_numpy(dtype=float)
        b = full[col].iloc[-n_exact:].to_numpy(dtype=float)
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-6, equal_nan=True, err_msg=col)


def test_poc_hal_tail_matches_full_history_last_rows():
    """End-to-end: bounded WPT → POC/HAL matches full-history on last rows."""
    df = _make_bars(800)
    wpt_full = extract_wpt_features(df)
    full = compute_poc_hal_features_from_series(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        volume=df["volume"],
        wpt_price_reconstructed=wpt_full["wpt_price_reconstructed"],
        poc_window=160,
    )
    with live_feature_tail(300):
        wpt_bounded = extract_wpt_features(df)
        bounded = compute_poc_hal_features_from_series(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            volume=df["volume"],
            wpt_price_reconstructed=wpt_bounded["wpt_price_reconstructed"],
            poc_window=160,
        )
    tail = 300
    n_exact = tail - 165  # poc_window=160 + margin
    for col in ["poc", "hal_high", "hal_low", "hal_mid"]:
        a = bounded[col].iloc[-n_exact:].to_numpy(dtype=float)
        b = full[col].iloc[-n_exact:].to_numpy(dtype=float)
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-6, equal_nan=True, err_msg=col)


def test_ofci_pct_tick_tail_matches_full_history_last_row():
    bars = _make_bars(400)
    ticks = _make_ticks(12000)
    fc = IncrementalFeatureComputer()
    info = {
        "compute_params": {
            "ofci_window": 100,
            "percentile_window": 540,
            "shift": 1,
        }
    }

    full = fc._compute_ofci_pct_from_trades(bars, ticks, info)
    with live_feature_tail(300):
        bounded = fc._compute_ofci_pct_from_trades(bars, ticks, info)

    assert "ofci_pct" in full.columns
    n_exact = 80
    a = bounded["ofci_pct"].iloc[-n_exact:].to_numpy(dtype=float)
    b = full["ofci_pct"].iloc[-n_exact:].to_numpy(dtype=float)
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-12, equal_nan=True)


def _make_dup_minute_ticks(n_min: int = 6000, seed: int = 17) -> pd.DataFrame:
    """Prod-shaped 1min ticks: buy+sell share each minute timestamp."""
    rng = np.random.default_rng(seed)
    base = pd.date_range("2024-01-01", periods=n_min, freq="1min")
    rows = []
    for ts in base:
        rows.append(
            {
                "timestamp": ts,
                "side": 1,
                "volume": float(rng.uniform(0.1, 5.0)),
            }
        )
        rows.append(
            {
                "timestamp": ts,
                "side": -1,
                "volume": float(rng.uniform(0.1, 5.0)),
            }
        )
    out = pd.DataFrame(rows).set_index("timestamp")
    assert out.index.has_duplicates
    return out


def test_ofci_pct_live_tail_ok_with_duplicate_minute_index():
    """Regression: live_tail percentile restore must not reindex on dup labels.

    Prod 1min aggregated ticks keep buy+sell on the same minute. Under
    ``live_feature_tail``, ``compute_percentile_rank_from_series`` used to
    ``reindex`` onto that duplicate index and raise, leaving ofci_pct missing
    and failing warmup with a misleading prepare_warmup_ticks hint.
    """
    bars = _make_bars(400)
    ticks = _make_dup_minute_ticks(6000)
    fc = IncrementalFeatureComputer()
    info = {
        "compute_params": {
            "ofci_window": 100,
            "percentile_window": 540,
            "shift": 1,
        },
        "output_columns": ["ofci_pct"],
    }

    full = fc._compute_ofci_pct_from_trades(bars, ticks, info)
    with live_feature_tail(300):
        bounded = fc._compute_ofci_pct_from_trades(bars, ticks, info)

    assert "ofci_pct" in bounded.columns
    assert np.isfinite(bounded["ofci_pct"].iloc[-1])
    n_exact = 80
    a = bounded["ofci_pct"].iloc[-n_exact:].to_numpy(dtype=float)
    b = full["ofci_pct"].iloc[-n_exact:].to_numpy(dtype=float)
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-12, equal_nan=True)

    joined = fc._join_loader_result_to_bars(bars.copy(), bounded, info)
    assert np.isfinite(joined["ofci_pct"].iloc[-1])


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(14, min_periods=1).mean().clip(lower=1e-6)


def _run_chain(df: pd.DataFrame) -> pd.DataFrame:
    """Full live DAG chain: WPT → POC/HAL → SR strength (single context call)."""
    wpt = extract_wpt_features(df)
    poc_hal = compute_poc_hal_features_from_series(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        volume=df["volume"],
        wpt_price_reconstructed=wpt["wpt_price_reconstructed"],
        poc_window=160,
    )
    sr = compute_sr_strength_max_from_series(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        atr=_atr(df),
        poc=poc_hal["poc"],
        hal_high=poc_hal["hal_high"],
        hal_low=poc_hal["hal_low"],
    )
    return sr


def test_chain_wpt_poc_sr_last_row_matches_full_history():
    """End-to-end: the *chain* under a single tail context must match full history.

    Isolated per-function tests pass fresh full-history inputs and so do NOT catch
    the compounding valid-region loss (WPT real last 300 → POC last ~140 → SR last
    ~80). Live only reads the LAST row of each requested feature, so this asserts
    the chained last row (what SRB filters actually consume, pointwise) is correct.
    """
    df = _make_bars(1200)
    full = _run_chain(df)
    with live_feature_tail(300):
        bounded = _run_chain(df)

    assert list(bounded.index) == list(full.index)
    # Last row is all that the live snapshot consumes; assert a small robust window.
    for col in ["sr_strength_max", "dist_to_nearest_sr", "direction_to_nearest_sr"]:
        a = bounded[col].iloc[-1]
        b = full[col].iloc[-1]
        assert np.isclose(
            a, b, rtol=0, atol=1e-5, equal_nan=True
        ), f"chained last row diverged on {col}: bounded={a} full={b}"
        # last ~40 rows (within the ~80-row chain-valid region) should also hold
        aa = bounded[col].iloc[-40:].to_numpy(dtype=float)
        bb = full[col].iloc[-40:].to_numpy(dtype=float)
        np.testing.assert_allclose(
            aa, bb, rtol=0, atol=1e-5, equal_nan=True, err_msg=col
        )


def test_get_live_tick_tail_rows_only_active_under_context():
    """Guard the dead-code regression: tick tail must be None outside the context.

    The live tick-dependent nodes must run *inside* ``live_feature_tail`` for the
    OFCI slice to activate. Here we assert the gating primitive directly.
    """
    from src.features.live_tail_context import get_live_tick_tail_rows

    assert get_live_tick_tail_rows() is None
    with live_feature_tail(300):
        assert get_live_tick_tail_rows() is not None
    assert get_live_tick_tail_rows() is None
