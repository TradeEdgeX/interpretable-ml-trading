import numpy as np
import pandas as pd

from src.features.time_series.utils_liquidity_features import (
    compute_liquidity_void_features_from_series,
)
from src.features.time_series.utils_order_flow_features import (
    compute_trade_clustering_from_ticks,
    compute_trade_cluster_buy_sell_avg_ratio_features_from_series,
    compute_trade_cluster_buy_sell_max_ratio_features_from_series,
)
from src.features.time_series.utils_spectrum_features import (
    extract_spectrum_features_from_series,
)
from src.features.time_series.utils_wpt_features import (
    extract_wpt_price_features_normalized,
)


def _assert_not_all_nan_or_constant(s: pd.Series, *, min_non_nan: int = 10) -> None:
    s = pd.to_numeric(s, errors="coerce")
    nn = s.dropna()
    assert len(nn) >= min_non_nan
    # allow binary-ish features; just ensure not all identical
    assert nn.nunique() > 1


def _assert_bounded_series(
    s: pd.Series, lo: float, hi: float, *, tol: float = 1e-9
) -> None:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return
    assert (s >= lo - tol).all()
    assert (s <= hi + tol).all()


def _assert_not_too_saturated(
    s: pd.Series,
    *,
    lo: float,
    hi: float,
    edge_eps: float = 1e-3,
    max_edge_frac: float = 0.995,
) -> None:
    """
    Saturation check: if a bounded feature is *almost always* at the boundaries,
    it likely lost information (over-clipped / bad scaling).
    """
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return
    edge = ((s <= lo + edge_eps) | (s >= hi - edge_eps)).mean()
    assert edge <= max_edge_frac


def test_spectrum_info_distribution_non_degenerate_and_bounded():
    n = 320
    idx = pd.date_range("2024-01-01", periods=n, freq="H")
    close = pd.Series(
        100 + np.cumsum(np.random.default_rng(0).normal(0, 0.3, size=n)), index=idx
    )
    out = extract_spectrum_features_from_series(close=close, rolling_window=64)

    for c in [
        "spectrum_price_flatness",
        "spectrum_price_high_freq_ratio",
        "spectrum_price_low_freq_ratio",
        "spectrum_price_entropy",
    ]:
        _assert_bounded_series(out[c], 0.0, 1.0)
        _assert_not_all_nan_or_constant(out[c], min_non_nan=50)
        _assert_not_too_saturated(out[c], lo=0.0, hi=1.0)


def test_wpt_price_features_normalized_non_degenerate_and_energy_ratios_bounded():
    n = 260
    idx = pd.date_range("2024-01-01", periods=n, freq="H")
    close = pd.Series(
        100 + np.cumsum(np.random.default_rng(1).normal(0, 0.5, size=n)), index=idx
    )
    df = pd.DataFrame({"close": close}, index=idx)
    out = extract_wpt_price_features_normalized(
        df, price_col="close", window=100, level=4, update_step=5
    )

    # unitless series should not be all-NaN / constant
    _assert_not_all_nan_or_constant(out["wpt_price_trend"], min_non_nan=20)
    _assert_not_all_nan_or_constant(out["wpt_price_fluctuation"], min_non_nan=20)

    for c in [
        "wpt_price_energy_low_ratio",
        "wpt_price_energy_mid_ratio",
        "wpt_price_energy_high_ratio",
    ]:
        _assert_bounded_series(out[c], 0.0, 1.0)


def test_liquidity_void_has_signal_variation_when_forced_event():
    """
    Construct a synthetic episode with:
    - sudden fast move (speed high)
    - low volume (relative to rolling mean)
    to ensure liquidity_void_detected is not constant.
    """
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="H")
    close = pd.Series(100.0, index=idx)
    # inject a fast move
    close.iloc[120:125] = close.iloc[120:125] + np.linspace(0.0, 5.0, 5)
    close.iloc[125:] = close.iloc[125:] + 5.0
    volume = pd.Series(1000.0, index=idx)
    volume.iloc[120:130] = 50.0  # low volume window
    atr = pd.Series(1.0, index=idx)

    out = compute_liquidity_void_features_from_series(
        close=close, volume=volume, atr=atr, lookback_window=20
    )
    det = out["liquidity_void_detected"]
    _assert_bounded_series(det, 0.0, 1.0)
    assert det.dropna().nunique() > 1


def test_trade_clustering_base_is_non_degenerate_on_random_ticks():
    n = 600
    idx = pd.date_range("2024-01-01", periods=n, freq="S")
    sides = np.random.default_rng(2).choice([1, -1], size=n)
    ticks = pd.DataFrame({"side": sides}, index=idx)
    df, _state = compute_trade_clustering_from_ticks(ticks, window_size=100)
    # A few representative outputs
    for c in [
        "trade_cluster_max_buy_run",
        "trade_cluster_max_sell_run",
        "trade_cluster_buy_run_count",
        "trade_cluster_sell_run_count",
        "trade_cluster_directional_entropy",
    ]:
        _assert_not_all_nan_or_constant(df[c], min_non_nan=50)


def test_trade_cluster_buy_sell_log_robust_ratios_are_finite_and_non_constant():
    n = 260
    idx = pd.date_range("2024-01-01", periods=n, freq="H")
    buy = pd.Series(
        np.random.default_rng(3).integers(0, 100, size=n), index=idx
    ).astype(float)
    sell = pd.Series(
        np.random.default_rng(4).integers(0, 100, size=n), index=idx
    ).astype(float)

    out_max = compute_trade_cluster_buy_sell_max_ratio_features_from_series(
        trade_cluster_max_buy_run=buy, trade_cluster_max_sell_run=sell
    )
    out_avg = compute_trade_cluster_buy_sell_avg_ratio_features_from_series(
        trade_cluster_avg_buy_run=buy, trade_cluster_avg_sell_run=sell
    )

    _assert_not_all_nan_or_constant(
        out_max["trade_cluster_buy_sell_max_ratio"], min_non_nan=50
    )
    _assert_not_all_nan_or_constant(
        out_avg["trade_cluster_buy_sell_avg_ratio"], min_non_nan=50
    )
