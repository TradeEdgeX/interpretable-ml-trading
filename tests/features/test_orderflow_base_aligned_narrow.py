import numpy as np
import pandas as pd

from src.features.time_series.utils_order_flow_features import (
    extract_order_flow_features,
    extract_trade_clustering_features,
    compute_vpin_base_aligned_features_from_series,
    compute_trade_cluster_base_aligned_features_from_series,
)


def _make_bars(n: int = 30) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01 00:00:00", periods=n, freq="5min")
    close = pd.Series(100 + np.linspace(0, 1, n), index=idx)
    df = pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close + 0.1,
            "low": close - 0.1,
            "close": close,
            "volume": 100.0,
        },
        index=idx,
    )
    return df


def _make_ticks(start: str, minutes: int = 200) -> pd.DataFrame:
    ts = pd.date_range(start, periods=minutes, freq="min")
    side = np.where(np.arange(minutes) % 2 == 0, 1, -1)
    ticks = pd.DataFrame(
        {
            "price": 100.0,
            "volume": 1.0,
            "side": side,
        },
        index=ts,
    )
    return ticks


def test_vpin_base_aligned_from_series_matches_df_entrypoint():
    bars = _make_bars()
    ticks = _make_ticks("2025-01-01 00:00:00", minutes=200)

    df_out = extract_order_flow_features(
        bars,
        ticks=ticks,
        include_trade_clustering=False,
        compute_vpin_derived=False,
        vpin_n_buckets=10,
        vpin_adaptive=False,
        monthly_cache_dir=None,
    )
    narrow = compute_vpin_base_aligned_features_from_series(
        open=bars["open"],
        close=bars["close"],
        high=bars["high"],
        low=bars["low"],
        volume=bars["volume"],
        ticks=ticks,
        include_trade_clustering=False,
        compute_vpin_derived=False,
        vpin_n_buckets=10,
        vpin_adaptive=False,
        monthly_cache_dir=None,
    )

    # Narrow output should be subset of df_out and aligned
    assert narrow.index.equals(bars.index)
    for col in narrow.columns:
        assert col in df_out.columns
        assert np.allclose(narrow[col].values, df_out[col].values, equal_nan=True)


def test_trade_cluster_base_aligned_from_series_matches_df_entrypoint():
    bars = _make_bars()
    ticks = _make_ticks("2025-01-01 00:00:00", minutes=200)

    df_out = extract_trade_clustering_features(
        bars,
        ticks=ticks,
        compute_trade_cluster_derived=False,
        monthly_cache_dir=None,
    )
    narrow = compute_trade_cluster_base_aligned_features_from_series(
        open=bars["open"],
        close=bars["close"],
        high=bars["high"],
        low=bars["low"],
        volume=bars["volume"],
        ticks=ticks,
        compute_trade_cluster_derived=False,
        monthly_cache_dir=None,
    )
    assert narrow.index.equals(bars.index)
    for col in narrow.columns:
        assert col in df_out.columns
        assert np.allclose(narrow[col].values, df_out[col].values, equal_nan=True)
