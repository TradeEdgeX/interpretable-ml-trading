import numpy as np
import pandas as pd

from src.features.time_series.utils_order_flow_features import (
    compute_trade_cluster_ratio_features_from_base,
    compute_trade_cluster_ratio_features_from_series,
    compute_trade_cluster_net_runs_counts_features_from_base,
    compute_trade_cluster_net_runs_counts_features_from_series,
    compute_trade_cluster_entropy_features_from_base,
    compute_trade_cluster_entropy_features_from_series,
    compute_trade_cluster_max_buy_run_ma_features_from_base,
    compute_trade_cluster_max_buy_run_ma_features_from_series,
    compute_trade_cluster_imbalance_zscore_features_from_base,
    compute_trade_cluster_imbalance_zscore_features_from_series,
)


def _make_series(
    name: str, n: int = 200, seed: int = 0, loc: float = 0.0, scale: float = 1.0
):
    rng = np.random.default_rng(seed)
    v = rng.normal(loc=loc, scale=scale, size=n).astype(float)
    v[2] = np.inf
    v[4] = -np.inf
    v[8] = np.nan
    idx = pd.date_range("2024-01-01", periods=n, freq="5min")
    return pd.Series(v, index=idx, name=name)


def test_trade_cluster_ratio_from_series_matches_from_base():
    df = pd.DataFrame(
        {
            "trade_cluster_max_buy_run": _make_series(
                "trade_cluster_max_buy_run", seed=1, loc=10, scale=2
            ),
            "trade_cluster_max_sell_run": _make_series(
                "trade_cluster_max_sell_run", seed=2, loc=9, scale=2
            ),
            "trade_cluster_avg_buy_run": _make_series(
                "trade_cluster_avg_buy_run", seed=3, loc=3, scale=1
            ),
            "trade_cluster_avg_sell_run": _make_series(
                "trade_cluster_avg_sell_run", seed=4, loc=3, scale=1
            ),
            "trade_cluster_buy_run_count": _make_series(
                "trade_cluster_buy_run_count", seed=5, loc=20, scale=5
            ),
            "trade_cluster_sell_run_count": _make_series(
                "trade_cluster_sell_run_count", seed=6, loc=18, scale=5
            ),
        }
    )
    base = compute_trade_cluster_ratio_features_from_base(df)
    narrow = compute_trade_cluster_ratio_features_from_series(
        **df.to_dict(orient="series")
    )
    assert list(narrow.columns) == list(base.columns)
    assert np.allclose(base.values, narrow.values, equal_nan=True)


def test_trade_cluster_net_runs_counts_from_series_matches_from_base():
    df = pd.DataFrame(
        {
            "trade_cluster_buy_run_count": _make_series(
                "trade_cluster_buy_run_count", seed=7, loc=20, scale=5
            ),
            "trade_cluster_sell_run_count": _make_series(
                "trade_cluster_sell_run_count", seed=8, loc=18, scale=5
            ),
        }
    )
    base = compute_trade_cluster_net_runs_counts_features_from_base(df)
    narrow = compute_trade_cluster_net_runs_counts_features_from_series(
        **df.to_dict(orient="series")
    )
    assert list(narrow.columns) == list(base.columns)
    assert np.allclose(base.values, narrow.values, equal_nan=True)


def test_trade_cluster_entropy_from_series_matches_from_base():
    df = pd.DataFrame(
        {
            "trade_cluster_directional_entropy": _make_series(
                "trade_cluster_directional_entropy", seed=9, loc=0.5, scale=0.2
            )
        }
    )
    base = compute_trade_cluster_entropy_features_from_base(df)
    narrow = compute_trade_cluster_entropy_features_from_series(
        **df.to_dict(orient="series")
    )
    assert list(narrow.columns) == list(base.columns)
    assert np.allclose(base.values, narrow.values, equal_nan=True)


def test_trade_cluster_max_buy_run_ma_from_series_matches_from_base():
    df = pd.DataFrame(
        {
            "trade_cluster_max_buy_run": _make_series(
                "trade_cluster_max_buy_run", seed=10, loc=10, scale=2
            ),
        }
    )
    base = compute_trade_cluster_max_buy_run_ma_features_from_base(df)
    narrow = compute_trade_cluster_max_buy_run_ma_features_from_series(
        **df.to_dict(orient="series")
    )
    assert list(narrow.columns) == list(base.columns)
    assert np.allclose(base.values, narrow.values, equal_nan=True)


def test_trade_cluster_imbalance_zscore_from_series_matches_from_base():
    df = pd.DataFrame(
        {
            "trade_cluster_imbalance_ratio": _make_series(
                "trade_cluster_imbalance_ratio", seed=11, loc=0.0, scale=1.0
            ),
        }
    )
    base = compute_trade_cluster_imbalance_zscore_features_from_base(df)
    narrow = compute_trade_cluster_imbalance_zscore_features_from_series(
        **df.to_dict(orient="series")
    )
    assert list(narrow.columns) == list(base.columns)
    assert np.allclose(base.values, narrow.values, equal_nan=True)
