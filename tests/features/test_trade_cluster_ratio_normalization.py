import numpy as np
import pandas as pd

from src.features.time_series.utils_order_flow_features import (
    compute_trade_cluster_buy_sell_avg_ratio_features_from_series,
    compute_trade_cluster_buy_sell_max_ratio_features_from_series,
)


def test_trade_cluster_buy_sell_ratio_features_are_finite_after_log_robust_scaling():
    n = 300
    idx = pd.date_range("2024-01-01", periods=n, freq="H")

    # Build positive series with occasional zeros to exercise eps paths
    buy = pd.Series(
        np.random.default_rng(0).integers(0, 100, size=n), index=idx
    ).astype(float)
    sell = pd.Series(
        np.random.default_rng(1).integers(0, 100, size=n), index=idx
    ).astype(float)

    out_max = compute_trade_cluster_buy_sell_max_ratio_features_from_series(
        trade_cluster_max_buy_run=buy,
        trade_cluster_max_sell_run=sell,
    )
    out_avg = compute_trade_cluster_buy_sell_avg_ratio_features_from_series(
        trade_cluster_avg_buy_run=buy,
        trade_cluster_avg_sell_run=sell,
    )

    for df in [out_max, out_avg]:
        assert df.shape[0] == n
        vals = df.fillna(0.0).to_numpy(dtype=float)
        assert np.isfinite(vals).all()
