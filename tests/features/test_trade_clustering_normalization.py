import numpy as np
import pandas as pd

from src.features.time_series.utils_order_flow_features import (
    compute_trade_clustering_from_ticks,
)


def test_trade_clustering_base_outputs_are_bounded_after_normalization():
    # Build synthetic tick stream: 500 ticks with random sides.
    n = 500
    idx = pd.date_range("2024-01-01", periods=n, freq="S")
    sides = np.random.default_rng(0).choice([1, -1], size=n)
    ticks = pd.DataFrame({"side": sides}, index=idx)

    window_size = 100
    df, _state = compute_trade_clustering_from_ticks(ticks, window_size=window_size)
    # base columns should exist
    cols_0_1 = [
        "trade_cluster_max_buy_run",
        "trade_cluster_max_sell_run",
        "trade_cluster_avg_buy_run",
        "trade_cluster_avg_sell_run",
        "trade_cluster_buy_run_count",
        "trade_cluster_sell_run_count",
        "trade_cluster_directional_entropy",
    ]
    for c in cols_0_1:
        assert c in df.columns
        vals = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(vals) == 0:
            continue
        assert np.isfinite(vals.to_numpy()).all()
        assert vals.min() >= -1e-9
        assert vals.max() <= 1.0 + 1e-9

    assert "trade_cluster_imbalance_ratio" in df.columns
    imb = pd.to_numeric(df["trade_cluster_imbalance_ratio"], errors="coerce").dropna()
    if len(imb) > 0:
        assert np.isfinite(imb.to_numpy()).all()
        assert imb.min() >= -1.0 - 1e-9
        assert imb.max() <= 1.0 + 1e-9
