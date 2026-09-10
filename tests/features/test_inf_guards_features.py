import numpy as np
import pandas as pd

from src.features.time_series.utils_order_flow_features import (
    extract_order_flow_features,
)
from src.features.time_series.utils_hurst_features import extract_hurst_features


def _base_df(n=200):
    idx = pd.date_range("2024-01-01", periods=n, freq="1H")
    price = np.linspace(100, 110, n)
    df = pd.DataFrame(
        {
            "open": price,
            "high": price + 0.5,
            "low": price - 0.5,
            "close": price,
            "volume": np.linspace(1000, 2000, n),
        },
        index=idx,
    )
    return df


def test_vpin_inf_guards():
    df = _base_df()
    # synthetic ticks_loader_json not needed; we only use vpin change/zscore guards on empty vpin cols
    df["vpin"] = np.linspace(0, 1, len(df))
    df["vpin_signed_imbalance"] = np.linspace(-0.5, 0.5, len(df))
    ticks = pd.DataFrame(
        {
            "price": np.linspace(100, 110, len(df)),
            "volume": np.linspace(1, 2, len(df)),
            "side": np.where(np.arange(len(df)) % 2 == 0, 1, -1),
        },
        index=df.index,
    )
    out = extract_order_flow_features(
        df,
        ticks=ticks,
        ticks_loader_json=None,
        include_trade_clustering=False,
        trade_clustering_window=100,
        freq="1H",
        vpin_bucket_volume=100,
        vpin_n_buckets=20,
        vpin_adaptive=False,
    )
    cols = [c for c in out.columns if c.startswith("vpin")]
    values = out[cols].to_numpy()
    # 允许 NaN，但不允许 inf/-inf
    assert not np.isinf(values).any()


def test_hurst_inf_guards():
    df = _base_df()
    out = extract_hurst_features(
        df,
        price_col="close",
        cvd_col=None,
        volume_col="volume",
        rolling_window=50,
        update_freq=5,
        clip_pct=0.5,
    )
    cols = [c for c in out.columns if c.startswith("hurst_")]
    values = out[cols].to_numpy()
    assert not np.isinf(values).any()
