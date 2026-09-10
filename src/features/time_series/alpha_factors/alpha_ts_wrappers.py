from __future__ import annotations

import pandas as pd

from src.features.registry import register_feature
from src.features.time_series.alpha_factors.alpha101_timeseries_adapted import (
    compute_adapted_alpha101_factors,
)


@register_feature(
    "compute_alpha101_ts_core_from_df",
    category="alpha_factors",
    description="A small, stable, TS-adapted Alpha101 subset (001/022/043/066). Unitless features derived from OHLCV.",
    outputs=[
        "alpha101_001_ts",
        "alpha101_022_ts",
        "alpha101_043_ts",
        "alpha101_066_ts",
    ],
)
def compute_alpha101_ts_core_from_df(
    df: pd.DataFrame,
    *,
    use_ts_rank: bool = True,
    alpha001_window: int = 5,
    alpha022_corr_window: int = 10,
    alpha022_delta_window: int = 5,
    alpha022_vol_window: int = 20,
    alpha043_vol_rank_window: int = 20,
    alpha043_mom_rank_window: int = 8,
    alpha043_adv_window: int = 20,
    alpha043_mom_period: int = 7,
) -> pd.DataFrame:
    """
    Narrow wrapper around `compute_adapted_alpha101_factors` so it can be used by:
    - feature registry
    - feature_dependencies.yaml
    - CS build-store (monthly FeatureStore partitions)

    Requirements:
    - df must have columns: open, high, low, close, volume
    - df must have DatetimeIndex (or downstream caller will set it)
    """
    return compute_adapted_alpha101_factors(
        df,
        use_ts_rank=bool(use_ts_rank),
        alpha001_window=int(alpha001_window),
        alpha022_corr_window=int(alpha022_corr_window),
        alpha022_delta_window=int(alpha022_delta_window),
        alpha022_vol_window=int(alpha022_vol_window),
        alpha043_vol_rank_window=int(alpha043_vol_rank_window),
        alpha043_mom_rank_window=int(alpha043_mom_rank_window),
        alpha043_adv_window=int(alpha043_adv_window),
        alpha043_mom_period=int(alpha043_mom_period),
    )


