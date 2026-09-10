"""
T5β liquidation cluster proxies on 2h bars (no tick-level liquidation feed).

``liquidation_cluster_score`` combines OI flow burst, funding stress, and
elevated volatility. Cascade / reversal sub-scores gate by OI scene semantics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.registry import register_feature


def _sigmoid01(x: pd.Series) -> pd.Series:
    return 1.0 / (1.0 + np.exp(-x.astype(float)))


@register_feature(
    "compute_liquidation_cluster_proxy_from_df",
    category="interaction",
    description=(
        "T5β 2h liquidation-cluster proxy: |oi_flow_z| × funding stress × vol. "
        "Sub-scores split cascade (ignition) vs reversal (exhaustion). "
        "Not tick liquidation data."
    ),
    outputs=[
        "liquidation_cluster_score",
        "liquidation_cascade_proxy_score",
        "liquidation_reversal_proxy_score",
    ],
)
def compute_liquidation_cluster_proxy_from_df(
    df: pd.DataFrame,
    *,
    oi_flow_z_col: str = "oi_flow_zscore",
    funding_abs_z_col: str = "funding_rate_abs_zscore_50",
    volatility_col: str = "atr_percentile",
    oi_ignition_col: str = "oi_ignition_score",
    oi_exhaustion_col: str = "oi_exhaustion_score",
    flow_shift: float = 2.0,
    flow_scale: float = 1.0,
    funding_shift: float = 2.0,
    funding_scale: float = 1.0,
) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df index must be a DatetimeIndex")

    def _col(name: str, default: float = 0.0) -> pd.Series:
        raw = df.get(name, pd.Series(dtype=float, index=df.index))
        if isinstance(raw, pd.Series):
            s = raw
        else:
            s = pd.Series(raw, index=df.index)
        return pd.to_numeric(s, errors="coerce").reindex(df.index).fillna(default)

    flow_z = _col(oi_flow_z_col)
    fund_z = _col(funding_abs_z_col)
    vol = _col(volatility_col, default=0.5).clip(0.0, 1.0)
    ignition = _col(oi_ignition_col).clip(0.0, 1.0)
    exhaustion = _col(oi_exhaustion_col).clip(0.0, 1.0)

    flow_burst = _sigmoid01((flow_z.abs() - float(flow_shift)) / float(flow_scale))
    fund_stress = _sigmoid01((fund_z - float(funding_shift)) / float(funding_scale))
    base = (flow_burst * fund_stress * vol).clip(0.0, 1.0)

    out = pd.DataFrame(index=df.index)
    out["liquidation_cluster_score"] = base
    out["liquidation_cascade_proxy_score"] = (base * ignition).clip(0.0, 1.0)
    out["liquidation_reversal_proxy_score"] = (base * exhaustion).clip(0.0, 1.0)
    return out
