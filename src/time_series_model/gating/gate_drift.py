from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .tree_gate import evaluate_gate_effect


@dataclass(frozen=True)
class GateDriftPoint:
    timestamp: Any
    activation_rate: float
    veto_loss_avoided: float
    false_reject_rate: float
    tail_loss_reduction: float
    n: int


def compute_gate_drift_series(
    df: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    gate_allow_col: str = "gate_allow",
    ret_col: str = "ret_used",
    window: int = 300,
    min_periods: int = 60,
    tail_q: float = 0.05,
) -> pd.DataFrame:
    """
    Compute a rolling drift dashboard for a gate.

    The key principle: drift metrics are *gate semantics*, not Sharpe/IC.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame()
    work = df.copy()
    work[timestamp_col] = pd.to_datetime(work[timestamp_col], utc=True, errors="coerce")
    work = (
        work.dropna(subset=[timestamp_col])
        .sort_values(timestamp_col)
        .reset_index(drop=True)
    )
    if work.empty:
        return pd.DataFrame()

    allow = (
        pd.to_numeric(work[gate_allow_col], errors="coerce")
        .fillna(0)
        .astype(int)
        .to_numpy()
    )
    ret = (
        pd.to_numeric(work[ret_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    )
    ts = work[timestamp_col].to_numpy()

    rows = []
    for i in range(len(work)):
        lo = max(0, i - int(window) + 1)
        a = allow[lo : i + 1]
        r = ret[lo : i + 1]
        if int(np.sum(np.isfinite(r))) < int(min_periods):
            continue
        met = evaluate_gate_effect(allow=a, ret_used=r, tail_q=float(tail_q))
        rows.append(
            {
                "timestamp": ts[i],
                "activation_rate": float(met.get("activation_rate", 0.0)),
                "veto_loss_avoided": float(met.get("veto_loss_avoided", 0.0)),
                "false_reject_rate": float(met.get("false_reject_rate", 0.0)),
                "tail_loss_reduction": float(met.get("tail_loss_reduction", 0.0)),
                "n": int(met.get("n", 0)),
            }
        )

    return pd.DataFrame(rows)


def degradation_decision(
    *,
    feature_available: bool,
    feature_stable: bool,
    policy: str = "degrade_to_notrade",
) -> str:
    """
    Deterministic degradation rule:
    - if feature unavailable/unstable -> degrade policy action
    """
    if bool(feature_available) and bool(feature_stable):
        return "normal"
    return str(policy)
