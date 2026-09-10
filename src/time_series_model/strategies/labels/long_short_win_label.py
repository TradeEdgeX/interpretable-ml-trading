"""Binary long/short win labels derived from signed forward RR."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from src.time_series_model.strategies.labels.forward_rr_signed_label import (
    compute_raw_signed_forward_rr,
)


def compute_long_short_win_labels(
    df: pd.DataFrame,
    *,
    horizon: int,
    rr_floor: float = 0.0,
    price_col: str = "close",
    atr_col: str = "atr14",
) -> pd.DataFrame:
    """Return ``long_win`` / ``short_win`` binary columns from signed forward RR."""
    fwd = compute_raw_signed_forward_rr(
        df,
        horizon=horizon,
        price_col=price_col,
        atr_col=atr_col,
    )
    out = pd.DataFrame(index=df.index)
    if rr_floor > 0:
        out["long_win"] = (fwd >= rr_floor).astype("float")
        out["short_win"] = (fwd <= -rr_floor).astype("float")
        out.loc[fwd.abs() < rr_floor, ["long_win", "short_win"]] = pd.NA
    else:
        out["long_win"] = (fwd > 0).astype("float")
        out["short_win"] = (fwd < 0).astype("float")
    out["forward_rr"] = fwd
    return out


def compute_long_win_label(
    df: pd.DataFrame,
    *,
    horizon: int,
    rr_floor: float = 0.0,
    price_col: str = "close",
    atr_col: str = "atr14",
) -> pd.Series:
    """Single-column label for long-only tree training (target_column: label)."""
    labels = compute_long_short_win_labels(
        df,
        horizon=horizon,
        rr_floor=rr_floor,
        price_col=price_col,
        atr_col=atr_col,
    )
    return labels["long_win"]


def compute_short_win_label(
    df: pd.DataFrame,
    *,
    horizon: int,
    rr_floor: float = 0.0,
    price_col: str = "close",
    atr_col: str = "atr14",
) -> pd.Series:
    """Single-column label for short-only tree training (target_column: label)."""
    labels = compute_long_short_win_labels(
        df,
        horizon=horizon,
        rr_floor=rr_floor,
        price_col=price_col,
        atr_col=atr_col,
    )
    return labels["short_win"]


def attach_long_short_win_targets(
    df: pd.DataFrame,
    *,
    horizon: int,
    rr_floor: float = 0.0,
    price_col: str = "close",
    atr_col: str = "atr14",
    long_col: str = "long_win",
    short_col: str = "short_win",
) -> pd.DataFrame:
    """Attach target columns in-place copy for dual-head tree training."""
    labels = compute_long_short_win_labels(
        df,
        horizon=horizon,
        rr_floor=rr_floor,
        price_col=price_col,
        atr_col=atr_col,
    )
    out = df.copy()
    out[long_col] = labels["long_win"]
    out[short_col] = labels["short_win"]
    return out
