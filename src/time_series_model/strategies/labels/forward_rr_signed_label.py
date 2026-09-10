"""Signed forward-RR label for direction-from-tree slugs.

Used by ``config/strategies/fast_scalp`` / ``config/strategies/short_term_swing``
where a single regression tree is trained to output an expected forward RR; a
positive ``score`` → LONG, negative → SHORT (rule lives in ``backtest.yaml`` as
``long/short_entry_threshold``).

Label semantics
---------------

``label[t] = (close[t + H] - close[t]) / max(atr[t], eps)``

- **Signed**: positive when the next H bars rallied, negative when they sold off.
- **ATR-normalized**: comparable across symbols / volatility regimes.
- **No path extreme** (unlike ``me_label.compute_path_extreme_rr``): we want the
  tree to learn directional drift over H bars, not MFE-vs-MAE asymmetry. For
  MFE-aware variants, point the slug at ``me_label.compute_path_extreme_rr``
  directly.

If ``rr_floor`` > 0, labels with ``|label| < rr_floor`` are set to NaN (drop
near-zero noise that destabilises the regressor).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

_EPS = 1e-9


def forward_rr_column_name(horizon: int) -> str:
    """Column name for unfloored signed forward RR at horizon H (IC / analysis)."""
    return f"forward_rr_h{horizon}"


def compute_raw_signed_forward_rr(
    df: pd.DataFrame,
    *,
    horizon: int,
    price_col: str = "close",
    atr_col: str = "atr14",
    drop_inf: bool = True,
) -> pd.Series:
    """Unfloored signed forward RR — aligns with execution-layer bar return."""
    return compute_signed_forward_rr_label(
        df,
        horizon=horizon,
        price_col=price_col,
        atr_col=atr_col,
        rr_floor=0.0,
        drop_inf=drop_inf,
    ).rename(forward_rr_column_name(horizon))


def compute_signed_forward_rr_label(
    df: pd.DataFrame,
    *,
    horizon: int,
    price_col: str = "close",
    atr_col: str = "atr14",
    rr_floor: float = 0.0,
    drop_inf: bool = True,
) -> pd.Series:
    """Return signed forward RR (≈ N(0, σ_H) after ATR norm).

    Caller is expected to attach this series under the ``target_column`` named
    in the slug's ``labels.yaml`` (default ``label``).
    """
    if horizon <= 0:
        raise ValueError(f"horizon must be > 0, got {horizon}")
    if price_col not in df.columns:
        raise KeyError(f"missing price column {price_col!r}")
    if atr_col not in df.columns:
        raise KeyError(f"missing atr column {atr_col!r}")

    close = pd.to_numeric(df[price_col], errors="coerce")
    atr = pd.to_numeric(df[atr_col], errors="coerce").where(lambda s: s > _EPS)
    sym_col = (
        "_symbol"
        if "_symbol" in df.columns
        else ("symbol" if "symbol" in df.columns else None)
    )
    if sym_col is not None:
        fwd = close.groupby(df[sym_col]).shift(-horizon)
    else:
        fwd = close.shift(-horizon)
    label = (fwd - close) / atr

    if drop_inf:
        label = label.replace([np.inf, -np.inf], np.nan)
    if rr_floor > 0:
        label = label.where(label.abs() >= rr_floor)
    return label.rename("label")
