"""AFML triple-barrier result + meta-label (0/1), given a primary side.

Primary side is an input (rule or model). This module does **not** infer
direction from the path. Meta y asks: did that side hit TP first?

Barrier result at t (scan t+1 .. t+H, no same-bar look-ahead):

- ``+1`` — take-profit first (primary side correct)
- ``-1`` — stop-loss first
- ``0``  — vertical (timeout), neither barrier
- NaN  — no side, or missing price / ATR

Same-bar both barriers: SL wins (conservative: that bet was not cleanly right).

Meta label: ``1`` iff barrier == +1, else ``0`` (NaN preserved).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_EPS = 1e-9


def compute_triple_barrier_result(
    df: pd.DataFrame,
    side: pd.Series,
    *,
    horizon: int = 20,
    pt_r: float = 1.0,
    sl_r: float = 1.0,
    price_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    atr_col: str = "atr",
) -> pd.Series:
    if horizon <= 0:
        raise ValueError(f"horizon must be > 0, got {horizon}")
    for col in (price_col, high_col, low_col, atr_col):
        if col not in df.columns:
            raise KeyError(f"missing column {col!r}")

    close = pd.to_numeric(df[price_col], errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(df[high_col], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(df[low_col], errors="coerce").to_numpy(dtype=float)
    atr = pd.to_numeric(df[atr_col], errors="coerce").to_numpy(dtype=float)
    side_arr = pd.to_numeric(side.reindex(df.index), errors="coerce").to_numpy(
        dtype=float
    )
    n = len(df)
    out = np.full(n, np.nan, dtype=float)
    pt = float(pt_r)
    sl = float(sl_r)

    for i in range(n - horizon):
        s = side_arr[i]
        if not np.isfinite(s) or s == 0.0:
            continue
        s = 1.0 if s > 0 else -1.0
        entry = close[i]
        risk = atr[i]
        if not np.isfinite(risk) or risk <= _EPS or not np.isfinite(entry):
            continue
        tp = entry + s * pt * risk
        sl_px = entry - s * sl * risk
        slc = slice(i + 1, i + 1 + horizon)
        fh = high[slc]
        fl = low[slc]
        if fh.size == 0:
            continue
        hit = 0.0
        for j in range(len(fh)):
            hj = fh[j]
            lj = fl[j]
            if not np.isfinite(hj) or not np.isfinite(lj):
                continue
            if s > 0:
                hit_tp = hj >= tp
                hit_sl = lj <= sl_px
            else:
                hit_tp = lj <= tp
                hit_sl = hj >= sl_px
            if hit_tp and hit_sl:
                hit = -1.0
                break
            if hit_tp:
                hit = 1.0
                break
            if hit_sl:
                hit = -1.0
                break
        out[i] = hit

    return pd.Series(out, index=df.index, name="barrier")


def compute_meta_label(barrier: pd.Series) -> pd.Series:
    """1 = primary side hit TP first; 0 = SL or timeout; NaN if no barrier."""
    b = pd.to_numeric(barrier, errors="coerce")
    y = pd.Series(np.nan, index=barrier.index, name="meta_label", dtype=float)
    known = b.notna()
    y.loc[known] = (b.loc[known] > 0).astype(float)
    return y


def primary_side_from_position(
    position: pd.Series,
    *,
    deadzone: float = 0.10,
) -> pd.Series:
    """Dumb locked side: sign(position) if |position| >= deadzone else 0."""
    pos = pd.to_numeric(position, errors="coerce")
    side = np.sign(pos)
    side = side.where(pos.abs() >= float(deadzone), 0.0)
    return side.rename("side")
