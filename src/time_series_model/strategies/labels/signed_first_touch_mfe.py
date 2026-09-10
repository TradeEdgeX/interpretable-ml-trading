"""Signed first-touch path amplitude in R (MFE / ATR).

One tree, one y: which side first reaches ``first_touch_r``, and how far
that side's MFE ran over the horizon. 2R / 3R are score cuts, not extra trees.

``label[t]``:
- Scan bars ``t+1 .. t+H`` (no same-bar look-ahead).
- First bar where long excursion or short excursion reaches ``first_touch_r``
  owns the path. Same-bar both sides: the larger excursion that bar wins.
- y = +MFE_long or −MFE_short over the full horizon (uncapped, so 2R/3R+ live
  in the magnitude).
- Neither side reaches ``first_touch_r``: y = +MFE_long if MFE_long ≥ MFE_short
  else −MFE_short (still path amplitude, not close-to-close).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_EPS = 1e-9


def compute_signed_first_touch_mfe_r(
    df: pd.DataFrame,
    *,
    horizon: int = 20,
    first_touch_r: float = 1.0,
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
    n = len(df)
    out = np.full(n, np.nan, dtype=float)
    touch = float(first_touch_r)

    for i in range(n - horizon):
        risk = atr[i]
        entry = close[i]
        if not np.isfinite(risk) or risk <= _EPS or not np.isfinite(entry):
            continue
        sl = slice(i + 1, i + 1 + horizon)
        fh = high[sl]
        fl = low[sl]
        if fh.size == 0:
            continue
        long_exc = (fh - entry) / risk
        short_exc = (entry - fl) / risk
        mfe_l = float(np.nanmax(long_exc)) if np.isfinite(long_exc).any() else np.nan
        mfe_s = float(np.nanmax(short_exc)) if np.isfinite(short_exc).any() else np.nan
        if not np.isfinite(mfe_l) or not np.isfinite(mfe_s):
            continue

        side = 0
        for j in range(len(long_exc)):
            le = long_exc[j]
            se = short_exc[j]
            le_ok = np.isfinite(le) and le >= touch
            se_ok = np.isfinite(se) and se >= touch
            if le_ok and se_ok:
                side = 1 if le >= se else -1
                break
            if le_ok:
                side = 1
                break
            if se_ok:
                side = -1
                break
        if side > 0:
            out[i] = mfe_l
        elif side < 0:
            out[i] = -mfe_s
        else:
            out[i] = mfe_l if mfe_l >= mfe_s else -mfe_s

    return pd.Series(out, index=df.index, name="label")
