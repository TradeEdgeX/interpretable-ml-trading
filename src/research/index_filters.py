"""Closed-bar index filters: Wilder ADX and 20-day momentum."""

from __future__ import annotations

import numpy as np
import pandas as pd


def mom_20(close: pd.Series) -> pd.Series:
    prev = close.shift(20)
    return (close / prev - 1.0).where(prev > 0)


def wilder_adx(
    high: pd.Series, low: pd.Series, close: pd.Series, *, n: int = 14
) -> pd.DataFrame:
    """Wilder ADX / +DI / −DI. Values at T use the T bar (closed-bar)."""
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0.0), 0.0)
    minus_dm = down.where((down > up) & (down > 0.0), 0.0)
    prev = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev).abs(), (low - prev).abs()],
        axis=1,
    ).max(axis=1)
    alpha = 1.0 / float(n)
    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr
    minus_di = 100.0 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    adx = dx.ewm(alpha=alpha, adjust=False).mean()
    return pd.DataFrame({"adx": adx, "plus_di": plus_di, "minus_di": minus_di})
