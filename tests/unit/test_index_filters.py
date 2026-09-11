"""ADX / mom_20 helpers (no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.index_filters import mom_20, wilder_adx


def test_mom_20_needs_20_bars():
    close = pd.Series(np.linspace(10, 12, 25))
    m = mom_20(close)
    assert m.iloc[:19].isna().all()
    assert np.isfinite(m.iloc[20])


def test_adx_is_finite_after_warmup():
    n = 80
    idx = pd.bdate_range("2020-01-02", periods=n)
    close = pd.Series(100 + np.arange(n) * 0.2, index=idx)
    high = close + 0.5
    low = close - 0.5
    out = wilder_adx(high, low, close, n=14)
    assert out["adx"].iloc[:14].isna().all() or out["adx"].iloc[20:].notna().any()
    assert out.loc[idx[40], "adx"] == out.loc[idx[40], "adx"]
    assert (out["plus_di"].iloc[40:] >= 0).all()
