"""Lean-CI stand-in for ``ema_talib`` when the C extension is not installed.

``requirements-ci.txt`` omits ``talib``. Production images have it; these tests
still need to exercise Champion recompute / parity contracts in Job 0b.
The recurrence matches TA-Lib (SMA seed, then IIR with ``2/(N+1)``).
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd


def ema_sma_seed_iir(close: pd.Series, *, timeperiod: int = 1200) -> pd.Series:
    period = int(timeperiod)
    if period < 2:
        raise ValueError(f"ema timeperiod must be >= 2, got {period}")
    c = pd.to_numeric(close, errors="coerce").astype("float64")
    if c.empty:
        return pd.Series(dtype=float, index=c.index)
    arr = np.asarray(c.to_numpy(), dtype=np.float64)
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return pd.Series(out, index=c.index, dtype=float)
    seed = float(np.mean(arr[:period]))
    out[period - 1] = seed
    alpha = 2.0 / (period + 1.0)
    for i in range(period, n):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return pd.Series(out, index=c.index, dtype=float)


def install_lean_ema_if_needed(monkeypatch) -> bool:
    """Patch ``ema_talib`` when ``talib`` is missing. Returns True if patched."""
    try:
        spec = importlib.util.find_spec("talib")
    except ModuleNotFoundError:
        spec = None
    if spec is not None:
        return False
    from features.time_series import ema_talib as mod

    monkeypatch.setattr(mod, "ema_talib", ema_sma_seed_iir)
    return True
