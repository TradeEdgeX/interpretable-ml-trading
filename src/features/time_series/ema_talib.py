"""Production EMA = TA-Lib IIR (FeatureStore ``ema_1200_value_f`` / live bus).

pandas ``ewm(span=N, adjust=False)`` uses the same alpha ``2/(N+1)`` but seeds
from the **first close**. TA-Lib seeds from the **SMA of the first N bars**
and leaves the first N-1 values NaN. Those are not interchangeable: a short
window ewm line will not match SRB / Rolling / TPC ``ema_1200``.

Call this helper (or ``compute_talib_indicator_from_series``) whenever a
consumer needs an EMA(1200) level. Do not reimplement with pandas ewm.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ema_talib(close: pd.Series, *, timeperiod: int = 1200) -> pd.Series:
    """TA-Lib EMA on ``close``; index preserved. Warmup rows are NaN.

    ``talib`` is imported here, not at module load. Lean CI
    (``requirements-ci.txt``) omits the C extension; live rolling / CMS
    modules must stay importable so collection does not explode.
    """
    import talib

    period = int(timeperiod)
    if period < 2:
        raise ValueError(f"ema_talib timeperiod must be >= 2, got {period}")
    c = pd.to_numeric(close, errors="coerce").astype("float64")
    if c.empty:
        return pd.Series(dtype=float, index=c.index)
    arr = np.asarray(c.to_numpy(), dtype=np.float64)
    out = talib.EMA(arr, timeperiod=period)
    return pd.Series(out, index=c.index, dtype=float)
