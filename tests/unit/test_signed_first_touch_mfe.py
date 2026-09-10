import numpy as np
import pandas as pd

from src.time_series_model.strategies.labels.signed_first_touch_mfe import (
    compute_signed_first_touch_mfe_r,
)


def _bars(path: list[tuple[float, float, float]]) -> pd.DataFrame:
    """path: (open=close, high, low) then pad to horizon+1."""
    rows = []
    for close, high, low in path:
        rows.append({"close": close, "high": high, "low": low, "atr": 1.0})
    return pd.DataFrame(rows)


def test_clean_long_3r_not_stopped() -> None:
    # t0 close=100. Then 20 bars grind to +3R, never −1R.
    path = [(100.0, 100.0, 100.0)]
    for i in range(1, 21):
        px = 100.0 + 3.0 * i / 20.0
        path.append((px, px, px - 0.2))
    y = compute_signed_first_touch_mfe_r(_bars(path), horizon=20, first_touch_r=1.0)
    assert y.iloc[0] == np.float64(y.iloc[0])
    assert y.iloc[0] >= 2.9


def test_sl_first_then_rally_stays_short() -> None:
    # Hit −1.2R on bar 1, then rally to +5R. First-touch owns the short side.
    path = [(100.0, 100.0, 100.0), (99.0, 99.2, 98.8)]
    for _ in range(19):
        path.append((105.0, 105.0, 104.5))
    y = compute_signed_first_touch_mfe_r(_bars(path), horizon=20, first_touch_r=1.0)
    assert y.iloc[0] < 0
    assert y.iloc[0] > -2.0


def test_neither_side_uses_larger_mfe() -> None:
    path = [(100.0, 100.0, 100.0)]
    for _ in range(20):
        path.append((100.2, 100.4, 99.9))
    y = compute_signed_first_touch_mfe_r(_bars(path), horizon=20, first_touch_r=1.0)
    assert y.iloc[0] > 0
    assert y.iloc[0] < 1.0
