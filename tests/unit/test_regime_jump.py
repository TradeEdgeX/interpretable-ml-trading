"""Closed-bar +3% flip and MA20 exit."""

from __future__ import annotations

import pandas as pd

from src.research.regime_jump import regime_positions


def test_plus_three_flips_long_then_ma20_exits_to_short():
    close = pd.Series(
        [100.0, 100.0, 104.0, 104.0, 90.0],
        index=pd.date_range("2024-01-02", periods=5, freq="B"),
    )
    ma200 = pd.Series([101.0] * 5, index=close.index)
    ma20 = pd.Series([95.0, 95.0, 95.0, 95.0, 95.0], index=close.index)
    pos = regime_positions(close, ma_slow=ma200, ma_fast=ma20, jump=0.03)
    # day0: below MA200, short
    assert pos.iloc[0] == -1.0
    # day2: +4% close → long
    assert pos.iloc[2] == 1.0
    # day4: close 90 < MA20 → back to short-rule, still below MA200
    assert pos.iloc[4] == -1.0
