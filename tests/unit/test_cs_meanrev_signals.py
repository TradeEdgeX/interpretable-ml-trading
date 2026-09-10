"""Unit tests for CS mean-reversion helpers."""

from __future__ import annotations

import pandas as pd

from ashare.cs_meanrev_signals import (
    consecutive_down_streak,
    cs_pct_rank,
    dedupe_symbol_dates,
    wilder_rsi,
)


def test_consecutive_down_streak():
    close = pd.Series([10.0, 9.0, 8.0, 7.0, 7.5, 7.0, 6.0])
    s = consecutive_down_streak(close)
    assert list(s) == [0, 1, 2, 3, 0, 1, 2]


def test_cs_pct_rank_weakest():
    v = pd.Series([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
    g = pd.Series(["a", "a", "a", "b", "b", "b"])
    r = cs_pct_rank(v, g, ascending=True)
    # within a: 1→lowest pct
    assert float(r.iloc[0]) < float(r.iloc[2])
    assert float(r.iloc[3]) < float(r.iloc[5])


def test_dedupe_symbol_dates():
    symbol = pd.Series(["x", "x", "x", "x", "x"])
    dates = pd.to_datetime(
        ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    )
    mask = pd.Series([True, True, True, True, True])
    keep = dedupe_symbol_dates(symbol, dates, mask, cooldown=2)
    # positions 0 keep, 1 skip (pos-0=1<=2), 2 skip, 3 keep (3-0=3>2), 4 skip
    assert list(keep) == [True, False, False, True, False]


def test_wilder_rsi_bounds():
    close = pd.Series([10.0, 10.5, 10.2, 10.8, 10.1, 11.0, 10.4, 11.2, 10.6, 11.5] * 4)
    rsi = wilder_rsi(close)
    valid = rsi.dropna()
    assert len(valid) > 0
    assert float(valid.iloc[-1]) >= 0.0
    assert float(valid.iloc[-1]) <= 100.0
