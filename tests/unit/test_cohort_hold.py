"""Closed-bar small-cap cohort helpers (no network)."""

from __future__ import annotations

import math

import pandas as pd

from src.research.cohort_hold import (
    book_kpis,
    cagr,
    circ_mcap_yi,
    hold_end,
    is_st_name,
    max_drawdown,
    snap_trading_day,
    trading_year_bars,
)


def test_circ_mcap_yi_from_amount_and_turnover():
    # 1.5e8 yuan turnover / 1.5% → 100 亿
    assert abs(circ_mcap_yi(1.5e8, 1.5) - 100.0) < 1e-6
    assert math.isnan(circ_mcap_yi(1.5e8, 0.0))
    assert math.isnan(circ_mcap_yi(0.0, 1.5))


def test_is_st_name():
    assert is_st_name("*ST假")
    assert is_st_name("ST假")
    assert not is_st_name("贵州茅台")


def test_hold_end_requires_full_window():
    cal = pd.bdate_range("2020-01-02", periods=20)
    assert trading_year_bars(3) == 756
    assert hold_end(cal, cal[0], 19) == cal[19]
    assert hold_end(cal, cal[0], 20) is None
    # weekend snaps to Friday
    assert snap_trading_day(cal, pd.Timestamp("2020-01-05")) == pd.Timestamp("2020-01-03")


def test_book_kpis_five_numbers():
    eq = pd.Series([1.0, 1.1, 1.05, 1.2], index=pd.bdate_range("2020-01-02", periods=4))
    k = book_kpis(eq)
    assert k["cagr"] == cagr(eq)
    assert k["maxdd"] == max_drawdown(eq)
    assert k["maxdd"] < 0
    assert k["calmar"] == k["cagr"] / abs(k["maxdd"])
    assert "sharpe" in k
