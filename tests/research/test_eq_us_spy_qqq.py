from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_tools.us_etf_daily import _parse_nasdaq_number
from src.research.eq_us_spy_qqq import (
    _first_breach_dd,
    _hold_until_level,
    _hold_windows,
    positions_from_close,
    rsi_wilder,
)
from src.research.harness_registry import required_harness


def test_nasdaq_number_strips_dollar_and_comma() -> None:
    assert _parse_nasdaq_number("$216.34") == 216.34
    assert _parse_nasdaq_number("45,512,740") == 45512740.0


def test_harness_is_eq_us_daily() -> None:
    assert required_harness("eq_us_spy_qqq") == "eq_us_daily"


def test_rsi_hold_is_closed_bar() -> None:
    close = pd.Series(np.linspace(100.0, 80.0, 30))
    rsi = rsi_wilder(close)
    hit = rsi <= 30.0
    assert bool(hit.any())
    pos = _hold_windows(hit, 5)
    first = int(hit.to_numpy().nonzero()[0][0])
    assert pos.iloc[first] == 0.0
    assert pos.iloc[first + 1] == 1.0


def test_first_dd_breach_fires_once_until_new_peak() -> None:
    close = pd.Series([100.0, 90.0, 79.0, 70.0, 85.0, 110.0, 80.0])
    hit = _first_breach_dd(close, -0.20)
    assert int(hit.sum()) == 2
    assert bool(hit.iloc[2])
    assert bool(hit.iloc[6])


def test_hold_until_level_skips_nan_target() -> None:
    close = pd.Series([10.0, 9.0, 8.0, 11.0, 12.0])
    trigger = pd.Series([True, False, False, False, False])
    level = pd.Series([float("nan"), 10.0, 10.0, 10.0, 10.0])
    pos = _hold_until_level(trigger, level, close)
    assert float(pos.sum()) == 0.0


def test_buy_hold_stays_fully_invested() -> None:
    close = pd.Series(np.linspace(50.0, 150.0, 250))
    pos = positions_from_close(close)
    assert float(pos["buy_hold"].mean()) == 1.0
    assert float(pos["rsi_hold_40"].sum()) == 0.0
