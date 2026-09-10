"""Contracts for short-harvest clock (next-open / TP10 / T+3 / fixed notional)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from auxiliary.research.short_harvest_clock import (
    MarketSpec,
    OhlcPanel,
    ShortHarvestPolicy,
    build_rsi_amp_events,
    cross_market_simulate,
    path_label_next_open_tp,
    simulate_short_harvest,
    trading_calendar,
)


def _panel_from_ohlc(
    dates: list[str],
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    *,
    rsi: float = 15.0,
    amp: float = 10.0,
) -> OhlcPanel:
    idx = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    return OhlcPanel(
        open=pd.Series(opens, index=idx, dtype=float),
        high=pd.Series(highs, index=idx, dtype=float),
        low=pd.Series(lows, index=idx, dtype=float),
        close=pd.Series(closes, index=idx, dtype=float),
        rsi=pd.Series([rsi] * len(idx), index=idx, dtype=float),
        amplitude=pd.Series([amp] * len(idx), index=idx, dtype=float),
    )


def test_path_label_next_open_not_signal_close() -> None:
    # Signal Mon close=100; Tue open=101; TP never hits; exit Wed close (day 3).
    dates = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    p = _panel_from_ohlc(
        dates,
        opens=[100, 101, 102, 103],
        highs=[100, 102, 103, 104],
        lows=[99, 100, 101, 102],
        closes=[100, 101.5, 102.5, 103.5],
    )
    lab = path_label_next_open_tp(
        p,
        "2024-01-02",
        take_profit=0.10,
        hold_trading_days=3,
        t_plus_one=True,
    )
    assert lab is not None
    assert lab["entry_price"] == 101.0
    assert lab["exit_reason"] == "time"
    # Entry Tue=day1, Wed=day2, Thu=day3 close → 2024-01-05
    assert lab["exit_date"] == pd.Timestamp("2024-01-05")
    assert abs(lab["ret"] - (103.5 / 101.0 - 1.0)) < 1e-9


def test_ashare_t_plus_one_blocks_entry_day_tp() -> None:
    # Entry Tue open=100; Tue high=120 would be +20% but A-share cannot sell same day.
    dates = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    p = _panel_from_ohlc(
        dates,
        opens=[100, 100, 100, 100],
        highs=[100, 120, 101, 101],
        lows=[99, 99, 99, 99],
        closes=[100, 110, 100, 100],
    )
    lab = path_label_next_open_tp(
        p, "2024-01-02", take_profit=0.10, hold_trading_days=3, t_plus_one=True
    )
    assert lab is not None
    # First sellable day is Wed; high=101 < 110 → continue; Thu is day 3 → time
    # Wait: entry Tue=day1, Wed=day2 can sell, high=101 < 110, Thu=day3 time
    assert lab["exit_date"] == pd.Timestamp("2024-01-05")
    assert lab["exit_reason"] == "time"


def test_hk_allows_same_day_tp() -> None:
    dates = ["2024-01-02", "2024-01-03", "2024-01-04"]
    p = _panel_from_ohlc(
        dates,
        opens=[100, 100, 100],
        highs=[100, 115, 101],
        lows=[99, 99, 99],
        closes=[100, 110, 100],
    )
    lab = path_label_next_open_tp(
        p, "2024-01-02", take_profit=0.10, hold_trading_days=3, t_plus_one=False
    )
    assert lab is not None
    assert lab["exit_reason"] == "tp_high"
    assert lab["exit_date"] == pd.Timestamp("2024-01-03")
    assert abs(lab["exit_price"] - 110.0 * (1 - 0.001)) < 1e-9


def test_tp_gap_open_uses_open() -> None:
    dates = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    p = _panel_from_ohlc(
        dates,
        opens=[100, 100, 112, 100],
        highs=[100, 101, 113, 101],
        lows=[99, 99, 111, 99],
        closes=[100, 100, 112, 100],
    )
    lab = path_label_next_open_tp(
        p, "2024-01-02", take_profit=0.10, hold_trading_days=3, t_plus_one=True
    )
    assert lab is not None
    assert lab["exit_reason"] == "tp_gap_open"
    assert lab["exit_price"] == 112.0


def test_limit_up_skip_on_entry() -> None:
    dates = ["2024-01-02", "2024-01-03", "2024-01-04"]
    p = _panel_from_ohlc(
        dates,
        opens=[100, 110, 100],
        highs=[100, 111, 101],
        lows=[99, 109, 99],
        closes=[100, 110, 100],
    )
    lab = path_label_next_open_tp(
        p,
        "2024-01-02",
        take_profit=0.10,
        hold_trading_days=3,
        t_plus_one=True,
        limit_up_pct=0.095,
    )
    assert lab is not None
    assert lab["skipped_limit"] is True


def test_simulate_fixed_notional_and_cash_cap() -> None:
    dates = pd.bdate_range("2024-01-02", periods=12)
    ds = [d.strftime("%Y-%m-%d") for d in dates]
    panels = {}
    for sym, base in [("AAA", 10.0), ("BBB", 20.0), ("CCC", 30.0)]:
        closes = [base] * len(ds)
        panels[sym] = _panel_from_ohlc(
            ds,
            opens=closes,
            highs=[base * 1.01] * len(ds),
            lows=[base * 0.99] * len(ds),
            closes=closes,
            rsi=10.0,
            amp=12.0,
        )
    # Fire on first day for all three; sleeve only 20k → 2 slots max by cash
    events = {
        pd.Timestamp(ds[0]): [
            {"symbol": "AAA", "score": 3.0},
            {"symbol": "BBB", "score": 2.0},
            {"symbol": "CCC", "score": 1.0},
        ]
    }
    cal = trading_calendar(panels, min_coverage=1.0)
    market = MarketSpec.ashare(friction=0.0)
    policy = ShortHarvestPolicy(
        slots=8,
        fixed_notional_cny=10_000.0,
        initial_equity_cny=20_000.0,
        hold_trading_days=3,
        take_profit=0.10,
    )
    res = simulate_short_harvest("toy", "rsi", panels, events, cal, market, policy)
    assert res.n_opens == 2
    assert res.n_skipped_cash >= 1
    assert abs(res.final_equity - 20_000.0) < 50  # flat path ≈ flat equity


def test_simulate_ashare_tp_on_second_day() -> None:
    dates = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]
    p = _panel_from_ohlc(
        dates,
        opens=[100, 100, 100, 100, 100],
        highs=[100, 101, 115, 101, 101],
        lows=[99, 99, 99, 99, 99],
        closes=[100, 100, 114, 100, 100],
        rsi=12.0,
        amp=15.0,
    )
    panels = {"S1": p}
    events = {pd.Timestamp("2024-01-02"): [{"symbol": "S1", "score": 1.0}]}
    cal = [pd.Timestamp(d) for d in dates]
    res = simulate_short_harvest(
        "toy",
        "v",
        panels,
        events,
        cal,
        MarketSpec.ashare(friction=0.0),
        ShortHarvestPolicy(initial_equity_cny=10_000.0, take_profit=0.10),
    )
    assert res.n_opens == 1
    assert res.n_tp_exits == 1
    assert res.trades[0].exit_reason == "tp_high"
    assert res.trades[0].entry_date == pd.Timestamp("2024-01-03")
    assert res.trades[0].exit_date == pd.Timestamp("2024-01-04")


def test_cross_market_calendar_isolation() -> None:
    # A-share has Mon-Wed; HK missing Tue → HK hold clock must not advance on Tue.
    a_dates = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    h_dates = ["2024-01-02", "2024-01-04", "2024-01-05", "2024-01-08"]
    a_panel = _panel_from_ohlc(
        a_dates,
        opens=[10, 10, 10, 10],
        highs=[10, 10, 10, 10],
        lows=[9, 9, 9, 9],
        closes=[10, 10, 10, 10],
        rsi=10,
        amp=12,
    )
    h_panel = _panel_from_ohlc(
        h_dates,
        opens=[20, 20, 20, 20],
        highs=[20, 20, 20, 20],
        lows=[19, 19, 19, 19],
        closes=[20, 20, 20, 20],
        rsi=10,
        amp=12,
    )
    a_events = {pd.Timestamp("2024-01-02"): [{"symbol": "A1", "score": 1.0}]}
    h_events = {pd.Timestamp("2024-01-02"): [{"symbol": "H1", "score": 1.0}]}
    res = cross_market_simulate(
        [
            (
                "ashare",
                {"A1": a_panel},
                a_events,
                [pd.Timestamp(d) for d in a_dates],
                MarketSpec.ashare(friction=0.0),
            ),
            (
                "hk",
                {"H1": h_panel},
                h_events,
                [pd.Timestamp(d) for d in h_dates],
                MarketSpec.hk(friction=0.0, fx_to_cny=1.0),
            ),
        ],
        policy=ShortHarvestPolicy(
            slots=8,
            initial_equity_cny=80_000.0,
            hold_trading_days=3,
            take_profit=None,
        ),
    )
    hk_trades = [t for t in res.trades if t.market == "hk"]
    assert hk_trades
    # Entry 01-04 (next HK session after 01-02); day1=01-04, day2=01-05, day3=01-08
    assert hk_trades[0].entry_date == pd.Timestamp("2024-01-04")
    assert hk_trades[0].exit_date == pd.Timestamp("2024-01-08")


def test_build_events_and_calendar_smoke() -> None:
    dates = pd.bdate_range("2024-01-02", periods=30)
    ds = [d.strftime("%Y-%m-%d") for d in dates]
    panels = {
        "X": _panel_from_ohlc(
            ds,
            opens=[10] * 30,
            highs=[11] * 30,
            lows=[9] * 30,
            closes=[10] * 30,
            rsi=10.0,
            amp=12.0,
        )
    }
    ev = build_rsi_amp_events(panels, amp_threshold=8.0, rsi_threshold=18.0, dedupe=5)
    assert len(ev) >= 1
    cal = trading_calendar(panels, min_coverage=1.0)
    assert len(cal) == 30


def test_hk_market_spec_has_no_limit_up() -> None:
    assert MarketSpec.hk().limit_up_pct is None
    assert MarketSpec.hk().t_plus_one is False
    assert MarketSpec.ashare().t_plus_one is True
    assert MarketSpec.ashare().limit_up_pct is not None
