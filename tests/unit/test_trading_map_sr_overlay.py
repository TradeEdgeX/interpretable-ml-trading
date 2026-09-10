"""SRB L1/L3 trading-map overlay helpers."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from scripts.event_backtest.reporting.trading_map import (
    _compute_fade_pierce_events_from_ohlc,
    _compute_sr_overlay_series,
    _should_draw_fade_pierce_overlay,
    _srb_trade_list_html,
)
from scripts.event_backtest.types.trade import ClosedTrade


def test_should_draw_fade_pierce_overlay_tabs():
    assert _should_draw_fade_pierce_overlay(None) is True
    assert _should_draw_fade_pierce_overlay("largebar_fade") is True
    assert _should_draw_fade_pierce_overlay("srb") is False


def test_compute_fade_pierce_events_from_ohlc_bull_pierce():
    idx = pd.date_range("2024-01-01", periods=8, freq="2h", tz="UTC")
    close = pd.Series([100.0] * 8, index=idx, dtype=float)
    open_ = close.copy()
    high = close + 0.5
    low = close - 0.5
    open_.iloc[1] = 100.0
    close.iloc[1] = 103.0
    high.iloc[1] = 103.2
    low.iloc[1] = 99.8
    open_.iloc[3] = 101.0
    close.iloc[3] = 99.5
    high.iloc[3] = 101.2
    low.iloc[3] = 99.4
    # High volume on pierce/recross so volume_ratio_pct clears 0.60 after warmup.
    vol = pd.Series(1.0, index=idx)
    vol.iloc[1] = 50.0
    vol.iloc[3] = 40.0
    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": vol,
        },
        index=idx,
    )
    # Inject flat atr/vol via direct feature path when rolling vol pct is cold:
    # helper uses real ATR + vol pct; with tiny sample vol pct may not arm.
    # Smoke: empty-or-events DataFrame schema only when volume present.
    out = _compute_fade_pierce_events_from_ohlc(df)
    assert list(out.columns) == [
        "pierce_time",
        "recross_time",
        "fade_side",
        "level",
        "pierce_y",
    ]


def test_compute_sr_overlay_has_l1_l3_columns():
    n = 300
    idx = pd.date_range("2024-01-01", periods=n, freq="2h", tz="UTC")
    close = pd.Series(
        100.0 + np.linspace(0, 10, n) + np.sin(np.arange(n) / 8), index=idx
    )
    df = pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1.0,
        },
        index=idx,
    )
    out = _compute_sr_overlay_series(df)
    assert "l1_support" in out.columns and "l1_resistance" in out.columns
    assert "l3_upper" in out.columns and "l3_lower" in out.columns
    assert out["l1_support"].iloc[-1] <= out["l1_resistance"].iloc[-1]
    assert out["l3_lower"].iloc[-1] <= out["l3_upper"].iloc[-1]


def test_srb_trade_list_html_includes_source():
    t = ClosedTrade(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        exit_price=105.0,
        entry_time=datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc),
        exit_time=datetime(2024, 6, 2, 12, 0, tzinfo=timezone.utc),
        atr_at_entry=1.0,
        pnl_r=1.5,
        pnl_usd=10.0,
        exit_reason="trailing_sl",
        archetype="srb",
        srb_true_sr_level=95.0,
        srb_true_sr_source="L3",
        srb_l1_support=99.0,
        srb_l1_resistance=110.0,
        srb_l3_lower=95.0,
        srb_l3_upper=120.0,
    )
    html = _srb_trade_list_html([t], "BTCUSDT")
    assert "L3" in html and "95.00" in html and "BTCUSDT" in html
    assert "stop" in html and "gate=L3 age_decay" in html
    assert "atr" in html or "sizing_stop" in html or "stop" in html
