"""Regression: macro VWAP level + 1m pv 用于 vwap1200 结构出场（仅 pv 符号穿越）。"""

from __future__ import annotations

import pandas as pd

from scripts.event_backtest import (
    PositionSimulator,
    _sync_macro_tp_vwap_from_feature_row,
)
from src.time_series_model.live.position_logic import enforce_position


def test_frozen_vwap_level_identity_from_feature_row() -> None:
    close = 1.0
    pv = 0.02
    vwap = close * (1.0 - pv)
    assert abs((close - vwap) / close - pv) < 1e-9


def test_live_pv_near_frozen_level() -> None:
    sim = PositionSimulator()
    row = pd.Series(
        {"close": 100.0, "macro_tp_vwap_1200_position": 0.02},
    )
    _sync_macro_tp_vwap_from_feature_row(sim, row)
    assert sim._macro_tp_vwap_level == 98.0
    bar_close = 98.2
    live_pv = (bar_close - float(sim._macro_tp_vwap_level)) / bar_close
    assert abs(live_pv) < 0.005


def test_position_simulator_passes_intraminute_pv_to_enforce() -> None:
    sim = PositionSimulator()
    sim._macro_tp_vwap_level = 100.0
    sim._macro_tp_vwap_position = 0.05
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    pos = {
        "side": "LONG",
        "entry_price": 99.0,
        "entry_time": now,
        "atr_at_entry": 1.0,
        "max_holding_bars": 0,
        "bar_minutes": 120,
        "stop_loss_price": 90.0,
        "structural_exit": "vwap1200",
        "breakeven_locked": True,
    }
    sim._positions["p1"] = pos
    closed = sim.update(
        {
            "timestamp": now,
            "open": 99.0,
            "high": 99.1,
            "low": 98.0,
            "close": 98.5,
        }
    )
    assert len(closed) == 1
    assert closed[0].exit_reason == "structural_exit_vwap1200"
