from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

import scripts.event_backtest.simulator.position as position_module
from scripts.event_backtest.simulator.position import PositionSimulator
from scripts.event_backtest.sizing import sync_event_backtest_sizing_equity
from scripts.event_backtest.spot.budget import (
    allocate_spot_accum_leg,
    build_spot_capital_budget_or_none,
)
from src.time_series_model.core.trade_intent import TradeIntent


def test_sync_compound_updates_risk_per_slot() -> None:
    sim = PositionSimulator(fee_rate=0.0)
    sims = {"BTC": sim}
    sync_event_backtest_sizing_equity(
        simulators=sims,
        equity_usdt=20_000.0,
        risk_per_slot=0.01,
        compound_sizing=True,
        initial_cash_usdt=10_000.0,
    )
    assert sim._risk_per_slot_usdt == 200.0
    assert sim._account_risk_equity == 20_000.0


def test_sync_fixed_base_ignores_equity_growth() -> None:
    sim = PositionSimulator(fee_rate=0.0)
    sims = {"BTC": sim}
    sync_event_backtest_sizing_equity(
        simulators=sims,
        equity_usdt=25_000.0,
        risk_per_slot=0.01,
        compound_sizing=False,
        initial_cash_usdt=10_000.0,
    )
    assert sim._risk_per_slot_usdt == 100.0
    assert sim._account_risk_equity == 10_000.0


def test_sync_updates_spot_budget_equity() -> None:
    sim = PositionSimulator(fee_rate=0.0)
    budget = {"equity_usdt": 10_000.0}
    sync_event_backtest_sizing_equity(
        simulators={"x": sim},
        equity_usdt=15_000.0,
        risk_per_slot=0.01,
        compound_sizing=True,
        initial_cash_usdt=10_000.0,
        spot_capital_budget=budget,
    )
    assert budget["equity_usdt"] == 15_000.0


def test_spot_backtest_refuses_missing_constitution_budget() -> None:
    with pytest.raises(ValueError, match="top-level 'spot' budget"):
        build_spot_capital_budget_or_none(
            constitution_raw={},
            strategy_names=["spot_accum_simple"],
            equity_anchor_usdt=100_000.0,
        )


def test_spot_allocator_refuses_unbounded_generic_sizing() -> None:
    sim = SimpleNamespace(_spot_capital_budget=None)
    with pytest.raises(RuntimeError, match="no capital budget"):
        allocate_spot_accum_leg(
            sim,
            archetype_lc="spot_accum_simple",
            symbol="BTCUSDT",
            intent_base_add_m=1.0,
            parent_pos_for_merge=None,
            now_ts=datetime(2026, 9, 7, tzinfo=timezone.utc),
        )


def test_spot_open_rejects_zero_quote_leg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        position_module,
        "_allocate_spot_accum_leg",
        lambda *args, **kwargs: (True, 1.0, 0.0, "", ""),
    )
    sim = PositionSimulator(fee_rate=0.0)
    intent = TradeIntent(
        action="LONG",
        symbol="BTCUSDT",
        archetype="spot_accum_simple",
        size_multiplier=1.0,
        execution_profile={"execution_constraints": {}},
    )
    opened = sim.open_position(
        intent,
        {
            "timestamp": datetime(2026, 9, 7, tzinfo=timezone.utc),
            "close": 50_000.0,
            "atr": 1_000.0,
        },
        {"atr": 1_000.0},
    )
    assert opened is None
    assert sim.last_open_reject_reason == "spot_budget_zero_leg"
    assert sim._positions == {}


def test_spot_close_never_uses_futures_multiplier_fallback() -> None:
    sim = PositionSimulator(fee_rate=0.0)
    sim._risk_per_slot_usdt = 1_000.0
    econ = sim._build_close_economics(
        pos={
            "archetype": "spot_accum_simple",
            "side": "LONG",
            "entry_price": 50_000.0,
            "_size_multiplier": 1_345.0,
            "_qty_base": 0.0,
            "_entry_notional_usdt": 0.0,
        },
        exit_price=60_000.0,
    )
    assert econ["notional_usdt"] == 0.0
    assert econ["qty_base"] == 0.0
    assert econ["pnl_usd_realized"] == 0.0
