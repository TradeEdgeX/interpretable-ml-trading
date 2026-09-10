from __future__ import annotations

from datetime import datetime, timezone

import pytest

from scripts.event_backtest.simulator.position import PositionSimulator
from src.time_series_model.portfolio.slot_sizing import compute_slot_size_from_risk


def _sim_with_budget(*, risk_per_slot_usdt: float = 100.0) -> PositionSimulator:
    sim = PositionSimulator(fee_rate=0.0)
    sim._risk_per_slot_usdt = float(risk_per_slot_usdt)
    return sim


def test_cap_slot_notional_reads_constitution_max_gross_leverage() -> None:
    sim = _sim_with_budget(risk_per_slot_usdt=100.0)
    sim._account_risk_equity = 10_000.0
    sim._account_risk_limits = {"enabled": True, "max_gross_leverage": 3.0}
    assert sim._slot_sizing_max_leverage() == pytest.approx(3.0)
    # uncapped 250k vs cap 30k
    assert sim._cap_slot_notional_usdt(250_000.0) == pytest.approx(30_000.0)


def test_estimate_entry_notional_applies_leverage_cap_on_tight_stop() -> None:
    sim = _sim_with_budget(risk_per_slot_usdt=100.0)
    sim._account_risk_equity = 10_000.0
    sim._account_risk_limits = {"max_gross_leverage": 3.0}
    pos = {"effective_stop_pct": 0.0005}  # 0.05% stop → risk-limited 200k notional
    notional = sim._estimate_entry_notional_usdt(
        pos=pos,
        entry_price=50_000.0,
        size_multiplier=1.0,
    )
    assert notional == pytest.approx(30_000.0)


def test_add_sizing_uses_compute_slot_size_from_risk_with_cap() -> None:
    sim = _sim_with_budget(risk_per_slot_usdt=100.0)
    sim._account_risk_equity = 10_000.0
    sim._account_risk_limits = {"max_gross_leverage": 3.0}
    risk_scale = 0.01 * 1.25  # 1.25R add
    sl_r = 0.5
    atr = 50.0
    price = 50_000.0
    sizing = compute_slot_size_from_risk(
        equity_usd=10_000.0,
        risk_frac=risk_scale,
        price=price,
        atr=atr,
        stop_atr=sl_r,
        max_leverage=sim._slot_sizing_max_leverage(),
    )
    assert sizing.notional_usd == pytest.approx(30_000.0)
    assert sizing.qty == pytest.approx(0.6)


def test_pnl_r_matches_realized_over_risk_budget_on_add_leg() -> None:
    sim = _sim_with_budget(risk_per_slot_usdt=100.0)
    pos = {
        "side": "LONG",
        "_size_multiplier": 3.0,
        "_entry_notional_usdt": 300.0,
        "_qty_base": 300.0 / 0.39,
        "entry_price": 0.39,
        "initial_risk_distance": 0.0065,
        "atr_at_entry": 0.0065,
        "effective_stop_pct": 0.0,
    }
    econ = sim._build_close_economics(pos=pos, exit_price=1.22)
    pnl_r = sim._pnl_r_from_economics(
        pos=pos,
        econ=econ,
        entry_price=0.39,
        exit_price=1.22,
    )
    realized = float(econ["pnl_usd_realized"])
    assert realized > 500.0
    assert pnl_r == pytest.approx(realized / 300.0, rel=1e-6)


def test_price_path_fallback_when_no_risk_budget() -> None:
    sim = _sim_with_budget(risk_per_slot_usdt=0.0)
    pos = {
        "side": "LONG",
        "_size_multiplier": 1.0,
        "initial_risk_distance": 10.0,
        "atr_at_entry": 10.0,
    }
    econ = {"pnl_usd_realized": 50.0}
    pnl_r = sim._pnl_r_from_economics(
        pos=pos,
        econ=econ,
        entry_price=100.0,
        exit_price=110.0,
    )
    assert pnl_r == pytest.approx(1.0)
