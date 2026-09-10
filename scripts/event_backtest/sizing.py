"""Event backtest position-sizing equity sync (compound vs fixed-base)."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional


def sync_event_backtest_sizing_equity(
    *,
    simulators: Mapping[str, Any],
    equity_usdt: float,
    risk_per_slot: float,
    compound_sizing: bool,
    initial_cash_usdt: float,
    spot_capital_budget: Optional[Dict[str, Any]] = None,
) -> float:
    """Align simulator risk budgets with current (or initial) equity.

    Constitution intent: ``risk_usd = equity × risk_per_slot``. Event backtests
    previously froze ``initial_cash × risk_per_slot`` for the whole run.
    """
    anchor = max(
        0.0,
        float(equity_usdt if compound_sizing else initial_cash_usdt),
    )
    risk_unit = anchor * float(risk_per_slot)
    for sim in simulators.values():
        sim._risk_per_slot_usdt = risk_unit
        sim._account_risk_equity = anchor
    if isinstance(spot_capital_budget, dict):
        spot_capital_budget["equity_usdt"] = anchor
    return anchor
