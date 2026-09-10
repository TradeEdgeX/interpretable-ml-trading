"""Spot-specific event backtest helpers (budget, metrics, benchmarks)."""

from scripts.event_backtest.spot.budget import (
    allocate_spot_accum_leg,
    build_spot_capital_budget_or_none,
    spot_regime_unit_multiplier,
)
from scripts.event_backtest.spot.metrics import (
    compute_spot_accum_accumulation_audit,
    compute_spot_buy_hold_benchmarks,
    compute_spot_inventory_metrics,
)

__all__ = [
    "allocate_spot_accum_leg",
    "build_spot_capital_budget_or_none",
    "compute_spot_accum_accumulation_audit",
    "compute_spot_buy_hold_benchmarks",
    "compute_spot_inventory_metrics",
    "spot_regime_unit_multiplier",
]
