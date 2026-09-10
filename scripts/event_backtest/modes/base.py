"""Event backtest mode classification (spot vs trend PCM vs external multi-leg)."""

from __future__ import annotations

from enum import Enum
from typing import Iterable, List, Set


class BacktestMode(str, Enum):
    """Which execution path owns the run.

      - SPOT: constitution ``spot`` budget, merge-by-archetype, partial profit ladder
      - TREND: default PCM + PositionSimulator (BPC/FER/ME/SRB, etc.)
    - MULTI_LEG: separate scripts (``chop_grid_backtest``, ``multi_leg_trading_map``);
        not handled by this engine.
    """

    SPOT = "spot"
    TREND = "trend"
    MULTI_LEG = "multi_leg"


_SPOT_STRATEGY_NAMES = frozenset({"spot_accum", "spot_accum_simple"})


def resolve_backtest_mode(strategy_names: Iterable[str]) -> BacktestMode:
    names = {str(x or "").strip().lower() for x in strategy_names}
    if names & _SPOT_STRATEGY_NAMES:
        return BacktestMode.SPOT
    return BacktestMode.TREND


def spot_strategy_names(strategy_names: Iterable[str]) -> Set[str]:
    return {
        str(x) for x in strategy_names if str(x).strip().lower() in _SPOT_STRATEGY_NAMES
    }
