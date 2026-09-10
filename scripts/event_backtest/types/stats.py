from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from scripts.event_backtest.types.trade import ClosedTrade
from src.time_series_model.core.constitution.add_position_rules import (
    resolve_add_position_size_multiplier as shared_resolve_add_position_size_multiplier,
)


def resolve_add_position_size_multiplier(add_rules, add_number, signal=None):
    return shared_resolve_add_position_size_multiplier(add_rules, add_number, signal)


def tail_contribution_rate(trades: List[ClosedTrade]) -> tuple[float, int, int]:
    """返回 top10% winners profit share 及计数。"""
    winners = sorted((float(t.pnl_r) for t in trades if t.pnl_r > 0), reverse=True)
    if not winners:
        return 0.0, 0, 0
    top_n = max(1, int(np.ceil(len(winners) * 0.1)))
    win_sum = float(np.sum(winners))
    top_sum = float(np.sum(winners[:top_n]))
    return (top_sum / win_sum) if win_sum > 1e-9 else 0.0, top_n, len(winners)


def clamp01(x: float) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, v))
