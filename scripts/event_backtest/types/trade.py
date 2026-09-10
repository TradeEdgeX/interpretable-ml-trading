from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Mapping


def srb_closed_trade_fields(pos: Mapping[str, Any]) -> Dict[str, Any]:
    """Copy SRB true-SR audit fields from an open position dict onto ClosedTrade kwargs."""

    def _f(key: str) -> float:
        try:
            v = float(pos.get(key, 0.0) or 0.0)
            return v if v == v else 0.0
        except (TypeError, ValueError):
            return 0.0

    return {
        "srb_true_sr_level": _f("_srb_true_sr_level"),
        "srb_true_sr_source": str(pos.get("_srb_true_sr_source") or ""),
        "srb_l1_support": _f("_srb_l1_support"),
        "srb_l1_resistance": _f("_srb_l1_resistance"),
        "srb_l3_lower": _f("_srb_l3_lower"),
        "srb_l3_upper": _f("_srb_l3_upper"),
    }


@dataclass
class ClosedTrade:
    symbol: str
    side: str  # LONG / SHORT
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    atr_at_entry: float
    pnl_r: float  # PnL in R-multiples
    pnl_usd: float  # realized USDT PnL (same as pnl_usd_realized when economics known)
    exit_reason: str
    pnl_usd_realized: float = 0.0  # realized PnL in USDT (qty/notional aware)
    notional_usdt: float = 0.0  # entry quote notional
    qty_base: float = 0.0  # base asset quantity
    entry_fee_usdt: float = 0.0
    exit_fee_usdt: float = 0.0
    exit_notional_usdt: float = 0.0
    archetype: str = ""
    bars_held: int = 0
    is_add_position: bool = False  # 加仓标记
    is_reverse: bool = False  # SRB 假突破反手标记
    size_multiplier: float = 1.0  # regime position scale
    atr_stop_pct: float = 0.0
    effective_stop_pct: float = 0.0
    sizing_stop_source: str = ""
    # 平仓时刻是否已触发保本锁（止损价已按 breakeven 规则上移/下移）
    breakeven_locked_at_exit: bool = False
    # SRB：入场时 true_sr 锚点（价位 + 来源 L1/L3/entry）及当时 L1/L3 快照
    srb_true_sr_level: float = 0.0
    srb_true_sr_source: str = ""
    srb_l1_support: float = 0.0
    srb_l1_resistance: float = 0.0
    srb_l3_lower: float = 0.0
    srb_l3_upper: float = 0.0
