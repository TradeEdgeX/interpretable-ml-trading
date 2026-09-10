"""CrossPoolGuard — 独立资金模式下的跨池风控.

多 PCM 架构下, 每个 LivePCM 实例只管理一个策略, 彼此不可见.
CrossPoolGuard 提供只读的跨池约束, 不修改各池的 slot_evidence.

A' 模式下唯一跨池约束: global exposure cap.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CrossPoolGuard:
    """跨池风控 (只读)."""

    def __init__(self, gross_leverage_cap: float = 3.0, same_side_only: bool = False):
        """
        Args:
            gross_leverage_cap: 总杠杆上限 (默认 3.0, 即总敞口不超过资本的 3x)
            same_side_only: True=同 symbol 只允许同向持仓 (A'-same-side 变体)
        """
        self.gross_leverage_cap = float(gross_leverage_cap)
        self.same_side_only = bool(same_side_only)

    def snapshot_all_positions(
        self, pcm_slot_snapshots: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """合并各池的 open positions 快照."""
        all_positions: List[Dict[str, Any]] = []
        for pool_name, positions in pcm_slot_snapshots.items():
            for pos in positions:
                pos_copy = dict(pos)
                pos_copy["_pool"] = pool_name
                all_positions.append(pos_copy)
        return all_positions

    def check_global_exposure(
        self,
        pcm_slot_snapshots: Dict[str, List[Dict[str, Any]]],
        total_capital: float = 1000.0,
        base_risk_per_slot: float = 0.01,
    ) -> tuple[bool, float, str]:
        """检查总敞口是否超过上限.

        Returns:
            (ok, current_leverage, reason)
        """
        total_notional = 0.0
        for positions in pcm_slot_snapshots.values():
            for pos in positions:
                notional = float(pos.get("notional_usdt", 0) or 0)
                if notional <= 0:
                    qty = float(pos.get("qty_base", 0) or 0)
                    entry = float(pos.get("entry_price", 0) or 0)
                    notional = qty * entry
                total_notional += abs(notional)

        leverage = total_notional / total_capital if total_capital > 0 else float("inf")
        ok = leverage <= self.gross_leverage_cap

        if not ok:
            reason = f"gross_leverage={leverage:.2f} > cap={self.gross_leverage_cap}"
        else:
            reason = ""
        return ok, leverage, reason

    def check_same_side(
        self,
        symbol: str,
        intent_side: str,
        pcm_slot_snapshots: Dict[str, List[Dict[str, Any]]],
    ) -> tuple[bool, str]:
        """A'-same-side 变体: 同 symbol 上已有反方向持仓时拒绝.

        Returns:
            (allowed, reason)
        """
        if not self.same_side_only:
            return True, ""

        intent_side_norm = str(intent_side).upper().strip()
        if intent_side_norm not in ("LONG", "SHORT"):
            return True, ""

        opposite = "SHORT" if intent_side_norm == "LONG" else "LONG"

        for positions in pcm_slot_snapshots.values():
            for pos in positions:
                if str(pos.get("symbol", "")).upper() != symbol.upper():
                    continue
                pos_side = str(pos.get("side", "")).upper().strip()
                pos_side_bucket = (
                    "LONG"
                    if pos_side in ("LONG", "BUY")
                    else ("SHORT" if pos_side in ("SHORT", "SELL") else "")
                )
                if pos_side_bucket == opposite:
                    return False, (
                        f"same_side_only: {symbol} has {opposite} position, "
                        f"reject {intent_side_norm}"
                    )
        return True, ""

    def check_intent(
        self,
        symbol: str,
        archetype: str,
        intent_side: str,
        intent_notional: float,
        pcm_slot_snapshots: Dict[str, List[Dict[str, Any]]],
        total_capital: float = 1000.0,
    ) -> tuple[bool, str]:
        """综合检查: same_side + 预估 exposure.

        Args:
            symbol: 交易对
            archetype: 策略原型
            intent_side: LONG/SHORT
            intent_notional: 拟开仓的名义价值
            pcm_slot_snapshots: 各池当前持仓快照
            total_capital: 总资本

        Returns:
            (allowed, reason)
        """
        # 1. same_side check
        allowed, reason = self.check_same_side(symbol, intent_side, pcm_slot_snapshots)
        if not allowed:
            return False, reason

        # 2. exposure check (预估: 加上 intent 的 notional)
        ok, leverage, exp_reason = self.check_global_exposure(
            pcm_slot_snapshots, total_capital
        )
        if not ok:
            # 预估加上新仓位后是否超限
            total_notional = sum(
                float(p.get("notional_usdt", 0) or 0)
                for positions in pcm_slot_snapshots.values()
                for p in positions
            )
            projected = (total_notional + abs(intent_notional)) / total_capital
            if projected > self.gross_leverage_cap:
                return False, (
                    f"projected leverage={projected:.2f} > cap={self.gross_leverage_cap}"
                )

        return True, ""
