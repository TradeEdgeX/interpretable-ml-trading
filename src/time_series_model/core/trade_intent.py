"""TradeIntent — 统一交易意图数据结构

GenericLiveStrategy 和 OrderFlowListener 共用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class TradeIntent:
    action: str  # LONG|SHORT|NO_TRADE
    symbol: str
    archetype: str
    execution_strategy: Optional[str] = None
    confidence: Optional[float] = None
    quantity: Optional[float] = None
    size_multiplier: Optional[float] = None
    position_id: Optional[str] = None
    add_position: bool = False
    parent_position_id: Optional[str] = None
    current_r: Optional[float] = None
    locked_profit: Optional[bool] = None
    execution_tags: Optional[List[str]] = None
    execution_evidence: Optional[Dict[str, bool]] = None
    execution_profile: Optional[Dict[str, Any]] = None
    pcm_budget: Optional[Dict[str, Any]] = None
