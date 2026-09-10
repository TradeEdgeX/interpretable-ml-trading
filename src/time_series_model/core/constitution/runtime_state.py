from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SlotRecord:
    position_id: str
    symbol: Optional[str] = None
    archetype: Optional[str] = None
    opened_at: Optional[str] = None
    closed_at: Optional[str] = None
    close_reason: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "position_id": str(self.position_id),
            "symbol": self.symbol,
            "archetype": self.archetype,
            "opened_at": self.opened_at,
            "closed_at": self.closed_at,
            "close_reason": self.close_reason,
        }


@dataclass
class SlotsRuntimeState:
    active: Dict[str, SlotRecord] = field(default_factory=dict)  # position_id -> record

    def active_count(self) -> int:
        return int(len(self.active))

    def as_dict(self) -> Dict[str, Any]:
        return {"active": {k: v.as_dict() for k, v in (self.active or {}).items()}}


@dataclass
class AddPositionRecord:
    position_id: str
    add_count: int = 0
    locked_profit: bool = False
    current_r: Optional[float] = None
    updated_at: Optional[str] = None
    #: ISO UTC timestamp of last successful add fill for this parent (live + persisted).
    #: Used with ``execution_constraints.min_order_interval_minutes`` (between adds only).
    last_add_at: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "position_id": str(self.position_id),
            "add_count": int(self.add_count),
            "locked_profit": bool(self.locked_profit),
            "current_r": self.current_r,
            "updated_at": self.updated_at,
            "last_add_at": self.last_add_at,
        }


@dataclass
class AddPositionRuntimeState:
    positions: Dict[str, AddPositionRecord] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "positions": {k: v.as_dict() for k, v in (self.positions or {}).items()}
        }


@dataclass
class ConstitutionRuntimeState:
    slots: SlotsRuntimeState = field(default_factory=SlotsRuntimeState)
    add_position: AddPositionRuntimeState = field(
        default_factory=AddPositionRuntimeState
    )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "slots": self.slots.as_dict(),
            "add_position": self.add_position.as_dict(),
        }
