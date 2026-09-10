from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class HumanOverride:
    tag: str
    reason: str

    def as_dict(self) -> Dict[str, Any]:
        return {"tag": str(self.tag), "reason": str(self.reason)}


@dataclass(frozen=True)
class SystemStateSnapshot:
    """
    Standard snapshot for attribution & replay.

    Keep it stable: add optional fields, do not break existing keys.
    """

    task_id: Optional[str]
    timestamp: Optional[str]

    constitution_hash: Optional[str]
    constitution_yaml: Optional[str]

    router_mode: Optional[str]
    gate_decisions: Dict[str, Any]
    pcm_budget: Dict[str, Any]

    active_slots: Optional[int]
    drawdown: Optional[float]

    # Observability / drift debug (optional, stable extension)
    observability: Optional[Dict[str, Any]] = None

    # Live dashboard: minimal monitoring (optional, stable extension)
    # Keys: active_archetype, size_cap, kill_switch_state, drawdown, daily_loss
    live_dashboard: Optional[Dict[str, Any]] = None

    kpi_gate: Optional[Dict[str, Any]] = None
    safety_state: Optional[Dict[str, Any]] = None
    overrides: Optional[List[HumanOverride]] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "timestamp": self.timestamp,
            "constitution_hash": self.constitution_hash,
            "constitution_yaml": self.constitution_yaml,
            "router_mode": self.router_mode,
            "gate_decisions": dict(self.gate_decisions or {}),
            "pcm_budget": dict(self.pcm_budget or {}),
            "active_slots": self.active_slots,
            "drawdown": self.drawdown,
            "observability": (
                dict(self.observability or {})
                if self.observability is not None
                else None
            ),
            "live_dashboard": (
                dict(self.live_dashboard or {})
                if self.live_dashboard is not None
                else None
            ),
            "kpi_gate": self.kpi_gate,
            "safety_state": (
                dict(self.safety_state or {}) if self.safety_state is not None else None
            ),
            "overrides": [o.as_dict() for o in (self.overrides or [])],
        }


def write_state_snapshot(
    *, out_path: str | Path, snapshot: SystemStateSnapshot
) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(snapshot.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
