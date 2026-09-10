from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ConstitutionState:
    """
    Minimal runtime snapshot to evaluate constitution constraints.

    Keep this stable: you can add optional fields, but avoid breaking changes.
    """

    task_id: Optional[str] = None
    timestamp: Optional[str] = None

    # Equity & drawdown snapshots (fractions, e.g. 0.07 = 7%).
    equity: Optional[float] = None
    drawdown: Optional[float] = None

    # Period losses (positive means loss fraction).
    daily_loss: float = 0.0
    weekly_loss: float = 0.0
    monthly_loss: float = 0.0

    # Execution/control layer flags.
    hard_violation: bool = False
    data_bad: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "timestamp": self.timestamp,
            "equity": self.equity,
            "drawdown": self.drawdown,
            "daily_loss": float(self.daily_loss),
            "weekly_loss": float(self.weekly_loss),
            "monthly_loss": float(self.monthly_loss),
            "hard_violation": bool(self.hard_violation),
            "data_bad": bool(self.data_bad),
        }
