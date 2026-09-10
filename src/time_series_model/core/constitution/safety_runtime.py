from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore

from src.order_management.storage import Storage


@dataclass
class SafetyRuntimeState:
    halted: bool = False
    halt_reason: List[str] = field(default_factory=list)
    halt_since: Optional[str] = None
    cooldown_until: Optional[str] = None
    last_metrics: Dict[str, Any] = field(default_factory=dict)
    last_reset_date: Optional[str] = None
    last_daily_halt_date: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "halted": bool(self.halted),
            "halt_reason": list(self.halt_reason or []),
            "halt_since": self.halt_since,
            "cooldown_until": self.cooldown_until,
            "last_metrics": dict(self.last_metrics or {}),
            "last_reset_date": self.last_reset_date,
            "last_daily_halt_date": self.last_daily_halt_date,
        }

    @staticmethod
    def from_dict(obj: Optional[Dict[str, Any]]) -> "SafetyRuntimeState":
        raw = dict(obj or {})
        return SafetyRuntimeState(
            halted=bool(raw.get("halted", False)),
            halt_reason=list(raw.get("halt_reason") or []),
            halt_since=raw.get("halt_since"),
            cooldown_until=raw.get("cooldown_until"),
            last_metrics=dict(raw.get("last_metrics") or {}),
            last_reset_date=raw.get("last_reset_date"),
            last_daily_halt_date=raw.get("last_daily_halt_date"),
        )


@dataclass(frozen=True)
class SafetyDecision:
    ok: bool
    reasons: List[str]
    state: SafetyRuntimeState


def _to_tz(now: datetime, tz_name: Optional[str]) -> datetime:
    if tz_name and ZoneInfo is not None:
        try:
            return now.astimezone(ZoneInfo(str(tz_name)))
        except Exception:
            pass
    return now


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except Exception:
        return None


def load_safety_state(*, db_path: str, state_id: str = "global") -> SafetyRuntimeState:
    storage = Storage(str(db_path))
    payload = storage.get_safety_state(state_id=state_id)
    return SafetyRuntimeState.from_dict(payload)


def save_safety_state(
    *, db_path: str, state: SafetyRuntimeState, state_id: str = "global"
) -> None:
    storage = Storage(str(db_path))
    storage.upsert_safety_state(state_id=state_id, payload=state.as_dict())


def evaluate_safety_state(
    *,
    state: SafetyRuntimeState,
    now: datetime,
    cooldown_minutes: int,
    daily_reset_tz: Optional[str],
    daily_loss: float,
    weekly_loss: float,
    monthly_loss: float,
    drawdown: Optional[float],
    hard_violation: bool,
    data_bad: bool,
    daily_cost_mean: Optional[float],
    daily_turnover_mean: Optional[float],
    limits: Dict[str, float],
) -> SafetyDecision:
    current_day = _to_tz(now, daily_reset_tz).date().isoformat()
    state.last_reset_date = current_day

    reasons: List[str] = []
    if drawdown is not None and float(drawdown) > float(limits["max_dd"]):
        reasons.append("max_dd")
    if float(daily_loss) >= float(limits["daily_loss_limit"]):
        reasons.append("daily_loss_limit")
    if float(weekly_loss) >= float(limits["weekly_loss_limit"]):
        reasons.append("weekly_loss_limit")
    if float(monthly_loss) >= float(limits["monthly_loss_limit"]):
        reasons.append("monthly_loss_limit")
    if daily_turnover_mean is not None and float(daily_turnover_mean) >= float(
        limits["max_turnover_mean"]
    ):
        reasons.append("max_turnover_mean")
    if daily_cost_mean is not None and float(daily_cost_mean) >= float(
        limits["max_cost_mean"]
    ):
        reasons.append("max_cost_mean")
    if bool(data_bad):
        reasons.append("data_bad")
    if bool(hard_violation):
        reasons.append("hard_violation")

    now_iso = now.astimezone(timezone.utc).replace(microsecond=0).isoformat()
    state.last_metrics = {
        "daily_loss": float(daily_loss),
        "weekly_loss": float(weekly_loss),
        "monthly_loss": float(monthly_loss),
        "drawdown": float(drawdown) if drawdown is not None else None,
        "daily_cost_mean": (
            float(daily_cost_mean) if daily_cost_mean is not None else None
        ),
        "daily_turnover_mean": (
            float(daily_turnover_mean) if daily_turnover_mean is not None else None
        ),
    }

    if reasons:
        state.halted = True
        state.halt_reason = list(sorted(set(reasons)))
        state.halt_since = state.halt_since or now_iso
        state.cooldown_until = (
            (now + timedelta(minutes=int(max(0, cooldown_minutes))))
            .replace(microsecond=0)
            .isoformat()
        )
        if "daily_loss_limit" in reasons:
            state.last_daily_halt_date = current_day
        return SafetyDecision(ok=False, reasons=state.halt_reason, state=state)

    if state.halted:
        cooldown_until = _parse_iso(state.cooldown_until)
        cooldown_ok = cooldown_until is None or now >= cooldown_until
        daily_ok = True
        if state.last_daily_halt_date and state.last_daily_halt_date == current_day:
            daily_ok = False
        if cooldown_ok and daily_ok:
            state.halted = False
            state.halt_reason = []
            state.halt_since = None
            state.cooldown_until = None
            state.last_daily_halt_date = None
            return SafetyDecision(ok=True, reasons=[], state=state)
        # If still halted but no current violations, keep previous reasons
        if not reasons:
            # Metrics recovered but cooldown/daily reset not satisfied yet
            return SafetyDecision(ok=False, reasons=state.halt_reason, state=state)

    return SafetyDecision(
        ok=not state.halted,
        reasons=state.halt_reason if state.halted else [],
        state=state,
    )
