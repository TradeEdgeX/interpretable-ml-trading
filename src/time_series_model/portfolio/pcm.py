from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Any

import yaml

from time_series_model.rl.router_types import RouterAction
from time_series_model.portfolio.ppath import assert_ppath_usage


@dataclass(frozen=True)
class SlotSnapshot:
    """
    Snapshot of an active slot for PCM decisions.
    Values are ex-post only (no predictive fields).
    """

    position_id: str
    symbol: str
    regime: str  # TREND/MEAN (kept for backward compatibility)
    archetype: str  # TC, TE, FR, ET (new: archetype as asset unit)

    # ppath is ex-post progress only (see ARCHITECTURE.md)
    ppath: float
    remaining_ppath: float

    # Risk release signals
    breakeven_locked: bool = False
    risk_released_r: float = 0.0

    # Slot validity
    trend_path_active: bool = True


@dataclass(frozen=True)
class CandidateSignal:
    """
    New opportunity evaluated by PCM (already passed Gate/Execution eligibility).
    """

    signal_id: str
    symbol: str
    regime: str  # TREND/MEAN (kept for backward compatibility)
    archetype: str  # TC, TE, FR, ET (new: archetype as asset unit)
    ppath: float
    features: Optional[Dict[str, Any]] = field(
        default=None
    )  # Optional features for reflexivity checks


@dataclass(frozen=True)
class AddOnRequest:
    """
    Add-on request for an existing position.
    """

    position_id: str
    regime: str  # TREND/MEAN (kept for backward compatibility)
    archetype: str  # TC, TE, FR, ET (new: archetype as asset unit)
    add_count: int
    locked_profit: bool


@dataclass(frozen=True)
class PCMPolicy:
    """
    V0 PCM policy:
    - max_slots=2 fixed
    - slot2 requires risk release on slot1 + trend path active
    - replacement only by ppath dominance (ex-post)
    - add-on only for TREND, max once, requires locked profit
    """

    max_slots: int = 2
    require_slot1_risk_release_for_slot2: bool = True
    risk_release_threshold_r: float = 0.0
    require_slot1_trend_active_for_slot2: bool = True
    replacement_margin: float = 0.0
    add_on_max_times: int = 1


@dataclass(frozen=True)
class PCMDecision:
    allow_entry: bool
    allow_add_on: bool
    replace_position_id: Optional[str]
    reasons: List[str]


def _is_slot_risk_released(slot: SlotSnapshot, *, policy: PCMPolicy) -> bool:
    if slot.breakeven_locked:
        return True
    return float(slot.risk_released_r) >= float(policy.risk_release_threshold_r)


def _pick_weakest_slot(slots: List[SlotSnapshot]) -> Optional[SlotSnapshot]:
    if not slots:
        return None
    return sorted(slots, key=lambda s: float(s.remaining_ppath))[0]


def _are_archetypes_compatible(arch1: str, arch2: str) -> bool:
    """
    Check if two archetypes are compatible (can coexist in different slots).

    Compatibility rules:
    - TC + TE: compatible (both are trend-following)
    - FR + ET: compatible (both are mean-reversion/reversal)
    - Other combinations: incompatible (semantically opposite)

    Args:
        arch1: First archetype (TC, TE, FR, ET)
        arch2: Second archetype (TC, TE, FR, ET)

    Returns:
        True if compatible, False otherwise
    """
    arch1_upper = str(arch1).upper().strip()
    arch2_upper = str(arch2).upper().strip()

    # Same archetype is always compatible (but should not happen in practice)
    if arch1_upper == arch2_upper:
        return True

    # Trend-following archetypes: TC and TE
    trend_archetypes = {"TC", "TRENDCONTINUATIONTC", "TE", "TRENDEXPANSIONTE"}
    # Mean-reversion/reversal archetypes: FR and ET
    mean_archetypes = {"FR", "FAILUREREVERSIONFR", "ET", "EXHAUSTIONTURNET"}

    arch1_is_trend = arch1_upper in trend_archetypes
    arch2_is_trend = arch2_upper in trend_archetypes
    arch1_is_mean = arch1_upper in mean_archetypes
    arch2_is_mean = arch2_upper in mean_archetypes

    # Both are trend-following: compatible
    if arch1_is_trend and arch2_is_trend:
        return True

    # Both are mean-reversion: compatible
    if arch1_is_mean and arch2_is_mean:
        return True

    # Mixed (trend + mean): incompatible
    return False


def decide_pcm(
    *,
    policy: PCMPolicy,
    active_slots: List[SlotSnapshot],
    candidate: Optional[CandidateSignal],
    add_on: Optional[AddOnRequest],
) -> PCMDecision:
    reasons: List[str] = []
    allow_entry = False
    allow_add_on = False
    replace_position_id: Optional[str] = None

    # Add-on decision (TREND only, locked profit, max once)
    if add_on is not None:
        if str(add_on.regime).upper() != "TREND":
            reasons.append("add_on:regime_not_trend")
        elif int(add_on.add_count) >= int(policy.add_on_max_times):
            reasons.append("add_on:max_times")
        elif not bool(add_on.locked_profit):
            reasons.append("add_on:locked_profit_required")
        else:
            allow_add_on = True

    # Entry / rotation decision
    if candidate is None:
        return PCMDecision(
            allow_entry=allow_entry,
            allow_add_on=allow_add_on,
            replace_position_id=replace_position_id,
            reasons=reasons,
        )
    assert_ppath_usage("rotation")

    # Reflexivity risk check (as final defense line)
    # Note: Hard veto should already be checked in Gate layer, but we check again here for safety
    if candidate.features is not None:
        try:
            from time_series_model.nnmultihead.gate_reflexivity_risk import (
                check_reflexivity_hard_veto,
            )

            reflexivity_features = {
                "ofci_p": candidate.features.get("ofci_pct", 0.0),
                "shd_p": candidate.features.get("shd_pct", 0.0),
                "lfi_p": candidate.features.get(
                    "lfi_pct", 0.0
                ),  # Reserved for future use
            }

            should_veto, veto_reason = check_reflexivity_hard_veto(reflexivity_features)
            if should_veto:
                reasons.append(f"reflexivity_veto:{veto_reason}")
                return PCMDecision(
                    allow_entry=False,
                    allow_add_on=allow_add_on,
                    replace_position_id=None,
                    reasons=reasons,
                )
        except ImportError:
            # If gate_reflexivity_risk is not available, skip reflexivity check
            pass
        except Exception:
            # If any error occurs, skip reflexivity check (fail-safe)
            pass

    slots = list(active_slots or [])

    # Check archetype compatibility with existing slots (simplified: no conflict rules)
    candidate_arch = (
        str(candidate.archetype).upper().strip()
        if hasattr(candidate, "archetype") and candidate.archetype
        else None
    )

    # Check if candidate is incompatible with existing slots
    for slot in slots:
        slot_arch = (
            str(slot.archetype).upper().strip()
            if hasattr(slot, "archetype") and slot.archetype
            else None
        )
        if not slot_arch or not candidate_arch:
            # Fallback to regime-based logic if archetype not available
            continue

        # Check if candidate is incompatible with this slot
        if not _are_archetypes_compatible(candidate_arch, slot_arch):
            reasons.append(f"archetype_incompatible:{candidate_arch}_vs_{slot_arch}")
            # If incompatible and no free slot, deny entry
            if len(slots) >= int(policy.max_slots):
                return PCMDecision(
                    allow_entry=False,
                    allow_add_on=allow_add_on,
                    replace_position_id=None,
                    reasons=reasons,
                )

    if len(slots) < int(policy.max_slots):
        # Slot1 always allowed if free slot exists
        if len(slots) == 0:
            allow_entry = True
            reasons.append("entry:free_slot")
        else:
            # Slot2: check archetype compatibility first
            slot1 = slots[0]
            slot1_arch = (
                str(slot1.archetype).upper().strip()
                if hasattr(slot1, "archetype") and slot1.archetype
                else None
            )

            if candidate_arch and slot1_arch:
                if not _are_archetypes_compatible(candidate_arch, slot1_arch):
                    reasons.append(
                        f"entry:archetype_incompatible:{candidate_arch}_vs_{slot1_arch}"
                    )
                elif (
                    policy.require_slot1_risk_release_for_slot2
                    and not _is_slot_risk_released(slot1, policy=policy)
                ):
                    reasons.append("entry:slot2_requires_risk_release")
                elif policy.require_slot1_trend_active_for_slot2 and not bool(
                    slot1.trend_path_active
                ):
                    reasons.append("entry:slot2_requires_trend_active")
                elif float(candidate.ppath) <= float(slot1.remaining_ppath):
                    reasons.append("entry:ppath_not_better_than_slot1")
                else:
                    allow_entry = True
                    reasons.append("entry:slot2_allowed")
            else:
                # Fallback to regime-based logic if archetype not available
                if (
                    policy.require_slot1_risk_release_for_slot2
                    and not _is_slot_risk_released(slot1, policy=policy)
                ):
                    reasons.append("entry:slot2_requires_risk_release")
                elif policy.require_slot1_trend_active_for_slot2 and not bool(
                    slot1.trend_path_active
                ):
                    reasons.append("entry:slot2_requires_trend_active")
                elif float(candidate.ppath) <= float(slot1.remaining_ppath):
                    reasons.append("entry:ppath_not_better_than_slot1")
                else:
                    allow_entry = True
                    reasons.append("entry:slot2_allowed")
        return PCMDecision(
            allow_entry=allow_entry,
            allow_add_on=allow_add_on,
            replace_position_id=replace_position_id,
            reasons=reasons,
        )

    # No free slot -> rotation by ppath dominance (ex-post only)
    # Simplified: no archetype compatibility check in rotation, just replace weakest slot
    weakest = _pick_weakest_slot(slots)
    if weakest is None:
        reasons.append("entry:no_active_slot_info")
        return PCMDecision(
            allow_entry=False,
            allow_add_on=allow_add_on,
            replace_position_id=None,
            reasons=reasons,
        )

    thr = float(weakest.remaining_ppath) * float(1.0 + policy.replacement_margin)
    if float(candidate.ppath) > thr:
        allow_entry = True
        replace_position_id = str(weakest.position_id)
        reasons.append("entry:replace_by_ppath")
    else:
        reasons.append("entry:ppath_not_better_than_weakest")

    return PCMDecision(
        allow_entry=allow_entry,
        allow_add_on=allow_add_on,
        replace_position_id=replace_position_id,
        reasons=reasons,
    )


@dataclass(frozen=True)
class ConstitutionKillSwitch:
    enabled: bool = True
    daily_loss_limit: float = 0.04
    weekly_loss_limit: float = 0.08
    monthly_loss_limit: float = 0.12
    kill_on_any_hard_violation: bool = True


@dataclass(frozen=True)
class ConstitutionSlots:
    enabled: bool = True
    slot_count: int = 2
    risk_per_slot: float = 0.01


@dataclass(frozen=True)
class Constitution:
    version: int = 1
    name: str = "Constitution_v1"
    kill_switch: ConstitutionKillSwitch = ConstitutionKillSwitch()
    slots: ConstitutionSlots = ConstitutionSlots()


@dataclass(frozen=True)
class RiskState:
    """
    Minimal risk state snapshot for capital policy.

    This is intentionally small; production systems should extend it, but keep
    the contract stable (add optional fields, don't break existing ones).
    """

    # Loss since period start, as a fraction of equity. Positive means loss.
    daily_loss: float = 0.0
    weekly_loss: float = 0.0
    monthly_loss: float = 0.0

    # Execution/control layer flags.
    hard_violation: bool = False
    data_bad: bool = False

    # Optional: drawdown (fraction) for external risk logic.
    recent_max_dd: Optional[float] = None


@dataclass(frozen=True)
class SymbolDecision:
    """
    Router/gate output for a symbol at a decision timestamp.
    """

    symbol: str
    mode: str  # NO_TRADE|MEAN|TREND
    gated: bool = True

    # A ranking signal (higher => more budget). This is NOT a probability.
    score: float = 0.0


@dataclass(frozen=True)
class CapitalPolicy:
    """
    A simple, explainable capital policy:
    - allocate a fixed budget per mode (MEAN vs TREND)
    - within each mode, split budget across symbols by soft ranking on score
    - apply constitution: kill-switch => global_pause
    """

    base_mode_budgets: Dict[str, float]


@dataclass(frozen=True)
class CapitalResult:
    global_pause: bool
    per_mode_budget: Dict[str, float]
    per_symbol_budget: Dict[str, float]
    reasons: List[str]

    def as_json(self) -> str:
        return json.dumps(
            {
                "global_pause": bool(self.global_pause),
                "per_mode_budget": dict(self.per_mode_budget),
                "per_symbol_budget": dict(self.per_symbol_budget),
                "reasons": list(self.reasons),
            },
            ensure_ascii=False,
            indent=2,
        )

    def as_router_action(self) -> RouterAction:
        """
        Optional bridge: represent mode budgets as RouterAction multipliers.

        Convention (3-action routers):
        - router_name == "MEAN" or "TREND"
        - capital_multiplier[mode] scales base position_size produced by router

        If the downstream router emits per-symbol decisions instead of per-mode,
        prefer using per_symbol_budget directly and ignore this action.
        """

        cm = {}
        for k, v in (self.per_mode_budget or {}).items():
            kk = str(k).upper()
            if kk in {"MEAN", "TREND"}:
                cm[kk] = float(max(0.0, v))
        return RouterAction(
            router_enabled={},
            capital_multiplier=cm,
            global_pause=bool(self.global_pause),
        )


def compute_pcm_budget_for_decisions(
    *,
    decisions: Iterable[SymbolDecision],
    policy: Optional[CapitalPolicy] = None,
    constitution: Optional[Constitution] = None,
    risk: Optional[RiskState] = None,
) -> CapitalResult:
    """
    Convenience wrapper for adapters (live/backtest) that only need budgets.
    """
    pol = policy or CapitalPolicy(base_mode_budgets={"MEAN": 1.0, "TREND": 1.0})
    con = constitution or Constitution()
    rs = risk or RiskState()
    return allocate_capital(constitution=con, policy=pol, risk=rs, decisions=decisions)


def _load_yaml(path: str | Path) -> dict:
    p = Path(path)
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def load_constitution(path: str | Path) -> Constitution:
    obj = _load_yaml(path)
    ks = obj.get("kill_switch") or {}
    slots = obj.get("slots") or {}

    return Constitution(
        version=int(obj.get("version", 1)),
        name=str(obj.get("name", "Constitution_v1")),
        kill_switch=ConstitutionKillSwitch(
            enabled=bool(ks.get("enabled", True)),
            daily_loss_limit=float(ks.get("daily_loss_limit", 0.04)),
            weekly_loss_limit=float(ks.get("weekly_loss_limit", 0.08)),
            monthly_loss_limit=float(ks.get("monthly_loss_limit", 0.12)),
            kill_on_any_hard_violation=bool(ks.get("kill_on_any_hard_violation", True)),
        ),
        slots=ConstitutionSlots(
            enabled=bool(slots.get("enabled", True)),
            slot_count=int(slots.get("slot_count", 2)),
            risk_per_slot=float(slots.get("risk_per_slot", 0.01)),
        ),
    )


def _normalize_mode_budgets(budgets: Dict[str, float]) -> Dict[str, float]:
    b = {str(k).upper(): float(max(0.0, v)) for k, v in (budgets or {}).items()}
    # Ensure keys exist
    for k in ["NO_TRADE", "MEAN", "TREND"]:
        b.setdefault(k, 0.0)
    # NO_TRADE is always 0 (budget is defined only for tradable modes)
    b["NO_TRADE"] = 0.0
    s = float(b.get("MEAN", 0.0) + b.get("TREND", 0.0))
    if s <= 0:
        return {"NO_TRADE": 0.0, "MEAN": 0.0, "TREND": 0.0}
    return {
        "NO_TRADE": 0.0,
        "MEAN": float(b["MEAN"] / s),
        "TREND": float(b["TREND"] / s),
    }


def _kill_switch_triggered(
    *, c: Constitution, risk: RiskState
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if not c.kill_switch.enabled:
        return False, reasons
    if float(risk.daily_loss) >= float(c.kill_switch.daily_loss_limit):
        reasons.append("kill_switch:daily_loss_limit")
    if float(risk.weekly_loss) >= float(c.kill_switch.weekly_loss_limit):
        reasons.append("kill_switch:weekly_loss_limit")
    if float(risk.monthly_loss) >= float(c.kill_switch.monthly_loss_limit):
        reasons.append("kill_switch:monthly_loss_limit")
    if bool(risk.data_bad):
        reasons.append("kill_switch:data_bad")
    if bool(risk.hard_violation):
        reasons.append("kill_switch:hard_violation")
    return (len(reasons) > 0), reasons


def allocate_capital(
    *,
    constitution: Constitution,
    policy: CapitalPolicy,
    risk: RiskState,
    decisions: Iterable[SymbolDecision],
) -> CapitalResult:
    """
    Compute per-mode + per-symbol capital budgets for a single decision timestamp.

    Budgets are *fractions* of total risk budget (not absolute cash),
    and MUST be interpreted under the slot/risk-per-slot constraints.
    """

    kill, kill_reasons = _kill_switch_triggered(c=constitution, risk=risk)
    if kill:
        return CapitalResult(
            global_pause=True,
            per_mode_budget={"NO_TRADE": 0.0, "MEAN": 0.0, "TREND": 0.0},
            per_symbol_budget={str(d.symbol): 0.0 for d in decisions},
            reasons=kill_reasons,
        )

    base = _normalize_mode_budgets(policy.base_mode_budgets)

    reasons: List[str] = []

    # Group decisions by mode, but only if gated.
    per_symbol_budget: Dict[str, float] = {}
    mode_to_items: Dict[str, List[SymbolDecision]] = {"MEAN": [], "TREND": []}
    for d in decisions:
        sym = str(d.symbol)
        mode = str(d.mode).upper()
        if (not bool(d.gated)) or mode == "NO_TRADE":
            per_symbol_budget[sym] = 0.0
            continue
        if mode in mode_to_items:
            mode_to_items[mode].append(d)
        else:
            # Unknown mode: treat as NO_TRADE in v1 (low freedom)
            per_symbol_budget[sym] = 0.0
            reasons.append(f"unknown_mode_as_no_trade:{mode}")

    # Split each mode budget across symbols in that mode by score.
    for mode in ["MEAN", "TREND"]:
        items = mode_to_items.get(mode) or []
        b = float(base.get(mode, 0.0))
        if not items or b <= 0:
            for it in items:
                per_symbol_budget[str(it.symbol)] = 0.0
            continue

        # Score-based soft split: shift by min score to keep non-negative weights.
        scores = [float(it.score) for it in items]
        mn = min(scores) if scores else 0.0
        weights = [max(0.0, s - mn) for s in scores]
        sw = float(sum(weights))
        if sw <= 1e-12:
            # fallback: equal split
            w = 1.0 / float(len(items))
            for it in items:
                per_symbol_budget[str(it.symbol)] = b * w
        else:
            for it, w in zip(items, weights):
                per_symbol_budget[str(it.symbol)] = b * float(w / sw)

    per_mode_budget = dict(base)
    return CapitalResult(
        global_pause=False,
        per_mode_budget=per_mode_budget,
        per_symbol_budget=per_symbol_budget,
        reasons=reasons,
    )
