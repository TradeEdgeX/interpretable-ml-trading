from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from src.time_series_model.core.constitution.execution_evidence import (
    compute_execution_evidence,
)


@dataclass(frozen=True)
class GateRules:
    rules: List[Dict[str, Any]]
    deny_if: List[str]
    allow_if: List[str]
    allow_mode: str
    default_action: str


def _parse_gate_rules(raw: Dict[str, Any]) -> GateRules:
    rules = list(raw.get("rules") or [])
    deny_if = [str(x) for x in (raw.get("deny_if") or [])]
    allow_if = [str(x) for x in (raw.get("allow_if") or [])]
    allow_mode = str(raw.get("allow_mode") or "any").lower()
    default_action = str(raw.get("default_action") or "allow").lower()
    return GateRules(
        rules=rules,
        deny_if=deny_if,
        allow_if=allow_if,
        allow_mode=allow_mode,
        default_action=default_action,
    )


def apply_gate_rules(
    *,
    gate_rules: Dict[str, Any],
    features: Dict[str, Any],
    quantiles: Dict[str, Any] | None = None,
) -> Tuple[bool, List[str]]:
    """
    Returns (ok, reasons). ok=False means veto (NO_TRADE).
    """
    when_then_rules = list(gate_rules.get("when_then_rules") or [])
    if when_then_rules:
        default_action = str(gate_rules.get("default_action") or "deny").lower()
        return apply_when_then_rules(
            when_then_rules=when_then_rules,
            features=features,
            quantiles=quantiles,
            default_action=default_action,
        )
    rules = _parse_gate_rules(gate_rules or {})
    flags = compute_execution_evidence(
        features=features, rules=list(rules.rules or []), quantiles=quantiles
    )
    deny_hits = [name for name in rules.deny_if if flags.get(name)]
    if deny_hits:
        return False, [f"gate_deny={deny_hits}"]
    allow_if = list(rules.allow_if or [])
    if allow_if:
        if rules.allow_mode == "all":
            ok = all(flags.get(name, False) for name in allow_if)
        elif rules.allow_mode.startswith("min"):
            raw = rules.allow_mode.replace("min", "").replace(":", "").strip()
            try:
                min_hits = int(raw)
            except Exception:
                min_hits = 1
            hit_count = sum(1 for name in allow_if if flags.get(name, False))
            ok = hit_count >= max(1, min_hits)
        elif rules.allow_mode.startswith("at_least_"):
            raw = rules.allow_mode.replace("at_least_", "").strip()
            try:
                min_hits = int(raw)
            except Exception:
                min_hits = 1
            hit_count = sum(1 for name in allow_if if flags.get(name, False))
            ok = hit_count >= max(1, min_hits)
        else:
            ok = any(flags.get(name, False) for name in allow_if)
        if not ok:
            return False, [f"gate_allow_not_met={allow_if}"]
        return True, []
    return rules.default_action != "deny", []


_PHASE_ORDER = ("safety", "exclusions", "preconditions", "evidence", "decision")


def _normalize_phase(phase: str) -> str:
    return str(phase or "exclusions").strip().lower()


def _rule_priority(rule: Dict[str, Any]) -> Tuple[int, str]:
    raw = rule.get("priority")
    try:
        prio = int(raw)
    except Exception:
        prio = 999
    return prio, str(rule.get("id") or rule.get("name") or "")


def _eval_leaf_condition(
    *,
    key: str,
    op: str,
    value: Any,
    features: Dict[str, Any],
    quantiles: Dict[str, Any] | None,
) -> bool:
    rule: Dict[str, Any] = {"name": "tmp_rule", "kind": op, "key": key}
    if op.startswith("quantile_"):
        rule["quantile"] = value
    elif op == "any_key_contains":
        rule["any_key_contains"] = list(value or [])
    else:
        rule["threshold"] = value
    flags = compute_execution_evidence(
        features=features, rules=[rule], quantiles=quantiles
    )
    return bool(flags.get("tmp_rule", False))


def _eval_when_clause(
    when: Any,
    *,
    features: Dict[str, Any],
    quantiles: Dict[str, Any] | None,
) -> bool:
    if when is None:
        return False
    if isinstance(when, list):
        return all(
            _eval_when_clause(w, features=features, quantiles=quantiles) for w in when
        )
    if not isinstance(when, dict):
        return False

    if "not" in when:
        return not _eval_when_clause(
            when.get("not"), features=features, quantiles=quantiles
        )

    if "all_of" in when:
        items = when.get("all_of") or []
        return all(
            _eval_when_clause(w, features=features, quantiles=quantiles) for w in items
        )

    if "any_of" in when:
        items = when.get("any_of") or []
        min_matches = int(when.get("min_matches") or 1)
        hits = sum(
            1
            for w in items
            if _eval_when_clause(w, features=features, quantiles=quantiles)
        )
        return hits >= max(1, min_matches)

    if "any_key_contains" in when:
        return _eval_leaf_condition(
            key="",
            op="any_key_contains",
            value=when.get("any_key_contains") or [],
            features=features,
            quantiles=quantiles,
        )

    if "key" in when and "op" in when:
        return _eval_leaf_condition(
            key=str(when.get("key") or ""),
            op=str(when.get("op") or ""),
            value=when.get("value"),
            features=features,
            quantiles=quantiles,
        )

    if len(when) == 1:
        key = next(iter(when.keys()))
        cond = when.get(key) or {}
        if isinstance(cond, dict):
            for op, val in cond.items():
                # Handle aliases for comparison operators
                actual_op = op
                if op == "value_le":  # alias for value_lte
                    actual_op = "value_lte"
                elif op == "value_ge":  # alias for value_gte
                    actual_op = "value_gte"

                return _eval_leaf_condition(
                    key=str(key),
                    op=str(actual_op),
                    value=val,
                    features=features,
                    quantiles=quantiles,
                )
    return False


def apply_when_then_rules(
    *,
    when_then_rules: List[Dict[str, Any]],
    features: Dict[str, Any],
    quantiles: Dict[str, Any] | None = None,
    default_action: str = "deny",
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if not when_then_rules:
        return default_action != "deny", reasons

    phase_map: Dict[str, List[Dict[str, Any]]] = {p: [] for p in _PHASE_ORDER}
    for rule in when_then_rules:
        if not isinstance(rule, dict):
            continue
        phase = _normalize_phase(rule.get("phase"))
        if phase not in phase_map:
            phase = "exclusions"
        phase_map[phase].append(rule)

    allow_hit = False
    for phase in _PHASE_ORDER:
        for rule in sorted(phase_map.get(phase, []), key=_rule_priority):
            when = rule.get("when")
            action = str(rule.get("then", {}).get("action") or "").lower()
            matched = _eval_when_clause(when, features=features, quantiles=quantiles)

            if phase in ("safety", "exclusions") and action == "deny" and matched:
                reason = str(rule.get("reason") or rule.get("id") or "")
                reasons.append(f"{phase}:{reason}")
                return False, reasons

            if phase in ("preconditions", "evidence") and action == "require":
                if not matched:
                    reason = str(rule.get("reason") or rule.get("id") or "")
                    reasons.append(f"{phase}_missing:{reason}")
                    return False, reasons

            if action == "allow" and matched:
                allow_hit = True

    if allow_hit:
        return True, reasons
    return default_action != "deny", reasons
