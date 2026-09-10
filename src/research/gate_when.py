"""Gate ``when`` clause parsing and threshold writeback (preserve ``all_of`` siblings)."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

GATE_DENY_OPS = frozenset({"lt", "le", "gt", "ge"})

# Gate KPI / yaml value_* aliases (NOT entry-filter pass semantics).
GATE_OPERATOR_ALIASES: Dict[str, str] = {
    ">": "gt",
    ">=": "ge",
    "<": "lt",
    "<=": "le",
    "value_gt": "gt",
    "value_gte": "ge",
    "value_lt": "lt",
    "value_lte": "le",
}


def resolve_gate_deny_operator(operator: str) -> str:
    """Normalize CLI/yaml operator to gate deny op (lt/le/gt/ge)."""
    raw = str(operator or "").strip().lower()
    if raw in GATE_DENY_OPS:
        return raw
    mapped = GATE_OPERATOR_ALIASES.get(raw)
    if mapped:
        return mapped
    return raw


def deny_op_to_value_key(deny_op: str) -> str:
    return {
        "lt": "value_lt",
        "le": "value_lte",
        "gt": "value_gt",
        "ge": "value_gte",
    }[deny_op]


def parse_gate_when_condition(
    when: Dict[str, Any],
) -> Tuple[Optional[str], Optional[str], Optional[float]]:
    """Parse first optimizable feature/op/threshold from a gate ``when`` block."""
    if not isinstance(when, dict):
        return None, None, None

    if "any_of" in when:
        conditions = when["any_of"]
        if conditions and isinstance(conditions, list):
            return parse_gate_when_condition(conditions[0])

    if "all_of" in when:
        conditions = when["all_of"]
        if conditions and isinstance(conditions, list):
            return parse_gate_when_condition(conditions[0])

    for feature_col, value_dict in when.items():
        if feature_col in ("any_of", "all_of"):
            continue
        if not isinstance(value_dict, dict):
            continue
        for op_key, threshold in value_dict.items():
            if op_key.startswith("value_"):
                op_suffix = op_key[6:]
                operator = op_suffix.replace("lte", "le").replace("gte", "ge")
                return feature_col, operator, float(threshold)
            if op_key.startswith("quantile_"):
                op_suffix = op_key[9:]
                operator = op_suffix.replace("lte", "le").replace("gte", "ge")
                return feature_col, operator, float(threshold)
    return None, None, None


def _clause_targets_feature(clause: Any, feature: str) -> bool:
    return isinstance(clause, dict) and feature in clause and len(clause) == 1


def _build_feature_clauses(
    feature: str,
    deny_op: str,
    recommended: float,
    *,
    interval: Optional[Tuple[float, float]] = None,
) -> List[Dict[str, Any]]:
    if interval is not None:
        lo, hi = float(interval[0]), float(interval[1])
        if deny_op in ("gt", "ge"):
            lo_key = deny_op_to_value_key("gt" if deny_op == "gt" else "ge")
            hi_key = "value_lt" if deny_op == "gt" else "value_lte"
            return [
                {feature: {lo_key: lo}},
                {feature: {hi_key: hi}},
            ]
        lo_key = "value_gt" if deny_op in ("lt", "le") else deny_op_to_value_key(deny_op)
        hi_key = deny_op_to_value_key("lt" if deny_op == "lt" else "le")
        return [{feature: {lo_key: lo}}, {feature: {hi_key: hi}}]

    key = deny_op_to_value_key(deny_op)
    return [{feature: {key: float(recommended)}}]


def gate_threshold_skip_reason(
    when: Dict[str, Any],
    feature: str,
    *,
    interval: Optional[Tuple[float, float]] = None,
) -> Optional[str]:
    """Machine-readable skip reason when a threshold rewrite would be unsafe."""
    if not isinstance(when, dict):
        return "invalid_when"
    if "any_of" in when:
        return "unsafe_any_of"
    if interval is None and isinstance(when.get("all_of"), list):
        n_feature_clauses = sum(
            1 for c in when["all_of"] if _clause_targets_feature(c, feature)
        )
        if n_feature_clauses >= 2:
            return "unsafe_band_no_interval"
    return None


def can_apply_gate_threshold(
    when: Dict[str, Any],
    feature: str,
    *,
    interval: Optional[Tuple[float, float]] = None,
) -> bool:
    """Return False when a single-point rewrite would change rule semantics.

    Unsafe cases (left unchanged by ``apply_gate_threshold_to_when``):
      - ``any_of`` disjunctions (rewriting one branch is ambiguous).
      - band rules (>=2 clauses on ``feature``) when no ``interval`` is given
        (a single point would drop the opposite bound).
    """
    return gate_threshold_skip_reason(when, feature, interval=interval) is None


def apply_gate_threshold_to_when(
    when: Dict[str, Any],
    feature: str,
    deny_op: str,
    recommended: float,
    *,
    interval: Optional[Tuple[float, float]] = None,
) -> Dict[str, Any]:
    """Update thresholds for ``feature`` only; preserve other ``all_of`` clauses.

    Returns ``when`` unchanged when the rewrite would be unsafe
    (see :func:`can_apply_gate_threshold`).
    """
    deny_op = resolve_gate_deny_operator(deny_op)
    out = deepcopy(when)

    # Empty when (single-feature draft from scratch) is always safe.
    if out and not can_apply_gate_threshold(out, feature, interval=interval):
        return out

    new_clauses = _build_feature_clauses(
        feature, deny_op, recommended, interval=interval
    )

    if "all_of" in out and isinstance(out["all_of"], list):
        kept = [c for c in out["all_of"] if not _clause_targets_feature(c, feature)]
        out["all_of"] = kept + new_clauses
        return out

    if feature in out and len(out) == 1:
        if len(new_clauses) == 1:
            return new_clauses[0]
        return {"all_of": new_clauses}

    other = {k: v for k, v in out.items() if k != feature}
    merged = new_clauses + ([{k: other[k]} for k in other] if other else [])
    if len(merged) == 1:
        return merged[0]
    return {"all_of": merged}


def load_allowed_gate_deny_features(
    strategy: Optional[str],
    *,
    strategies_root: Path | str = "config/strategies",
    project_root: Optional[Path] = None,
) -> List[str]:
    if not strategy:
        return []
    try:
        import yaml as _yaml
    except Exception:
        return []
    root = Path(project_root) if project_root else Path(__file__).resolve().parents[2]
    path = root / strategies_root / strategy / "features_gate.yaml"
    if not path.exists():
        return []
    try:
        raw = _yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return []
    fp = raw.get("feature_pipeline", {}) or {}
    allowed = fp.get("allowed_gate_deny_features", []) or []
    patterns: List[str] = []
    for x in allowed:
        if isinstance(x, (str, int)):
            s = str(x).strip()
            if s:
                patterns.append(s)
    return patterns


def is_feature_allowed_for_gate_deny(feature: str, patterns: List[str]) -> bool:
    if not patterns:
        return True
    if not feature:
        return False
    import fnmatch as _fn

    return any(_fn.fnmatchcase(feature, pat) for pat in patterns)
