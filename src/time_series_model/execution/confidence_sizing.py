"""Feature-threshold confidence → size_multiplier (entry set unchanged).

Used by ``ExecutionParamGenerator`` when ``execution.yaml`` has::

    confidence_sizing:
      enabled: true
      high_size_multiplier: 1.0
      low_size_multiplier: 0.5
      high_when:
        mode: all   # or any
        conditions:
          - feature: volume_participation_score
            operator: ">="
            value: 0.45
          - feature: ema_1200_position
            abs: true
            operator: ">="
            value: 0.18

Missing / non-finite feature values fail the condition (fail-closed → low size
when ``mode=all``; for ``mode=any`` a missing feature simply does not match).
"""

from __future__ import annotations

from typing import Any, Mapping, Optional


_OPS = {
    ">=": lambda a, b: a >= b,
    ">": lambda a, b: a > b,
    "<=": lambda a, b: a <= b,
    "<": lambda a, b: a < b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


def _cond_ok(cond: Mapping[str, Any], features: Mapping[str, Any]) -> bool:
    feat = str(cond.get("feature") or "")
    if not feat:
        return False
    raw = features.get(feat)
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return False
    if val != val:  # NaN
        return False
    if bool(cond.get("abs")):
        val = abs(val)
    op = str(cond.get("operator") or ">=")
    fn = _OPS.get(op)
    if fn is None:
        return False
    try:
        thr = float(cond.get("value"))
    except (TypeError, ValueError):
        return False
    return bool(fn(val, thr))


def resolve_confidence_size_scale(
    cfg: Optional[Mapping[str, Any]],
    features: Optional[Mapping[str, Any]],
) -> Optional[float]:
    """Return multiplicative scale in (0, +inf), or None if sizing inactive.

    When enabled, always returns ``high_size_multiplier`` or ``low_size_multiplier``
    (defaults 1.0 / 0.5). Caller multiplies onto the current size_multiplier.
    """
    if not cfg or not cfg.get("enabled"):
        return None
    feats = features or {}
    try:
        hi = float(cfg.get("high_size_multiplier", 1.0) or 1.0)
        lo = float(cfg.get("low_size_multiplier", 0.5) or 0.5)
    except (TypeError, ValueError):
        hi, lo = 1.0, 0.5
    if hi <= 0 or lo <= 0:
        return None

    hw = cfg.get("high_when") or {}
    conds = list(hw.get("conditions") or [])
    if not conds:
        return hi
    mode = str(hw.get("mode") or "all").lower()
    oks = [_cond_ok(c, feats) for c in conds if isinstance(c, Mapping)]
    if not oks:
        return lo
    matched = all(oks) if mode == "all" else any(oks)
    return hi if matched else lo
