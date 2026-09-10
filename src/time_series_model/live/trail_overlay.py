"""Optional trailing overlay (research / EB). Default off when config missing.

Bear dump → absorption → bounce can sweep ATR trails. Overlay modes:

- ``pause``: skip trail SL ratchet for N bars after trigger
- ``widen``: multiply ``trail_r`` while trigger feature is hot
- ``bank``: force flatten when adverse absorption / divergence fires

Prod ``execution.yaml`` must leave ``trail_overlay.enabled: false`` (or omit).
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple


def _f(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if x != x:
        return None
    return x


def overlay_cfg(pos: Mapping[str, Any]) -> Dict[str, Any]:
    raw = pos.get("trail_overlay")
    return dict(raw) if isinstance(raw, dict) else {}


def overlay_enabled(pos: Mapping[str, Any]) -> bool:
    cfg = overlay_cfg(pos)
    return bool(cfg.get("enabled", False))


def _feat(feats: Optional[Mapping[str, Any]], name: str) -> Optional[float]:
    if not feats or not name:
        return None
    return _f(feats.get(name))


def _trigger_hit(
    *,
    mode_cfg: Mapping[str, Any],
    feats: Optional[Mapping[str, Any]],
    is_long: bool,
) -> bool:
    """Evaluate a mode trigger. ``side``: both | long | short."""
    side = str(mode_cfg.get("side", "both")).strip().lower()
    if side == "long" and not is_long:
        return False
    if side == "short" and is_long:
        return False

    name = str(mode_cfg.get("feature") or "").strip()
    if not name:
        return False
    val = _feat(feats, name)
    if val is None:
        return False

    # Signed bank-style: short wants high score, long wants low (neg)
    signed = str(mode_cfg.get("compare", "min")).strip().lower()
    if signed == "abs_min":
        thr = _f(mode_cfg.get("abs_min"))
        if thr is None:
            return False
        if is_long:
            return val <= -abs(thr)
        return val >= abs(thr)

    if signed == "max":
        thr = _f(mode_cfg.get("max"))
        if thr is None:
            return False
        return val <= thr

    thr = _f(mode_cfg.get("min"))
    if thr is None:
        return False
    return val >= thr


def resolve_trail_overlay(
    pos: Dict[str, Any],
    *,
    trail_r: float,
    is_long: bool,
    feats: Optional[Mapping[str, Any]],
    trailing_activated: bool,
) -> Tuple[float, bool, bool, str]:
    """Apply overlay for one bar.

    Returns:
        (effective_trail_r, pause_ratchet, bank_now, tag)
    """
    if not overlay_enabled(pos) or not trailing_activated:
        return float(trail_r), False, False, ""

    cfg = overlay_cfg(pos)
    modes = cfg.get("modes") or {}
    if not isinstance(modes, dict):
        return float(trail_r), False, False, ""

    tag_parts = []
    eff = float(trail_r)
    pause = False
    bank = False

    # bank first (hard exit)
    bank_cfg = modes.get("bank")
    if isinstance(bank_cfg, dict) and bank_cfg.get("enabled", True):
        if _trigger_hit(mode_cfg=bank_cfg, feats=feats, is_long=is_long):
            bank = True
            tag_parts.append("bank")

    widen_cfg = modes.get("widen")
    if isinstance(widen_cfg, dict) and widen_cfg.get("enabled", True):
        if _trigger_hit(mode_cfg=widen_cfg, feats=feats, is_long=is_long):
            mult = _f(widen_cfg.get("trail_r_mult")) or 1.5
            eff = float(trail_r) * float(mult)
            tag_parts.append("widen")

    pause_cfg = modes.get("pause")
    if isinstance(pause_cfg, dict) and pause_cfg.get("enabled", True):
        bars_left = int(pos.get("_trail_overlay_pause_left") or 0)
        if bars_left > 0:
            pause = True
            pos["_trail_overlay_pause_left"] = bars_left - 1
            tag_parts.append("pause")
        elif _trigger_hit(mode_cfg=pause_cfg, feats=feats, is_long=is_long):
            n = int(pause_cfg.get("bars") or 6)
            pos["_trail_overlay_pause_left"] = max(0, n - 1)
            pause = True
            tag_parts.append("pause")

    return eff, pause, bank, "+".join(tag_parts)
