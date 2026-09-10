"""Regime ``box_prefilter`` gates from YAML thresholds + Feature Store box_* columns."""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional, Sequence

_BOX_POS_WINDOW = re.compile(r"box_pos_(\d+)", re.I)


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out:
        return None
    return out


def regime_box_window(
    regime: Mapping[str, Any], rules: Sequence[Any] | None = None
) -> int:
    """Window N for ``box_*_{N}`` columns (align with ``rules`` e.g. box_pos_60)."""
    if regime.get("box_window") is not None:
        return max(1, int(regime["box_window"]))
    for block in rules or []:
        if not isinstance(block, dict):
            continue
        clauses = block.get("all_of") or block.get("any_of") or []
        if not isinstance(clauses, list):
            continue
        for clause in clauses:
            if not isinstance(clause, dict):
                continue
            feat = str(clause.get("feature") or "")
            m = _BOX_POS_WINDOW.search(feat)
            if m:
                return max(1, int(m.group(1)))
    return 120


def is_stable_box_bar(
    features: Mapping[str, Any],
    box: Mapping[str, Any],
    *,
    box_window: int,
) -> bool:
    """True when bar matches ``regime.box_prefilter`` (stable, credible box)."""
    w = max(1, int(box_window))
    stability = _as_float(features.get(f"box_stability_{w}"))
    width = _as_float(features.get(f"box_width_pct_{w}"))
    touches_hi = _as_float(features.get(f"box_touches_hi_{w}"))
    touches_lo = _as_float(features.get(f"box_touches_lo_{w}"))
    if stability is None or width is None or touches_hi is None or touches_lo is None:
        return bool(features.get("box_prefilter", False))

    stability_min = float(box.get("stability_min", 0.85))
    width_min = float(box.get("width_min", 0.04))
    width_max = float(box.get("width_max", 0.30))
    touches_min = float(box.get("touches_min", 5))
    return (
        stability >= stability_min
        and width >= width_min
        and width <= width_max
        and touches_hi >= touches_min
        and touches_lo >= touches_min
    )


def stable_box_blocks_chop_entry(
    features: Mapping[str, Any],
    regime: Mapping[str, Any],
    *,
    rules: Sequence[Any] | None = None,
) -> bool:
    """Chop grid: block new grid when stable box and ``exclude_box_prefilter`` is false."""
    if bool(regime.get("exclude_box_prefilter", True)):
        return False
    box = regime.get("box_prefilter")
    if not isinstance(box, dict) or not box:
        return bool(features.get("box_prefilter", False))
    return is_stable_box_bar(
        features,
        box,
        box_window=regime_box_window(regime, rules),
    )


def stable_box_blocks_trend_entry(
    features: Mapping[str, Any],
    regime: Mapping[str, Any],
    *,
    rules: Sequence[Any] | None = None,
) -> bool:
    """Trend scalp: block when stable box and ``exclude_box_prefilter`` is true."""
    if not bool(regime.get("exclude_box_prefilter", True)):
        return False
    box = regime.get("box_prefilter")
    if not isinstance(box, dict) or not box:
        return bool(features.get("box_prefilter", False))
    return is_stable_box_bar(
        features,
        box,
        box_window=regime_box_window(regime, rules),
    )
