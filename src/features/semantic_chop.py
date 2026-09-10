"""Canonical semantic_chop — always uses bpc_semantic_chop directly.

No fallback to proxy ``semantic_chop``.
Production must use ``bpc_semantic_chop`` (ML-based) exclusively.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, MutableMapping, Optional, Sequence, Tuple

# The ONE and ONLY semantic chop column used in production.
SEMANTIC_CHOP_KEY: str = "bpc_semantic_chop"

# Backward-compatible alias (used by chop_grid_config.py et al).
SEMANTIC_CHOP_COLUMNS: Tuple[str, ...] = ("bpc_semantic_chop",)

# Runtime aliases for multileg regime/gate feature resolution.
# "semantic_chop" → "bpc_semantic_chop" (legacy consumers that haven't been updated)
_MULTILEG_RUNTIME_ALIASES: dict[str, tuple[str, ...]] = {
    "semantic_chop": ("bpc_semantic_chop",),
    "bpc_semantic_chop": ("bpc_semantic_chop",),
    "grid_semantic_chop": ("grid_semantic_chop",),
    "trend_confidence": ("trend_confidence_f",),
}


def multileg_feature_aliases(feature: str) -> tuple[str, ...]:
    """Runtime alias columns for a multileg regime/gate feature name."""
    key = str(feature or "").strip()
    if not key:
        return ()
    return _MULTILEG_RUNTIME_ALIASES.get(key, ())


def as_finite_float(value: Any) -> Optional[float]:
    """Return float when finite; None for missing, NaN, or inf."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def resolve_semantic_chop(
    features: Mapping[str, Any],
    *,
    default: Optional[float] = None,
) -> Optional[float]:
    """Direct read of bpc_semantic_chop. No fallback to proxy semantic_chop."""
    return as_finite_float(features.get("bpc_semantic_chop")) or default


def set_canonical_semantic_chop(features: MutableMapping[str, Any]) -> None:
    """Set canonical ``semantic_chop`` = ``bpc_semantic_chop`` for consumers that read the alias."""
    val = as_finite_float(features.get("bpc_semantic_chop"))
    if val is None:
        return
    features["semantic_chop"] = val


def normalize_semantic_chop_aliases(features: MutableMapping[str, Any]) -> None:
    """Set canonical ``semantic_chop`` from ``bpc_semantic_chop`` (no fallback)."""
    val = as_finite_float(features.get("bpc_semantic_chop"))
    if val is None:
        return
    features["semantic_chop"] = val
    if "bpc_semantic_chop" in features:
        features["semantic_chop"] = val


def resolve_feature_float(
    features: Mapping[str, Any],
    keys: Sequence[str],
    *,
    default: float = 0.0,
) -> float:
    """Resolve first finite float from ordered keys."""
    for key in keys:
        val = as_finite_float(features.get(key))
        if val is not None:
            return val
    return float(default)
