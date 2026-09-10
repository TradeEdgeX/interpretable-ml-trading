"""Resolve spot_accum symbol budgets from weights × live/offline equity."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

UNIT_MODE_FIXED = "fixed"
UNIT_MODE_BUDGET_TRANCHES = "budget_tranches"
_VALID_UNIT_MODES = {UNIT_MODE_FIXED, UNIT_MODE_BUDGET_TRANCHES}


def clamp01(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, v))


def parse_symbol_budget_pct(raw: Any) -> Dict[str, float]:
    """Parse constitution ``symbol_budget_pct`` into weights that sum to 1.

    Accepts ``0.40`` or ``40`` (percent points). Empty / invalid → ``{}``.
    """
    if not isinstance(raw, Mapping):
        return {}
    parsed: Dict[str, float] = {}
    for key, val in raw.items():
        sym = str(key or "").strip().upper()
        if not sym:
            continue
        try:
            fv = float(val or 0.0)
        except (TypeError, ValueError):
            continue
        if fv > 0.0:
            parsed[sym] = fv
    if not parsed:
        return {}
    if max(parsed.values()) > 1.0:
        parsed = {k: v / 100.0 for k, v in parsed.items()}
    total = sum(parsed.values())
    if total <= 0.0:
        return {}
    return {k: v / total for k, v in parsed.items()}


def parse_symbol_budgets_usdt(raw: Any) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not isinstance(raw, Mapping):
        return out
    for key, val in raw.items():
        sym = str(key or "").strip().upper()
        if not sym:
            continue
        try:
            fv = float(val or 0.0)
        except (TypeError, ValueError):
            continue
        if fv > 0.0:
            out[sym] = fv
    return out


def parse_unit_mode(raw: Any) -> str:
    """Return ``budget_tranches`` or ``fixed`` (default)."""
    mode = str(raw or "").strip().lower()
    if mode in _VALID_UNIT_MODES:
        return mode
    return UNIT_MODE_FIXED


def parse_tranches_per_symbol(raw: Any, *, default: int = 0) -> int:
    try:
        n = int(raw or 0)
    except (TypeError, ValueError):
        return int(default)
    return max(0, n)


def resolve_symbol_budgets_usdt(
    *,
    equity_usdt: float,
    target_deploy_pct: float = 1.0,
    symbol_budget_pct: Optional[Mapping[str, float]] = None,
    symbol_budgets_fallback: Optional[Mapping[str, float]] = None,
) -> Dict[str, float]:
    """``budget[s] = equity × target_deploy_pct × weight[s]``.

    When weights are missing, return the hard-USDT fallback (legacy recharge).
    """
    weights = {
        str(k).upper(): float(v)
        for k, v in dict(symbol_budget_pct or {}).items()
        if float(v or 0.0) > 0.0
    }
    if weights:
        scale = max(0.0, float(equity_usdt or 0.0)) * clamp01(target_deploy_pct)
        return {sym: scale * w for sym, w in weights.items()}
    return dict(symbol_budgets_fallback or {})


def resolve_symbol_units_usdt(
    *,
    symbol_budgets_usdt: Mapping[str, float],
    unit_mode: str = UNIT_MODE_FIXED,
    tranches_per_symbol: int = 0,
    symbol_units_fallback: Optional[Mapping[str, float]] = None,
    min_unit_usdt: float = 1.0,
) -> Dict[str, float]:
    """Resolve per-symbol base deploy unit (before decay / remain caps).

    ``budget_tranches``: ``unit[s] = max(min_unit, budget[s] / tranches)``.
    Recompute whenever live equity refreshes the symbol budget map so account
    scale keeps a roughly fixed deployment calendar.

    ``fixed``: use the explicit ``symbol_unit_notional_usdt`` map as-is.
    """
    mode = parse_unit_mode(unit_mode)
    if mode == UNIT_MODE_BUDGET_TRANCHES:
        n = int(tranches_per_symbol or 0)
        if n <= 0:
            raise ValueError(
                "unit_mode=budget_tranches requires tranches_per_symbol > 0"
            )
        floor = max(0.0, float(min_unit_usdt or 0.0))
        out: Dict[str, float] = {}
        for key, budget in dict(symbol_budgets_usdt or {}).items():
            sym = str(key or "").strip().upper()
            try:
                sb = float(budget or 0.0)
            except (TypeError, ValueError):
                continue
            if not sym or sb <= 0.0:
                continue
            out[sym] = max(floor, sb / float(n))
        return out

    fallback = parse_symbol_budgets_usdt(symbol_units_fallback)
    return dict(fallback)
