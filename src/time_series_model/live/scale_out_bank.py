"""SRB research: one-shot partial scale-out after adds are full (default off).

Modes (``execution.yaml`` → ``scale_out_bank``):
  - ``pct``: mother price return ≥ ``min_profit_pct`` (e.g. 0.20 = +20%)
  - ``wyckoff``: ret ≥ floor + high volume + weak advance (FER / false expansion)

Residual legs keep structural exit (ema1200) / trailing — not a full take-profit.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple


def scale_out_bank_cfg(pos: Mapping[str, Any]) -> Dict[str, Any]:
    raw = pos.get("scale_out_bank")
    return dict(raw) if isinstance(raw, dict) else {}


def scale_out_bank_enabled(pos: Mapping[str, Any]) -> bool:
    return bool(scale_out_bank_cfg(pos).get("enabled", False))


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


def mother_return_pct(
    *,
    entry_price: float,
    close: float,
    is_long: bool,
) -> Optional[float]:
    ep = _f(entry_price)
    px = _f(close)
    if ep is None or px is None or ep <= 0:
        return None
    if is_long:
        return px / ep - 1.0
    return 1.0 - px / ep


def _feat(feats: Optional[Mapping[str, Any]], name: str) -> Optional[float]:
    if not feats or not name:
        return None
    return _f(feats.get(name))


def _weak_advance(feats: Optional[Mapping[str, Any]], mode: str) -> bool:
    """Fail-closed: missing required columns → no trigger."""
    if not feats:
        return False
    tag = str(mode or "fer_or_me_false_expansion").strip().lower()
    if tag != "fer_or_me_false_expansion":
        return False
    fer = _feat(feats, "fer_signed_efficiency_pct")
    me_fe = _feat(feats, "me_false_expansion")
    if fer is None and me_fe is None:
        return False
    if fer is not None and fer <= 0.30:
        return True
    if me_fe is not None and me_fe >= 0.5:
        return True
    return False


def evaluate_scale_out_bank(
    *,
    cfg: Mapping[str, Any],
    entry_price: float,
    close: float,
    is_long: bool,
    add_leg_count: int,
    already_done: bool,
    features: Optional[Mapping[str, Any]] = None,
) -> Tuple[Optional[float], str]:
    """Return (sell_fraction, reason) or (None, '') when no scale-out."""
    if already_done or not bool(cfg.get("enabled", False)):
        return None, ""
    if close <= 0:
        return None, ""

    min_adds = int(cfg.get("min_add_count", 3))
    if bool(cfg.get("require_adds_full", True)) and int(add_leg_count) < min_adds:
        return None, ""

    ret = mother_return_pct(entry_price=entry_price, close=close, is_long=is_long)
    if ret is None:
        return None, ""

    mode = str(cfg.get("mode", "pct")).strip().lower()
    sell_frac = float(cfg.get("sell_fraction", 0.5))
    sell_frac = max(0.0, min(1.0, sell_frac))
    if sell_frac <= 0:
        return None, ""

    if mode == "pct":
        thr = _f(cfg.get("min_profit_pct", 0.20))
        if thr is None:
            return None, ""
        if ret + 1e-12 < thr:
            return None, ""
        return sell_frac, (
            f"scale_out_bank|mode=pct|ret={ret:.3f}|thr={thr:.3f}|adds={add_leg_count}"
        )

    if mode == "wyckoff":
        floor = _f(cfg.get("min_profit_pct", 0.15))
        if floor is None or ret + 1e-12 < floor:
            return None, ""
        vol_min = _f(cfg.get("volume_ratio_pct_min", 0.80))
        vol = _feat(features, "volume_ratio_pct")
        if vol_min is None or vol is None or vol + 1e-12 < vol_min:
            return None, ""
        if not _weak_advance(features, str(cfg.get("weak_advance", ""))):
            return None, ""
        return sell_frac, (
            f"scale_out_bank|mode=wyckoff|ret={ret:.3f}|vol={vol:.2f}|adds={add_leg_count}"
        )

    return None, ""


def apply_partial_sell_to_sim_position(
    pos: Dict[str, Any],
    *,
    sell_qty: float,
    exit_price: float,
) -> None:
    """Reduce EB futures/spot position book after a partial scale-out."""
    qty_full = float(pos.get("_qty_base", 0.0) or 0.0)
    sell_qty = min(max(0.0, float(sell_qty)), qty_full)
    if sell_qty <= 0.0 or qty_full <= 0.0:
        return
    ratio = sell_qty / qty_full
    keep = 1.0 - ratio
    pos["_qty_base"] = float(qty_full - sell_qty)
    pos["_entry_notional_usdt"] = (
        float(pos.get("_entry_notional_usdt", 0.0) or 0.0) * keep
    )
    pos["_entry_fee_usdt"] = float(pos.get("_entry_fee_usdt", 0.0) or 0.0) * keep
    pos["_size_multiplier"] = float(pos.get("_size_multiplier", 1.0) or 1.0) * keep
    if pos.get("_contracts") is not None:
        pos["_contracts"] = float(pos.get("_contracts", 0.0) or 0.0) * keep
    if pos.get("_risk_coin_budget") is not None:
        pos["_risk_coin_budget"] = (
            float(pos.get("_risk_coin_budget", 0.0) or 0.0) * keep
        )
    if pos.get("_entry_fee_coin") is not None:
        pos["_entry_fee_coin"] = float(pos.get("_entry_fee_coin", 0.0) or 0.0) * keep
    pos["_last_scale_out_price"] = float(exit_price)
