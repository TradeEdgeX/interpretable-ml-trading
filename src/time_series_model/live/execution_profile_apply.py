from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple


import re

# ATR key pattern: exact 'atr' or timeframe prefix like '15T_atr', '60T_atr', '240T_atr'
_ATR_PATTERN = re.compile(r"^(?:\d+[TtHhDd]_)?atr$")


def pick_atr(feats: Dict[str, Any]) -> Optional[float]:
    """Pick the ATR value from features.

    Only matches exact 'atr' or timeframe-prefixed keys like '15T_atr', '60T_atr'.
    Excludes false positives like 'macd_atr', 'rsi_atr', etc.
    """
    if not feats:
        return None
    # Prefer timeframe-prefixed ATR (e.g. '240T_atr')
    for k in feats:
        if k != "atr" and _ATR_PATTERN.match(k):
            try:
                v = float(feats[k])
                if v > 0:
                    return v
            except Exception:
                continue
    # Fallback to plain 'atr'
    if "atr" in feats:
        try:
            v = float(feats["atr"])
            if v > 0:
                return v
        except Exception:
            pass
    return None


def compute_rr_prices(
    *,
    side: str,
    entry_price: float,
    atr: float,
    stop_loss_r: float,
    take_profit_r: float,
) -> Tuple[Optional[float], Optional[float]]:
    if entry_price <= 0 or atr <= 0:
        return None, None
    sl_r = float(stop_loss_r)
    tp_r = float(take_profit_r)
    if str(side).upper() in {"BUY", "LONG"}:
        stop_loss = entry_price - sl_r * atr
        take_profit = entry_price + tp_r * atr
    else:
        stop_loss = entry_price + sl_r * atr
        take_profit = entry_price - tp_r * atr
    return float(stop_loss), float(take_profit)


def compute_trailing_stop(
    *,
    side: str,
    current_price: float,
    atr: float,
    trailing_atr: float,
) -> Optional[float]:
    if current_price <= 0 or atr <= 0 or trailing_atr is None:
        return None
    trail = float(trailing_atr)
    if str(side).upper() in {"BUY", "LONG"}:
        return float(current_price - trail * atr)
    return float(current_price + trail * atr)


def holding_expired(
    *,
    entry_time: datetime,
    now: datetime,
    max_holding_bars: Optional[int],
    bar_minutes: int,
) -> bool:
    if max_holding_bars is None or max_holding_bars <= 0:
        return False
    if bar_minutes <= 0:
        return False
    max_age = timedelta(minutes=int(max_holding_bars) * int(bar_minutes))
    return now >= entry_time + max_age


def rr_constraints_from_exec_params(exec_params: Dict[str, Any]) -> Dict[str, Any]:
    """Build ``execution_profile['rr_constraints']`` from ExecutionParamGenerator output."""
    return {
        "stop_loss_r": exec_params.get("initial_r", 2.0),
        "take_profit_r": exec_params.get("take_profit_r", 2.5),
        "stop_loss_type": exec_params.get("stop_loss_type", "fixed"),
        "take_profit_type": exec_params.get("take_profit_type", "fixed"),
        "box_window": exec_params.get("box_window", 120),
        "box_stop_buffer_frac": exec_params.get("box_stop_buffer_frac", 0.25),
        "box_target_edge_frac": exec_params.get("box_target_edge_frac", 0.15),
        "box_hi": exec_params.get("box_hi"),
        "box_lo": exec_params.get("box_lo"),
        "box_width_pct": exec_params.get("box_width_pct"),
        "box_pos": exec_params.get("box_pos"),
        "box_hi_120": exec_params.get("box_hi_120"),
        "box_lo_120": exec_params.get("box_lo_120"),
        "box_width_pct_120": exec_params.get("box_width_pct_120"),
        "box_pos_120": exec_params.get("box_pos_120"),
        "allow_trailing": bool(exec_params.get("allow_trailing", True)),
        "activation_r": exec_params.get("activation_r"),
        "trailing_atr": exec_params.get("trail_r"),
        "trail_r_far": exec_params.get("trail_r_far"),
        "trail_r_near": exec_params.get("trail_r_near"),
        "l3_near_threshold_atr": exec_params.get("l3_near_threshold_atr"),
        "max_holding_bars": exec_params.get(
            "max_holding_bars", exec_params.get("time_stop_bars", 0)
        ),
        "structural_exit": exec_params.get("structural_exit"),
        "regime_lifecycle_exit": exec_params.get("regime_lifecycle_exit") or {},
        "regime_exit": exec_params.get("regime_exit") or {},
        "cycle_close": exec_params.get("cycle_close") or {},
        "profit_take_ladder": exec_params.get("profit_take_ladder") or {},
        "sr_exit_price": exec_params.get("sr_exit_price"),
        "sr_exit_buffer_atr": exec_params.get("sr_exit_buffer_atr"),
        "min_stop_pct": exec_params.get("min_stop_pct"),
        "max_stop_pct": exec_params.get("max_stop_pct"),
        "trail_expand_primary_atr": bool(
            exec_params.get("trail_expand_primary_atr", False)
        ),
        "structural_sl": exec_params.get("structural_sl") or {},
        "breakeven_enabled": bool(exec_params.get("breakeven_enabled", False)),
        "breakeven_trigger_r": float(
            exec_params.get("breakeven_trigger_r", 1.0) or 1.0
        ),
        "breakeven_lock_level_r": float(
            exec_params.get("breakeven_lock_level_r", 0.0) or 0.0
        ),
        "breakeven_measure": str(exec_params.get("breakeven_measure", "initial_risk")),
        "time_stop_uncap_mfe_r": exec_params.get("time_stop_uncap_mfe_r"),
        "l3_structural_exit_enabled": bool(
            exec_params.get("l3_structural_exit_enabled", False)
        ),
        "l3_structural_exit_buffer_atr": float(
            exec_params.get("l3_structural_exit_buffer_atr", 0.25) or 0.25
        ),
        # Research trail overlay (default {} / enabled false — no prod effect)
        "trail_overlay": exec_params.get("trail_overlay") or {},
        # SRB scale-out bank (research; default {} / enabled false)
        "scale_out_bank": exec_params.get("scale_out_bank") or {},
    }
