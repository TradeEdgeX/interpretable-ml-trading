from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class HeuristicDecision:
    ok: bool
    side: Optional[str]  # "BUY" | "SELL"
    reasons: List[str]
    risk_multiplier: float = 1.0


def _pick_atr(feats: Dict[str, Any]) -> Optional[float]:
    # Prefer timeframe ATR keys like "15T_atr".
    for k, v in feats.items():
        if str(k).endswith("_atr"):
            try:
                return float(v)
            except Exception:
                continue
    if "atr" in feats:
        try:
            return float(feats["atr"])
        except Exception:
            return None
    return None


def _pick_rsi(feats: Dict[str, Any]) -> Optional[float]:
    for k, v in feats.items():
        if str(k).endswith("_rsi"):
            try:
                return float(v)
            except Exception:
                continue
    if "rsi" in feats:
        try:
            return float(feats["rsi"])
        except Exception:
            return None
    return None


def _rolling_high_low(
    bars: List[Dict[str, Any]], lookback: int
) -> Tuple[Optional[float], Optional[float]]:
    if len(bars) < max(2, lookback + 1):
        return None, None
    xs = bars[-(lookback + 1) : -1]
    try:
        hh = max(float(b["high"]) for b in xs)
        ll = min(float(b["low"]) for b in xs)
        return hh, ll
    except Exception:
        return None, None


def _breakout_side_from_bars(
    *, bars: List[Dict[str, Any]], lookback: int, atr: float, atr_mult: float
) -> Optional[str]:
    """
    Return "BUY" if upside breakout, "SELL" if downside breakout, else None.
    """
    if not bars:
        return None
    last = bars[-1]
    hh, ll = _rolling_high_low(bars, lookback=lookback)
    if hh is None or ll is None:
        return None
    close = float(last["close"])
    if close > hh + atr_mult * atr:
        return "BUY"
    if close < ll - atr_mult * atr:
        return "SELL"
    return None


def _failed_breakout_side_from_bars(
    *, bars: List[Dict[str, Any]], lookback: int, atr: float, atr_mult: float
) -> Optional[str]:
    """
    Return fade side:
      - If swept above HH but closed back below HH => SELL
      - If swept below LL but closed back above LL => BUY
    """
    if not bars:
        return None
    last = bars[-1]
    hh, ll = _rolling_high_low(bars, lookback=lookback)
    if hh is None or ll is None:
        return None
    high = float(last["high"])
    low = float(last["low"])
    close = float(last["close"])
    if high > hh + atr_mult * atr and close < hh:
        return "SELL"
    if low < ll - atr_mult * atr and close > ll:
        return "BUY"
    return None


def evaluate_required_conditions(
    *,
    archetype_name: str,
    regime: str,
    required_conditions: List[str],
    feats: Dict[str, Any],
    bars: List[Dict[str, Any]],
) -> HeuristicDecision:
    """
    Minimal, conservative implementation:
    - Uses only live-available primitives + a few bar-derived structure checks.
    - Missing inputs => fail closed (NO_TRADE).
    """
    reasons: List[str] = []
    if not required_conditions:
        return HeuristicDecision(ok=True, side=None, reasons=reasons)
    rr_min = 2.0
    lookback = 48
    atr_mult = 0.3
    pullback_max_atr = 1.0
    vol_expansion_atr_frac = 0.003  # ATR/price
    vpin_absorption_min = 0.35

    atr = _pick_atr(feats)
    if atr is None:
        reasons.append("missing_atr")
        return HeuristicDecision(ok=False, side=None, reasons=reasons)

    pred_dir_prob = float(feats.get("pred_dir_prob", 0.5))
    pred_mfe = float(feats.get("pred_mfe_atr", 0.0))
    pred_mae = float(feats.get("pred_mae_atr", 0.0))
    pred_ttm = float(feats.get("pred_t_to_mfe", 0.0))
    eff = pred_mfe / (pred_mae + 1e-9)

    # Side defaults: trend follows dir_prob; mean fades when pattern indicates.
    side_default = "BUY" if pred_dir_prob >= 0.5 else "SELL"

    # Bar-derived structure hints.
    hh, ll = _rolling_high_low(bars, lookback=lookback)
    last = bars[-1] if bars else None
    if last is None or hh is None or ll is None:
        reasons.append("insufficient_bars_for_structure")
        # For some conditions (pure primitives) we can still proceed, but keep conservative.

    # Condition implementations
    for c in required_conditions:
        cc = str(c).strip()
        if not cc:
            continue

        if cc == "rr_geq_2":
            if eff < rr_min:
                reasons.append(f"rr_lt_{rr_min:.2f}")

        elif cc == "ttm_above_threshold":
            if pred_ttm < 5.0:
                reasons.append("ttm_too_small")

        elif cc == "vol_expansion":
            if last is None:
                reasons.append("no_last_bar_for_vol_expansion")
            else:
                price = float(last["close"])
                if price <= 0 or (atr / price) < vol_expansion_atr_frac:
                    reasons.append("no_vol_expansion")

        elif cc == "no_reverse_absorption":
            # For TREND: require orderflow not strongly against direction.
            imb = float(feats.get("imbalance", 0.0))
            if side_default == "BUY" and imb < -0.25:
                reasons.append("reverse_absorption_against_trend")
            if side_default == "SELL" and imb > 0.25:
                reasons.append("reverse_absorption_against_trend")

        elif cc == "structure_breakout":
            bside = _breakout_side_from_bars(
                bars=bars, lookback=lookback, atr=atr, atr_mult=atr_mult
            )
            if bside is None:
                reasons.append("no_structure_breakout")

        elif cc == "healthy_pullback":
            if last is None or hh is None or ll is None:
                reasons.append("no_structure_for_pullback")
            else:
                close = float(last["close"])
                # Approx: require close not too deep into the range after breakout.
                if side_default == "BUY":
                    if close < hh - pullback_max_atr * atr:
                        reasons.append("pullback_too_deep")
                else:
                    if close > ll + pullback_max_atr * atr:
                        reasons.append("pullback_too_deep")

        elif cc == "htf_trend_confirmed":
            # Minimal: require dir_prob confidence.
            if abs(pred_dir_prob - 0.5) < 0.08:
                reasons.append("htf_trend_not_confident")

        elif cc == "ltf_precision_entry":
            # Minimal: require favorable RR.
            if eff < 1.5:
                reasons.append("ltf_entry_rr_too_low")

        elif cc == "not_counter_trend":
            # Minimal: ensure orderflow aligns with dir.
            imb = float(feats.get("imbalance", 0.0))
            if side_default == "BUY" and imb < -0.1:
                reasons.append("counter_trend_orderflow")
            if side_default == "SELL" and imb > 0.1:
                reasons.append("counter_trend_orderflow")

        elif cc == "breakout_failed_close_back":
            fade = _failed_breakout_side_from_bars(
                bars=bars, lookback=lookback, atr=atr, atr_mult=atr_mult
            )
            if fade is None:
                reasons.append("no_failed_breakout")

        elif cc == "no_follow_through":
            # Minimal: require dir_prob not too confident (mean setups are "corpse recycle").
            if abs(pred_dir_prob - 0.5) > 0.2:
                reasons.append("too_confident_trend_followthrough")

        elif cc == "absorption_present":
            vpin = float(feats.get("vpin", 0.0))
            if vpin < vpin_absorption_min:
                reasons.append("no_absorption_vpin")

        elif cc == "ttm_break":
            # Mean must revert quickly; if predicted ttm too long, skip.
            if pred_ttm > 30.0:
                reasons.append("mean_ttm_too_long")

        elif cc == "stops_swept":
            fade = _failed_breakout_side_from_bars(
                bars=bars, lookback=lookback, atr=atr, atr_mult=atr_mult
            )
            if fade is None:
                reasons.append("no_stops_swept")

        elif cc == "high_volume_no_progress":
            tv = float(feats.get("total_vol", 0.0))
            if tv <= 0:
                reasons.append("no_orderflow_volume")
            # Placeholder: if total_vol exists, accept for now.

        elif cc == "rejection_close":
            if last is None:
                reasons.append("no_last_bar_for_rejection")
            else:
                o = float(last["open"])
                h = float(last["high"])
                l = float(last["low"])
                cpx = float(last["close"])
                body = abs(cpx - o)
                rng = max(1e-9, h - l)
                lower_wick = min(o, cpx) - l
                upper_wick = h - max(o, cpx)
                if body / rng > 0.65:
                    reasons.append("rejection_body_too_large")
                if side_default == "BUY" and upper_wick < lower_wick:
                    reasons.append("no_rejection_upper_wick")
                if side_default == "SELL" and lower_wick < upper_wick:
                    reasons.append("no_rejection_lower_wick")

        else:
            # Unknown condition: fail closed.
            reasons.append(f"unknown_required_condition:{cc}")

    ok = len(reasons) == 0
    return HeuristicDecision(
        ok=ok, side=side_default if ok else None, reasons=reasons, risk_multiplier=1.0
    )
