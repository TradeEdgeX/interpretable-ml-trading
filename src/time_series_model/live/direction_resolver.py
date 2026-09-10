from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class DirectionDecision:
    ok: bool
    side: Optional[str]  # "BUY" | "SELL"
    source: str  # "structure"
    method: str
    reason: str


def _pick_atr(feats: Dict[str, Any]) -> Optional[float]:
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


def _rolling_high_low(
    bars: List[Dict[str, Any]], lookback: int
) -> tuple[Optional[float], Optional[float]]:
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


def _recent_return_side(bars: List[Dict[str, Any]], lookback: int) -> Optional[str]:
    if len(bars) < lookback + 1:
        return None
    try:
        last_close = float(bars[-1]["close"])
        prev_close = float(bars[-(lookback + 1)]["close"])
    except Exception:
        return None
    delta = last_close - prev_close
    if delta > 0:
        return "BUY"
    if delta < 0:
        return "SELL"
    return None


def _trend_sign_side(
    bars: List[Dict[str, Any]],
    feats: Dict[str, Any],
    lookback: int,
    min_consistency: float,
) -> Optional[str]:
    try:
        consistency = float(feats.get("price_dir_consistency_pct", 0.0))
    except Exception:
        consistency = 0.0
    if consistency < min_consistency:
        return None
    return _recent_return_side(bars, lookback=lookback)


def _invert_side(side: Optional[str]) -> Optional[str]:
    if side == "BUY":
        return "SELL"
    if side == "SELL":
        return "BUY"
    return None


def resolve_direction(
    *,
    archetype_name: str,
    policy: Dict[str, Any],
    feats: Dict[str, Any],
    bars: List[Dict[str, Any]],
) -> DirectionDecision:
    method = str(
        (policy.get("structure_direction") or {}).get("method") or "recent_return"
    )
    fallback = str(
        (policy.get("structure_direction") or {}).get("fallback") or ""
    ).strip()
    lookback = int((policy.get("structure_direction") or {}).get("lookback_bars") or 5)
    min_consistency = float(
        (policy.get("structure_direction") or {}).get("min_consistency") or 0.5
    )
    atr_mult = float((policy.get("structure_direction") or {}).get("atr_mult") or 0.3)

    side: Optional[str] = None
    atr = _pick_atr(feats) or 0.0

    if method == "recent_return":
        side = _recent_return_side(bars, lookback=lookback)
    elif method == "trend_sign":
        side = _trend_sign_side(
            bars, feats, lookback=lookback, min_consistency=min_consistency
        )
    elif method == "breakout_sign":
        if atr <= 0:
            side = None
        else:
            side = _breakout_side_from_bars(
                bars=bars, lookback=lookback, atr=atr, atr_mult=atr_mult
            )
    elif method == "failed_breakout":
        if atr <= 0:
            side = None
        else:
            side = _failed_breakout_side_from_bars(
                bars=bars, lookback=lookback, atr=atr, atr_mult=atr_mult
            )
    elif method == "sweep_side":
        raw = feats.get("sweep_side")
        if isinstance(raw, str):
            side = "BUY" if raw.upper() in ["BUY", "LONG"] else "SELL"
        else:
            side = None
    elif method == "reverse_of":
        base = str((policy.get("structure_direction") or {}).get("base") or "").strip()
        if base:
            side = resolve_direction(
                archetype_name=archetype_name,
                policy={
                    "structure_direction": {
                        "method": base,
                        "lookback_bars": lookback,
                        "min_consistency": min_consistency,
                        "atr_mult": atr_mult,
                    }
                },
                feats=feats,
                bars=bars,
            ).side
            side = _invert_side(side)
        else:
            side = None

    if side is None and fallback:
        side = resolve_direction(
            archetype_name=archetype_name,
            policy={
                "structure_direction": {
                    "method": fallback,
                    "lookback_bars": lookback,
                    "min_consistency": min_consistency,
                    "atr_mult": atr_mult,
                }
            },
            feats=feats,
            bars=bars,
        ).side

    if side is None:
        return DirectionDecision(
            ok=False,
            side=None,
            source="structure",
            method=method,
            reason=f"direction_unresolved:{archetype_name}",
        )

    return DirectionDecision(
        ok=True,
        side=side,
        source="structure",
        method=method,
        reason=f"direction:{method}",
    )
