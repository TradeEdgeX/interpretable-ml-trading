from __future__ import annotations

from typing import Any, Dict


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def build_execution_profile(
    *,
    archetype_name: str,
    feats: Dict[str, Any],
    constraints: Dict[str, Any] | None,
) -> Dict[str, Any]:
    """
    Build execution profile from MLP outputs.

    Principles:
    - Structural direction only (no side flips).
    - MLP can only *weaken* structural RR, not strengthen it.
    - dir only affects sizing / timing tolerances.
    """

    fixed_rr = (constraints or {}).get("fixed_rr") or {}
    structural_sl = float(fixed_rr.get("stop_loss_r", 1.0))
    structural_tp = float(fixed_rr.get("take_profit_r", 2.0))
    max_holding = fixed_rr.get("max_holding_bars")
    min_holding = fixed_rr.get("min_holding_bars")

    dir_prob = float(feats.get("pred_dir_prob", 0.5))
    pred_mfe = float(feats.get("pred_mfe_atr", 0.0))
    pred_mae = float(feats.get("pred_mae_atr", 0.0))
    pred_mtt = float(feats.get("pred_t_to_mfe", 0.0))

    # dir -> size (only shrink/scale, never flip)
    confidence = _clip(abs(dir_prob - 0.5) * 2.0, 0.0, 1.0)
    size_mult = _clip((dir_prob - 0.5) * 2.0, 0.0, 1.0)

    # mfe/mae -> take profit cap (never exceed structural RR)
    expected_rr = pred_mfe / (abs(pred_mae) + 1e-9) if pred_mae != 0 else structural_tp
    take_profit_r = min(structural_tp, expected_rr)

    # mae -> stop loss / size adjustment
    stop_loss_r = structural_sl
    if pred_mae > structural_sl:
        size_mult *= 0.5
        stop_loss_r = pred_mae * 1.1

    # mtt -> holding time tolerance
    max_holding_bars = max_holding
    if pred_mtt > 0:
        hold_mult = 1.5 if dir_prob > 0.7 else 1.0
        max_holding_bars = int(pred_mtt * hold_mult)
        if min_holding is not None and max_holding_bars < int(min_holding):
            max_holding_bars = int(min_holding)

    # Allow trailing in trend archetypes only if dir is confident
    trend_archetypes = {
        "BREAKOUTPULLBACKCONTINUATION",
        "HTFBIASLTFENTRY",
        "MOMENTUMEXPANSION",
    }
    allow_trailing = str(archetype_name).upper() in trend_archetypes and dir_prob > 0.65
    trailing_atr = None
    if allow_trailing:
        mae_ref = pred_mae if pred_mae > 0 else stop_loss_r
        trailing_atr = max(0.3, min(float(stop_loss_r), float(mae_ref)) * 0.6)

    rr_constraints = {
        "stop_loss_r": float(stop_loss_r),
        "take_profit_r": float(take_profit_r),
        "max_holding_bars": max_holding_bars,
        "min_holding_bars": min_holding,
        "allow_trailing": allow_trailing,
        "trailing_atr": trailing_atr,
    }

    return {
        "size_multiplier": float(size_mult),
        "rr_constraints": rr_constraints,
        "signals": {
            "confidence": confidence,
            "pred_dir_prob": dir_prob,
            "pred_mfe_atr": pred_mfe,
            "pred_mae_atr": pred_mae,
            "pred_t_to_mfe": pred_mtt,
        },
        "notes": {
            "structural_tp": structural_tp,
            "structural_sl": structural_sl,
        },
    }
