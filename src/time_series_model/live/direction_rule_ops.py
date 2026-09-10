"""
Shared direction rule helpers — single implementation for live, validation, backtest.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

METHOD_DUAL_POSITION_AGREE_DEADBAND = "dual_position_agree_deadband"
METHOD_SINGLE_POSITION_BAND = "single_position_band"
METHOD_SIGNAL_MATCH_POSITION_BAND = "signal_match_position_band"


def dual_position_agree_deadband_scalar(v1: Any, v2: Any, epsilon: float) -> int:
    """
    eps >= 0 (deadband):
      Long: v1 > eps and v2 > eps -> +1
      Short: v1 < -eps and v2 < -eps -> -1

    eps < 0 (relaxed band, e.g. SRB 启动期 / 下跌初段 EMA 滞后):
      Long: v1 > eps and v2 > eps (easier than eps>=0 case)
      Short: v1 < -eps and v2 < -eps  with threshold -eps > 0
      If both match, long wins (same order as two separate if branches).
    """
    eps = float(epsilon)
    try:
        p1 = float(v1)
        p2 = float(v2)
    except (TypeError, ValueError):
        return 0
    if np.isnan(p1) or np.isnan(p2):
        return 0
    if eps >= 0:
        if p1 > eps and p2 > eps:
            return 1
        if p1 < -eps and p2 < -eps:
            return -1
        return 0
    if p1 > eps and p2 > eps:
        return 1
    thr = -eps
    if p1 < thr and p2 < thr:
        return -1
    return 0


def dual_position_agree_deadband_series(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    epsilon: float,
) -> pd.Series:
    """Per-row {-1,0,+1} aligned to df.index; missing columns -> all zeros."""
    if col_a not in df.columns or col_b not in df.columns:
        return pd.Series(0.0, index=df.index, dtype=float)

    eps = float(epsilon)

    s1 = pd.to_numeric(df[col_a], errors="coerce").astype(float)
    s2 = pd.to_numeric(df[col_b], errors="coerce").astype(float)

    out = pd.Series(0.0, index=df.index, dtype=float)
    valid = s1.notna() & s2.notna()
    if eps >= 0:
        long_m = valid & (s1 > eps) & (s2 > eps)
        short_m = valid & (s1 < -eps) & (s2 < -eps)
    else:
        thr = -eps
        long_m = valid & (s1 > eps) & (s2 > eps)
        short_m = valid & (s1 < thr) & (s2 < thr)
    out.loc[long_m] = 1.0
    out.loc[short_m & ~long_m] = -1.0
    return out


def single_position_band_scalar(pos: Any, inner_abs: float, outer_abs: float) -> int:
    """Band-pass on normalized position (e.g. macro_tp_vwap_1200_position).

    Long: inner < pos < outer. Short: -outer < pos < -inner.
    Else (too near VWAP, overextended, NaN): 0.
    Requires inner < outer (inner may be negative, e.g. -0.10 for macro pullback long).
    """
    try:
        inner = float(inner_abs)
        outer = float(outer_abs)
    except (TypeError, ValueError):
        return 0
    if outer <= inner:
        return 0
    try:
        p = float(pos)
    except (TypeError, ValueError):
        return 0
    if np.isnan(p):
        return 0
    if inner < p < outer:
        return 1
    if -outer < p < -inner:
        return -1
    return 0


def single_position_band_series(
    df: pd.DataFrame,
    col: str,
    inner_abs: float,
    outer_abs: float,
) -> pd.Series:
    if col not in df.columns:
        return pd.Series(0.0, index=df.index, dtype=float)
    try:
        inner = float(inner_abs)
        outer = float(outer_abs)
    except (TypeError, ValueError):
        return pd.Series(0.0, index=df.index, dtype=float)
    if outer <= inner:
        return pd.Series(0.0, index=df.index, dtype=float)

    s = pd.to_numeric(df[col], errors="coerce").astype(float)
    out = pd.Series(0.0, index=df.index, dtype=float)
    valid = s.notna()
    long_m = valid & (s > inner) & (s < outer)
    short_m = valid & (s > -outer) & (s < -inner)
    out.loc[long_m] = 1.0
    out.loc[short_m & ~long_m] = -1.0
    return out


def parse_signal_match_position_band_rule(rule: dict) -> Optional[Dict[str, Any]]:
    """Compound rule: signal_rules + position_band same sign; optional require_sign_agreement."""
    if not isinstance(rule, dict):
        return None
    if str(rule.get("method", "")).strip().lower() != METHOD_SIGNAL_MATCH_POSITION_BAND:
        return None
    srs = rule.get("signal_rules")
    if not isinstance(srs, list) or not srs:
        return None
    pb = rule.get("position_band")
    if not isinstance(pb, dict):
        return None
    feat = pb.get("feature")
    if not isinstance(feat, str) or not feat.strip():
        return None
    try:
        inner = float(pb.get("inner_abs", pb.get("inner", 0.0)))
        outer = float(pb.get("outer_abs", pb.get("outer", 0.0)))
    except (TypeError, ValueError):
        return None
    consensus = str(rule.get("consensus_mode", "first")).strip().lower()
    rsa_raw = rule.get("require_sign_agreement")
    rsa: Optional[Dict[str, Any]] = None
    if isinstance(rsa_raw, dict):
        rsa_feat = rsa_raw.get("feature")
        if isinstance(rsa_feat, str) and rsa_feat.strip():
            try:
                dead = float(rsa_raw.get("deadband", 0.0) or 0.0)
            except (TypeError, ValueError):
                dead = 0.0
            rsa = {"feature": rsa_feat.strip(), "deadband": dead}
    return {
        "signal_rules": list(srs),
        "band_feature": feat.strip(),
        "inner_abs": inner,
        "outer_abs": outer,
        "consensus_mode": consensus,
        "require_sign_agreement": rsa,
    }


def signal_match_position_band_series(
    df: pd.DataFrame,
    *,
    signal_rules: List[dict],
    band_feature: str,
    inner_abs: float,
    outer_abs: float,
    consensus_mode: str = "first",
    require_sign_agreement: Optional[Dict[str, Any]] = None,
) -> pd.Series:
    """Vectorized compound direction: signal cascade ∩ band same sign.

    consensus_mode:
        "first" (default) — first non-zero signal per row determines direction.
        "all" — ALL non-zero signals must agree; any disagreement → 0.
    require_sign_agreement:
        When set, cand must also satisfy sign(feature) == cand and |feature| > deadband.
    """
    all_signals: List[pd.Series] = []
    for sr in signal_rules:
        if not is_direction_rule_enabled(sr):
            continue
        if (
            isinstance(sr, dict)
            and str(sr.get("method", "")).strip().lower()
            == METHOD_SIGNAL_MATCH_POSITION_BAND
        ):
            continue
        dual = parse_dual_rule(sr)
        if dual is not None:
            col_a, col_b, eps = dual
            if col_a not in df.columns or col_b not in df.columns:
                continue
            d = dual_position_agree_deadband_series(df, col_a, col_b, eps)
        else:
            band = parse_single_position_band_rule(sr)
            if band is not None:
                fcol, ia, oa = band
                if fcol not in df.columns:
                    continue
                d = single_position_band_series(df, fcol, ia, oa)
            else:
                feature = sr.get("feature", "")
                transform = str(sr.get("transform", "raw"))
                if not feature or feature not in df.columns:
                    continue
                series = pd.to_numeric(df[feature], errors="coerce").fillna(0.0)
                if transform == "sign":
                    raw = np.sign(series).astype(float)
                elif transform == "negate_sign":
                    raw = (-np.sign(series)).astype(float)
                elif transform == "center_sign":
                    raw = np.sign(series - 0.5).astype(float)
                elif transform == "threshold":
                    thr = float(sr.get("threshold", 0.0))
                    raw = np.where(series > thr, 1.0, -1.0).astype(float)
                else:
                    raw = series.astype(float)
                d = pd.Series(raw, index=df.index, dtype=float)
        all_signals.append(d)

    if consensus_mode == "all" and all_signals:
        cand = pd.Series(0.0, index=df.index, dtype=float)
        for i, s in enumerate(all_signals):
            non_zero = s != 0
            if i == 0:
                cand = s.copy()
            else:
                disagree = non_zero & (cand != 0) & (s != cand)
                cand = cand.where(~disagree, 0.0)
                new_info = non_zero & (cand == 0) & (~disagree)
                cand = cand.where(~new_info, s)
        has_nonzero = pd.Series(False, index=df.index)
        for s in all_signals:
            has_nonzero = has_nonzero | (s != 0)
        cand = cand.where(has_nonzero, 0.0)
    else:
        cand = pd.Series(0.0, index=df.index, dtype=float)
        for d in all_signals:
            cand = cand.where(cand != 0, d)

    if band_feature not in df.columns:
        return pd.Series(0.0, index=df.index, dtype=float)
    band_d = single_position_band_series(df, band_feature, inner_abs, outer_abs)
    mask = (cand != 0) & (cand == band_d)
    cand = cand.where(mask, 0.0)
    if require_sign_agreement:
        rsa_feat = str(require_sign_agreement.get("feature", "")).strip()
        try:
            dead = float(require_sign_agreement.get("deadband", 0.0) or 0.0)
        except (TypeError, ValueError):
            dead = 0.0
        if not rsa_feat or rsa_feat not in df.columns:
            return pd.Series(0.0, index=df.index, dtype=float)
        sser = pd.to_numeric(df[rsa_feat], errors="coerce")
        slope_ok = sser.notna() & (sser.abs() > dead) & (np.sign(sser) == cand)
        cand = cand.where(slope_ok, 0.0)
    return cand


def parse_single_position_band_rule(
    rule: dict,
) -> Optional[tuple[str, float, float]]:
    """Return (feature_col, inner_abs, outer_abs) for single_position_band."""
    if not isinstance(rule, dict):
        return None
    if rule.get("method") != METHOD_SINGLE_POSITION_BAND:
        return None
    feat = rule.get("feature")
    if not isinstance(feat, str) or not feat.strip():
        return None
    try:
        inner = float(rule.get("inner_abs", rule.get("inner", 0.0)))
        outer = float(rule.get("outer_abs", rule.get("outer", 0.0)))
    except (TypeError, ValueError):
        return None
    return (feat.strip(), inner, outer)


def parse_dual_rule(rule: dict) -> Optional[tuple[str, str, float]]:
    """Return (col_a, col_b, epsilon) if rule is a valid dual deadband rule."""
    if not isinstance(rule, dict):
        return None
    if rule.get("method") != METHOD_DUAL_POSITION_AGREE_DEADBAND:
        return None
    feats = rule.get("features")
    if not isinstance(feats, list) or len(feats) != 2:
        return None
    a, b = str(feats[0]).strip(), str(feats[1]).strip()
    if not a or not b:
        return None
    try:
        eps = float(rule.get("epsilon", 0.0))
    except (TypeError, ValueError):
        eps = 0.0
    return (a, b, eps)


def direction_rule_ft_key(rule: dict) -> Tuple[Any, ...]:
    """Dedup key for direction rules (dual: method+cols+eps; else feature+transform)."""
    if not isinstance(rule, dict):
        return (None, None)
    parsed = parse_dual_rule(rule)
    if parsed is not None:
        a, b, e = parsed
        return ("dual_position_agree_deadband", a, b, float(e))
    band = parse_single_position_band_rule(rule)
    if band is not None:
        f, inn, out = band
        return ("single_position_band", f, float(inn), float(out))
    cmp = parse_signal_match_position_band_rule(rule)
    if cmp is not None:
        return (
            METHOD_SIGNAL_MATCH_POSITION_BAND,
            cmp["band_feature"],
            float(cmp["inner_abs"]),
            float(cmp["outer_abs"]),
            len(cmp["signal_rules"]),
        )
    return (rule.get("feature"), rule.get("transform"))


def is_direction_rule_enabled(rule: dict) -> bool:
    """False only when enabled is explicitly false (same idea as entry_filter.enabled)."""
    if not isinstance(rule, dict):
        return True
    return rule.get("enabled", True) is not False
