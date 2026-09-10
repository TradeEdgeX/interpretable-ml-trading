"""Drift / stability kernels shared by research and monitor scripts."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.research.stat_kernels.ic import rank_ic


def series_percentile(series: pd.Series, q: float, *, min_n: int = 5) -> Optional[float]:
    s = (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .astype(float)
    )
    if len(s) < min_n:
        return None
    return float(s.quantile(q))


def compute_psi(
    reference: pd.Series,
    current: pd.Series,
    *,
    n_bins: int = 10,
    min_ref: int = 100,
    min_cur: int = 50,
) -> Optional[float]:
    """Population Stability Index between reference and current distributions."""
    ref = (
        pd.to_numeric(reference, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    cur = (
        pd.to_numeric(current, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if len(ref) < min_ref or len(cur) < min_cur:
        return None
    edges = np.unique(np.quantile(ref, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(edges) < 3:
        return None
    ref_hist, _ = np.histogram(ref, bins=edges)
    cur_hist, _ = np.histogram(cur, bins=edges)
    ref_pct = ref_hist / max(ref_hist.sum(), 1)
    cur_pct = cur_hist / max(cur_hist.sum(), 1)
    eps = 1e-6
    return float(np.sum((cur_pct - ref_pct) * np.log((cur_pct + eps) / (ref_pct + eps))))


def ic_drift_item(
    *,
    feature: str,
    baseline_ic: float,
    x: pd.Series,
    y: pd.Series,
    ic_flip_min_abs: float = 0.02,
    min_n: int = 100,
) -> Optional[Dict[str, Any]]:
    """Rank-IC vs baseline; flags sign-flip when both |IC| exceed threshold."""
    cur_ic, _, n = rank_ic(
        pd.to_numeric(x, errors="coerce"),
        pd.to_numeric(y, errors="coerce"),
        min_n=min_n,
    )
    if n < min_n or np.isnan(cur_ic):
        return None
    delta = cur_ic - baseline_ic
    sign_flip = (
        baseline_ic * cur_ic < 0
        and abs(cur_ic) > ic_flip_min_abs
        and abs(baseline_ic) > ic_flip_min_abs
    )
    return {
        "kind": "ic_drift",
        "feature": feature,
        "baseline_ic": baseline_ic,
        "current_ic": cur_ic,
        "delta": delta,
        "n": n,
        "sign_flip": sign_flip,
    }


def evaluate_ic_drift_vs_baseline(
    *,
    window_df: pd.DataFrame,
    ic_baseline: Dict[str, Any],
    ic_flip_min_abs: float = 0.02,
    min_n: int = 100,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """IC drift vs factor_ic_baseline JSON (bucket=all rows only)."""
    items: List[Dict[str, Any]] = []
    alerts: List[str] = []
    target = str(ic_baseline.get("target", "forward_rr"))
    baseline_rows = [
        r
        for r in (ic_baseline.get("rows") or [])
        if isinstance(r, dict) and r.get("bucket") == "all"
    ]
    if target not in window_df.columns:
        # Live feature bus has no forward labels — skip IC (not an ALERT).
        items.append(
            {
                "kind": "ic_drift",
                "skipped": (
                    f"target {target!r} not in window "
                    "(expected on live bus; IC drift disabled)"
                ),
            }
        )
        return items, alerts

    y_win = window_df[target]
    for row in baseline_rows:
        feat = str(row.get("feature", ""))
        if not feat or feat not in window_df.columns:
            continue
        base_ic = float(row.get("rank_ic", 0.0))
        item = ic_drift_item(
            feature=feat,
            baseline_ic=base_ic,
            x=window_df[feat],
            y=y_win,
            ic_flip_min_abs=ic_flip_min_abs,
            min_n=min_n,
        )
        if item is None:
            continue
        items.append(item)
        if item["sign_flip"]:
            alerts.append(
                f"IC_SIGN_FLIP: {feat} {item['current_ic']:+.4f} "
                f"vs baseline {base_ic:+.4f}"
            )
    return items, alerts


def evaluate_psi_features(
    *,
    window_df: pd.DataFrame,
    reference_df: Optional[pd.DataFrame],
    psi_features: List[str],
    psi_tol: float,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    items: List[Dict[str, Any]] = []
    alerts: List[str] = []
    if reference_df is None:
        for feat in psi_features:
            if feat not in window_df.columns:
                items.append(
                    {"kind": "psi", "feature": feat, "skipped": "missing in window"}
                )
            else:
                items.append(
                    {
                        "kind": "psi",
                        "feature": feat,
                        "skipped": (
                            "no reference parquet on host "
                            "(PSI disabled; deploy train_final baseline or ref snapshot)"
                        ),
                    }
                )
        return items, alerts

    ref_df = reference_df
    for feat in psi_features:
        if feat not in window_df.columns:
            items.append(
                {"kind": "psi", "feature": feat, "skipped": "missing in window"}
            )
            continue
        ref_series = ref_df[feat] if feat in ref_df.columns else window_df[feat]
        psi = compute_psi(ref_series, window_df[feat])
        items.append({"kind": "psi", "feature": feat, "psi": psi})
        if psi is not None and psi > psi_tol:
            alerts.append(f"PSI_DRIFT: {feat} psi={psi:.3f} > {psi_tol}")
    return items, alerts


def plateau_mid_in_band(
    *,
    series: pd.Series,
    plateau_start: float,
    plateau_end: float,
    drift_quantile: float = 0.5,
    tail_band_q: Tuple[float, float] = (0.25, 0.75),
    min_n: int = 5,
) -> Dict[str, Any]:
    """Check whether window median lies inside plateau band."""
    p_low = series_percentile(series, tail_band_q[0], min_n=min_n)
    p_mid = series_percentile(series, drift_quantile, min_n=min_n)
    p_high = series_percentile(series, tail_band_q[1], min_n=min_n)
    if p_mid is None:
        return {
            "window_p25": p_low,
            "window_p50": p_mid,
            "window_p75": p_high,
            "status": "INSUFFICIENT_DATA",
            "in_band": False,
        }
    in_band = plateau_start <= p_mid <= plateau_end
    return {
        "window_p25": p_low,
        "window_p50": p_mid,
        "window_p75": p_high,
        "status": "OK" if in_band else "DRIFT",
        "in_band": in_band,
    }
