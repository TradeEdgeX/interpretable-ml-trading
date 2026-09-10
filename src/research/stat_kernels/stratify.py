"""Feature stratification kernels (bad-rate / median RR by threshold split)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

DEFAULT_MIN_SAMPLES = 30


def compute_stratification(
    df: pd.DataFrame,
    feature: str,
    threshold: float,
    operator: str,
    rr_col: str,
    label_col: str,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> Optional[Dict[str, Any]]:
    """
    Split by threshold; compare bad rate and median RR between signal vs rest.

    operator="high": signal when feature >= threshold
    operator="low": signal when feature <= threshold
    """
    cols_needed = [feature, label_col]
    if rr_col in df.columns and df[rr_col].notna().any():
        cols_needed.append(rr_col)
    cols_needed = list(dict.fromkeys(cols_needed))
    valid = df[cols_needed].copy()
    if valid.columns.duplicated().any():
        valid = valid.loc[:, ~valid.columns.duplicated()]
    valid = valid.dropna().reset_index(drop=True)
    if len(valid) < min_samples * 2:
        return None

    if operator == "high":
        signal_mask = (valid[feature] >= threshold).values
    else:
        signal_mask = (valid[feature] <= threshold).values

    signal_df = valid.loc[signal_mask]
    rest_df = valid.loc[~signal_mask]
    if len(signal_df) < min_samples or len(rest_df) < min_samples:
        return None

    signal_bad_rate = float((signal_df[label_col] == 0).mean())
    rest_bad_rate = float((rest_df[label_col] == 0).mean())
    signal_med_rr = (
        float(signal_df[rr_col].median()) if rr_col in signal_df.columns else float("nan")
    )
    rest_med_rr = (
        float(rest_df[rr_col].median()) if rr_col in rest_df.columns else float("nan")
    )

    return {
        "n_signal": len(signal_df),
        "n_rest": len(rest_df),
        "bad_rate_signal": round(signal_bad_rate, 4),
        "bad_rate_rest": round(rest_bad_rate, 4),
        "bad_rate_diff": round(signal_bad_rate - rest_bad_rate, 4),
        "bad_rate_diff_abs": round(abs(signal_bad_rate - rest_bad_rate), 4),
        "median_rr_signal": round(signal_med_rr, 2),
        "median_rr_rest": round(rest_med_rr, 2),
        "threshold": round(threshold, 4),
    }


def analyze_feature_percentiles(
    df: pd.DataFrame,
    feature: str,
    percentiles: List[int],
    rr_col: str,
    label_col: str,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> List[Dict[str, Any]]:
    """Stratify one feature at multiple percentile thresholds."""
    valid = df[feature].dropna()
    if len(valid) < min_samples * 2:
        return []

    results: List[Dict[str, Any]] = []
    for pct in percentiles:
        threshold = float(np.percentile(valid, pct))
        if pct >= 50:
            row = compute_stratification(
                df, feature, threshold, "high", rr_col, label_col, min_samples
            )
            if row:
                row.update(
                    {"percentile": f"P{pct}", "direction": "high", "feature": feature}
                )
                results.append(row)
        if pct <= 50:
            row = compute_stratification(
                df, feature, threshold, "low", rr_col, label_col, min_samples
            )
            if row:
                row.update(
                    {"percentile": f"P{pct}", "direction": "low", "feature": feature}
                )
                results.append(row)
    return results
