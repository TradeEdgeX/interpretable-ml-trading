"""Gate threshold lift metrics (deny-operator semantics)."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def compute_lift_for_threshold(
    df: pd.DataFrame,
    feature_col: str,
    operator: str,
    threshold: float,
    label_col: str = "is_good",
) -> Dict[str, float]:
    """Compute lift at a single deny-threshold (pass = complement of deny)."""
    if feature_col not in df.columns:
        return {
            "lift": 0.0,
            "lift_valid": True,
            "pass_rate_good": 0.0,
            "pass_rate_bad": 0.0,
            "pass_rate_all": 0.0,
            "n_good": 0,
            "n_bad": 0,
            "n_passed": 0,
            "n_valid": 0,
            "valid_good": 0,
            "valid_bad": 0,
        }

    feat_values = df[feature_col]
    valid_mask = feat_values.notna()
    n_valid = int(valid_mask.sum())

    if operator == "lt":
        passed = valid_mask & (feat_values >= threshold)
    elif operator == "le":
        passed = valid_mask & (feat_values > threshold)
    elif operator == "gt":
        passed = valid_mask & (feat_values <= threshold)
    elif operator == "ge":
        passed = valid_mask & (feat_values < threshold)
    else:
        raise ValueError(
            f"Invalid operator: {operator}. Expected one of: lt, le, gt, ge"
        )

    is_good = df[label_col] == 1
    is_bad = df[label_col] == 0
    n_good = int(is_good.sum())
    n_bad = int(is_bad.sum())
    n_all = len(df)

    if n_good == 0 or n_bad == 0 or n_all == 0:
        return {
            "lift": 0.0,
            "lift_valid": True,
            "pass_rate_good": 0.0,
            "pass_rate_bad": 0.0,
            "pass_rate_all": 0.0,
            "n_good": n_good,
            "n_bad": n_bad,
            "n_passed": 0,
            "n_valid": n_valid,
            "valid_good": 0,
            "valid_bad": 0,
        }

    valid_good = int((is_good & valid_mask).sum())
    valid_bad = int((is_bad & valid_mask).sum())

    pass_rate_good = (passed & is_good).sum() / valid_good if valid_good > 0 else 0.0
    pass_rate_bad = (passed & is_bad).sum() / valid_bad if valid_bad > 0 else 0.0
    pass_rate_all = passed.sum() / n_valid if n_valid > 0 else 0.0

    if pass_rate_bad < 0.01:
        lift = float("nan")
    elif pass_rate_bad > 0:
        lift = pass_rate_good / pass_rate_bad - 1.0
    else:
        lift = float("nan")

    return {
        "lift": lift,
        "lift_valid": bool(np.isfinite(lift)),
        "pass_rate_good": pass_rate_good,
        "pass_rate_bad": pass_rate_bad,
        "pass_rate_all": pass_rate_all,
        "n_good": n_good,
        "n_bad": n_bad,
        "n_passed": int(passed.sum()),
        "n_valid": n_valid,
        "valid_good": valid_good,
        "valid_bad": valid_bad,
    }


def scan_thresholds_for_lift(
    df: pd.DataFrame,
    feature_col: str,
    operator: str,
    threshold_range: Tuple[float, float],
    step: float,
    label_col: str = "is_good",
) -> List[Dict[str, Any]]:
    """Scan threshold grid and return lift metrics per point."""
    results: List[Dict[str, Any]] = []
    low, high = threshold_range
    thresholds = np.arange(low, high + step / 2, step)

    for th in thresholds:
        metrics = compute_lift_for_threshold(df, feature_col, operator, float(th), label_col)
        metrics["threshold"] = float(th)
        results.append(metrics)

    return results
