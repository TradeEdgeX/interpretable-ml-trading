"""Gate robustness scoring (decision-boundary stability)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.research.stat_kernels.gate_lift import compute_lift_for_threshold


@dataclass
class RobustnessScore:
    """Robustness score for gate parameter stability."""

    param_stability: float
    temporal_stability: float
    sample_efficiency: float
    overall_score: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "param_stability": self.param_stability,
            "temporal_stability": self.temporal_stability,
            "sample_efficiency": self.sample_efficiency,
            "overall_score": self.overall_score,
        }


@dataclass
class UnifiedOptimizationConfig:
    """Unified gate optimization configuration."""

    min_lift: float = 1.0
    min_pass_rate: float = 0.20
    max_pass_rate: float = 0.80
    min_plateau_width: float = 0.05
    max_lift_std_ratio: float = 0.3
    min_samples_good: int = 50
    min_samples_bad: int = 50
    param_sensitivity_epsilon: float = 0.02
    temporal_cv_folds: int = 5
    threshold_step: float = 0.05
    threshold_range: Tuple[float, float] = (0.05, 0.95)
    strict_hard: bool = False
    allow_hard_nan_lift: bool = True
    nan_lift_max_pass_rate_bad: float = 0.01
    nan_lift_min_pass_rate_good: float = 0.20
    nan_lift_min_coverage: float = 0.15
    nan_lift_min_robustness: float = 0.60
    nan_lift_min_plateau_width: float = 0.03
    require_positive_effect: bool = False
    positive_effect_tol: float = 0.0


def compute_robustness_score(
    df: pd.DataFrame,
    feature_col: str,
    operator: str,
    threshold: float,
    label_col: str = "is_good",
    config: UnifiedOptimizationConfig | None = None,
) -> RobustnessScore:
    """Execution-robust gate score v2 (decision boundary stability)."""
    if config is None:
        config = UnifiedOptimizationConfig()

    base_metrics = compute_lift_for_threshold(
        df, feature_col, operator, threshold, label_col
    )

    valid_good = base_metrics.get("valid_good", base_metrics["n_good"])
    valid_bad = base_metrics.get("valid_bad", base_metrics["n_bad"])

    if valid_bad < config.min_samples_bad or valid_good < config.min_samples_good:
        return RobustnessScore(0.0, 0.0, 0.0, 0.0)

    base_pass_rate_all = base_metrics["pass_rate_all"]
    base_pass_rate_good = base_metrics["pass_rate_good"]
    base_pass_rate_bad = base_metrics["pass_rate_bad"]

    perturbations = [
        config.param_sensitivity_epsilon,
        -config.param_sensitivity_epsilon,
    ]
    pass_rate_changes = []

    for eps in perturbations:
        new_threshold = threshold + eps
        new_threshold = max(
            config.threshold_range[0],
            min(config.threshold_range[1], new_threshold),
        )
        perturbed_metrics = compute_lift_for_threshold(
            df, feature_col, operator, new_threshold, label_col
        )
        delta_all = abs(perturbed_metrics["pass_rate_all"] - base_pass_rate_all)
        delta_good = abs(perturbed_metrics["pass_rate_good"] - base_pass_rate_good)
        delta_bad = abs(perturbed_metrics["pass_rate_bad"] - base_pass_rate_bad)
        combined_change = delta_all + 0.5 * (delta_good + delta_bad)
        pass_rate_changes.append(combined_change)

    if pass_rate_changes and base_pass_rate_all > 0:
        avg_pass_rate_change = float(np.mean(pass_rate_changes))
        denom = max(base_pass_rate_all, 0.1)
        relative_change = avg_pass_rate_change / denom
        param_stability = 1.0 / (1.0 + 10 * relative_change)
        if base_pass_rate_bad < 0.05:
            param_stability *= 0.7
    else:
        param_stability = 1.0

    temporal_stability = 1.0

    if len(df) >= 100:
        if "symbol" in df.columns and df["symbol"].nunique() > 1:
            per_symbol_scores: List[float] = []
            per_symbol_weights: List[int] = []
            for _, sym_df in df.groupby("symbol"):
                if len(sym_df) >= 50:
                    if "timestamp" in sym_df.columns:
                        sym_df = sym_df.sort_values("timestamp")
                    mid_point = len(sym_df) // 2
                    df_first = sym_df.iloc[:mid_point]
                    df_second = sym_df.iloc[mid_point:]
                    m1 = compute_lift_for_threshold(
                        df_first, feature_col, operator, threshold, label_col
                    )
                    m2 = compute_lift_for_threshold(
                        df_second, feature_col, operator, threshold, label_col
                    )
                    diff = abs(m1["pass_rate_all"] - m2["pass_rate_all"]) + 0.5 * (
                        abs(m1["pass_rate_good"] - m2["pass_rate_good"])
                        + abs(m1["pass_rate_bad"] - m2["pass_rate_bad"])
                    )
                    valid_ratio_1 = (
                        m1["n_valid"] / len(df_first) if len(df_first) > 0 else 0
                    )
                    valid_ratio_2 = (
                        m2["n_valid"] / len(df_second) if len(df_second) > 0 else 0
                    )
                    diff += 0.5 * abs(valid_ratio_1 - valid_ratio_2)
                    score = 1.0 / (1.0 + 5 * diff)
                    per_symbol_scores.append(score)
                    per_symbol_weights.append(len(sym_df))

            if per_symbol_scores:
                temporal_stability = float(
                    np.average(per_symbol_scores, weights=per_symbol_weights)
                )
            else:
                temporal_stability = 0.5
        else:
            work = df.sort_values("timestamp") if "timestamp" in df.columns else df
            mid_point = len(work) // 2
            df_first_half = work.iloc[:mid_point].copy()
            df_second_half = work.iloc[mid_point:].copy()
            metrics_first = compute_lift_for_threshold(
                df_first_half, feature_col, operator, threshold, label_col
            )
            metrics_second = compute_lift_for_threshold(
                df_second_half, feature_col, operator, threshold, label_col
            )
            combined_temporal_diff = abs(
                metrics_first["pass_rate_all"] - metrics_second["pass_rate_all"]
            ) + 0.5 * (
                abs(metrics_first["pass_rate_good"] - metrics_second["pass_rate_good"])
                + abs(metrics_first["pass_rate_bad"] - metrics_second["pass_rate_bad"])
            )
            valid_ratio_first = (
                metrics_first["n_valid"] / len(df_first_half)
                if len(df_first_half) > 0
                else 0
            )
            valid_ratio_second = (
                metrics_second["n_valid"] / len(df_second_half)
                if len(df_second_half) > 0
                else 0
            )
            combined_temporal_diff += 0.5 * abs(valid_ratio_first - valid_ratio_second)
            temporal_stability = 1.0 / (1.0 + 5 * combined_temporal_diff)

    n_good = base_metrics.get("valid_good", base_metrics["n_good"])
    n_bad = base_metrics.get("valid_bad", base_metrics["n_bad"])
    n_passed = base_metrics["n_passed"]
    min_samples = min(config.min_samples_good, config.min_samples_bad)
    eff_good = (
        min(np.log(max(n_good, 1)) / np.log(max(min_samples, 1)), 1.0)
        if min_samples > 0
        else 1.0
    )
    eff_bad = (
        min(np.log(max(n_bad, 1)) / np.log(max(min_samples, 1)), 1.0)
        if min_samples > 0
        else 1.0
    )
    passed_ratio = n_passed / len(df) if len(df) > 0 else 0
    passed_penalty = 1.0 if passed_ratio > 0.1 else passed_ratio / 0.1
    sample_efficiency = min(eff_good, eff_bad) * passed_penalty

    overall_score = (
        0.45 * param_stability
        + 0.35 * temporal_stability
        + 0.20 * sample_efficiency
    )

    return RobustnessScore(
        param_stability=param_stability,
        temporal_stability=temporal_stability,
        sample_efficiency=sample_efficiency,
        overall_score=overall_score,
    )
