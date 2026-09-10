"""Plateau detection kernels (snotio + metric scan results)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from src.research.stat_kernels.snotio_calc import find_snotio_plateau

__all__ = ["find_snotio_plateau", "find_stable_lift_plateau"]


def find_stable_lift_plateau(
    results: List[Dict[str, Any]],
    config: Any,
    actual_step: float | None = None,
) -> Optional[Dict[str, Any]]:
    """Find stable lift plateau interval from threshold scan results."""
    if not results:
        return None

    results = [r for r in results if np.isfinite(r.get("lift", float("nan")))]
    if not results:
        return None

    results_sorted = sorted(results, key=lambda x: x["threshold"])
    step_for_continuity = (
        actual_step if actual_step is not None else config.threshold_step
    )

    valid_thresholds = []
    for r in results_sorted:
        if (
            r["lift"] >= config.min_lift
            and config.min_pass_rate <= r["pass_rate_all"] <= config.max_pass_rate
            and r.get("n_good", 0) >= config.min_samples_good
            and r.get("n_bad", 0) >= config.min_samples_bad
        ):
            valid_thresholds.append(r)

    if len(valid_thresholds) < 2:
        return None

    stable_intervals = []
    i = 0
    while i < len(valid_thresholds):
        start_idx = i
        anchor_lift = valid_thresholds[i]["lift"]
        current_threshold = valid_thresholds[i]["threshold"]
        j = i + 1
        while j < len(valid_thresholds):
            next_threshold = valid_thresholds[j]["threshold"]
            next_lift = valid_thresholds[j]["lift"]
            if next_threshold - current_threshold > step_for_continuity * 2:
                break
            anchor_pass_rate = valid_thresholds[start_idx]["pass_rate_all"]
            if abs(valid_thresholds[j]["pass_rate_all"] - anchor_pass_rate) > 0.15:
                break
            if abs(next_lift - anchor_lift) > config.max_lift_std_ratio * abs(
                anchor_lift
            ):
                break
            anchor_n_valid = valid_thresholds[start_idx].get("n_valid", 0)
            current_n_valid = valid_thresholds[j].get("n_valid", 0)
            if anchor_n_valid > 0:
                if abs(current_n_valid - anchor_n_valid) / anchor_n_valid > 0.1:
                    break
            current_threshold = next_threshold
            j += 1

        if j - start_idx >= 2:
            interval = valid_thresholds[start_idx:j]
            lifts = [r["lift"] for r in interval]
            lift_mean = float(np.mean(lifts))
            lift_std = float(np.std(lifts)) if len(lifts) > 1 else 0.0
            interval_width = interval[-1]["threshold"] - interval[0]["threshold"]
            if len(valid_thresholds) > 1:
                total_range = (
                    valid_thresholds[-1]["threshold"] - valid_thresholds[0]["threshold"]
                )
                relative_width = (
                    interval_width / total_range if total_range > 0 else 0.0
                )
            else:
                relative_width = 0.0
            if (
                relative_width >= config.min_plateau_width
                or interval_width >= config.min_plateau_width
            ):
                stable_intervals.append(
                    {
                        "interval": interval,
                        "start_threshold": interval[0]["threshold"],
                        "end_threshold": interval[-1]["threshold"],
                        "width": interval_width,
                        "relative_width": relative_width,
                        "lift_mean": lift_mean,
                        "lift_std": lift_std,
                        "lift_min": float(np.min(lifts)),
                        "lift_max": float(np.max(lifts)),
                        "lift_stability_ratio": (
                            lift_std / max(lift_mean, 0.2)
                            if lift_mean != 0
                            else float("inf")
                        ),
                        "num_points": len(interval),
                    }
                )
        i = j if j > i else i + 1

    if not stable_intervals:
        return None

    best_interval = min(
        stable_intervals,
        key=lambda x: (x["lift_stability_ratio"], -x["width"], -x["lift_mean"]),
    )
    interval = best_interval["interval"]
    start_th = best_interval["start_threshold"]
    end_th = best_interval["end_threshold"]
    mid_th = (start_th + end_th) / 2
    mid_metrics = min(interval, key=lambda x: abs(x["threshold"] - mid_th))

    return {
        "plateau_start": start_th,
        "plateau_end": end_th,
        "plateau_mid": mid_th,
        "recommended_threshold": mid_th,
        "recommended_threshold_type": "plateau_mid",
        "lift_mean": best_interval["lift_mean"],
        "lift_std": best_interval["lift_std"],
        "lift_min": best_interval["lift_min"],
        "lift_max": best_interval["lift_max"],
        "lift_stability_ratio": best_interval["lift_stability_ratio"],
        "pass_rate_at_mid": mid_metrics["pass_rate_all"],
        "lift_at_mid": mid_metrics["lift"],
        "plateau_width": best_interval["width"],
        "plateau_relative_width": best_interval.get("relative_width", 0.0),
        "num_valid_thresholds": best_interval["num_points"],
        "interval_details": list(interval),
    }
