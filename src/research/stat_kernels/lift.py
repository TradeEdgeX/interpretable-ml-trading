"""Lift metric kernel."""

from __future__ import annotations


def compute_lift(pass_rate_good: float, pass_rate_bad: float) -> float:
    if pass_rate_bad <= 0:
        return float("inf") if pass_rate_good > 0 else 0.0
    return pass_rate_good / pass_rate_bad - 1.0
