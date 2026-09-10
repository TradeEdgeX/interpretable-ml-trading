"""Kernel correctness + legacy-CLI delegation tests for gate optimization.

After the gate-optimization logic was extracted into
``src.research.stat_kernels``, the legacy ``optimize_gate_unified`` CLI no
longer owns any of this math; it delegates to the canonical kernel. These
tests assert (a) the kernels are numerically correct on their own and
(b) the legacy CLI delegates to the same kernel object (no forked code path).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts import optimize_gate_unified as legacy
from src.research.stat_kernels import gate_optimize
from src.research.stat_kernels.gate_lift import (
    compute_lift_for_threshold,
    scan_thresholds_for_lift,
)
from src.research.stat_kernels.plateau import find_stable_lift_plateau
from src.research.stat_kernels.robustness import (
    UnifiedOptimizationConfig,
    compute_robustness_score,
)
from src.research.stat_kernels.stratify import compute_stratification


def _synthetic_gate_df(n: int = 400, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    feat = rng.normal(size=n)
    is_good = (feat > 0).astype(int)
    # flip some labels for mixed pass rates
    flip = rng.random(n) < 0.25
    is_good = np.where(flip, 1 - is_good, is_good)
    return pd.DataFrame(
        {
            "pulse_z": feat,
            "is_good": is_good,
            "forward_rr": rng.normal(0.2, 1.0, size=n),
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
        }
    )


def test_legacy_cli_delegates_to_canonical_kernel():
    # The legacy CLI must not fork the optimization math; it imports the
    # canonical kernel object directly.
    assert legacy.optimize_gate_rule_unified is gate_optimize.optimize_gate_rule_unified


def test_lift_kernel_values_are_sane():
    df = _synthetic_gate_df()
    for op, th in [("lt", 0.0), ("gt", 0.5), ("le", -0.2)]:
        res = compute_lift_for_threshold(df, "pulse_z", op, th)
        for key in ("pass_rate_good", "pass_rate_bad", "pass_rate_all"):
            if np.isnan(res[key]):
                continue
            assert 0.0 <= res[key] <= 1.0


def test_scan_thresholds_and_plateau():
    df = _synthetic_gate_df(n=500)
    cfg = UnifiedOptimizationConfig(
        min_samples_good=20, min_samples_bad=20, min_lift=0.5
    )
    scan = scan_thresholds_for_lift(df, "pulse_z", "lt", (0.0, 1.0), 0.1)
    assert len(scan) > 0
    plateau = find_stable_lift_plateau(scan, cfg, actual_step=0.1)
    if plateau is not None:
        assert (
            plateau["plateau_start"] <= plateau["plateau_mid"] <= plateau["plateau_end"]
        )


def test_robustness_score_bounds():
    df = _synthetic_gate_df(n=600)
    cfg = UnifiedOptimizationConfig(min_samples_good=20, min_samples_bad=20)
    score = compute_robustness_score(df, "pulse_z", "lt", 0.0, config=cfg)
    assert 0.0 <= score.overall_score <= 1.0
    assert 0.0 <= score.param_stability <= 1.0


def test_stratification_kernel():
    df = _synthetic_gate_df(n=200)
    row = compute_stratification(
        df, "pulse_z", 0.0, "high", "forward_rr", "is_good", min_samples=20
    )
    assert row is not None
    assert row["n_signal"] + row["n_rest"] <= len(df)
    assert 0.0 <= row["bad_rate_signal"] <= 1.0


def test_rr_simulate_import():
    from src.research.execution_kernel.rr_simulate import simulate_rr_execution

    assert simulate_rr_execution is not None
