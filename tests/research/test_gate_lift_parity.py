"""Parity smoke: single-feature gate lift vs optimize_gate_rule_unified."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.stat_kernels.gate_optimize import optimize_gate_rule_unified
from scripts.research.gate_lift_scan import gate_lift_plateau_payload
from src.research.stat_kernels.robustness import UnifiedOptimizationConfig
from src.time_series_model.archetype import GateRule


def _df(n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 1, n)
    good = (x < 0.4).astype(int)
    return pd.DataFrame({"vol_persistence": x, "is_good": good})


def test_lift_parity_recommended_threshold_within_tolerance() -> None:
    df = _df()
    cfg = UnifiedOptimizationConfig(
        min_lift=0.05, min_pass_rate=0.1, max_pass_rate=0.95
    )
    rule = GateRule(
        id="gate_test",
        tag="TEST",
        phase="hard_gate",
        priority=1,
        reason="test",
        when={"vol_persistence": {"value_gt": 0.5}},
        then={"action": "deny"},
    )
    old = optimize_gate_rule_unified(df, rule, "is_good", cfg, step=0.05, strategy=None)
    payload = gate_lift_plateau_payload(
        df,
        "vol_persistence",
        "gt",
        base_mask=pd.Series(True, index=df.index),
        label_col="is_good",
        grid=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        config=cfg,
    )
    if old.get("status") == "stable_plateau_found" and payload.get("is_plateau"):
        old_rec = float(old["recommended_threshold"])
        new_rec = float(payload["recommended"])
        assert abs(old_rec - new_rec) < 0.15
