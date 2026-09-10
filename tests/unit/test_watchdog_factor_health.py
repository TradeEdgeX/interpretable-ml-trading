"""Unit tests for regime_watchdog factor health (drift kernels)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.regime_watchdog import evaluate_factor_health


def _ic_baseline(feature: str, rank_ic: float, target: str = "forward_rr") -> dict:
    return {
        "target": target,
        "rows": [{"bucket": "all", "feature": feature, "rank_ic": rank_ic}],
    }


def test_factor_health_ok_stable_distribution():
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.normal(0, 1, 400))
    cur = pd.Series(rng.normal(0, 1, 250))
    x = pd.Series(rng.normal(0, 1, 250))
    y = pd.Series(rng.normal(0, 1, 250))
    window = pd.DataFrame({"feat_a": x, "forward_rr": y})
    ref_df = pd.DataFrame({"feat_a": ref, "forward_rr": ref})

    r = evaluate_factor_health(
        window_df=window,
        reference_df=ref_df,
        ic_baseline=_ic_baseline("feat_a", 0.05),
        psi_features=["feat_a"],
        psi_tol=0.5,
        ic_flip_min_abs=0.02,
    )
    assert r["any_alert"] is False


def test_factor_health_psi_alert():
    rng = np.random.default_rng(1)
    ref = pd.Series(rng.normal(0, 1, 400))
    cur = pd.Series(rng.normal(4, 1, 250))
    window = pd.DataFrame({"feat_a": cur, "forward_rr": cur})
    ref_df = pd.DataFrame({"feat_a": ref, "forward_rr": ref})

    r = evaluate_factor_health(
        window_df=window,
        reference_df=ref_df,
        ic_baseline=_ic_baseline("feat_a", 0.3),
        psi_features=["feat_a"],
        psi_tol=0.05,
        ic_flip_min_abs=0.02,
    )
    assert r["any_alert"] is True
    assert any("PSI_DRIFT" in a for a in r["alerts"])


def test_factor_health_ic_sign_flip_alert():
    n = 200
    x = pd.Series(np.linspace(-1, 1, n))
    window = pd.DataFrame({"feat_a": x, "forward_rr": -x})
    ref_df = pd.DataFrame({"feat_a": x, "forward_rr": x})

    r = evaluate_factor_health(
        window_df=window,
        reference_df=ref_df,
        ic_baseline=_ic_baseline("feat_a", 0.4),
        psi_features=[],
        psi_tol=0.25,
        ic_flip_min_abs=0.02,
    )
    assert r["any_alert"] is True
    assert any("IC_SIGN_FLIP" in a for a in r["alerts"])


def test_factor_health_missing_target_skips_not_alerts():
    """Live bus has no forward_rr — IC drift disabled, not a false ALERT."""
    window = pd.DataFrame({"feat_a": [1.0, 2.0, 3.0]})
    r = evaluate_factor_health(
        window_df=window,
        reference_df=None,
        ic_baseline=_ic_baseline("feat_a", 0.1),
        psi_features=["feat_a"],
        psi_tol=0.25,
        ic_flip_min_abs=0.02,
    )
    assert r["any_alert"] is False
    assert not r["alerts"]
    ic_items = [it for it in r["items"] if it.get("kind") == "ic_drift"]
    assert ic_items and "skipped" in ic_items[0]
    psi_items = [it for it in r["items"] if it.get("kind") == "psi"]
    assert psi_items and "skipped" in psi_items[0]
