"""Unit tests for src.research.stat_kernels.drift."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.stat_kernels.drift import (
    compute_psi,
    evaluate_ic_drift_vs_baseline,
    evaluate_psi_features,
    plateau_mid_in_band,
    series_percentile,
)


def test_compute_psi_stable_distribution_near_zero():
    rng = np.random.default_rng(42)
    ref = pd.Series(rng.normal(0, 1, 500))
    cur = pd.Series(rng.normal(0, 1, 300))
    psi = compute_psi(ref, cur)
    assert psi is not None
    assert psi < 0.25


def test_ic_sign_flip_detected():
    n = 200
    x = pd.Series(np.linspace(-1, 1, n))
    y_pos = pd.Series(np.linspace(-1, 1, n))
    y_neg = -y_pos
    baseline = {
        "target": "forward_rr",
        "rows": [{"bucket": "all", "feature": "feat_a", "rank_ic": 0.5}],
    }
    df_ok = pd.DataFrame({"feat_a": x, "forward_rr": y_pos})
    items_ok, alerts_ok = evaluate_ic_drift_vs_baseline(
        window_df=df_ok, ic_baseline=baseline, min_n=50
    )
    assert not alerts_ok

    df_flip = pd.DataFrame({"feat_a": x, "forward_rr": y_neg})
    _, alerts_flip = evaluate_ic_drift_vs_baseline(
        window_df=df_flip, ic_baseline=baseline, ic_flip_min_abs=0.02, min_n=50
    )
    assert any("IC_SIGN_FLIP" in a for a in alerts_flip)


def test_plateau_mid_in_band():
    s = pd.Series(np.linspace(0.2, 0.8, 50))
    band = plateau_mid_in_band(series=s, plateau_start=0.3, plateau_end=0.7, min_n=5)
    assert band["status"] == "OK"
    assert band["in_band"] is True

    band_out = plateau_mid_in_band(
        series=s, plateau_start=0.9, plateau_end=1.0, min_n=5
    )
    assert band_out["status"] == "DRIFT"
    assert band_out["in_band"] is False


def test_series_percentile_insufficient():
    assert series_percentile(pd.Series([1.0, 2.0]), 0.5, min_n=5) is None


def test_ic_missing_target_skips_without_alert():
    baseline = {
        "target": "forward_rr",
        "rows": [{"bucket": "all", "feature": "feat_a", "rank_ic": 0.5}],
    }
    df = pd.DataFrame({"feat_a": [1.0, 2.0, 3.0]})
    items, alerts = evaluate_ic_drift_vs_baseline(window_df=df, ic_baseline=baseline)
    assert not alerts
    assert items[0].get("skipped")


def test_evaluate_psi_skips_without_reference():
    df = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0, 5.0]})
    items, alerts = evaluate_psi_features(
        window_df=df,
        reference_df=None,
        psi_features=["f1"],
        psi_tol=0.01,
    )
    assert not alerts
    assert items[0]["skipped"]


def test_evaluate_psi_features_alert():
    ref = pd.Series(np.random.default_rng(0).normal(0, 1, 300))
    cur = pd.Series(np.random.default_rng(1).normal(3, 1, 200))
    df = pd.DataFrame({"f1": cur})
    ref_df = pd.DataFrame({"f1": ref})
    items, alerts = evaluate_psi_features(
        window_df=df,
        reference_df=ref_df,
        psi_features=["f1"],
        psi_tol=0.01,
    )
    assert items[0]["psi"] is not None
    assert alerts
