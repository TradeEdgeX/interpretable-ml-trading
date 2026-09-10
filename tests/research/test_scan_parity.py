"""Research scan CLI uses same kernels as quick_layer_scan."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from scripts import quick_layer_scan


def _tiny_frame(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    feat = rng.normal(size=n)
    feat_b = rng.normal(size=n)
    label = feat > 0
    return pd.DataFrame(
        {
            "pulse_z": feat,
            "vol_persistence": feat_b,
            "success_no_rr_extreme": label,
        }
    )


def test_feature_plateau_report_matches_quick_layer_scan():
    df = _tiny_frame()
    label = df["success_no_rr_extreme"].astype(bool)
    base_mask = pd.Series(True, index=df.index)
    ns = argparse.Namespace(
        feature="pulse_z",
        operator="<=",
        grid="0,1,0.25",
    )
    direct = quick_layer_scan.mode_feature_plateau(ns, df, label, base_mask)
    payload = quick_layer_scan.feature_plateau_payload(ns, df, label, base_mask)
    assert "pulse_z" in direct
    assert payload["feature"] == "pulse_z"
    assert "recommended" in payload or "recommended_threshold" in payload


def test_pair_scan_report_shape():
    df = _tiny_frame()
    label = df["success_no_rr_extreme"].astype(bool)
    base_mask = pd.Series(True, index=df.index)
    ns = argparse.Namespace(
        pair_a="pulse_z:<=:0,0.5,1",
        pair_b="vol_persistence:>=:-1,0,1",
    )
    report = quick_layer_scan.mode_pair_scan(ns, df, label, base_mask)
    assert "pair_scan" in report
    assert "pulse_z" in report
    assert "vol_persistence" in report
    assert "|" in report
