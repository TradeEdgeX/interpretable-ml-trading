"""Tests for feature plateau JSON payload (calibrate chain)."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from scripts.quick_layer_scan import feature_plateau_payload


def test_feature_plateau_payload_recommended():
    rng = np.random.default_rng(0)
    n = 500
    feat = rng.normal(size=n)
    df = pd.DataFrame(
        {
            "pulse_z": feat,
            "success_no_rr_extreme": (feat > 0).astype(int),
        }
    )
    label = df["success_no_rr_extreme"].astype(bool)
    mask = pd.Series(True, index=df.index)
    ns = argparse.Namespace(feature="pulse_z", operator=">=", grid="0,0.5,1.0")
    payload = feature_plateau_payload(ns, df, label, mask)
    assert payload["feature"] == "pulse_z"
    assert payload["recommended"] is not None
    assert "rows" in payload and len(payload["rows"]) == 3
