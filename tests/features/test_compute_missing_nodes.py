"""Post-FS merge fill for bar-only dependents of tick-fed columns."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.loader.compute_missing_nodes import compute_missing_feature_nodes
from src.features.registry import ensure_features_registered


def test_compute_missing_archive_of_after_cvd_present():
    ensure_features_registered()
    n = 80
    idx = pd.date_range("2024-01-01", periods=n, freq="2h")
    df = pd.DataFrame(
        {
            "close": np.linspace(100.0, 110.0, n),
            "volume": np.full(n, 1000.0),
            "src_breakout_side": np.zeros(n),
            "cvd": np.linspace(0.0, 5.0, n),
            "price_to_vwap_ratio": np.full(n, 1.02),
            "ofci_pct": np.full(n, 0.65),
        },
        index=idx,
    )
    df.loc[idx[40:], "volume"] = 5000.0
    df.loc[idx[40:], "src_breakout_side"] = 1.0

    assert "src_archive_of_confirm_ok" not in df.columns
    out, filled = compute_missing_feature_nodes(df, ["src_archive_orderflow_confirm_f"])
    assert "src_archive_orderflow_confirm_f" in filled
    assert "src_archive_of_confirm_ok" in out.columns
    assert float(out["src_archive_of_confirm_ok"].sum()) >= 0.0

    # Idempotent
    out2, filled2 = compute_missing_feature_nodes(
        out, ["src_archive_orderflow_confirm_f"]
    )
    assert filled2 == []
    assert out2["src_archive_of_confirm_ok"].equals(out["src_archive_of_confirm_ok"])
