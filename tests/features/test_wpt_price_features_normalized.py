import numpy as np
import pandas as pd

from src.features.time_series.utils_wpt_features import (
    extract_wpt_price_features_normalized,
)


def test_wpt_price_features_normalized_are_unitless_and_energy_ratios_bounded():
    n = 260
    idx = pd.date_range("2024-01-01", periods=n, freq="H")
    close = pd.Series(
        100 + np.cumsum(np.random.default_rng(0).normal(0, 0.5, size=n)), index=idx
    )
    df = pd.DataFrame({"close": close}, index=idx)

    out = extract_wpt_price_features_normalized(
        df, price_col="close", window=100, level=4, update_step=5
    )
    for c in ["wpt_price_trend", "wpt_price_fluctuation"]:
        assert c in out.columns
        vals = pd.to_numeric(out[c], errors="coerce").dropna()
        if len(vals) > 0:
            assert np.isfinite(vals.to_numpy()).all()

    for c in [
        "wpt_price_energy_low_ratio",
        "wpt_price_energy_mid_ratio",
        "wpt_price_energy_high_ratio",
    ]:
        assert c in out.columns
        vals = pd.to_numeric(out[c], errors="coerce").dropna()
        if len(vals) > 0:
            assert (vals >= -1e-9).all()
            assert (vals <= 1.0 + 1e-9).all()
