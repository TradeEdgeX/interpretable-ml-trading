import numpy as np
import pandas as pd

from src.features.time_series.utils_spectrum_features import (
    extract_spectrum_features_from_series,
)


def test_spectrum_core_outputs_are_bounded_0_1():
    n = 300
    idx = pd.date_range("2024-01-01", periods=n, freq="H")
    close = pd.Series(
        100 + np.cumsum(np.random.default_rng(1).normal(0, 0.2, size=n)), index=idx
    )
    out = extract_spectrum_features_from_series(close=close, rolling_window=64)

    for c in [
        "spectrum_price_flatness",
        "spectrum_price_high_freq_ratio",
        "spectrum_price_low_freq_ratio",
        "spectrum_price_entropy",
    ]:
        vals = pd.to_numeric(out[c], errors="coerce").dropna()
        if len(vals) == 0:
            continue
        assert np.isfinite(vals.to_numpy()).all()
        assert vals.min() >= -1e-9
        assert vals.max() <= 1.0 + 1e-9
