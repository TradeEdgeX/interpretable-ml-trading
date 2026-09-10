import numpy as np
import pandas as pd

from src.features.time_series.utils_volatility_features import (
    extract_volatility_features_from_vp,
)


class _VP:
    """Minimal VolumeProfileResult-like object for unit tests."""

    def __init__(
        self, hist: np.ndarray, centers: np.ndarray, price_min: float, price_max: float
    ):
        self.hist = hist
        self.centers = centers
        self.price_min = float(price_min)
        self.price_max = float(price_max)


def test_vp_volatility_features_are_bounded_and_finite():
    # synthetic unimodal distribution
    hist = np.array([1, 2, 4, 8, 4, 2, 1], dtype=float)
    centers = np.linspace(90, 110, len(hist))
    vp = _VP(
        hist=hist,
        centers=centers,
        price_min=float(centers.min()),
        price_max=float(centers.max()),
    )

    feats = extract_volatility_features_from_vp(vp, current_price=100.0)
    assert all(
        k in feats
        for k in [
            "vp_width_ratio",
            "vp_poc_deviation",
            "vp_skewness",
            "vp_entropy",
            "vp_lv_ratio",
            "vp_hv_ratio",
        ]
    )
    vals = np.array(list(feats.values()), dtype=float)
    assert np.isfinite(vals).all()

    assert 0.0 <= feats["vp_width_ratio"] <= 1.0
    assert -1.0 <= feats["vp_poc_deviation"] <= 1.0
    assert -1.0 <= feats["vp_skewness"] <= 1.0
    assert 0.0 <= feats["vp_entropy"] <= 1.0
    assert 0.0 <= feats["vp_lv_ratio"] <= 1.0
    assert 0.0 <= feats["vp_hv_ratio"] <= 1.0
