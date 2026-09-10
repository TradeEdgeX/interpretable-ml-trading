import numpy as np
import pandas as pd
import pytest

from src.time_series_model.portfolio.asset_stats import compute_asset_stats


@pytest.mark.unit
def test_asset_stats_basic():
    r = np.asarray([0.01] * 100, dtype=float)
    st = compute_asset_stats(r)
    assert st.mu == pytest.approx(0.01)
    assert st.max_dd == pytest.approx(0.0)
    assert st.stability > 0.9


@pytest.mark.unit
def test_asset_stats_fragility_regime_variance():
    r = pd.Series([0.01] * 50 + [-0.01] * 50)
    regime = pd.Series(["A"] * 50 + ["B"] * 50)
    st = compute_asset_stats(r, regime=regime)
    assert st.fragility > 0.0
