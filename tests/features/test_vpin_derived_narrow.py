import numpy as np
import pandas as pd

from src.features.time_series.utils_order_flow_features import (
    compute_vpin_ma_max_features_from_base,
    compute_vpin_change_features_from_base,
    compute_vpin_zscore_features_from_base,
    compute_vpin_quantile_rank_features_from_base,
    compute_vpin_volatility_features_from_base,
    compute_vpin_spike_features_from_base,
    compute_vpin_momentum_features_from_base,
    compute_vpin_signed_zscore_features_from_base,
    compute_vpin_ma_max_features_from_series,
    compute_vpin_change_features_from_series,
    compute_vpin_zscore_features_from_series,
    compute_vpin_quantile_rank_features_from_series,
    compute_vpin_volatility_features_from_series,
    compute_vpin_spike_features_from_series,
    compute_vpin_momentum_features_from_series,
    compute_vpin_signed_zscore_features_from_series,
)


def _make_vpin_series(n: int = 200, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    v = rng.normal(loc=0.5, scale=0.2, size=n).astype(float)
    v[5] = np.inf
    v[7] = -np.inf
    v[9] = np.nan
    idx = pd.date_range("2024-01-01", periods=n, freq="5min")
    return pd.Series(v, index=idx, name="vpin")


def _make_vpin_signed_series(n: int = 200, seed: int = 1) -> pd.Series:
    rng = np.random.default_rng(seed)
    v = rng.normal(loc=0.0, scale=1.0, size=n).astype(float)
    v[3] = np.inf
    v[4] = -np.inf
    v[8] = np.nan
    idx = pd.date_range("2024-01-01", periods=n, freq="5min")
    return pd.Series(v, index=idx, name="vpin_signed_imbalance")


def test_vpin_ma_max_from_series_matches_from_base():
    vpin = _make_vpin_series()
    base = compute_vpin_ma_max_features_from_base(pd.DataFrame({"vpin": vpin}))
    narrow = compute_vpin_ma_max_features_from_series(vpin=vpin)
    assert list(narrow.columns) == list(base.columns)
    assert np.allclose(base.values, narrow.values, equal_nan=True)


def test_vpin_change_from_series_matches_from_base():
    vpin = _make_vpin_series()
    base = compute_vpin_change_features_from_base(pd.DataFrame({"vpin": vpin}))
    narrow = compute_vpin_change_features_from_series(vpin=vpin)
    assert list(narrow.columns) == list(base.columns)
    assert np.allclose(base.values, narrow.values, equal_nan=True)


def test_vpin_zscore_from_series_matches_from_base():
    vpin = _make_vpin_series()
    base = compute_vpin_zscore_features_from_base(pd.DataFrame({"vpin": vpin}))
    narrow = compute_vpin_zscore_features_from_series(vpin=vpin)
    assert list(narrow.columns) == list(base.columns)
    assert np.allclose(base.values, narrow.values, equal_nan=True)


def test_vpin_quantile_rank_from_series_matches_from_base():
    vpin = _make_vpin_series()
    base = compute_vpin_quantile_rank_features_from_base(pd.DataFrame({"vpin": vpin}))
    narrow = compute_vpin_quantile_rank_features_from_series(vpin=vpin)
    assert list(narrow.columns) == list(base.columns)
    assert np.allclose(base.values, narrow.values, equal_nan=True)


def test_vpin_volatility_from_series_matches_from_base():
    vpin = _make_vpin_series()
    base = compute_vpin_volatility_features_from_base(pd.DataFrame({"vpin": vpin}))
    narrow = compute_vpin_volatility_features_from_series(vpin=vpin)
    assert list(narrow.columns) == list(base.columns)
    assert np.allclose(base.values, narrow.values, equal_nan=True)


def test_vpin_spike_from_series_matches_from_base():
    vpin = _make_vpin_series()
    base = compute_vpin_spike_features_from_base(pd.DataFrame({"vpin": vpin}))
    narrow = compute_vpin_spike_features_from_series(vpin=vpin)
    assert list(narrow.columns) == list(base.columns)
    assert np.allclose(base.values, narrow.values, equal_nan=True)


def test_vpin_momentum_from_series_matches_from_base():
    vpin = _make_vpin_series()
    base = compute_vpin_momentum_features_from_base(pd.DataFrame({"vpin": vpin}))
    narrow = compute_vpin_momentum_features_from_series(vpin=vpin)
    assert list(narrow.columns) == list(base.columns)
    assert np.allclose(base.values, narrow.values, equal_nan=True)


def test_vpin_signed_zscore_from_series_matches_from_base():
    vsi = _make_vpin_signed_series()
    base = compute_vpin_signed_zscore_features_from_base(
        pd.DataFrame({"vpin_signed_imbalance": vsi})
    )
    narrow = compute_vpin_signed_zscore_features_from_series(vpin_signed_imbalance=vsi)
    assert list(narrow.columns) == list(base.columns)
    assert np.allclose(base.values, narrow.values, equal_nan=True)
