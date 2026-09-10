import numpy as np
import pandas as pd

from src.features.time_series.utils_interaction_features import (
    compute_macdext_atr_normalized_from_series,
    compute_macdfix_atr_normalized_from_series,
)


def test_macdext_atr_normalized_matches_formula():
    idx = pd.date_range("2025-01-01", periods=5, freq="H", tz="UTC")
    close = pd.Series([100, 100, 100, 100, 100], index=idx, dtype=float)
    atr = pd.Series([2, 2, 0, 2, np.nan], index=idx, dtype=float)
    macdext = pd.Series(
        [0.01, 0.02, 0.03, np.nan, 0.05], index=idx, dtype=float
    )  # rel_close
    macdext_signal = pd.Series([0.01, 0.01, 0.01, 0.01, 0.01], index=idx, dtype=float)
    macdext_histogram = pd.Series([0.0, 0.01, 0.02, 0.03, 0.04], index=idx, dtype=float)

    out = compute_macdext_atr_normalized_from_series(
        macdext=macdext,
        macdext_signal=macdext_signal,
        macdext_histogram=macdext_histogram,
        close=close,
        atr=atr,
    )

    # scale = close/atr, but atr==0 or NaN => treated as 0
    scale = (
        (close / atr.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )
    exp = (macdext * scale).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    assert np.allclose(out["macdext_atr_norm"].values, exp.values, equal_nan=False)


def test_macdfix_atr_normalized_has_expected_columns():
    idx = pd.date_range("2025-01-01", periods=3, freq="H", tz="UTC")
    out = compute_macdfix_atr_normalized_from_series(
        macdfix=pd.Series([0.01, 0.02, 0.03], index=idx),
        macdfix_signal=pd.Series([0.01, 0.01, 0.01], index=idx),
        macdfix_histogram=pd.Series([0.0, 0.01, 0.02], index=idx),
        close=pd.Series([100, 100, 100], index=idx),
        atr=pd.Series([2, 2, 2], index=idx),
    )
    assert set(out.columns) == {
        "macdfix_atr_norm",
        "macdfix_signal_atr_norm",
        "macdfix_histogram_atr_norm",
    }
