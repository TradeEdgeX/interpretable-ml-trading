"""EMA(1200) is TA-Lib IIR — same as FeatureStore / live bus. Not pandas ewm."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import talib

from src.features.loader.talib_feature_wrappers import (
    compute_talib_indicator_from_series,
)
from src.features.time_series.ema_talib import ema_talib


def test_ema_talib_matches_feature_store_wrapper() -> None:
    close = pd.Series(
        np.linspace(100.0, 140.0, 1500),
        index=pd.date_range("2020-01-01", periods=1500, freq="2h"),
    )
    helper = ema_talib(close, timeperiod=1200)
    wrapped = compute_talib_indicator_from_series(
        indicator_name="EMA",
        timeperiod=1200,
        output_column="ema_1200",
        real=close,
    )["ema_1200"]
    pd.testing.assert_series_equal(helper, wrapped, check_names=False)


def test_ema_talib_matches_raw_talib() -> None:
    close = pd.Series(np.linspace(50.0, 80.0, 1300), dtype=float)
    got = ema_talib(close, timeperiod=1200)
    raw = pd.Series(talib.EMA(close.to_numpy(), timeperiod=1200), index=close.index)
    pd.testing.assert_series_equal(got, raw, check_names=False)
    assert pd.isna(got.iloc[1198])
    assert pd.notna(got.iloc[1199])


def test_pandas_ewm_diverges_from_talib_during_warmup() -> None:
    """Same alpha, different seed → early gold-cross dates will not match SRB/live."""
    close = pd.Series(np.linspace(10_000.0, 12_000.0, 1500), dtype=float)
    ewm = close.ewm(span=1200, adjust=False).mean()
    tal = ema_talib(close, timeperiod=1200)
    both = pd.concat({"ewm": ewm, "tal": tal}, axis=1).dropna()
    rel = (both["ewm"] - both["tal"]).abs() / both["tal"]
    assert float(rel.iloc[0]) > 1e-3
    assert float(rel.iloc[0]) > float(rel.iloc[-1])
