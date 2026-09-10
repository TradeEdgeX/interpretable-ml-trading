"""Regime chop column selection for chop_grid / dual_add diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.chop_grid_config import (
    GridConfig,
    regime_chop_column,
    regime_chop_series,
)


def test_regime_chop_series_raw_vs_ts_quantile() -> None:
    ix = pd.date_range("2020-01-01", periods=4, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "semantic_chop": [0.1, 0.2, 0.3, 0.4],
            "semantic_chop_ts_q": [np.nan, 0.9, 0.1, 0.5],
        },
        index=ix,
    )
    raw_cfg = GridConfig(chop_signal="raw")
    q_cfg = GridConfig(chop_signal="ts_quantile")
    pd.testing.assert_series_equal(
        regime_chop_series(df, raw_cfg), df["semantic_chop"], check_names=False
    )
    pd.testing.assert_series_equal(
        regime_chop_series(df, q_cfg), df["semantic_chop_ts_q"], check_names=False
    )
    assert regime_chop_column(raw_cfg) == "semantic_chop"
    assert regime_chop_column(q_cfg) == "semantic_chop_ts_q"
