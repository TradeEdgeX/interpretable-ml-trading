"""Closed-quarter tests for Epoch AI chip-spend features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.features.time_series.ai_chip_spend_features import (
    compute_ai_chip_spend_from_df,
    load_ai_chip_sales_quarterly,
)


def test_pinned_tape_loads() -> None:
    tape = load_ai_chip_sales_quarterly()
    assert len(tape) >= 16
    assert tape["cost_usd"].min() > 0
    assert (tape["available_at"] > tape["end_date"]).all()


def test_same_quarter_cannot_see_own_print() -> None:
    tape = load_ai_chip_sales_quarterly()
    q1 = tape.iloc[0]
    idx = pd.date_range(q1["end_date"] - pd.Timedelta(days=10), periods=20, freq="D", tz="UTC")
    df = pd.DataFrame({"close": np.linspace(1.0, 2.0, len(idx))}, index=idx)
    out = compute_ai_chip_spend_from_df(df)
    before = out.index < q1["available_at"]
    # Before Q1 is available, no prior quarter exists.
    assert out.loc[before, "ai_chip_spend_usd"].isna().all()
    on = out.index >= q1["available_at"]
    assert np.isclose(out.loc[on, "ai_chip_spend_usd"].iloc[0], float(q1["cost_usd"]))


def test_future_quarter_does_not_change_past() -> None:
    tmp = Path("/tmp")  # unused; use prefix of the real tape via drop
    df = pd.DataFrame(
        {"close": np.linspace(1.0, 2.0, 200)},
        index=pd.date_range("2023-01-01", periods=200, freq="D", tz="UTC"),
    )
    full = compute_ai_chip_spend_from_df(df)
    early = compute_ai_chip_spend_from_df(df.iloc[:80])
    pd.testing.assert_frame_equal(full.iloc[:80], early)
    del tmp
