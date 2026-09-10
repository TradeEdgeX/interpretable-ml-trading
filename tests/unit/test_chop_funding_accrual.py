"""Chop grid coin_m funding parquet accrual."""

from __future__ import annotations

import pandas as pd

from scripts.event_backtest.funding.chop_accrual import aggregated_funding_rate_decimal


def test_aggregated_funding_rate_sums_events_in_hold_window() -> None:
    idx = pd.to_datetime(
        ["2024-01-01 08:00:00+00:00", "2024-01-01 16:00:00+00:00"],
        utc=True,
    )
    series = pd.Series([0.0001, 0.0002], index=idx)
    entry = pd.Timestamp("2024-01-01 07:00:00+00:00")
    exit_ = pd.Timestamp("2024-01-01 20:00:00+00:00")
    rate = aggregated_funding_rate_decimal(entry, exit_, series, side="LONG")
    assert abs(rate - 0.0003) < 1e-12
