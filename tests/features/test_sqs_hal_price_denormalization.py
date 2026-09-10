import numpy as np
import pandas as pd

from src.features.time_series import baseline_features as bf


def test_sqs_hal_high_denormalizes_hal_level_to_raw_price(monkeypatch):
    idx = pd.date_range("2024-01-01", periods=8, freq="H")
    close = pd.Series(np.linspace(100, 107, len(idx)), index=idx)
    high = close + 1.0
    low = close - 1.0
    volume = pd.Series(1000.0, index=idx)
    atr = pd.Series(2.0, index=idx)

    # HAL high is normalized as (level - close) / ATR.
    hal_high_norm = pd.Series(1.5, index=idx)  # level = close + 1.5*ATR = close + 3

    seen = []

    def _fake_calculate_sqs(*, sr_price, df, **kwargs):
        # record the sr_price that SQS sees
        seen.append(float(sr_price))
        return 0.0

    monkeypatch.setattr(bf, "calculate_sqs", _fake_calculate_sqs)

    bf.compute_sqs_hal_high_from_series(
        high=high,
        low=low,
        close=close,
        volume=volume,
        atr=atr,
        hal_high=hal_high_norm,
        window=3,
        tolerance_factor=0.5,
        sr_type="resistance",
    )

    assert len(seen) > 0
    # last call should use last bar's denormalized SR price
    expected_last = float(close.iloc[-1] + hal_high_norm.iloc[-1] * atr.iloc[-1])
    assert abs(seen[-1] - expected_last) < 1e-9


def test_sqs_hal_low_denormalizes_hal_level_to_raw_price(monkeypatch):
    idx = pd.date_range("2024-01-01", periods=8, freq="H")
    close = pd.Series(np.linspace(200, 207, len(idx)), index=idx)
    high = close + 1.0
    low = close - 1.0
    volume = pd.Series(1000.0, index=idx)
    atr = pd.Series(4.0, index=idx)

    # HAL low normalized: (level - close) / ATR (will be negative if below close)
    hal_low_norm = pd.Series(-1.25, index=idx)  # level = close - 5

    seen = []

    def _fake_calculate_sqs(*, sr_price, df, **kwargs):
        seen.append(float(sr_price))
        return 0.0

    monkeypatch.setattr(bf, "calculate_sqs", _fake_calculate_sqs)

    bf.compute_sqs_hal_low_from_series(
        high=high,
        low=low,
        close=close,
        volume=volume,
        atr=atr,
        hal_low=hal_low_norm,
        window=3,
        tolerance_factor=0.5,
        sr_type="support",
    )

    assert len(seen) > 0
    expected_last = float(close.iloc[-1] + hal_low_norm.iloc[-1] * atr.iloc[-1])
    assert abs(seen[-1] - expected_last) < 1e-9
