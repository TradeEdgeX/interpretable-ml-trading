"""weekly_ema_200_position uses live base close vs weekly EMA (not weekly close only)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.features.time_series.baseline_features import (
    compute_weekly_ema_position_from_ohlc,
)


def test_weekly_ema_position_uses_current_close_not_weekly_close_ratio() -> None:
    # Old bug: (weekly_close - wk_ema)/weekly_close ffill → 0.0 all week when last Sunday closed at EMA.
    # New: intraweek 2h close drop must turn position negative while weekly EMA stays ~100.
    idx = pd.date_range("2024-01-01", periods=90 * 84, freq="2h", tz="UTC")  # ~90 weeks
    close = pd.Series(100.0, index=idx)
    close.iloc[-6:] = 70.0
    high = close * 1.01
    low = close * 0.99

    out = compute_weekly_ema_position_from_ohlc(
        close=close,
        high=high,
        low=low,
        ema_span_weeks=8,
    )["weekly_ema_200_position"]

    last = float(out.iloc[-1])
    assert last < -0.05, f"expected negative position, got {last}"


def test_weekly_ema_position_negative_when_close_below_ffilled_ema(
    tmp_path: Path,
) -> None:
    idx = pd.date_range("2026-01-01", periods=500, freq="2h", tz="UTC")
    # Ramp then drop: current price well below slow weekly EMA.
    close = pd.Series(np.linspace(3000, 3500, 450), index=idx[:450])
    close = pd.concat([close, pd.Series(2100.0, index=idx[450:])])
    high = close * 1.001
    low = close * 0.999

    seed_dir = tmp_path / "seed"
    seed_dir.mkdir()
    weekly = pd.DataFrame(
        {
            "week_ts": pd.date_range("2024-01-07", periods=80, freq="W-SUN", tz="UTC"),
            "weekly_close": 3000.0,
            "weekly_ema_200": 2540.0,
        }
    )
    weekly.to_parquet(seed_dir / "ETHUSDT.parquet", index=False)

    out = compute_weekly_ema_position_from_ohlc(
        close=close,
        high=high,
        low=low,
        ema_span_weeks=200,
        weekly_ema_context_dir=str(seed_dir),
        symbol="ETHUSDT",
    )["weekly_ema_200_position"]

    assert float(out.iloc[-1]) < 0.0


def test_weekly_ema_position_uses_explicit_seed_root_env(
    tmp_path: Path, monkeypatch
) -> None:
    idx = pd.date_range("2026-01-01", periods=3, freq="1D", tz="UTC")
    close = pd.Series(80.0, index=idx)
    seed_dir = tmp_path / "research_seeds"
    seed_dir.mkdir()
    pd.DataFrame(
        {
            "week_ts": [pd.Timestamp("2025-12-28", tz="UTC")],
            "weekly_close": [100.0],
            "weekly_ema_200": [100.0],
        }
    ).to_parquet(seed_dir / "BTCUSDT.parquet", index=False)
    monkeypatch.setenv("MLBOT_WEEKLY_EMA_SEED_ROOT", str(seed_dir))

    out = compute_weekly_ema_position_from_ohlc(
        close=close,
        high=close,
        low=close,
        symbol="BTCUSDT",
    )["weekly_ema_200_position"]

    assert float(out.iloc[-1]) == -0.25


def test_short_buffer_insufficient_span_is_nan() -> None:
    idx = pd.date_range("2025-06-01", periods=90 * 12, freq="2h", tz="UTC")
    close = pd.Series(2500.0, index=idx)
    out = compute_weekly_ema_position_from_ohlc(
        close=close,
        high=close * 1.001,
        low=close * 0.999,
        ema_span_weeks=200,
    )["weekly_ema_200_position"]
    assert pd.isna(out.iloc[-1])
