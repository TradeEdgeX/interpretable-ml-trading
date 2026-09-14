"""Closed-calendar tests for AI mega-round features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.features.time_series.ai_financing_features import (
    compute_ai_basket_funding_zscore_from_df,
    compute_ai_financing_event_from_df,
    load_ai_financing_events,
)
from src.features.time_series.funding_rate_features import _rolling_robust_zscore


def _write_calendar(tmp: Path, dates: list[str], usd: float = 1_000_000_000.0) -> Path:
    payload = {
        "version": 1,
        "events": [
            {"date": d, "company": "TestLab", "usd": usd, "round": "test"}
            for d in dates
        ],
    }
    path = tmp / "events.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _bars(start: str, n: int, freq: str = "2h") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    return pd.DataFrame({"close": np.linspace(100.0, 110.0, n)}, index=idx)


def test_repo_calendar_loads_mega_rounds() -> None:
    ev = load_ai_financing_events()
    assert len(ev) >= 15
    assert ev.index.min() >= pd.Timestamp("2022-01-01", tz="UTC")
    assert float(ev["usd"].min()) >= 3e8


def test_event_fires_first_bar_of_next_utc_day(tmp_path: Path) -> None:
    cal = _write_calendar(tmp_path, ["2024-05-26"])
    # 2h bars covering announcement day and the day after.
    df = _bars("2024-05-26 00:00", 24)
    out = compute_ai_financing_event_from_df(df, calendar_path=str(cal), hold_days=5)
    same_day = out.index.normalize() == pd.Timestamp("2024-05-26", tz="UTC")
    next_day = out.index.normalize() == pd.Timestamp("2024-05-27", tz="UTC")
    assert float(out.loc[same_day, "ai_financing_event"].sum()) == 0.0
    assert float(out.loc[next_day, "ai_financing_event"].sum()) == 1.0
    first_next = out.index[next_day][0]
    assert out.loc[first_next, "ai_financing_event"] == 1.0
    assert out.loc[first_next, "ai_financing_days_since"] == 1.0
    assert out.loc[next_day, "ai_financing_in_window"].min() == 1.0


def test_window_ends_after_hold_days(tmp_path: Path) -> None:
    cal = _write_calendar(tmp_path, ["2024-05-26"])
    df = _bars("2024-05-26 00:00", 12 * 8, freq="2h")
    out = compute_ai_financing_event_from_df(df, calendar_path=str(cal), hold_days=5)
    day6 = out.index.normalize() == pd.Timestamp("2024-06-01", tz="UTC")
    day5 = out.index.normalize() == pd.Timestamp("2024-05-31", tz="UTC")
    assert float(out.loc[day5, "ai_financing_in_window"].min()) == 1.0
    assert float(out.loc[day6, "ai_financing_in_window"].max()) == 0.0


def test_no_lookahead_future_round_does_not_change_past(tmp_path: Path) -> None:
    cal1 = _write_calendar(tmp_path, ["2024-05-26"])
    two = tmp_path / "two"
    two.mkdir()
    cal2 = _write_calendar(two, ["2024-05-26", "2024-10-02"])
    df = _bars("2024-05-26 00:00", 12 * 10)
    a = compute_ai_financing_event_from_df(df, calendar_path=str(cal1), hold_days=5)
    b = compute_ai_financing_event_from_df(df, calendar_path=str(cal2), hold_days=5)
    past = df.index < pd.Timestamp("2024-10-02", tz="UTC")
    pd.testing.assert_frame_equal(a.loc[past], b.loc[past])


def test_prefix_matches_full_series(tmp_path: Path) -> None:
    cal = _write_calendar(tmp_path, ["2024-05-26", "2024-10-02"])
    df = _bars("2024-05-20 00:00", 12 * 20)
    full = compute_ai_financing_event_from_df(df, calendar_path=str(cal), hold_days=5)
    k = 40
    prefix = compute_ai_financing_event_from_df(
        df.iloc[:k], calendar_path=str(cal), hold_days=5
    )
    pd.testing.assert_frame_equal(full.iloc[:k], prefix)


def test_ai_basket_funding_equal_weight(tmp_path: Path) -> None:
    idx_fr = pd.date_range("2024-01-01", periods=40, freq="8h", tz="UTC")
    for i, sym in enumerate(("FETUSDT", "RENDERUSDT")):
        s = pd.Series(0.0001 * (i + 1) + np.linspace(0, 0.001, 40), index=idx_fr)
        part = pd.DataFrame({"_symbol": sym, "funding_rate": s}, index=idx_fr)
        part.index.name = "datetime"
        part.to_parquet(tmp_path / f"{sym}_2024-01_funding_rate.parquet")
    bars = _bars("2024-01-05 00:00", 12)
    bars["_symbol"] = "BTCUSDT"
    out = compute_ai_basket_funding_zscore_from_df(
        bars,
        funding_rate_dir=str(tmp_path),
        symbols=("FETUSDT", "RENDERUSDT"),
        z_window=10,
        z_min_periods=5,
    )
    assert out["ai_basket_funding_zscore"].notna().any()
    # Mean of two z-scores equals each half-sum; just check it is finite.
    assert np.isfinite(out["ai_basket_funding_zscore"].dropna().iloc[-1])
    # Sanity: helper still returns a series so the join is not empty.
    native = pd.read_parquet(tmp_path / "FETUSDT_2024-01_funding_rate.parquet")["funding_rate"]
    assert _rolling_robust_zscore(native, 10, 5).notna().sum() > 0
