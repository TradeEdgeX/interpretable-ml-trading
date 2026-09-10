"""Smoke test: _build_grid_segment_labels joins seg + features and adds C KPIs."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts._build_grid_segment_labels import build_segment_labels


def _segments() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": "BTCUSDT",
                "start": "2026-01-01 00:00",
                "end": "2026-01-01 12:00",
                "pnl_per_capital": 0.04,
                "max_drawdown": -0.02,
                "trades": 4,
                "forced_exits": 1,
            },
            {
                "symbol": "BTCUSDT",
                "start": "2026-02-01 00:00",
                "end": "2026-02-01 12:00",
                "pnl_per_capital": -0.01,
                "max_drawdown": -0.05,
                "trades": 5,
                "forced_exits": 4,
            },
            {
                "symbol": "ETHUSDT",
                "start": "2026-01-15 00:00",
                "end": "2026-01-15 12:00",
                "pnl_per_capital": 0.10,
                "max_drawdown": 0.0,
                "trades": 2,
                "forced_exits": 0,
            },
        ]
    )


def _features() -> pd.DataFrame:
    rows = []
    for sym in ("BTCUSDT", "ETHUSDT"):
        idx = pd.date_range("2025-12-31", periods=80, freq="h", tz="UTC")
        for i, ts in enumerate(idx):
            rows.append(
                {
                    "datetime": ts,
                    "symbol": sym,
                    "bpc_semantic_chop": 0.40 + (i % 5) * 0.05,
                    "tpc_semantic_chop": 0.30 + (i % 4) * 0.05,
                }
            )
    return pd.DataFrame(rows)


def test_build_segment_labels_columns_and_kpis() -> None:
    out = build_segment_labels(
        segments=_segments(),
        features=_features(),
        tolerance=pd.Timedelta("2h"),
    )

    expected_seg_cols = {
        "seg_pnl_per_capital",
        "seg_max_drawdown",
        "seg_total_r_over_dd",
        "seg_adverse_break_rate",
        "seg_maker_return_per_round",
        "seg_segment_total_r",
        "seg_period_5_ok",
    }
    assert expected_seg_cols.issubset(out.columns)
    assert "bpc_semantic_chop" in out.columns
    assert "tpc_semantic_chop" in out.columns

    assert len(out) == 3
    btc_first = out[out["symbol"] == "BTCUSDT"].iloc[0]
    assert btc_first["seg_pnl_per_capital"] == pytest.approx(0.04)
    assert btc_first["seg_total_r_over_dd"] == pytest.approx(0.04 / 0.02)
    assert btc_first["seg_adverse_break_rate"] == pytest.approx(0.25)
    assert btc_first["seg_maker_return_per_round"] == pytest.approx(0.04 / 4)
    assert btc_first["seg_period_5_ok"] == 1

    btc_second = out[out["symbol"] == "BTCUSDT"].iloc[1]
    assert btc_second["seg_period_5_ok"] == 0

    eth = out[out["symbol"] == "ETHUSDT"].iloc[0]
    assert pd.isna(eth["seg_total_r_over_dd"])


def test_build_segment_labels_empty_features_returns_empty() -> None:
    feats = _features()
    feats = feats.iloc[0:0]
    out = build_segment_labels(
        segments=_segments(),
        features=feats,
        tolerance=pd.Timedelta("2h"),
    )
    assert out.empty


def test_build_segment_labels_requires_pnl_column() -> None:
    bad = _segments().drop(columns=["pnl_per_capital"])
    with pytest.raises(KeyError):
        build_segment_labels(
            segments=bad,
            features=_features(),
            tolerance=pd.Timedelta("2h"),
        )
