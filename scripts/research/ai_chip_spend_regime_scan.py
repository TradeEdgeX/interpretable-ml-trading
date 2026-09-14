#!/usr/bin/env python3
"""Phase-1: Epoch chip-spend QoQ vs BTC closed-bar MA200 state. Not a close."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.time_series.ai_chip_spend_features import (  # noqa: E402
    compute_ai_chip_spend_from_df,
)


def _load_segments() -> list[dict[str, Any]]:
    raw = yaml.safe_load(
        (PROJECT_ROOT / "config/market_segment.yaml").read_text(encoding="utf-8")
    )
    return [
        s
        for s in (raw.get("segments") or [])
        if s.get("id") in {"bear_2022", "bull_2023_2024", "recent_range_to_bear"}
    ]


def _load_btc_daily(start: str, end: str) -> pd.DataFrame:
    from src.data_tools.data_utils import load_raw_data

    parquet = PROJECT_ROOT / "data" / "parquet_data"
    if parquet.exists():
        df = load_raw_data(
            data_path=str(parquet),
            symbol="BTCUSDT",
            start_date=start,
            end_date=end,
            timeframe="1D",
        )
        close = pd.to_numeric(df["close"], errors="coerce")
        close.index = pd.to_datetime(df.index, utc=True)
        return pd.DataFrame({"close": close}).sort_index()
    vision = PROJECT_ROOT / "data" / "klines_vision" / "BTCUSDT"
    parts = sorted(vision.glob("BTCUSDT_*_1d.parquet"))
    if not parts:
        raise FileNotFoundError("no BTC daily parquet or Vision 1d tape")
    daily = pd.concat([pd.read_parquet(p) for p in parts]).sort_index()
    daily = daily[~daily.index.duplicated(keep="last")]
    daily.index = pd.to_datetime(daily.index, utc=True)
    return daily.loc[start:end, ["close"]]


def _rates(mask_high: pd.Series, bear: pd.Series) -> dict[str, float]:
    m = mask_high.notna() & bear.notna()
    high = m & (mask_high == 1.0)
    low = m & (mask_high == 0.0)
    return {
        "n_high": int(high.sum()),
        "n_low": int(low.sum()),
        "bear_share_high": float(bear[high].mean()) if high.any() else float("nan"),
        "bear_share_low": float(bear[low].mean()) if low.any() else float("nan"),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Chip-spend QoQ vs BTC MA200 state")
    p.add_argument("--start-date", default="2021-01-01")
    p.add_argument("--end-date", default="2026-05-31")
    p.add_argument(
        "--out",
        default=(
            "config/experiments/20260914_ai_chip_spend_btc_regime/"
            "quick_scan/chip_spend_regime.json"
        ),
    )
    args = p.parse_args(argv)

    daily = _load_btc_daily(args.start_date, args.end_date)
    feats = compute_ai_chip_spend_from_df(daily)
    close = pd.to_numeric(daily["close"], errors="coerce")
    sma = close.rolling(200, min_periods=200).mean()
    bear = (close < sma).astype(float)
    bear[sma.isna()] = np.nan

    joined = feats.join(pd.DataFrame({"bear": bear, "close": close}))
    # Need a completed quarter and a defined QoQ-high flag.
    valid = joined.dropna(subset=["ai_chip_spend_qoq_high", "bear"])
    pooled = _rates(valid["ai_chip_spend_qoq_high"], valid["bear"])
    pooled["delta_pp"] = 100.0 * (
        pooled["bear_share_high"] - pooled["bear_share_low"]
    )

    segs = []
    for seg in _load_segments():
        start = pd.Timestamp(seg["start_date"], tz="UTC")
        end = pd.Timestamp(seg["end_date"], tz="UTC")
        sub = valid.loc[(valid.index >= start) & (valid.index < end)]
        row = _rates(sub["ai_chip_spend_qoq_high"], sub["bear"])
        row["segment"] = seg["id"]
        row["delta_pp"] = 100.0 * (row["bear_share_high"] - row["bear_share_low"])
        segs.append(row)

    payload = {
        "experiment_id": "20260914_ai_chip_spend_btc_regime",
        "note": "phase1 flashlight only; does not promote",
        "x": "ai_chip_spend_qoq_high (Epoch chip cost QoQ > expanding median)",
        "y": "BTC daily close < SMA200 (closed bar, not market_segment labels)",
        "source": "config/research/ai_chip_sales_quarterly.csv",
        "attribution": "Epoch AI chip sales, CC BY",
        "summary": {
            "ic": None,
            "lift_pp": pooled["delta_pp"],
            **pooled,
        },
        "segments": segs,
    }
    out = PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    for row in segs:
        print(
            f"{row['segment']}: high_bear={row['bear_share_high']:.1%} "
            f"low_bear={row['bear_share_low']:.1%} "
            f"delta={row['delta_pp']:+.1f}pp nH={row['n_high']} nL={row['n_low']}"
        )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
