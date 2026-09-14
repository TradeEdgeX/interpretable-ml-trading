#!/usr/bin/env python3
"""Phase-1 flashlight for 20260914_ai_financing_btc. Not a close."""

from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.time_series.ai_financing_features import (  # noqa: E402
    compute_ai_basket_funding_zscore_from_df,
    compute_ai_financing_event_from_df,
    load_ai_financing_events,
)
from src.features.time_series.funding_rate_features import (  # noqa: E402
    compute_funding_rate_features_from_df,
)
from src.research.stat_kernels.ic import rank_ic  # noqa: E402

VISION_KLINE = (
    "https://data.binance.vision/data/futures/um/monthly/klines/"
    "{symbol}/1d/{symbol}-1d-{year}-{month:02d}.zip"
)


def _load_segments(path: Path) -> list[dict[str, Any]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rows = []
    for seg in raw.get("segments") or []:
        sid = str(seg.get("id") or "")
        if sid in {"bear_2022", "bull_2023_2024", "recent_range_to_bear"}:
            rows.append(seg)
    return rows


_VISION_KLINE_COLS = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
)


def _read_vision_kline_csv(csv_bytes: bytes) -> pd.DataFrame:
    """Parse Vision monthly kline CSV (headered or headerless)."""
    preview = pd.read_csv(io.BytesIO(csv_bytes), header=None, nrows=1)
    first = str(preview.iloc[0, 0]).strip().lower()
    if first in {"open_time", "opentime"}:
        raw = pd.read_csv(io.BytesIO(csv_bytes))
    else:
        raw = pd.read_csv(io.BytesIO(csv_bytes), header=None)
        raw.columns = list(_VISION_KLINE_COLS[: len(raw.columns)])
    rename = {str(c).strip().lower(): str(c).strip().lower() for c in raw.columns}
    raw = raw.rename(columns=rename)
    if "open_time" not in raw.columns:
        raw = raw.rename(columns={raw.columns[0]: "open_time"})
    return raw


def _vision_open_time_to_utc(series: pd.Series) -> pd.Series:
    num = pd.to_numeric(series, errors="coerce")
    if num.notna().any():
        return pd.to_datetime(num, unit="ms", utc=True)
    return pd.to_datetime(series, utc=True, errors="coerce")


def _download_vision_daily(symbol: str, dest: Path, start: str, end: str) -> pd.DataFrame:
    dest.mkdir(parents=True, exist_ok=True)
    months = pd.period_range(start[:7], end[:7], freq="M")
    parts: list[pd.DataFrame] = []
    for per in months:
        y, m = int(per.year), int(per.month)
        parquet = dest / f"{symbol}_{y}-{m:02d}_1d.parquet"
        if parquet.exists() and parquet.stat().st_size > 0:
            parts.append(pd.read_parquet(parquet))
            continue
        url = VISION_KLINE.format(symbol=symbol, year=y, month=m)
        try:
            with urlopen(url, timeout=60) as resp:
                blob = resp.read()
        except Exception:
            continue
        try:
            with zipfile.ZipFile(io.BytesIO(blob)) as zf:
                names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
                if not names:
                    continue
                csv_bytes = zf.read(names[0])
        except Exception:
            continue
        raw = _read_vision_kline_csv(csv_bytes)
        if raw.empty:
            continue
        raw["datetime"] = _vision_open_time_to_utc(raw["open_time"])
        keep = raw[["datetime", "open", "high", "low", "close", "volume"]].copy()
        keep = keep.dropna(subset=["datetime"]).set_index("datetime").sort_index()
        keep.to_parquet(parquet)
        parts.append(keep)
    if not parts:
        raise FileNotFoundError(f"no Vision 1d klines for {symbol}")
    df = pd.concat(parts).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df.loc[start:end]


def _event_study(
    daily: pd.DataFrame,
    events: pd.DataFrame,
    *,
    hold_days: int,
    start: str,
    end: str,
) -> list[dict[str, Any]]:
    close = pd.to_numeric(daily["close"], errors="coerce")
    ret1 = close.pct_change()
    rows: list[dict[str, Any]] = []
    for dt, ev in events.iterrows():
        d0 = pd.Timestamp(dt).tz_convert("UTC").normalize()
        if d0 < pd.Timestamp(start, tz="UTC") or d0 > pd.Timestamp(end, tz="UTC"):
            continue
        entry = d0 + pd.Timedelta(days=1)
        exit_d = d0 + pd.Timedelta(days=hold_days)
        if entry not in close.index or exit_d not in close.index:
            continue
        win = ret1.loc[entry:exit_d]
        rows.append(
            {
                "date": str(d0.date()),
                "company": ev["company"],
                "usd": float(ev["usd"]),
                "ret_dplus1_to_hold": float(close.loc[exit_d] / close.loc[entry] - 1.0),
                "mean_daily_ret": float(win.mean()) if len(win) else float("nan"),
                "n_days": int(win.notna().sum()),
            }
        )
    return rows


def _funding_window(
    funding: pd.Series,
    events: pd.DataFrame,
    *,
    hold_days: int,
    start: str,
    end: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dt, ev in events.iterrows():
        d0 = pd.Timestamp(dt).tz_convert("UTC").normalize()
        if d0 < pd.Timestamp(start, tz="UTC") or d0 > pd.Timestamp(end, tz="UTC"):
            continue
        pre = funding.loc[d0 - pd.Timedelta(days=5) : d0 - pd.Timedelta(days=1)]
        post = funding.loc[d0 + pd.Timedelta(days=1) : d0 + pd.Timedelta(days=hold_days)]
        rows.append(
            {
                "date": str(d0.date()),
                "company": ev["company"],
                "pre_mean": float(pre.mean()) if pre.notna().any() else float("nan"),
                "post_mean": float(post.mean()) if post.notna().any() else float("nan"),
            }
        )
    return rows


def _seg_mask(index: pd.DatetimeIndex, seg: dict[str, Any]) -> pd.Series:
    start = pd.Timestamp(seg["start_date"], tz="UTC")
    end = pd.Timestamp(seg["end_date"], tz="UTC")
    return pd.Series((index >= start) & (index < end), index=index)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Phase-1 AI-financing vs BTC scan")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--start-date", default="2022-01-01")
    p.add_argument("--end-date", default="2026-05-31")
    p.add_argument("--hold-days", type=int, default=5)
    p.add_argument(
        "--kline-dir",
        default="data/klines_vision",
        help="Phase-1 daily tape (Binance Vision). Not parquet_data.",
    )
    p.add_argument("--funding-dir", default="data/funding_rate/parquet")
    p.add_argument(
        "--out",
        default=(
            "config/experiments/20260914_ai_financing_btc/"
            "quick_scan/ai_financing_scan.json"
        ),
    )
    args = p.parse_args(argv)

    events = load_ai_financing_events()
    daily = _download_vision_daily(
        args.symbol, PROJECT_ROOT / args.kline_dir / args.symbol,
        args.start_date, args.end_date,
    )
    daily["_symbol"] = args.symbol
    feats = compute_ai_financing_event_from_df(
        daily, hold_days=int(args.hold_days)
    )
    try:
        fr = compute_funding_rate_features_from_df(
            daily, funding_rate_dir=str(PROJECT_ROOT / args.funding_dir)
        )
    except Exception:
        fr = pd.DataFrame(index=daily.index)
    try:
        basket = compute_ai_basket_funding_zscore_from_df(
            daily, funding_rate_dir=str(PROJECT_ROOT / args.funding_dir)
        )
    except Exception:
        basket = pd.DataFrame(index=daily.index)

    close = pd.to_numeric(daily["close"], errors="coerce")
    fwd1 = close.pct_change().shift(-1)
    ic, p_ic, n_ic = rank_ic(
        feats["ai_financing_in_window"], fwd1, min_n=50
    )

    segs = _load_segments(PROJECT_ROOT / "config/market_segment.yaml")
    segment_rows: list[dict[str, Any]] = []
    for seg in segs:
        mask = _seg_mask(daily.index, seg)
        sub_events = _event_study(
            daily.loc[mask.values],
            events,
            hold_days=int(args.hold_days),
            start=str(seg["start_date"]),
            end=str(seg["end_date"]),
        )
        rets = [r["ret_dplus1_to_hold"] for r in sub_events]
        base = close.pct_change().loc[mask.values]
        # 5-day baseline: rolling product of 5 subsequent daily returns.
        base5 = (1.0 + base).rolling(int(args.hold_days)).apply(
            lambda x: float(np.prod(x) - 1.0), raw=True
        ).shift(-(int(args.hold_days) - 1))
        funding_rows = []
        if "funding_rate" in fr.columns:
            funding_rows = _funding_window(
                fr["funding_rate"],
                events,
                hold_days=int(args.hold_days),
                start=str(seg["start_date"]),
                end=str(seg["end_date"]),
            )
        post = [r["post_mean"] for r in funding_rows if np.isfinite(r["post_mean"])]
        pre = [r["pre_mean"] for r in funding_rows if np.isfinite(r["pre_mean"])]
        segment_rows.append(
            {
                "segment": seg["id"],
                "n_events": len(sub_events),
                "mean_event_ret": float(np.mean(rets)) if rets else float("nan"),
                "median_event_ret": float(np.median(rets)) if rets else float("nan"),
                "hit_rate": float(np.mean([r > 0 for r in rets])) if rets else float("nan"),
                "baseline_5d_mean": float(base5.mean()) if base5.notna().any() else float("nan"),
                "funding_pre_mean": float(np.mean(pre)) if pre else float("nan"),
                "funding_post_mean": float(np.mean(post)) if post else float("nan"),
                "events": sub_events,
            }
        )

    lead = {}
    if "ai_basket_funding_zscore" in basket.columns and "funding_rate_zscore_50" in fr.columns:
        b = basket["ai_basket_funding_zscore"]
        z = fr["funding_rate_zscore_50"]
        r1 = close.pct_change().shift(-1)
        ic_b_r, p_b_r, n_b_r = rank_ic(b, r1, min_n=50)
        ic_b_z, p_b_z, n_b_z = rank_ic(b.shift(1), z, min_n=50)
        lead = {
            "basket_z_vs_btc_next_ret_ic": ic_b_r,
            "basket_z_vs_btc_next_ret_p": p_b_r,
            "basket_z_vs_btc_next_ret_n": n_b_r,
            "basket_z_lag1_vs_btc_funding_z_ic": ic_b_z,
            "basket_z_lag1_vs_btc_funding_z_p": p_b_z,
            "basket_z_lag1_vs_btc_funding_z_n": n_b_z,
        }

    payload = {
        "experiment_id": "20260914_ai_financing_btc",
        "note": "phase1 flashlight only; does not promote",
        "clock": "closed calendar D+1; Vision 1d klines + Binance funding",
        "n_events_calendar": int(len(events)),
        "summary": {
            "ic": ic,
            "ic_pvalue": p_ic,
            "n": n_ic,
            "lift_pp": None,
        },
        "segments": segment_rows,
        "ai_basket_proxy": lead,
    }
    # Fill lift_pp = event mean minus baseline, pooled recent-first is wrong;
    # report mean of (event - baseline) across segments that have events.
    lifts = []
    for row in segment_rows:
        if np.isfinite(row["mean_event_ret"]) and np.isfinite(row["baseline_5d_mean"]):
            lifts.append(100.0 * (row["mean_event_ret"] - row["baseline_5d_mean"]))
    if lifts:
        payload["summary"]["lift_pp"] = float(np.mean(lifts))

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    for row in segment_rows:
        print(
            f"{row['segment']}: n={row['n_events']} "
            f"mean={row['mean_event_ret']:+.3%} "
            f"base5={row['baseline_5d_mean']:+.3%} "
            f"fund_post={row['funding_post_mean']}"
        )
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
