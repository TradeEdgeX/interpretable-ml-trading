#!/usr/bin/env python3
"""
Poll Binance UM futures REST depth and save wall snapshot Parquet (T5α Phase 1B).

API: GET /fapi/v1/depth?symbol=&limit=1000  (weight 20, public)

Binance does **not** publish historical L2 archives. This tool **polls live**
snapshots and appends to daily parquet files for later merge_asof onto 2h bars.

Output:
  data/orderbook/parquet/<SYMBOL>_YYYY-MM-DD_depth_snap.parquet

Schema (DatetimeIndex UTC):
  _symbol, mid, spread_bps, bucket_pct,
  wall_bid_notional_usd_max, wall_ask_notional_usd_max,
  wall_bid_price, wall_ask_price, best_bid, best_ask, depth_limit
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests

from src.data_tools.depth_wall_aggregate import aggregate_walls_from_depth


def _normalize_symbol(symbol: str) -> str:
    s = str(symbol or "").strip().upper().replace("-", "").replace("/", "")
    if not s:
        raise ValueError("empty symbol")
    if not s.endswith("USDT"):
        s = f"{s}USDT"
    return s


@dataclass
class DepthSnapshotDownloader:
    parquet_dir: Path
    base_url: str = "https://fapi.binance.com/fapi/v1/depth"
    depth_limit: int = 1000
    bucket_pct: float = 0.005
    retries: int = 4
    timeout_sec: int = 30
    session: requests.Session | None = None

    def __post_init__(self) -> None:
        self.parquet_dir.mkdir(parents=True, exist_ok=True)
        if self.session is None:
            self.session = requests.Session()

    def _daily_path(self, symbol: str, day: datetime) -> Path:
        d = day.astimezone(timezone.utc).date().isoformat()
        return self.parquet_dir / f"{symbol}_{d}_depth_snap.parquet"

    def _fetch_depth(self, symbol: str) -> dict:
        assert self.session is not None
        params = {"symbol": symbol, "limit": int(self.depth_limit)}
        for attempt in range(self.retries):
            try:
                resp = self.session.get(
                    self.base_url, params=params, timeout=self.timeout_sec
                )
                if resp.status_code == 429:
                    time.sleep(min(2 ** (attempt + 1), 30))
                    continue
                resp.raise_for_status()
                data = resp.json()
                if not isinstance(data, dict) or "bids" not in data:
                    raise ValueError(f"unexpected depth response for {symbol}")
                return data
            except Exception:
                time.sleep(min(2**attempt, 10))
        raise RuntimeError(f"depth fetch failed for {symbol}")

    def poll_once(self, symbol: str, *, ts: Optional[datetime] = None) -> pd.DataFrame:
        sym = _normalize_symbol(symbol)
        raw = self._fetch_depth(sym)
        wall = aggregate_walls_from_depth(
            raw["bids"], raw["asks"], bucket_pct=self.bucket_pct
        )
        now = ts or datetime.now(tz=timezone.utc)
        row = {
            "datetime": now,
            "_symbol": sym,
            "mid": wall.mid,
            "spread_bps": wall.spread_bps,
            "bucket_pct": wall.bucket_pct,
            "wall_bid_notional_usd_max": wall.wall_bid_notional_usd_max,
            "wall_ask_notional_usd_max": wall.wall_ask_notional_usd_max,
            "wall_bid_price": wall.wall_bid_price,
            "wall_ask_price": wall.wall_ask_price,
            "best_bid": wall.best_bid,
            "best_ask": wall.best_ask,
            "depth_limit": int(self.depth_limit),
        }
        df = pd.DataFrame([row]).set_index("datetime")
        idx = df.index
        df.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
        return df

    def append_snapshot(self, symbol: str, df: pd.DataFrame) -> Path:
        sym = _normalize_symbol(symbol)
        if df.empty:
            raise ValueError("empty snapshot")
        ts = df.index[-1].to_pydatetime()
        out_path = self._daily_path(sym, ts)
        part = df.copy()
        if out_path.exists() and out_path.stat().st_size > 0:
            old = pd.read_parquet(out_path)
            if not isinstance(old.index, pd.DatetimeIndex):
                old.index = pd.to_datetime(old.index, utc=True)
            combined = pd.concat([old, part])
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        else:
            combined = part.sort_index()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(out_path)
        return out_path

    def run_poll_loop(
        self,
        *,
        symbols: List[str],
        poll_count: int,
        poll_interval_sec: float,
    ) -> None:
        total = len(symbols) * int(poll_count)
        n = 0
        for i in range(int(poll_count)):
            for sym_raw in symbols:
                n += 1
                sym = _normalize_symbol(sym_raw)
                try:
                    snap = self.poll_once(sym)
                    path = self.append_snapshot(sym, snap)
                    print(
                        f"[{n}/{total}] ok {sym} "
                        f"bid_wall=${snap['wall_bid_notional_usd_max'].iloc[0]:,.0f} "
                        f"ask_wall=${snap['wall_ask_notional_usd_max'].iloc[0]:,.0f} "
                        f"→ {path.name}"
                    )
                except Exception as e:
                    print(f"[{n}/{total}] error {sym}: {e}")
                time.sleep(float(poll_interval_sec))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Poll Binance depth → wall snapshot parquet"
    )
    ap.add_argument("--symbols", nargs="+", required=True)
    ap.add_argument(
        "--poll-count",
        type=int,
        default=1,
        help="Polls per symbol (historical backfill N/A; use cron/loop to accumulate)",
    )
    ap.add_argument("--poll-interval-sec", type=float, default=60.0)
    ap.add_argument(
        "--depth-limit", type=int, default=1000, choices=[5, 10, 20, 50, 100, 500, 1000]
    )
    ap.add_argument("--bucket-pct", type=float, default=0.005)
    ap.add_argument(
        "--parquet-dir",
        default="data/orderbook/parquet",
        help="Daily snapshot parquet output dir",
    )
    args = ap.parse_args()

    dl = DepthSnapshotDownloader(
        parquet_dir=Path(args.parquet_dir),
        depth_limit=int(args.depth_limit),
        bucket_pct=float(args.bucket_pct),
    )
    dl.run_poll_loop(
        symbols=list(args.symbols),
        poll_count=int(args.poll_count),
        poll_interval_sec=float(args.poll_interval_sec),
    )


if __name__ == "__main__":
    main()
