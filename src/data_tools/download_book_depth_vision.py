#!/usr/bin/env python3
"""
Download Binance futures UM daily bookDepth (Vision ZIP) → wall snapshot Parquet.

Vision path:
  https://data.binance.vision/data/futures/um/daily/bookDepth/<SYMBOL>/

CSV columns: timestamp, percentage, depth, notional
  percentage: -5..-1 (bid bands below mid), +1..+5 (ask bands above mid)

Output (per calendar day, one row per Vision snapshot timestamp):
  data/book_depth/parquet/<SYMBOL>_YYYY-MM-DD_book_depth.parquet

Compatible with ``compute_wall_features_from_df`` (same wall_* column names).
"""

from __future__ import annotations

import argparse
import io
import time
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import requests


def _normalize_symbol(symbol: str) -> str:
    s = str(symbol or "").strip().upper().replace("-", "").replace("/", "")
    if not s:
        raise ValueError("empty symbol")
    if not s.endswith("USDT"):
        s = f"{s}USDT"
    return s


def _date_list(start: date, end: date) -> List[date]:
    days: List[date] = []
    cur = start
    while cur <= end:
        days.append(cur)
        cur += timedelta(days=1)
    return days


def _aggregate_book_depth_csv(df: pd.DataFrame, *, symbol: str) -> pd.DataFrame:
    """Collapse percentage bands to one row per timestamp with max bid/ask notional."""
    required = {"timestamp", "percentage", "depth", "notional"}
    if not required.issubset(df.columns):
        raise ValueError(f"unexpected bookDepth schema: {list(df.columns)}")

    work = df.copy()
    work["datetime"] = pd.to_datetime(work["timestamp"], utc=True)
    work["percentage"] = pd.to_numeric(work["percentage"], errors="coerce")
    work["notional"] = pd.to_numeric(work["notional"], errors="coerce")
    work["depth"] = pd.to_numeric(work["depth"], errors="coerce")

    rows: list[dict] = []
    for ts, grp in work.groupby("datetime", sort=True):
        bids = grp[grp["percentage"] < 0]
        asks = grp[grp["percentage"] > 0]
        bid_n = float(bids["notional"].max()) if not bids.empty else 0.0
        ask_n = float(asks["notional"].max()) if not asks.empty else 0.0
        bid_pct = (
            float(bids.loc[bids["notional"].idxmax(), "percentage"])
            if bid_n > 0
            else 0.0
        )
        ask_pct = (
            float(asks.loc[asks["notional"].idxmax(), "percentage"])
            if ask_n > 0
            else 0.0
        )
        rows.append(
            {
                "datetime": ts,
                "_symbol": symbol,
                "wall_bid_notional_usd_max": bid_n,
                "wall_ask_notional_usd_max": ask_n,
                "wall_bid_pct_band": bid_pct,
                "wall_ask_pct_band": ask_pct,
                "wall_bid_price": float("nan"),
                "wall_ask_price": float("nan"),
                "mid": float("nan"),
                "spread_bps": float("nan"),
                "bucket_pct": float("nan"),
                "source": "vision_book_depth",
            }
        )

    out = pd.DataFrame(rows).set_index("datetime")
    out.index = pd.to_datetime(out.index, utc=True)
    return out.sort_index()


@dataclass
class BookDepthVisionDownloader:
    data_dir: Path
    parquet_dir: Path
    base_url: str = "https://data.binance.vision/data/futures/um/daily/bookDepth"
    retries: int = 4
    timeout_sec: int = 120
    sleep_sec: float = 0.2
    progress_every: int = 25
    session: requests.Session | None = None

    def __post_init__(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.parquet_dir.mkdir(parents=True, exist_ok=True)
        if self.session is None:
            self.session = requests.Session()

    def _zip_name(self, symbol: str, day: date) -> str:
        return f"{symbol}-bookDepth-{day.isoformat()}.zip"

    def _zip_url(self, symbol: str, day: date) -> str:
        return f"{self.base_url}/{symbol}/{self._zip_name(symbol, day)}"

    def _local_zip(self, symbol: str, day: date) -> Path:
        d = self.data_dir / symbol
        d.mkdir(parents=True, exist_ok=True)
        return d / self._zip_name(symbol, day)

    def _parquet_path(self, symbol: str, day: date) -> Path:
        return self.parquet_dir / f"{symbol}_{day.isoformat()}_book_depth.parquet"

    def _should_skip(self, zip_path: Path, parquet_path: Path, *, force: bool) -> bool:
        if force:
            return False
        if parquet_path.exists() and parquet_path.stat().st_size > 0:
            return True
        return zip_path.exists() and zip_path.stat().st_size > 1024

    def _download_zip(self, symbol: str, day: date) -> bool:
        url = self._zip_url(symbol, day)
        out = self._local_zip(symbol, day)
        assert self.session is not None
        for attempt in range(self.retries):
            try:
                resp = self.session.get(url, timeout=self.timeout_sec)
                if resp.status_code == 404:
                    return False
                resp.raise_for_status()
                out.write_bytes(resp.content)
                return True
            except Exception:
                time.sleep(min(2**attempt, 10))
        return False

    def _zip_to_parquet(self, zip_path: Path, *, symbol: str) -> pd.DataFrame:
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
            if not members:
                raise ValueError(f"no CSV in {zip_path}")
            raw = zf.read(members[0])
        df = pd.read_csv(io.BytesIO(raw))
        return _aggregate_book_depth_csv(df, symbol=symbol)

    def run(
        self,
        *,
        symbols: List[str],
        start_date: date,
        end_date: date,
        force: bool = False,
    ) -> None:
        days = _date_list(start_date, end_date)
        total = len(symbols) * len(days)
        ok = skipped = missing = failed = 0
        n = 0
        for sym_raw in symbols:
            sym = _normalize_symbol(sym_raw)
            for day in days:
                n += 1
                zpath = self._local_zip(sym, day)
                ppath = self._parquet_path(sym, day)
                if self._should_skip(zpath, ppath, force=force):
                    skipped += 1
                    if self.progress_every and (
                        n % self.progress_every == 0 or n == total
                    ):
                        print(f"[{n}/{total}] skip {sym} {day}")
                    continue
                if not zpath.exists() or force:
                    got = self._download_zip(sym, day)
                    if not got:
                        missing += 1
                        if self.progress_every and (
                            n % self.progress_every == 0 or n == total
                        ):
                            print(f"[{n}/{total}] 404 {sym} {day}")
                        time.sleep(self.sleep_sec)
                        continue
                    time.sleep(self.sleep_sec)
                try:
                    dfp = self._zip_to_parquet(zpath, symbol=sym)
                    ppath.parent.mkdir(parents=True, exist_ok=True)
                    dfp.to_parquet(ppath)
                    ok += 1
                    if self.progress_every and (
                        n % self.progress_every == 0 or n == total
                    ):
                        print(f"[{n}/{total}] ok {sym} {day} rows={len(dfp)}")
                except Exception as e:
                    failed += 1
                    print(f"[{n}/{total}] error {sym} {day}: {e}")

        print(
            "bookDepthVision done: "
            f"ok={ok}, skipped={skipped}, missing_404={missing}, failed={failed}, total={total}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Download Vision bookDepth → wall parquet")
    ap.add_argument("--symbols", nargs="+", required=True)
    ap.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end-date", default=None, help="YYYY-MM-DD (default: today UTC)")
    ap.add_argument("--data-dir", default="data/book_depth/zip")
    ap.add_argument("--parquet-dir", default="data/book_depth/parquet")
    ap.add_argument("--sleep-sec", type=float, default=0.2)
    ap.add_argument("--progress-every", type=int, default=25)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    start = date.fromisoformat(args.start_date)
    end = (
        date.fromisoformat(args.end_date)
        if args.end_date
        else datetime.now(tz=timezone.utc).date()
    )
    dl = BookDepthVisionDownloader(
        data_dir=Path(args.data_dir),
        parquet_dir=Path(args.parquet_dir),
        sleep_sec=float(args.sleep_sec),
        progress_every=int(args.progress_every),
    )
    dl.run(
        symbols=list(args.symbols),
        start_date=start,
        end_date=end,
        force=bool(args.force),
    )


if __name__ == "__main__":
    main()
