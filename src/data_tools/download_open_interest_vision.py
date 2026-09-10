#!/usr/bin/env python3
"""
Download Binance futures UM daily metrics (Vision ZIP) and write monthly OI Parquet.

Vision path:
  https://data.binance.vision/data/futures/um/daily/metrics/<SYMBOL>/

Each daily CSV has 5m ``sum_open_interest`` / ``sum_open_interest_value`` rows.
Output schema matches ``download_open_interest.py``::

    DatetimeIndex('datetime', UTC)
    columns: _symbol, oi_contracts, oi_usd

File pattern: ``<parquet_dir>/<SYMBOL>_YYYY-MM_oi_5m.parquet``
"""

from __future__ import annotations

import argparse
import io
import time
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests


def _normalize_symbol(symbol: str) -> str:
    s = str(symbol or "").strip().upper().replace("-", "").replace("/", "")
    if not s:
        raise ValueError("Empty symbol")
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


def _month_key(d: date) -> Tuple[int, int]:
    return d.year, d.month


@dataclass
class OpenInterestVisionDownloader:
    data_dir: Path
    parquet_dir: Path
    base_url: str = "https://data.binance.vision/data/futures/um/daily/metrics"
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
        return f"{symbol}-metrics-{day.isoformat()}.zip"

    def _zip_url(self, symbol: str, day: date) -> str:
        return f"{self.base_url}/{symbol}/{self._zip_name(symbol, day)}"

    def _local_zip_path(self, symbol: str, day: date) -> Path:
        sym_dir = self.data_dir / symbol
        sym_dir.mkdir(parents=True, exist_ok=True)
        return sym_dir / self._zip_name(symbol, day)

    def _parquet_path(self, symbol: str, year: int, month: int) -> Path:
        return self.parquet_dir / f"{symbol}_{year}-{month:02d}_oi_5m.parquet"

    def _should_skip_day(self, zip_path: Path, *, force: bool) -> bool:
        if force:
            return False
        return zip_path.exists() and zip_path.stat().st_size > 1024

    def _download_day(self, symbol: str, day: date) -> bool:
        url = self._zip_url(symbol, day)
        out = self._local_zip_path(symbol, day)
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
                continue
        return False

    def _zip_to_oi_df(self, zip_path: Path, *, symbol: str) -> pd.DataFrame:
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
            if not members:
                raise ValueError(f"No CSV found inside {zip_path}")
            with zf.open(members[0], "r") as f:
                raw = f.read()
        df = pd.read_csv(io.BytesIO(raw))
        required = {"create_time", "sum_open_interest", "sum_open_interest_value"}
        if not required.issubset(df.columns):
            raise ValueError(
                f"Unexpected metrics CSV schema in {zip_path}. "
                f"Columns={list(df.columns)[:20]}"
            )
        df["datetime"] = pd.to_datetime(df["create_time"], utc=True)
        df["oi_contracts"] = pd.to_numeric(df["sum_open_interest"], errors="coerce")
        df["oi_usd"] = pd.to_numeric(df["sum_open_interest_value"], errors="coerce")
        df["_symbol"] = symbol
        out = df[["datetime", "_symbol", "oi_contracts", "oi_usd"]].copy()
        out = out.sort_values("datetime").drop_duplicates(
            subset=["datetime"], keep="last"
        )
        out = out.set_index("datetime")
        return out

    def _merge_monthly_parquet(
        self,
        symbol: str,
        year: int,
        month: int,
        vision_df: pd.DataFrame,
    ) -> pd.DataFrame:
        ppath = self._parquet_path(symbol, year, month)
        parts = [vision_df]
        if ppath.exists() and ppath.stat().st_size > 0:
            existing = pd.read_parquet(ppath)
            parts.insert(0, existing)
        merged = pd.concat(parts, axis=0, ignore_index=False)
        if not isinstance(merged.index, pd.DatetimeIndex):
            if "datetime" in merged.columns:
                merged.index = pd.to_datetime(merged["datetime"], utc=True)
            else:
                raise ValueError(f"Invalid OI parquet schema: {ppath}")
        idx = merged.index
        merged.index = (
            idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
        )
        merged = merged.sort_index()
        merged = merged[~merged.index.duplicated(keep="last")]
        return merged

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
        ok = 0
        skipped = 0
        missing = 0
        failed = 0
        n = 0
        month_frames: Dict[Tuple[str, int, int], List[pd.DataFrame]] = {}

        for sym_raw in symbols:
            sym = _normalize_symbol(sym_raw)
            for day in days:
                n += 1
                zpath = self._local_zip_path(sym, day)
                if self._should_skip_day(zpath, force=force):
                    skipped += 1
                    got = zpath.exists()
                else:
                    got = self._download_day(sym, day)
                    if got:
                        ok += 1
                        time.sleep(self.sleep_sec)
                    else:
                        missing += 1

                if self.progress_every and (n % self.progress_every == 0 or n == total):
                    state = (
                        "skip"
                        if self._should_skip_day(zpath, force=False) and zpath.exists()
                        else ("ok" if got else "404")
                    )
                    print(f"[{n}/{total}] {state} {sym} {day.isoformat()}")

                if not zpath.exists() or zpath.stat().st_size <= 1024:
                    continue
                try:
                    day_df = self._zip_to_oi_df(zpath, symbol=sym)
                except Exception as e:
                    failed += 1
                    print(f"[{n}/{total}] ❌ parse failed {sym} {day.isoformat()}: {e}")
                    continue
                key = (sym, day.year, day.month)
                month_frames.setdefault(key, []).append(day_df)

        months_written = 0
        for (sym, year, month), frames in sorted(month_frames.items()):
            vision_df = pd.concat(frames, axis=0).sort_index()
            vision_df = vision_df[~vision_df.index.duplicated(keep="last")]
            merged = self._merge_monthly_parquet(sym, year, month, vision_df)
            ppath = self._parquet_path(sym, year, month)
            merged.to_parquet(ppath)
            months_written += 1
            print(
                f"✅ monthly {sym} {year}-{month:02d} rows={len(merged)} "
                f"range={merged.index.min()} .. {merged.index.max()}"
            )

        print(
            "✅ openInterestVision done: "
            f"days_ok={ok}, days_skipped={skipped}, days_404={missing}, "
            f"parse_failed={failed}, months_written={months_written}, total_days={total}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Download Binance Vision daily metrics → monthly OI Parquet (5m)"
    )
    ap.add_argument(
        "--symbols", nargs="+", required=True, help="Symbols like HYPEUSDT ETHUSDT …"
    )
    ap.add_argument("--start-date", required=True, help="YYYY-MM-DD (inclusive)")
    ap.add_argument(
        "--end-date",
        default=None,
        help="YYYY-MM-DD inclusive (default: today UTC)",
    )
    ap.add_argument(
        "--data-dir",
        default="data/open_interest/vision_zip",
        help="Cache directory for daily metrics ZIP files",
    )
    ap.add_argument(
        "--parquet-dir",
        default="data/open_interest/parquet",
        help="Output directory for monthly OI Parquet",
    )
    ap.add_argument("--sleep-sec", type=float, default=0.2)
    ap.add_argument("--progress-every", type=int, default=25)
    ap.add_argument(
        "--force", action="store_true", help="Re-download cached daily ZIPs"
    )
    args = ap.parse_args()

    start = date.fromisoformat(str(args.start_date))
    if args.end_date:
        end = date.fromisoformat(str(args.end_date))
    else:
        end = datetime.now(tz=timezone.utc).date()

    dl = OpenInterestVisionDownloader(
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
