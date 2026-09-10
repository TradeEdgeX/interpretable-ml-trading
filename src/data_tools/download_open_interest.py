#!/usr/bin/env python3
"""
Download Binance futures Open Interest historical data via REST API and save as Parquet.

Binance API endpoint:
  GET /futures/data/openInterestHist
  - symbol: e.g. BTCUSDT
  - period: 5m | 15m | 30m | 1h | 2h | 4h | 6h | 12h | 1d
  - limit: max 500
  - startTime / endTime: epoch ms

Response example:
  [
    {
      "symbol": "BTCUSDT",
      "sumOpenInterest": "32212.30700000",      # OI in base asset (contracts)
      "sumOpenInterestValue": "2160113255.77",   # OI in USD
      "timestamp": 1583127900000
    }
  ]

Usage:
  python -m src.data_tools.download_open_interest \
    --symbols BTCUSDT ETHUSDT \
    --start-year 2023 --start-month 1 \
    --period 5m \
    --parquet-dir data/open_interest/parquet
"""

from __future__ import annotations

import argparse
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import requests


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

VALID_PERIODS = ("5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d")


def _month_list(
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
) -> List[Tuple[int, int]]:
    months: List[Tuple[int, int]] = []
    y, m = start_year, start_month
    while (y < end_year) or (y == end_year and m <= end_month):
        months.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


def _normalize_symbol(symbol: str) -> str:
    s = str(symbol or "").strip().upper().replace("-", "").replace("/", "")
    if not s:
        raise ValueError("Empty symbol")
    if not s.endswith("USDT"):
        s = f"{s}USDT"
    return s


def _month_start_ms(year: int, month: int) -> int:
    return int(datetime(year, month, 1, tzinfo=timezone.utc).timestamp() * 1000)


def _month_end_ms(year: int, month: int) -> int:
    """Return the first millisecond of the NEXT month (exclusive)."""
    if month == 12:
        return int(datetime(year + 1, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    return int(datetime(year, month + 1, 1, tzinfo=timezone.utc).timestamp() * 1000)


# ─────────────────────────────────────────────────────────────
# Downloader
# ─────────────────────────────────────────────────────────────


@dataclass
class OpenInterestDownloader:
    """Download Binance futures OI history via paginated API calls, save monthly Parquet."""

    parquet_dir: Path
    base_url: str = "https://fapi.binance.com/futures/data/openInterestHist"
    period: str = "5m"
    retries: int = 4
    timeout_sec: int = 60
    sleep_sec: float = 0.35  # respect Binance rate limits
    page_limit: int = 500  # max 500 per request
    progress_every: int = 25
    session: requests.Session = field(default_factory=requests.Session, init=False)

    def __post_init__(self) -> None:
        self.parquet_dir.mkdir(parents=True, exist_ok=True)
        if self.period not in VALID_PERIODS:
            raise ValueError(
                f"Invalid period '{self.period}'. Must be one of {VALID_PERIODS}"
            )

    # ── file naming ────────────────────────────────────────
    def _parquet_path(self, symbol: str, year: int, month: int) -> Path:
        return (
            self.parquet_dir / f"{symbol}_{year}-{month:02d}_oi_{self.period}.parquet"
        )

    def _should_skip(self, path: Path, *, force: bool) -> bool:
        if force:
            return False
        return path.exists() and path.stat().st_size > 0

    # ── single API page ────────────────────────────────────
    def _fetch_page(self, symbol: str, start_ms: int, end_ms: int) -> List[dict]:
        """Fetch one page (up to 500 rows) from Binance."""
        params = {
            "symbol": symbol,
            "period": self.period,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": self.page_limit,
        }
        for attempt in range(self.retries):
            try:
                resp = self.session.get(
                    self.base_url, params=params, timeout=self.timeout_sec
                )
                if resp.status_code == 404:
                    return []
                if resp.status_code == 429:
                    # rate-limited — back off
                    wait = min(2 ** (attempt + 2), 60)
                    print(f"  ⚠️  429 rate-limit, sleeping {wait}s …")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                if not isinstance(data, list):
                    return []
                return data
            except Exception:
                time.sleep(min(2**attempt, 10))
                continue
        return []

    # ── download one month with pagination ─────────────────
    def _download_month(
        self, symbol: str, year: int, month: int
    ) -> Optional[pd.DataFrame]:
        """Paginate through one calendar month, return DataFrame or None."""
        ms_start = _month_start_ms(year, month)
        ms_end = _month_end_ms(year, month) - 1  # inclusive
        all_rows: list[dict] = []
        cursor = ms_start

        while cursor < ms_end:
            page = self._fetch_page(symbol, cursor, ms_end)
            if not page:
                break
            all_rows.extend(page)
            # advance cursor past last timestamp
            last_ts = max(int(r["timestamp"]) for r in page)
            if last_ts <= cursor:
                break  # safety: no progress
            cursor = last_ts + 1
            time.sleep(self.sleep_sec)

        if not all_rows:
            return None

        df = pd.DataFrame(all_rows)
        df["datetime"] = pd.to_datetime(
            df["timestamp"].astype(int), unit="ms", utc=True
        )
        df["oi_contracts"] = pd.to_numeric(df["sumOpenInterest"], errors="coerce")
        df["oi_usd"] = pd.to_numeric(df["sumOpenInterestValue"], errors="coerce")
        df["_symbol"] = symbol

        out = df[["datetime", "_symbol", "oi_contracts", "oi_usd"]].copy()
        out = out.sort_values("datetime").drop_duplicates(
            subset=["datetime"], keep="last"
        )
        out = out.set_index("datetime")
        return out

    # ── main loop ──────────────────────────────────────────
    def run(
        self,
        *,
        symbols: List[str],
        start_year: int,
        start_month: int,
        end_year: int,
        end_month: int,
        force: bool = False,
    ) -> None:
        months = _month_list(start_year, start_month, end_year, end_month)
        total = len(symbols) * len(months)
        ok = 0
        skipped = 0
        empty = 0
        failed = 0
        n = 0

        for sym_raw in symbols:
            sym = _normalize_symbol(sym_raw)
            for y, m in months:
                n += 1
                ppath = self._parquet_path(sym, y, m)
                if self._should_skip(ppath, force=force):
                    skipped += 1
                    if self.progress_every and (
                        n % self.progress_every == 0 or n == total
                    ):
                        print(f"[{n}/{total}] ⏩ skip {sym} {y}-{m:02d} (cached)")
                    continue

                try:
                    df = self._download_month(sym, y, m)
                except Exception as e:
                    failed += 1
                    print(f"[{n}/{total}] ❌ error {sym} {y}-{m:02d}: {e}")
                    continue

                if df is None or df.empty:
                    empty += 1
                    if self.progress_every and (
                        n % self.progress_every == 0 or n == total
                    ):
                        print(f"[{n}/{total}] ⚪ empty {sym} {y}-{m:02d}")
                    continue

                ppath.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(ppath)
                ok += 1
                if self.progress_every and (n % self.progress_every == 0 or n == total):
                    print(f"[{n}/{total}] ✅ ok {sym} {y}-{m:02d}  rows={len(df)}")

        print(
            "✅ openInterest done: "
            f"ok={ok}, skipped={skipped}, empty={empty}, failed={failed}, total={total}"
        )


def hist_rows_to_frame(rows: List[dict], symbol: str) -> pd.DataFrame:
    """Normalize Binance ``openInterestHist`` rows to the downloader parquet schema."""
    if not rows:
        return pd.DataFrame(columns=["_symbol", "oi_contracts", "oi_usd"])
    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms", utc=True)
    df["oi_contracts"] = pd.to_numeric(df["sumOpenInterest"], errors="coerce")
    df["oi_usd"] = pd.to_numeric(df["sumOpenInterestValue"], errors="coerce")
    df["_symbol"] = _normalize_symbol(symbol)
    out = df[["datetime", "_symbol", "oi_contracts", "oi_usd"]].copy()
    out = out.sort_values("datetime").drop_duplicates(subset=["datetime"], keep="last")
    return out.set_index("datetime")


def fetch_recent_open_interest(
    symbol: str = "BTCUSDT",
    *,
    period: str = "1d",
    limit: int = 8,
    timeout_sec: float = 6,
) -> pd.DataFrame:
    """Last ``limit`` hist rows — same ``openInterestHist`` URL as the monthly downloader."""
    now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    lookback_ms = max(2, int(limit) + 2) * 24 * 3600 * 1000
    dl = OpenInterestDownloader(
        parquet_dir=Path(tempfile.gettempdir()),
        period=period,
        retries=2,
        timeout_sec=max(2, int(timeout_sec)),
        sleep_sec=0.0,
    )
    rows = dl._fetch_page(_normalize_symbol(symbol), now_ms - lookback_ms, now_ms)
    if rows and len(rows) > int(limit):
        rows = rows[-int(limit) :]
    return hist_rows_to_frame(rows, symbol)


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Download Binance futures OI history → Parquet"
    )
    ap.add_argument(
        "--symbols", nargs="+", required=True, help="Symbols like BTCUSDT ETHUSDT …"
    )
    ap.add_argument("--start-year", type=int, required=True)
    ap.add_argument("--start-month", type=int, required=True)
    ap.add_argument("--end-year", type=int, default=None)
    ap.add_argument("--end-month", type=int, default=None)
    ap.add_argument(
        "--period",
        default="5m",
        choices=list(VALID_PERIODS),
        help="OI aggregation period (default: 5m)",
    )
    ap.add_argument(
        "--parquet-dir",
        default="data/open_interest/parquet",
        help="Output directory for Parquet files",
    )
    ap.add_argument(
        "--sleep-sec",
        type=float,
        default=0.35,
        help="Sleep between API calls (respect rate limits)",
    )
    ap.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print progress every N tasks (0 disables)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if cached parquet exists",
    )
    args = ap.parse_args()

    now = datetime.now(tz=timezone.utc)
    end_year = args.end_year if args.end_year is not None else now.year
    end_month = args.end_month if args.end_month is not None else now.month

    dl = OpenInterestDownloader(
        parquet_dir=Path(args.parquet_dir),
        period=str(args.period),
        sleep_sec=float(args.sleep_sec),
        progress_every=int(args.progress_every),
    )
    dl.run(
        symbols=list(args.symbols),
        start_year=int(args.start_year),
        start_month=int(args.start_month),
        end_year=int(end_year),
        end_month=int(end_month),
        force=bool(args.force),
    )


if __name__ == "__main__":
    main()
