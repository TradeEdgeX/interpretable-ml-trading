#!/usr/bin/env python3
"""
Download Binance futures monthly fundingRate data (ZIP) and optionally convert to Parquet.

Binance Data Portal:
  UM (USDT-M / fapi):  https://data.binance.vision/?prefix=data/futures/um/monthly/fundingRate/<SYMBOL>/
  CM (COIN-M / dapi):  https://data.binance.vision/?prefix=data/futures/cm/monthly/fundingRate/<SYMBOL>_PERP/

dapi funding rates are a *different* series from fapi (see
docs/design/fapi_dapi_glossary_CN.md — "dapi funding 通常更低"); use
``--margin-kind cm`` to fetch the real COIN-M series instead of proxying with
the USDT-M one in ``scripts/event_backtest/funding/loader.py``.
"""

from __future__ import annotations

import argparse
import io
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import requests


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


def _normalize_symbol(symbol: str, margin_kind: str = "um") -> str:
    s = str(symbol or "").strip().upper().replace("-", "").replace("/", "")
    if not s:
        raise ValueError("Empty symbol")
    if margin_kind == "cm":
        if s.endswith("_PERP"):
            return s
        # Accept the usual USDT-style signal symbol (BTCUSDT) and map it to the
        # dapi inverse contract name (BTCUSD_PERP) via the shared symbol table —
        # see .cursor/rules/um-coinm-code-reuse.mdc (one mapping, both venues).
        from order_management.exchange.symbol_map import exec_symbol_for

        base = s[: -len("USDT")] if s.endswith("USDT") else s
        try:
            return exec_symbol_for(f"{base}USDT")
        except KeyError as exc:
            raise ValueError(
                f"no dapi mapping for {symbol!r} — pass the *_PERP symbol "
                f"directly or add it to config/order_management/dapi_symbol_overrides.yaml"
            ) from exc
    if not s.endswith("USDT"):
        s = f"{s}USDT"
    return s


def _should_skip_month(
    *,
    zip_path: Path,
    parquet_path: Path | None,
    force: bool,
) -> bool:
    """
    Skip rules:
    - if force: never skip
    - if parquet exists and non-empty: skip (preferred)
    - else if zip exists and non-empty: skip
    """
    if force:
        return False
    if (
        parquet_path is not None
        and parquet_path.exists()
        and parquet_path.stat().st_size > 0
    ):
        return True
    if zip_path.exists() and zip_path.stat().st_size > 1024:
        return True
    return False


_BASE_URL_BY_MARGIN_KIND = {
    "um": "https://data.binance.vision/data/futures/um/monthly/fundingRate",
    "cm": "https://data.binance.vision/data/futures/cm/monthly/fundingRate",
}


@dataclass
class FundingRateDownloader:
    data_dir: Path
    parquet_dir: Optional[Path]
    margin_kind: str = "um"  # "um" (USDT-M / fapi) or "cm" (COIN-M / dapi)
    base_url: Optional[str] = None
    retries: int = 4
    timeout_sec: int = 600
    sleep_sec: float = 0.2
    progress_every: int = 25

    def __post_init__(self) -> None:
        if self.margin_kind not in _BASE_URL_BY_MARGIN_KIND:
            raise ValueError(
                f"margin_kind must be one of {sorted(_BASE_URL_BY_MARGIN_KIND)}"
            )
        if self.base_url is None:
            self.base_url = _BASE_URL_BY_MARGIN_KIND[self.margin_kind]
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if self.parquet_dir is not None:
            self.parquet_dir.mkdir(parents=True, exist_ok=True)

        self.session = requests.Session()

    def _zip_name(self, symbol: str, year: int, month: int) -> str:
        return f"{symbol}-fundingRate-{year}-{month:02d}.zip"

    def _zip_url(self, symbol: str, year: int, month: int) -> str:
        return f"{self.base_url}/{symbol}/{self._zip_name(symbol, year, month)}"

    def _local_zip_path(self, symbol: str, year: int, month: int) -> Path:
        sym_dir = self.data_dir / symbol
        sym_dir.mkdir(parents=True, exist_ok=True)
        return sym_dir / self._zip_name(symbol, year, month)

    def _local_parquet_path(self, symbol: str, year: int, month: int) -> Path:
        assert self.parquet_dir is not None
        return self.parquet_dir / f"{symbol}_{year}-{month:02d}_funding_rate.parquet"

    def _download_one(self, symbol: str, year: int, month: int) -> bool:
        url = self._zip_url(symbol, year, month)
        out = self._local_zip_path(symbol, year, month)

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

    def _zip_to_parquet(self, zip_path: Path, *, symbol: str) -> pd.DataFrame:
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Prefer the first CSV inside
            members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
            if not members:
                raise ValueError(f"No CSV found inside {zip_path}")
            with zf.open(members[0], "r") as f:
                raw = f.read()
        df = pd.read_csv(io.BytesIO(raw))

        # Expected columns (as per Binance fundingRate export)
        # - calc_time: epoch ms
        # - funding_interval_hours
        # - last_funding_rate
        if "calc_time" not in df.columns or "last_funding_rate" not in df.columns:
            raise ValueError(
                f"Unexpected funding CSV schema in {zip_path}. Columns={list(df.columns)[:20]}"
            )

        df["datetime"] = pd.to_datetime(df["calc_time"], unit="ms", utc=True)
        df["funding_rate"] = pd.to_numeric(df["last_funding_rate"], errors="coerce")
        if "funding_interval_hours" in df.columns:
            df["funding_interval_hours"] = pd.to_numeric(
                df["funding_interval_hours"], errors="coerce"
            ).astype("Int64")
        df["_symbol"] = symbol

        out = df[
            ["datetime", "_symbol", "funding_rate"]
            + (
                ["funding_interval_hours"]
                if "funding_interval_hours" in df.columns
                else []
            )
        ].copy()
        out = out.sort_values("datetime").dropna(subset=["datetime"])
        out = out.set_index("datetime")
        return out

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
        missing = 0
        failed = 0
        n = 0
        for sym_raw in symbols:
            sym = _normalize_symbol(sym_raw, margin_kind=self.margin_kind)
            for y, m in months:
                n += 1
                zpath = self._local_zip_path(sym, y, m)
                ppath = (
                    self._local_parquet_path(sym, y, m)
                    if self.parquet_dir is not None
                    else None
                )
                if _should_skip_month(zip_path=zpath, parquet_path=ppath, force=force):
                    skipped += 1
                    if self.progress_every and (
                        n % self.progress_every == 0 or n == total
                    ):
                        print(f"[{n}/{total}] ⏩ skip {sym} {y}-{m:02d} (cached)")
                    continue

                got = self._download_one(sym, y, m)
                if not got:
                    # Most likely 404 (Binance doesn't have that month)
                    missing += 1
                    if self.progress_every and (
                        n % self.progress_every == 0 or n == total
                    ):
                        print(f"[{n}/{total}] ⚪ missing {sym} {y}-{m:02d} (404)")
                    continue

                if self.parquet_dir is not None:
                    try:
                        dfp = self._zip_to_parquet(zpath, symbol=sym)
                        ppath = self._local_parquet_path(sym, y, m)
                        ppath.parent.mkdir(parents=True, exist_ok=True)
                        dfp.to_parquet(ppath)
                    except Exception as e:
                        failed += 1
                        print(f"[{n}/{total}] ❌ convert failed {sym} {y}-{m:02d}: {e}")
                        time.sleep(self.sleep_sec)
                        continue

                ok += 1
                if self.progress_every and (n % self.progress_every == 0 or n == total):
                    print(f"[{n}/{total}] ✅ ok {sym} {y}-{m:02d}")
                time.sleep(self.sleep_sec)

        print(
            "✅ fundingRate done: "
            f"ok={ok}, skipped={skipped}, missing_404={missing}, failed={failed}, total={total}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--symbols", nargs="+", required=True, help="Symbols like BTCUSDT ETHUSDT ..."
    )
    ap.add_argument("--start-year", type=int, required=True)
    ap.add_argument("--start-month", type=int, required=True)
    ap.add_argument("--end-year", type=int, default=None)
    ap.add_argument("--end-month", type=int, default=None)
    ap.add_argument(
        "--data-dir",
        default="data/funding_rate/zip",
        help="Output directory for ZIP files",
    )
    ap.add_argument(
        "--parquet-dir",
        default="data/funding_rate/parquet",
        help="Output directory for Parquet (set empty to disable)",
    )
    ap.add_argument(
        "--sleep-sec",
        type=float,
        default=0.2,
        help="Sleep between requests (avoid rate limits)",
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
        help="Force re-download even if cached ZIP/parquet exists",
    )
    ap.add_argument(
        "--margin-kind",
        choices=["um", "cm"],
        default="um",
        help="um=USDT-M/fapi funding (default); cm=COIN-M/dapi funding "
        "(auto-maps BTCUSDT-style symbols to BTCUSD_PERP)",
    )
    args = ap.parse_args()

    now = datetime.utcnow()
    end_year = args.end_year if args.end_year is not None else now.year
    end_month = args.end_month if args.end_month is not None else now.month

    parquet_dir = str(args.parquet_dir).strip()
    dl = FundingRateDownloader(
        data_dir=Path(args.data_dir),
        parquet_dir=Path(parquet_dir) if parquet_dir else None,
        margin_kind=str(args.margin_kind),
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
