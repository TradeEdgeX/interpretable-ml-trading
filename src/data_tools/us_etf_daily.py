"""US ETF daily OHLCV for the SPY/QQQ court.

Writes ``data/eq/us/daily/{SYMBOL}.parquet``.
Yahoo Chart API first; Stooq daily CSV if Yahoo is blocked.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
_STD_COLS = ["date", "symbol", "open", "high", "low", "close", "adjclose", "volume"]


def _http_get(url: str, *, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _parse_nasdaq_number(raw) -> float:
    if raw is None or raw == "":
        return float("nan")
    s = str(raw).replace("$", "").replace(",", "").strip()
    if s in {"", "--", "N/A"}:
        return float("nan")
    return float(s)


def _nasdaq_historical(symbol: str, start: date, end: date) -> pd.DataFrame:
    """Nasdaq public quote history. About ten years; split-adjusted close, not total-return."""
    url = (
        "https://api.nasdaq.com/api/quote/"
        f"{symbol}/historical?assetclass=etf"
        f"&fromdate={start.isoformat()}&todate={end.isoformat()}&limit=9999"
    )
    raw = json.loads(_http_get(url).decode("utf-8"))
    rows = ((raw.get("data") or {}).get("tradesTable") or {}).get("rows") or []
    if not rows:
        raise ValueError(f"nasdaq empty history for {symbol}")
    recs = []
    for row in rows:
        recs.append(
            {
                "date": pd.to_datetime(row.get("date"), format="%m/%d/%Y"),
                "open": _parse_nasdaq_number(row.get("open")),
                "high": _parse_nasdaq_number(row.get("high")),
                "low": _parse_nasdaq_number(row.get("low")),
                "close": _parse_nasdaq_number(row.get("close")),
                "volume": _parse_nasdaq_number(row.get("volume")),
            }
        )
    df = pd.DataFrame(recs)
    df["adjclose"] = df["close"]
    return df


def _yahoo_chart(symbol: str, start: date, end: date) -> pd.DataFrame:
    p1 = int(datetime(start.year, start.month, start.day, tzinfo=timezone.utc).timestamp())
    p2 = int(datetime(end.year, end.month, end.day, 23, 59, tzinfo=timezone.utc).timestamp())
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?period1={p1}&period2={p2}&interval=1d&events=div%2Csplit"
    )
    raw = json.loads(_http_get(url).decode("utf-8"))
    result = ((raw.get("chart") or {}).get("result") or [None])[0]
    if not result:
        raise ValueError(f"yahoo empty chart for {symbol}")
    ts = result.get("timestamp") or []
    quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
    adj = ((result.get("indicators") or {}).get("adjclose") or [{}])[0]
    adjclose = adj.get("adjclose") or quote.get("close")
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(ts, unit="s", utc=True).tz_localize(None).normalize(),
            "open": quote.get("open"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "close": quote.get("close"),
            "adjclose": adjclose,
            "volume": quote.get("volume"),
        }
    )
    return df


def _stooq_daily(symbol: str) -> pd.DataFrame:
    slug = f"{symbol.lower()}.us"
    url = f"https://stooq.com/q/d/l/?s={slug}&i=d"
    raw = _http_get(url).decode("utf-8")
    from io import StringIO

    df = pd.read_csv(StringIO(raw))
    df = df.rename(columns={c: str(c).strip().lower() for c in df.columns})
    if "date" not in df.columns or "close" not in df.columns:
        raise ValueError(f"stooq unexpected columns for {symbol}: {list(df.columns)}")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    if "adjclose" not in df.columns:
        df["adjclose"] = df["close"]
    return df


def fetch_us_etf_daily(
    symbol: str,
    *,
    start: date,
    end: date,
) -> pd.DataFrame:
    """Adjusted daily bars. ``adjclose`` is the return series."""
    sym = str(symbol or "").strip().upper()
    last_err: Optional[Exception] = None
    for loader in (
        lambda: _nasdaq_historical(sym, start, end),
        lambda: _yahoo_chart(sym, start, end),
        lambda: _stooq_daily(sym),
    ):
        try:
            df = loader()
        except (
            urllib.error.URLError,
            TimeoutError,
            ValueError,
            json.JSONDecodeError,
            KeyError,
        ) as exc:
            last_err = exc
            logger.warning("us etf download %s failed: %s", sym, exc)
            continue
        df = df.copy()
        df["symbol"] = sym
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        for col in ("open", "high", "low", "close", "adjclose", "volume"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "adjclose" not in df.columns:
            df["adjclose"] = df["close"]
        df = df.dropna(subset=["date", "adjclose"])
        df = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]
        df = df.sort_values("date").drop_duplicates("date")
        if df.empty:
            last_err = ValueError(f"no rows for {sym}")
            continue
        return df[_STD_COLS]
    raise RuntimeError(f"could not download {sym}: {last_err}")


def download_us_etf_daily(
    symbols: Iterable[str],
    *,
    output_dir: str | Path = "data/eq/us/daily",
    start_date: str = "2013-01-01",
    end_date: Optional[str] = None,
    resume: bool = True,
) -> dict:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date) if end_date else date.today()
    stats = {
        "total": 0,
        "success": 0,
        "skipped": 0,
        "failed": 0,
        "failed_symbols": [],
        "output_dir": str(out_dir),
        "elapsed_sec": 0,
    }
    t0 = time.time()
    for raw in symbols:
        sym = str(raw).strip().upper()
        if not sym:
            continue
        stats["total"] += 1
        dest = out_dir / f"{sym}.parquet"
        if resume and dest.is_file():
            stats["skipped"] += 1
            continue
        try:
            df = fetch_us_etf_daily(sym, start=start, end=end)
            df.to_parquet(dest, index=False)
            stats["success"] += 1
        except Exception as exc:  # noqa: BLE001 — keep other symbols going
            logger.exception("download %s", sym)
            stats["failed"] += 1
            stats["failed_symbols"].append(f"{sym}:{exc}")
    stats["elapsed_sec"] = round(time.time() - t0, 1)
    return stats


def load_us_etf_daily(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df.sort_values("date").reset_index(drop=True)
