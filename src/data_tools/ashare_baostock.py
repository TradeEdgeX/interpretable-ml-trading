"""Baostock helpers for A-share daily qfq + point-in-time universe.

Court-only: no CMS, no proxy patch. Year-chunked k-line (the API truncates
long windows). Optional dependency: ``baostock``.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

logger = logging.getLogger(__name__)

STOCK_TYPE = "1"
_K_FIELDS = "date,open,high,low,close,volume,amount,turn,pctChg"
_STD_COLS = [
    "date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "amplitude",
    "pct_change",
    "change",
    "turnover",
]


def baostock_code_to_symbol(code: object) -> str:
    text = str(code or "").strip()
    if "." in text:
        text = text.split(".")[-1]
    return text.zfill(6) if text.isdigit() else text


def code_to_baostock(code: str) -> str:
    digits = baostock_code_to_symbol(code)
    if digits.startswith(("6", "5", "9")):
        return f"sh.{digits}"
    if digits.startswith(("0", "3", "2")):
        return f"sz.{digits}"
    if digits.startswith(("4", "8")):
        return f"bj.{digits}"
    return f"sh.{digits}"


def is_shsz_a_share(symbol: str) -> bool:
    s = baostock_code_to_symbol(symbol)
    return bool(s) and s[0] in "036" and len(s) == 6 and s.isdigit()


def quarter_end_dates(start: str, end: str) -> list[date]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    out: list[date] = []
    year, q = start_ts.year, (start_ts.month - 1) // 3 + 1
    while True:
        month = q * 3
        day = (pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)).date()
        if day >= start_ts.date() and day <= end_ts.date():
            out.append(day)
        if day > end_ts.date():
            break
        q += 1
        if q > 4:
            q = 1
            year += 1
    return out


def _year_windows(start_date: str, end_date: str) -> list[tuple[str, str]]:
    start = date.fromisoformat(start_date[:10])
    end = date.fromisoformat(end_date[:10])
    out: list[tuple[str, str]] = []
    cur = start
    while cur <= end:
        nxt = date(cur.year, 12, 31)
        if nxt > end:
            nxt = end
        out.append((cur.isoformat(), nxt.isoformat()))
        cur = date(cur.year + 1, 1, 1)
    return out


def _rows_from_rs(rs) -> pd.DataFrame:
    fields = list(rs.fields)
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        return pd.DataFrame(columns=fields)
    return pd.DataFrame(rows, columns=fields)


def _normalize_kline(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=_STD_COLS)
    out = df.copy()
    out["symbol"] = symbol
    out = out.rename(columns={"pctChg": "pct_change", "turn": "turnover"})
    out["date"] = pd.to_datetime(out["date"])
    for col in (
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "turnover",
        "pct_change",
    ):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    close = out["close"].replace(0, pd.NA)
    out["amplitude"] = ((out["high"] - out["low"]) / close * 100).round(2)
    out["change"] = out["close"].diff().round(2)
    existing = [c for c in _STD_COLS if c in out.columns]
    out = out[existing].dropna(subset=["open", "close"])
    out = out.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    return out


def _login():
    import baostock as bs

    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"baostock login failed: {lg.error_msg}")
    return bs


def fetch_stock_basic(out: str | Path) -> pd.DataFrame:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    bs = _login()
    try:
        rs = bs.query_stock_basic()
        if rs.error_code != "0":
            raise RuntimeError(f"query_stock_basic: {rs.error_msg}")
        df = _rows_from_rs(rs)
    finally:
        bs.logout()
    if df.empty:
        raise RuntimeError("query_stock_basic returned 0 rows")
    df = df.copy()
    df["symbol"] = df["code"].map(baostock_code_to_symbol)
    for col in ("ipoDate", "outDate"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    df.to_parquet(out, index=False)
    return df


def fetch_stock_industry(out: str | Path) -> pd.DataFrame:
    """Latest baostock industry map (not point-in-time)."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    bs = _login()
    try:
        rs = bs.query_stock_industry()
        if rs.error_code != "0":
            raise RuntimeError(f"query_stock_industry: {rs.error_msg}")
        df = _rows_from_rs(rs)
    finally:
        bs.logout()
    if df.empty:
        raise RuntimeError("query_stock_industry returned 0 rows")
    df = df.copy()
    df["symbol"] = df["code"].map(baostock_code_to_symbol)
    if "industry" not in df.columns:
        raise RuntimeError("query_stock_industry missing industry")
    df["industry"] = df["industry"].astype(str).str.strip()
    df = df[df["industry"].ne("") & df["industry"].ne("nan")]
    df.to_parquet(out, index=False)
    return df


def listed_a_shares(basic: pd.DataFrame) -> pd.DataFrame:
    df = basic
    if "type" in df.columns:
        df = df[df["type"].astype(str) == STOCK_TYPE]
    return df.reset_index(drop=True)


def universe_symbols(
    basic: pd.DataFrame,
    *,
    include: str = "listed,delisted",
    delist_since: str = "2014-06-01",
    shsz_only: bool = True,
) -> list[str]:
    """Return 6-digit A-share codes for download."""
    stocks = listed_a_shares(basic).copy()
    stocks["symbol"] = stocks["symbol"].map(baostock_code_to_symbol)
    if shsz_only:
        stocks = stocks[stocks["symbol"].map(is_shsz_a_share)]
    want = {p.strip().lower() for p in include.split(",") if p.strip()}
    parts: list[pd.DataFrame] = []
    if "listed" in want and "status" in stocks.columns:
        parts.append(stocks[stocks["status"].astype(str) == "1"])
    if "delisted" in want and "status" in stocks.columns:
        dead = stocks[stocks["status"].astype(str) == "0"]
        if "outDate" in dead.columns:
            out = pd.to_datetime(dead["outDate"], errors="coerce")
            dead = dead[out.ge(pd.Timestamp(delist_since)) | out.isna()]
        parts.append(dead)
    if not parts:
        parts = [stocks]
    out = pd.concat(parts, ignore_index=True).drop_duplicates("symbol")
    return sorted(out["symbol"].tolist())


def fetch_universe_day(bs, day: date) -> tuple[date, pd.DataFrame]:
    for i in range(11):
        cand = day - timedelta(days=i)
        rs = bs.query_all_stock(day=cand.isoformat())
        if rs.error_code != "0":
            continue
        df = _rows_from_rs(rs)
        if df.empty:
            continue
        out = df.copy()
        out["symbol"] = out["code"].map(baostock_code_to_symbol)
        out["asof"] = pd.Timestamp(cand)
        out["requested"] = pd.Timestamp(day)
        return cand, out
    return day, pd.DataFrame()


def fetch_universe_quarters(
    out_dir: str | Path,
    start: str,
    end: str,
    *,
    sleep: float = 0.15,
) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    days = quarter_end_dates(start, end)
    written: list[Path] = []
    pending = [d for d in days if not (out_dir / f"{d.isoformat()}.parquet").is_file()]
    if not pending:
        return [out_dir / f"{d.isoformat()}.parquet" for d in days]
    bs = _login()
    try:
        for day in days:
            dest = out_dir / f"{day.isoformat()}.parquet"
            if dest.is_file():
                written.append(dest)
                continue
            _used, df = fetch_universe_day(bs, day)
            if df.empty:
                logger.warning("universe %s empty", day)
                continue
            df.to_parquet(dest, index=False)
            written.append(dest)
            time.sleep(sleep)
    finally:
        bs.logout()
    return written


def _download_one_raw(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    import baostock as bs

    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(lg.error_msg)
    try:
        parts: list[pd.DataFrame] = []
        bs_code = code_to_baostock(symbol)
        for w0, w1 in _year_windows(start_date, end_date):
            rs = bs.query_history_k_data_plus(
                bs_code,
                _K_FIELDS,
                start_date=w0,
                end_date=w1,
                frequency="d",
                adjustflag="2",
            )
            if rs.error_code != "0":
                continue
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
            if not rows:
                continue
            raw = pd.DataFrame(rows, columns=_K_FIELDS.split(","))
            parts.append(raw)
        if not parts:
            return pd.DataFrame()
        raw = pd.concat(parts, ignore_index=True)
        return _normalize_kline(raw, baostock_code_to_symbol(symbol))
    finally:
        bs.logout()


def _worker_download(payload: dict) -> tuple[str, str, int]:
    """Process entry: (symbol, status, rows)."""
    symbol = payload["symbol"]
    dest = Path(payload["dest"])
    try:
        df = _download_one_raw(symbol, payload["start"], payload["end"])
        if df.empty:
            return symbol, "empty", 0
        dest.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dest, index=False)
        return symbol, "ok", int(len(df))
    except Exception as exc:  # noqa: BLE001
        return symbol, f"err:{exc}"[:80], 0


def _history_ok(path: Path, start: str, *, min_rows: int = 60) -> bool:
    if not path.is_file():
        return False
    try:
        df = pd.read_parquet(path, columns=["date"])
    except Exception:  # noqa: BLE001
        return False
    if df.empty or len(df) < min_rows:
        return False
    first = pd.to_datetime(df["date"]).min()
    want = pd.Timestamp(start)
    # Listed after ``start``: any file is enough. Older names must reach start+180d.
    return bool(first <= want + pd.Timedelta(days=180))


def download_ashare_universe(
    symbols: Iterable[str],
    *,
    output_dir: str | Path = "data/ashare/daily",
    start_date: str = "2016-01-01",
    end_date: Optional[str] = None,
    resume: bool = True,
    workers: int = 3,
    refresh_short: bool = True,
) -> dict:
    """Download qfq daily bars for an explicit symbol list (resume-safe)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    end = end_date or date.today().isoformat()
    start = start_date[:10]
    end = end[:10]
    todo: list[str] = []
    skipped = 0
    for raw in symbols:
        sym = baostock_code_to_symbol(raw)
        if not sym:
            continue
        dest = output_dir / f"{sym}.parquet"
        if resume and dest.exists() and (not refresh_short or _history_ok(dest, start)):
            skipped += 1
            continue
        todo.append(sym)

    success = failed = empty = 0
    failed_symbols: list[str] = []
    t0 = time.monotonic()
    if not todo:
        return {
            "total": skipped,
            "success": 0,
            "skipped": skipped,
            "failed": 0,
            "empty": 0,
            "failed_symbols": [],
            "elapsed_sec": 0.0,
            "output_dir": str(output_dir),
        }

    payloads = [
        {
            "symbol": sym,
            "dest": str(output_dir / f"{sym}.parquet"),
            "start": start,
            "end": end,
        }
        for sym in todo
    ]
    workers = max(1, int(workers))
    logger.info(
        "baostock qfq: %d to fetch, %d skipped, workers=%d, %s → %s",
        len(todo),
        skipped,
        workers,
        start,
        end,
    )
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_worker_download, p): p["symbol"] for p in payloads}
        done = 0
        for fut in as_completed(futs):
            done += 1
            try:
                sym, status, rows = fut.result()
            except Exception as exc:  # noqa: BLE001
                failed += 1
                failed_symbols.append(futs[fut])
                logger.error("[%d/%d] crash %s: %s", done, len(todo), futs[fut], exc)
                continue
            if status == "ok":
                success += 1
                if done % 50 == 0 or done == len(todo):
                    logger.info("[%d/%d] %s rows=%d", done, len(todo), sym, rows)
            elif status == "empty":
                empty += 1
                failed_symbols.append(sym)
            else:
                failed += 1
                failed_symbols.append(sym)
                logger.warning("[%d/%d] %s %s", done, len(todo), sym, status)

    return {
        "total": skipped + len(todo),
        "success": success,
        "skipped": skipped,
        "failed": failed + empty,
        "empty": empty,
        "failed_symbols": failed_symbols,
        "elapsed_sec": round(time.monotonic() - t0, 1),
        "output_dir": str(output_dir),
    }
