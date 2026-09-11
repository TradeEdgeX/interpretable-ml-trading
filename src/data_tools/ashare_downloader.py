"""Minimal A-share / index daily OHLCV download via AKShare.

Writes ``data/ashare/daily/{symbol}.parquet`` for court examples.
No proxy monkey-patch, no full-market pool, no Lab auxiliary panels.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_CN_TO_EN = {
    "日期": "date",
    "股票代码": "symbol",
    "开盘": "open",
    "最高": "high",
    "最低": "low",
    "收盘": "close",
    "成交量": "volume",
    "成交额": "amount",
    "振幅": "amplitude",
    "涨跌幅": "pct_change",
    "涨跌额": "change",
    "换手率": "turnover",
}

_STD_COLS = [
    "date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "turnover",
]


def normalize_ashare_symbol(raw: str) -> str:
    """Normalize to ``000300.SH`` / ``000001.SZ`` / six-digit stock code."""
    s = str(raw or "").strip().upper()
    if not s:
        return ""
    if s.startswith("SH") and s[2:].isdigit():
        return f"{s[2:].zfill(6)}.SH"
    if s.startswith("SZ") and s[2:].isdigit():
        return f"{s[2:].zfill(6)}.SZ"
    if "." in s:
        code, exch = s.split(".", 1)
        digits = "".join(ch for ch in code if ch.isdigit()).zfill(6)[-6:]
        exch = exch.strip().upper()
        if exch in ("SH", "SS"):
            return f"{digits}.SH"
        if exch in ("SZ",):
            return f"{digits}.SZ"
        return f"{digits}.{exch}"
    digits = "".join(ch for ch in s if ch.isdigit())
    if not digits:
        return ""
    return digits.zfill(6)[-6:]


def _is_index(symbol: str) -> bool:
    s = normalize_ashare_symbol(symbol)
    if s.endswith(".SH") or s.endswith(".SZ"):
        code = s.split(".", 1)[0]
        return code.startswith(("000", "399"))
    return False


def _ak_index_symbol(symbol: str) -> str:
    s = normalize_ashare_symbol(symbol)
    code, exch = s.split(".", 1)
    prefix = "sh" if exch == "SH" else "sz"
    return f"{prefix}{code}"


def _normalize_ohlcv(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    rename_map = {
        k: v for k, v in _CN_TO_EN.items() if k in df.columns and v not in df.columns
    }
    df = df.rename(columns=rename_map)
    if "symbol" not in df.columns:
        df["symbol"] = symbol
    existing = [c for c in _STD_COLS if c in df.columns]
    df = df[existing].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _download_one(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    import akshare as ak

    sym = normalize_ashare_symbol(symbol)
    if _is_index(sym) and "." in sym:
        raw = ak.stock_zh_index_daily(symbol=_ak_index_symbol(sym))
        if raw is None or raw.empty:
            return pd.DataFrame()
        raw = raw.copy()
        raw["date"] = pd.to_datetime(raw["date"])
        start_ts = pd.to_datetime(start_date)
        end_ts = pd.to_datetime(end_date)
        raw = raw[(raw["date"] >= start_ts) & (raw["date"] <= end_ts)]
        return _normalize_ohlcv(raw, sym)

    code = sym.split(".", 1)[0] if "." in sym else sym
    raw = ak.stock_zh_a_hist(
        symbol=code,
        period="daily",
        start_date=start_date,
        end_date=end_date,
        adjust="qfq",
    )
    return _normalize_ohlcv(raw, sym)


def _download_with_retry(
    symbol: str,
    start_date: str,
    end_date: str,
    max_retries: int = 3,
    base_sleep: float = 0.5,
) -> pd.DataFrame:
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            return _download_one(symbol, start_date, end_date)
        except Exception as e:  # noqa: BLE001 — network / upstream flakiness
            last_exc = e
            wait = base_sleep * (2**attempt)
            logger.warning(
                "Retry %d/%d for %s after %.1fs: %s",
                attempt + 1,
                max_retries,
                symbol,
                wait,
                str(e)[:120],
            )
            time.sleep(wait)
    raise last_exc  # type: ignore[misc]


def _sina_symbol(symbol: str) -> str:
    code = normalize_ashare_symbol(symbol)
    digits = code.split(".", 1)[0] if "." in code else code
    if digits.startswith(("6", "5", "9")):
        return f"sh{digits}"
    if digits.startswith(("4", "8")):
        return f"bj{digits}"
    return f"sz{digits}"


def download_one_sina(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Sina daily qfq. Turnover is stored as a fraction; convert to percent."""
    import akshare as ak

    start = start_date.replace("-", "")[:8]
    end = end_date.replace("-", "")[:8]
    raw = ak.stock_zh_a_daily(
        symbol=_sina_symbol(symbol),
        start_date=start,
        end_date=end,
        adjust="qfq",
    )
    if raw is None or raw.empty:
        return pd.DataFrame()
    out = raw.copy()
    out = out.rename(columns={c: str(c).strip() for c in out.columns})
    if "turnover" in out.columns:
        out["turnover"] = pd.to_numeric(out["turnover"], errors="coerce") * 100.0
    digits = normalize_ashare_symbol(symbol)
    digits = digits.split(".", 1)[0] if "." in digits else digits
    return _normalize_ohlcv(out, digits)


def download_ashare_daily(
    symbols: list[str],
    *,
    output_dir: str | Path = "data/ashare/daily",
    years: int = 5,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    sleep_between: float = 0.3,
    resume: bool = True,
) -> dict:
    """Download daily OHLCV for a small explicit symbol list.

    Returns counts: total / success / skipped / failed / elapsed_sec.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    end = date.today() if not end_date else date.fromisoformat(str(end_date)[:10])
    if start_date:
        start = date.fromisoformat(str(start_date)[:10])
    else:
        start = end - timedelta(days=int(years) * 365)
    start_s = start.strftime("%Y%m%d")
    end_s = end.strftime("%Y%m%d")

    total = len(symbols)
    success = skipped = failed = 0
    failed_symbols: list[str] = []
    t0 = time.monotonic()

    for i, raw in enumerate(symbols):
        sym = normalize_ashare_symbol(raw)
        if not sym:
            failed += 1
            failed_symbols.append(str(raw))
            continue
        out_file = output_dir / f"{sym}.parquet"
        if resume and out_file.exists():
            logger.info("[%d/%d] skip existing %s", i + 1, total, sym)
            skipped += 1
            continue
        try:
            df = _download_with_retry(sym, start_s, end_s)
            if df.empty:
                failed += 1
                failed_symbols.append(sym)
                logger.warning("[%d/%d] empty %s", i + 1, total, sym)
            else:
                df.to_parquet(out_file, index=False)
                success += 1
                logger.info("[%d/%d] wrote %s (%d rows)", i + 1, total, out_file, len(df))
        except Exception as e:  # noqa: BLE001
            failed += 1
            failed_symbols.append(sym)
            logger.error("[%d/%d] failed %s: %s", i + 1, total, sym, e)
        time.sleep(sleep_between)

    return {
        "total": total,
        "success": success,
        "skipped": skipped,
        "failed": failed,
        "failed_symbols": failed_symbols,
        "elapsed_sec": round(time.monotonic() - t0, 1),
        "output_dir": str(output_dir),
    }


def _file_covers_start(path: Path, start: str, *, min_rows: int = 20) -> bool:
    """Resume if the parquet already has a usable daily tape.

    Names listed after ``start`` correctly begin at IPO — do not treat that
    as incomplete history.
    """
    del start
    if not path.is_file():
        return False
    try:
        df = pd.read_parquet(path)
    except Exception:  # noqa: BLE001
        return False
    if df.empty or len(df) < min_rows or "close" not in df.columns:
        return False
    if "turnover" not in df.columns or "amount" not in df.columns:
        return False
    return True


def download_ashare_universe_sina(
    symbols: Iterable[str],
    *,
    output_dir: str | Path = "data/ashare/daily",
    start_date: str = "2016-01-01",
    end_date: Optional[str] = None,
    resume: bool = True,
    workers: int = 4,
    sleep_between: float = 0.05,
) -> dict:
    """Thread-pooled Sina qfq download. Resume skips files that already reach start."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    end = (end_date or date.today().isoformat())[:10]
    start = start_date[:10]
    todo: list[str] = []
    skipped = 0
    for raw in symbols:
        sym = normalize_ashare_symbol(raw)
        if "." in sym:
            sym = sym.split(".", 1)[0]
        if not sym:
            continue
        dest = output_dir / f"{sym}.parquet"
        if resume and _file_covers_start(dest, start):
            skipped += 1
            continue
        todo.append(sym)

    success = failed = 0
    failed_symbols: list[str] = []
    t0 = time.monotonic()
    if not todo:
        return {
            "total": skipped,
            "success": 0,
            "skipped": skipped,
            "failed": 0,
            "failed_symbols": [],
            "elapsed_sec": 0.0,
            "output_dir": str(output_dir),
        }

    def _one(sym: str) -> tuple[str, str, int]:
        dest = output_dir / f"{sym}.parquet"
        try:
            df = download_one_sina(sym, start.replace("-", ""), end.replace("-", ""))
            time.sleep(sleep_between)
            if df.empty:
                return sym, "empty", 0
            df.to_parquet(dest, index=False)
            return sym, "ok", int(len(df))
        except Exception as exc:  # noqa: BLE001
            return sym, f"err:{exc}"[:80], 0

    logger.info("sina qfq: %d to fetch, %d skipped, workers=%d", len(todo), skipped, workers)
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futs = {pool.submit(_one, s): s for s in todo}
        done = 0
        for fut in as_completed(futs):
            done += 1
            try:
                sym, status, rows = fut.result()
            except Exception as exc:  # noqa: BLE001
                failed += 1
                failed_symbols.append(futs[fut])
                logger.error("[%d/%d] %s %s", done, len(todo), futs[fut], exc)
                continue
            if status == "ok":
                success += 1
                if done % 50 == 0 or done == len(todo):
                    logger.info("[%d/%d] %s rows=%d", done, len(todo), sym, rows)
            else:
                failed += 1
                failed_symbols.append(sym)
                if done % 50 == 0:
                    logger.warning("[%d/%d] %s %s", done, len(todo), sym, status)

    return {
        "total": skipped + len(todo),
        "success": success,
        "skipped": skipped,
        "failed": failed,
        "failed_symbols": failed_symbols,
        "elapsed_sec": round(time.monotonic() - t0, 1),
        "output_dir": str(output_dir),
    }


def load_ashare_daily_bars(
    data_path: str | Path,
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Load one A-share daily parquet as OHLCV with UTC DatetimeIndex (bar close)."""
    root = Path(data_path)
    sym = normalize_ashare_symbol(symbol)
    candidates = [
        root / f"{sym}.parquet",
        root / f"{sym.split('.', 1)[0]}.parquet",
        root / "daily" / f"{sym}.parquet",
        root / "daily" / f"{sym.split('.', 1)[0]}.parquet",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return pd.DataFrame()

    df = pd.read_parquet(path)
    if "date" in df.columns:
        idx = pd.DatetimeIndex(pd.to_datetime(df["date"]))
    elif isinstance(df.index, pd.DatetimeIndex):
        idx = df.index
    else:
        raise ValueError(f"ashare daily parquet missing date column: {path}")

    # Session close ~ 15:00 Asia/Shanghai → UTC for court alignment.
    if getattr(idx, "tz", None) is None:
        idx = (idx + pd.Timedelta(hours=15)).tz_localize("Asia/Shanghai").tz_convert(
            "UTC"
        )
    else:
        idx = idx.tz_convert("UTC")

    out = pd.DataFrame(
        {
            "open": pd.to_numeric(df["open"], errors="coerce").to_numpy(),
            "high": pd.to_numeric(df["high"], errors="coerce").to_numpy(),
            "low": pd.to_numeric(df["low"], errors="coerce").to_numpy(),
            "close": pd.to_numeric(df["close"], errors="coerce").to_numpy(),
            "volume": pd.to_numeric(df["volume"], errors="coerce")
            .fillna(0.0)
            .to_numpy(),
        },
        index=idx,
    )
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    if start_date:
        out = out[out.index >= pd.to_datetime(start_date, utc=True)]
    if end_date:
        # inclusive calendar day
        end_ts = pd.to_datetime(end_date, utc=True) + pd.Timedelta(days=1)
        out = out[out.index < end_ts]
    out["timestamp"] = out.index
    out["_symbol"] = sym
    out["symbol"] = sym
    return out


def is_ashare_daily_path(data_path: str | Path, symbol: str) -> bool:
    """True when ``data_path`` holds ashare daily files for ``symbol``."""
    root = Path(data_path)
    sym = normalize_ashare_symbol(symbol)
    for p in (
        root / f"{sym}.parquet",
        root / f"{sym.split('.', 1)[0]}.parquet",
        root / "daily" / f"{sym}.parquet",
        root / "daily" / f"{sym.split('.', 1)[0]}.parquet",
    ):
        if p.exists():
            return True
    # Directory named ashare/daily with any parquet
    if root.name == "daily" and (root.parent.name == "ashare") and list(root.glob("*.parquet")):
        return True
    return False
