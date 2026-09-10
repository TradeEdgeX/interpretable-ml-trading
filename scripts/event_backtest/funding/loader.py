"""Load historical funding rate parquet for backtest overlay."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# scripts/event_backtest/funding/loader.py -> repo root is 3 levels up (parents[2]
# pointed at scripts/, so the default dir never existed and funding silently no-op'd).
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_PARQUET_DIR = _REPO_ROOT / "data" / "funding_rate" / "parquet"


def _read_funding_glob(root: Path, glob_sym: str) -> Optional[pd.Series]:
    files = sorted(root.glob(f"{glob_sym}_*funding_rate.parquet"))
    if not files:
        return None
    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames).sort_index()
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[~df.index.duplicated(keep="last")]
    if "funding_rate" not in df.columns:
        return None
    return pd.to_numeric(df["funding_rate"], errors="coerce").dropna()


def load_funding_rate_series(
    symbol: str,
    *,
    parquet_dir: Optional[Path] = None,
    margin_mode: str = "usd_m",
) -> Optional[pd.Series]:
    """Return funding_rate series indexed by UTC timestamps, or None if missing.

    For ``margin_mode="coin_m"`` this prefers the *real* dapi (COIN-M) funding
    series — COIN-M funding parquet under the dapi symbol name (e.g. BTCUSD_PERP).
    dapi funding is a genuinely different series from fapi (see
    docs/design/fapi_dapi_glossary_CN.md — "通常更低"); if it hasn't been
    downloaded yet we fall back to the USDT-M series as a documented
    approximation and log a warning (never a silent substitution).
    """
    sym = str(symbol).upper()
    root = Path(parquet_dir) if parquet_dir else _DEFAULT_PARQUET_DIR
    if not root.is_dir():
        return None

    if str(margin_mode).lower() == "coin_m":
        dapi_sym = sym
        try:
            from order_management.exchange.symbol_map import exec_symbol_for

            dapi_sym = exec_symbol_for(sym)
        except KeyError:
            pass
        if dapi_sym != sym:
            native = _read_funding_glob(root, dapi_sym)
            if native is not None and not native.empty:
                return native
            logger.warning(
                "no native dapi funding parquet for %s (%s) — falling back to "
                "fapi %s funding as a proxy (public court is usd_m)",
                sym,
                dapi_sym,
                sym,
            )

    return _read_funding_glob(root, sym)


def load_funding_rates_for_symbols(
    symbols: list[str],
    *,
    parquet_dir: Optional[Path] = None,
    margin_mode: str = "usd_m",
) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    for sym in symbols:
        fr = load_funding_rate_series(
            sym, parquet_dir=parquet_dir, margin_mode=margin_mode
        )
        if fr is not None and not fr.empty:
            out[str(sym).upper()] = fr
    return out
