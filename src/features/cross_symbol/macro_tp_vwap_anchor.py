"""Cross-symbol anchor for macro_tp_vwap_1200_position (default: BTCUSDT)."""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_ANCHOR_SYMBOL = "BTCUSDT"
ANCHOR_COLUMN = "macro_tp_vwap_1200_position"

# Process-wide last value per (timeframe, anchor_symbol) for live multi-symbol overlay.
_LIVE_MACRO_TP_VWAP_CACHE: Dict[str, float] = {}


def _live_cache_key(timeframe: str, anchor_symbol: str) -> str:
    return f"{str(timeframe).strip().upper()}::{str(anchor_symbol).strip().upper()}"


def live_set_macro_tp_vwap(
    timeframe: str, anchor_symbol: str, value: float
) -> None:
    _LIVE_MACRO_TP_VWAP_CACHE[_live_cache_key(timeframe, anchor_symbol)] = float(value)


def live_get_macro_tp_vwap(
    timeframe: str, anchor_symbol: str
) -> Optional[float]:
    v = _LIVE_MACRO_TP_VWAP_CACHE.get(_live_cache_key(timeframe, anchor_symbol))
    return None if v is None else float(v)


def apply_live_macro_tp_vwap_overlay(
    *,
    archetypes_dir: Optional[str],
    symbol: str,
    features_by_timeframe: Dict[str, Dict[str, Any]],
) -> None:
    """Mutates each inner features dict: non-anchor symbols get BTC (anchor) macro_tp.

    Reads strategy ``meta.yaml`` next to ``archetypes_dir`` parent.
    """
    if not archetypes_dir or not features_by_timeframe:
        return
    try:
        from pathlib import Path

        import yaml

        strat_dir = Path(archetypes_dir).parent
        meta_path = strat_dir / "meta.yaml"
        meta_full: Dict[str, Any] = {}
        if meta_path.is_file():
            meta_full = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
        ms = meta_full.get("strategy")
        if not isinstance(ms, dict):
            ms = meta_full if isinstance(meta_full, dict) else {}
        en, anchor_sym = parse_macro_tp_vwap_anchor_config(
            meta_strategy=ms,
            meta_yaml_full=meta_full,
        )
        if not en:
            return
        au = str(anchor_sym).strip().upper()
        su = str(symbol).strip().upper()
        for tf, feats in list(features_by_timeframe.items()):
            if not feats or ANCHOR_COLUMN not in feats:
                continue
            try:
                raw_v = feats[ANCHOR_COLUMN]
                v = float(raw_v) if raw_v is not None and not pd.isna(raw_v) else None
            except (TypeError, ValueError):
                v = None
            if su == au:
                if v is not None:
                    live_set_macro_tp_vwap(str(tf), anchor_sym, v)
            else:
                cached = live_get_macro_tp_vwap(str(tf), anchor_sym)
                if cached is not None:
                    feats[ANCHOR_COLUMN] = cached
                else:
                    logger.debug(
                        "live macro_tp_vwap_anchor: no cache for tf=%s anchor=%s sym=%s",
                        tf,
                        anchor_sym,
                        symbol,
                    )
    except Exception as exc:
        logger.warning("live macro_tp_vwap_anchor overlay skipped: %s", exc)


def ensure_datetime_column(df: pd.DataFrame, col: str = "datetime") -> pd.DataFrame:
    """Ensure ``col`` exists for time alignment (copy-on-write safe)."""
    if df is None or df.empty or col in df.columns:
        return df
    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        out[col] = out.index
        return out
    for alt in ("timestamp", "date"):
        if alt in out.columns:
            out[col] = pd.to_datetime(out[alt], utc=True, errors="coerce")
            return out
    return out


def parse_macro_tp_vwap_anchor_config(
    *,
    meta_strategy: Optional[Mapping[str, Any]] = None,
    meta_yaml_full: Optional[Mapping[str, Any]] = None,
) -> Tuple[bool, str]:
    """Resolve (enabled, anchor_symbol). Default: enabled True, symbol BTCUSDT.

    Accepts config under top-level ``macro_tp_vwap_anchor`` or ``strategy.macro_tp_vwap_anchor``
    in meta.yaml, or ``macro_tp_vwap_anchor`` inside the strategy meta dict passed as meta_strategy.
    """
    raw: Any = None
    if meta_yaml_full:
        raw = meta_yaml_full.get("macro_tp_vwap_anchor")
        if raw is None:
            strat = meta_yaml_full.get("strategy")
            if isinstance(strat, dict):
                raw = strat.get("macro_tp_vwap_anchor")
    if raw is None and meta_strategy:
        raw = meta_strategy.get("macro_tp_vwap_anchor")

    if raw is False:
        return False, DEFAULT_ANCHOR_SYMBOL
    if raw is None:
        return True, DEFAULT_ANCHOR_SYMBOL
    if isinstance(raw, dict):
        en = raw.get("enabled", True)
        if isinstance(en, str):
            en = str(en).strip().lower() not in ("false", "0", "no", "off")
        sym = str(raw.get("symbol", DEFAULT_ANCHOR_SYMBOL)).strip().upper()
        if not sym:
            sym = DEFAULT_ANCHOR_SYMBOL
        return bool(en), sym
    return True, DEFAULT_ANCHOR_SYMBOL


def apply_macro_tp_vwap_anchor(
    df: pd.DataFrame,
    *,
    anchor_symbol: str = DEFAULT_ANCHOR_SYMBOL,
    enabled: bool = True,
    column: str = ANCHOR_COLUMN,
    symbol_col: str = "symbol",
    time_col: str = "datetime",
) -> pd.DataFrame:
    """Overwrite ``column`` for non-anchor rows using anchor-symbol values aligned on ``time_col``.

    Anchor rows keep their original ``column`` values. If anchor symbol is missing from
    ``df``, logs a warning and returns ``df`` unchanged.
    """
    if not enabled or df is None or df.empty:
        return df
    if column not in df.columns:
        return df

    sym_u = str(anchor_symbol).strip().upper()
    if not sym_u:
        sym_u = DEFAULT_ANCHOR_SYMBOL

    if symbol_col not in df.columns:
        logger.warning(
            "macro_tp_vwap_anchor: column %r missing, skip overlay", symbol_col
        )
        return df

    if time_col not in df.columns:
        logger.warning(
            "macro_tp_vwap_anchor: column %r missing, skip overlay", time_col
        )
        return df

    sc = df[symbol_col].astype(str).str.upper()
    anchor_mask = sc == sym_u
    if not anchor_mask.any():
        logger.warning(
            "macro_tp_vwap_anchor: anchor symbol %s not in data, skip overlay",
            sym_u,
        )
        return df

    btc = df.loc[anchor_mask, [time_col, column]].copy()
    btc[time_col] = pd.to_datetime(btc[time_col], utc=True, errors="coerce")
    btc = btc.dropna(subset=[time_col])
    if btc.empty:
        logger.warning("macro_tp_vwap_anchor: anchor rows have no valid timestamps")
        return df
    lut = btc.sort_values(time_col).drop_duplicates(subset=[time_col], keep="last")
    lut = lut.rename(columns={column: "__macro_anchor_src"})

    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col], utc=True, errors="coerce")
    out = out.merge(lut[[time_col, "__macro_anchor_src"]], on=time_col, how="left")
    non_anchor = ~anchor_mask
    out.loc[non_anchor, column] = out.loc[non_anchor, "__macro_anchor_src"].to_numpy()
    out = out.drop(columns=["__macro_anchor_src"])
    return out


def apply_macro_tp_vwap_from_anchor_frame(
    df: pd.DataFrame,
    anchor_features: pd.DataFrame,
    *,
    column: str = ANCHOR_COLUMN,
    time_col: str = "datetime",
) -> pd.DataFrame:
    """Set ``df[column]`` from anchor_features aligned on ``time_col`` (ffill after reindex).

    Used when the anchor symbol is not present in ``df`` (e.g. single-alt training).
    """
    if df is None or df.empty or anchor_features is None or anchor_features.empty:
        return df
    if column not in df.columns or column not in anchor_features.columns:
        return df
    if time_col not in df.columns:
        logger.warning(
            "apply_macro_tp_vwap_from_anchor_frame: missing %r on target", time_col
        )
        return df
    if time_col not in anchor_features.columns:
        logger.warning(
            "apply_macro_tp_vwap_from_anchor_frame: missing %r on anchor", time_col
        )
        return df

    af = anchor_features[[time_col, column]].copy()
    af[time_col] = pd.to_datetime(af[time_col], utc=True, errors="coerce")
    af = af.dropna(subset=[time_col])
    lut = af.sort_values(time_col).drop_duplicates(subset=[time_col], keep="last")
    lut = lut.rename(columns={column: "__macro_anchor_src"})

    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col], utc=True, errors="coerce")
    out = out.merge(lut[[time_col, "__macro_anchor_src"]], on=time_col, how="left")
    out[column] = out["__macro_anchor_src"].ffill().bfill().to_numpy()
    out = out.drop(columns=["__macro_anchor_src"])
    return out


def series_overlay_macro_tp_vwap(
    target_times: pd.DatetimeIndex | pd.Series,
    anchor_series: pd.Series,
    *,
    column: str = ANCHOR_COLUMN,
) -> np.ndarray:
    """Map anchor_series (DatetimeIndex -> value) onto target_times; forward-fill gaps."""
    if anchor_series is None or len(anchor_series) == 0:
        return np.full(len(target_times), np.nan, dtype=float)
    s = anchor_series.copy()
    s.index = pd.to_datetime(s.index, utc=True, errors="coerce")
    s = s[~s.index.isna()].sort_index()
    tt = pd.to_datetime(target_times, utc=True, errors="coerce")
    aligned = s.reindex(tt.values)
    return pd.Series(aligned.values).ffill().to_numpy()
