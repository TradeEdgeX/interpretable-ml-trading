"""Merge live feature-bus snapshots into monthly FeatureStore partitions.

Live SRB writes rolling ``shared_feature_bus/features/120T/*.parquet``. CMS
historical funnel reads ``feature_store/features_srb_120T_*`` when the bus
tail is too short or off-cadence. This module archives bus rows into the SRB
layer (merge per month) so prod accumulates history without manual uploads.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from src.feature_store.feature_store import FeatureStore, FeatureStoreSpec
from src.live_data_stream.feature_bus import (
    normalize_timeframe,
    resolve_disk_primary_timeframe,
)

logger = logging.getLogger(__name__)

DEFAULT_SRB_LAYER = "features_srb_120T_186d0cf9ef"
DEFAULT_SYMBOLS = ("BTCUSDT", "BNBUSDT", "SOLUSDT", "ETHUSDT", "XRPUSDT")


def resolve_srb_layer_name(store_root: Path, explicit: Optional[str] = None) -> str:
    if explicit and str(explicit).strip():
        name = Path(str(explicit).strip()).name
        if name.endswith(".meta.json"):
            name = name.replace(".meta.json", "")
        return name
    preferred = store_root / DEFAULT_SRB_LAYER
    if preferred.is_dir():
        return DEFAULT_SRB_LAYER
    cands = sorted(
        (p.name for p in store_root.glob("features_srb_120T_*") if p.is_dir()),
        reverse=True,
    )
    if cands:
        return cands[0]
    return DEFAULT_SRB_LAYER


def normalize_bus_frame_to_bars(
    df: pd.DataFrame,
    *,
    bar_minutes: int = 120,
) -> pd.DataFrame:
    """Collapse live-bus ticks to one row per closed bar (default 120T)."""
    if df is None or df.empty:
        return pd.DataFrame()

    work = df.copy()
    if "timestamp" in work.columns:
        ts = pd.to_datetime(work["timestamp"], utc=True)
    elif isinstance(work.index, pd.DatetimeIndex):
        ts = pd.to_datetime(work.index, utc=True)
        work = work.reset_index(drop=True)
    else:
        return pd.DataFrame()

    if "_bar_timestamp" in work.columns:
        bar_ts = pd.to_datetime(work["_bar_timestamp"], utc=True, errors="coerce")
        ts = ts.where(bar_ts.isna(), bar_ts)

    work["timestamp"] = ts
    work = work.sort_values("timestamp")
    if int(bar_minutes) > 0:
        work["_bar_key"] = work["timestamp"].dt.floor(f"{int(bar_minutes)}min")
    else:
        work["_bar_key"] = work["timestamp"]

    drop_cols = [
        c
        for c in work.columns
        if c.startswith("_") and c not in ("_bar_key",)
    ]
    work = work.drop(columns=[c for c in drop_cols if c in work.columns], errors="ignore")

    grouped = work.groupby("_bar_key", sort=True).last()
    out = grouped.drop(columns=["timestamp"], errors="ignore")
    out.index = pd.to_datetime(out.index, utc=True)
    out.index.name = None
    return out.sort_index()


def load_bus_feature_frame(
    bus_root: Path,
    symbol: str,
    *,
    strategy_timeframe: str = "120T",
    lookback_days: int = 0,
) -> pd.DataFrame:
    root = Path(bus_root)
    disk_tf, _legacy = resolve_disk_primary_timeframe(root, strategy_timeframe)
    tf_key = normalize_timeframe(disk_tf)
    path = root / "features" / tf_key / f"{symbol.upper()}.parquet"
    if not path.is_file():
        return pd.DataFrame()
    raw = pd.read_parquet(path)
    if raw.empty:
        return pd.DataFrame()
    if "timestamp" not in raw.columns:
        if isinstance(raw.index, pd.DatetimeIndex):
            raw = raw.reset_index().rename(columns={"index": "timestamp"})
        else:
            return pd.DataFrame()
    raw = raw.copy()
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    if int(lookback_days) > 0:
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=int(lookback_days))
        raw = raw[raw["timestamp"] >= cutoff]
    if raw.empty:
        return pd.DataFrame()
    return normalize_bus_frame_to_bars(raw)


def merge_bus_frame_to_store(
    store: FeatureStore,
    spec: FeatureStoreSpec,
    df: pd.DataFrame,
    *,
    source: str = "feature_bus",
) -> Dict[str, int]:
    """Write bus rows into monthly FS partitions (merge_existing)."""
    stats: Dict[str, int] = {"months": 0, "rows": 0}
    if df is None or df.empty:
        return stats

    work = df.copy()
    if not isinstance(work.index, pd.DatetimeIndex):
        if "timestamp" in work.columns:
            work = work.set_index(pd.to_datetime(work["timestamp"], utc=True))
        else:
            return stats
    if work.index.tz is not None:
        work.index = work.index.tz_convert("UTC")
    work = work.sort_index()
    work = work[~work.index.duplicated(keep="last")]

    meta_base = {"source": source, "merged_from": "feature_bus"}
    for period, month_df in work.groupby(work.index.to_period("M")):
        if month_df.empty:
            continue
        month_str = str(period)
        month_out = month_df.copy()
        if not isinstance(month_out.index, pd.DatetimeIndex):
            continue
        if month_out.index.tz is not None:
            month_out = month_out.copy()
            month_out.index = month_out.index.tz_localize(None)
        if store.has_month(spec, month_str):
            try:
                existing = store.read_month(spec, month_str)
                if not isinstance(existing.index, pd.DatetimeIndex):
                    if "timestamp" in existing.columns:
                        existing = existing.set_index(
                            pd.to_datetime(existing["timestamp"], utc=False)
                        )
                    else:
                        existing = existing.set_index(existing.columns[0])
                if isinstance(existing.index, pd.DatetimeIndex) and existing.index.tz is not None:
                    existing = existing.copy()
                    existing.index = existing.index.tz_localize(None)
                idx = existing.index.union(month_out.index)
                left = existing.reindex(idx)
                right = month_out.reindex(idx)
                month_out = left.combine_first(right)
            except Exception:
                month_out = month_df.copy()
                if month_out.index.tz is not None:
                    month_out.index = month_out.index.tz_localize(None)
        store.write_month(
            spec,
            month_str,
            month_out,
            overwrite=True,
            metadata={**meta_base, "month": month_str},
        )
        stats["months"] += 1
        stats["rows"] += int(len(month_df))
    return stats


def merge_bus_symbol_to_store(
    *,
    bus_root: Path,
    store_root: Path,
    layer: str,
    symbol: str,
    timeframe: str = "120T",
    lookback_days: int = 0,
) -> Dict[str, Any]:
    sym = symbol.upper()
    df = load_bus_feature_frame(
        bus_root,
        sym,
        strategy_timeframe=timeframe,
        lookback_days=lookback_days,
    )
    if df.empty:
        return {
            "symbol": sym,
            "skipped": True,
            "reason": "no_bus_rows",
            "months": 0,
            "rows": 0,
        }

    store = FeatureStore(str(store_root))
    spec = FeatureStoreSpec(layer=layer, symbol=sym, timeframe=timeframe)
    stats = merge_bus_frame_to_store(store, spec, df)
    return {
        "symbol": sym,
        "skipped": False,
        "months": stats["months"],
        "rows": stats["rows"],
        "from": str(df.index.min()),
        "to": str(df.index.max()),
    }


def merge_bus_to_store(
    *,
    bus_root: Path,
    store_root: Path,
    symbols: Iterable[str],
    layer: Optional[str] = None,
    timeframe: str = "120T",
    lookback_days: int = 0,
) -> Dict[str, Any]:
    layer_name = resolve_srb_layer_name(store_root, layer)
    store_root.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []
    for sym in symbols:
        s = str(sym or "").strip().upper()
        if not s:
            continue
        try:
            results.append(
                merge_bus_symbol_to_store(
                    bus_root=bus_root,
                    store_root=store_root,
                    layer=layer_name,
                    symbol=s,
                    timeframe=timeframe,
                    lookback_days=lookback_days,
                )
            )
        except Exception as exc:
            logger.exception("bus→FS merge failed for %s", s)
            results.append(
                {"symbol": s, "skipped": True, "reason": "error", "error": str(exc)}
            )
    return {"layer": layer_name, "results": results}
