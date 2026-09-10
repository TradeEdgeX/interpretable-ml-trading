"""Load FeatureStore monthly partitions into a flat panel (timestamp, symbol)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd


@dataclass(frozen=True)
class FeatureStorePanelConfig:
    root: str = "feature_store"
    layer: str = ""
    timeframe: str = "1D"
    timestamp_col: str = "timestamp"
    symbol_col: str = "symbol"


def _month_key(ts: pd.Timestamp) -> str:
    return f"{ts.year:04d}-{ts.month:02d}"


def _month_starts(start: pd.Timestamp, end: pd.Timestamp) -> List[pd.Timestamp]:
    start = pd.Timestamp(start).normalize().replace(day=1)
    end = pd.Timestamp(end).normalize().replace(day=1)
    return list(pd.date_range(start=start, end=end, freq="MS"))


def load_feature_store_panel(
    *,
    symbols: Sequence[str],
    cfg: FeatureStorePanelConfig,
    start_date: str,
    end_date: str,
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    if not cfg.layer:
        raise ValueError("FeatureStorePanelConfig.layer is required")
    if not symbols:
        raise ValueError("symbols is empty")

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    root = Path(cfg.root)
    parts: List[pd.DataFrame] = []

    for sym in symbols:
        sym = str(sym).strip().upper()
        for ms in _month_starts(start_ts, end_ts):
            p = root / cfg.layer / sym / cfg.timeframe / f"{_month_key(ms)}.parquet"
            if not p.exists():
                continue
            df = pd.read_parquet(p)
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
                idx_col = df.columns[0]
                if idx_col != cfg.timestamp_col:
                    df = df.rename(columns={idx_col: cfg.timestamp_col})
            elif cfg.timestamp_col not in df.columns:
                raise ValueError(f"Missing '{cfg.timestamp_col}' in {p}")
            df[cfg.timestamp_col] = pd.to_datetime(
                df[cfg.timestamp_col], errors="coerce"
            )
            df = df.dropna(subset=[cfg.timestamp_col])
            if cfg.symbol_col not in df.columns:
                df[cfg.symbol_col] = sym
            if columns:
                extra = [
                    c
                    for c in columns
                    if c in df.columns and c not in (cfg.timestamp_col, cfg.symbol_col)
                ]
                keep = list(dict.fromkeys([cfg.timestamp_col, cfg.symbol_col] + extra))
                df = df[keep]
            parts.append(df)

    if not parts:
        raise FileNotFoundError(
            f"No FeatureStore partitions for layer={cfg.layer}, timeframe={cfg.timeframe}"
        )

    out = pd.concat(parts, axis=0, ignore_index=True)
    out = out.sort_values([cfg.timestamp_col, cfg.symbol_col])
    mask = (out[cfg.timestamp_col] >= start_ts) & (out[cfg.timestamp_col] < end_ts)
    return out.loc[mask].copy()
