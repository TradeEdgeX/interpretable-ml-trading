from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


@dataclass(frozen=True)
class FeatureStoreSpec:
    """
    Identifies a feature-store dataset.

    We keep this intentionally small: layer + symbol + timeframe.
    """

    layer: str
    symbol: str
    timeframe: str


class FeatureStore:
    """
    Simple partitioned Parquet feature store.

    Layout:
      {root}/{layer}/{symbol}/{timeframe}/{YYYY-MM}.parquet
      {root}/{layer}/{symbol}/{timeframe}/{YYYY-MM}.meta.json
    """

    SCHEMA_VERSION = 1

    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _dataset_dir(self, spec: FeatureStoreSpec) -> Path:
        return self.root_dir / spec.layer / spec.symbol / spec.timeframe

    def _month_key(self, idx: pd.DatetimeIndex) -> str:
        # monthly partition key: YYYY-MM
        if len(idx) == 0:
            raise ValueError("Empty index")
        ts = pd.Timestamp(idx[0]).to_period("M")
        return f"{ts.year:04d}-{ts.month:02d}"

    def _file_paths(self, spec: FeatureStoreSpec, month: str) -> tuple[Path, Path]:
        ds = self._dataset_dir(spec)
        return ds / f"{month}.parquet", ds / f"{month}.meta.json"

    def has_month(self, spec: FeatureStoreSpec, month: str) -> bool:
        parquet_path, _ = self._file_paths(spec, month)
        return parquet_path.exists()

    def delete_month(self, spec: FeatureStoreSpec, month: str) -> bool:
        """Delete cached parquet and meta files for a given month.
        
        Returns:
            True if files were deleted, False if no files existed.
        """
        parquet_path, meta_path = self._file_paths(spec, month)
        deleted = False
        if parquet_path.exists():
            parquet_path.unlink()
            deleted = True
        if meta_path.exists():
            meta_path.unlink()
            deleted = True
        return deleted

    def read_month_meta(self, spec: FeatureStoreSpec, month: str) -> dict:
        """Read the sidecar meta json for a given month."""
        _, meta_path = self._file_paths(spec, month)
        if not meta_path.exists():
            raise FileNotFoundError(f"FeatureStore missing meta file: {meta_path}")
        return json.loads(meta_path.read_text(encoding="utf-8"))

    def write_month(
        self,
        spec: FeatureStoreSpec,
        month: str,
        df: pd.DataFrame,
        *,
        feature_columns: Optional[Iterable[str]] = None,
        base_columns: Optional[Iterable[str]] = None,
        overwrite: bool = False,
        merge_existing: bool = False,
        metadata: Optional[dict] = None,
    ) -> Path:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("FeatureStore expects a DatetimeIndex")
        # Normalize to tz-naive timestamps to avoid mixed tz-aware/tz-naive partitions
        # causing pandas sort/concat failures later.
        if df.index.tz is not None:
            df = df.copy()
            df.index = df.index.tz_convert(None)
        df = df.sort_index()

        parquet_path, meta_path = self._file_paths(spec, month)
        parquet_path.parent.mkdir(parents=True, exist_ok=True)

        if parquet_path.exists() and not overwrite and not merge_existing:
            return parquet_path

        keep_cols: list[str] = []
        if base_columns:
            keep_cols.extend(list(base_columns))
        if feature_columns:
            keep_cols.extend(list(feature_columns))
        keep_cols = list(dict.fromkeys([c for c in keep_cols if c]))

        # Select columns from df (if provided), else keep all df columns.
        if keep_cols:
            df_out = df.loc[:, [c for c in keep_cols if c in df.columns]]
        else:
            df_out = df

        # Optional merge: append/update columns into an existing month file without dropping old columns.
        if parquet_path.exists() and merge_existing:
            try:
                existing = pd.read_parquet(parquet_path)
                if not isinstance(existing.index, pd.DatetimeIndex):
                    # If index was persisted as a column for some reason, fall back to overwrite.
                    existing = existing.set_index(existing.columns[0])
                if isinstance(existing.index, pd.DatetimeIndex) and existing.index.tz is not None:
                    existing = existing.copy()
                    existing.index = existing.index.tz_convert(None)
                existing = existing.sort_index()
                # Union of columns: existing cols + newly requested keep cols (or all new cols).
                col_union = list(dict.fromkeys(list(existing.columns) + list(df_out.columns)))
                # Union of index (defensive): keep all rows observed.
                idx_union = existing.index.union(df_out.index)
                merged = existing.reindex(idx_union)
                df_aligned = df_out.reindex(idx_union)
                for c in df_aligned.columns:
                    merged[c] = df_aligned[c]
                # Ensure column union order
                out = merged.loc[:, [c for c in col_union if c in merged.columns]]
            except Exception:
                # Safe fallback: overwrite with df_out only.
                out = df_out
        else:
            out = df_out

        out.to_parquet(parquet_path)

        meta = {
            "schema_version": self.SCHEMA_VERSION,
            "layer": spec.layer,
            "symbol": spec.symbol,
            "timeframe": spec.timeframe,
            "month": month,
            "rows": int(len(out)),
            "cols": int(len(out.columns)),
            "columns": list(out.columns),
        }
        if metadata:
            meta["metadata"] = metadata
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        return parquet_path

    def read_month(
        self,
        spec: FeatureStoreSpec,
        month: str,
        *,
        columns: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        parquet_path, _ = self._file_paths(spec, month)
        if not parquet_path.exists():
            raise FileNotFoundError(f"FeatureStore missing month file: {parquet_path}")
        df = pd.read_parquet(parquet_path, columns=columns)
        # Defensive: normalize to tz-naive index (we treat FeatureStore as tz-naive UTC).
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            df = df.copy()
            df.index = df.index.tz_convert(None)
        return df

    def read_range(
        self,
        spec: FeatureStoreSpec,
        start: pd.Timestamp,
        end: pd.Timestamp,
        *,
        columns: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        # Read all months covering [start, end]
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        if end_ts < start_ts:
            raise ValueError("end < start")

        # period_range does not need timezone; strip tz for month enumeration
        start_m = start_ts.tz_convert(None) if start_ts.tzinfo is not None else start_ts
        end_m = end_ts.tz_convert(None) if end_ts.tzinfo is not None else end_ts
        months = pd.period_range(start=start_m, end=end_m, freq="M")
        parts: list[pd.DataFrame] = []
        for p in months:
            month = f"{p.year:04d}-{p.month:02d}"
            if not self.has_month(spec, month):
                continue
            df_m = self.read_month(spec, month, columns=columns)
            parts.append(df_m)
        if not parts:
            return pd.DataFrame()
        df = pd.concat(parts)
        # Ensure we never end up with mixed tz-aware / tz-naive indexes across months.
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            df = df.copy()
            df.index = df.index.tz_convert(None)
        df = df.sort_index()

        # Clip to [start, end] (FeatureStore is treated as tz-naive UTC).
        start_a = start_ts.tz_convert(None) if start_ts.tzinfo is not None else start_ts
        end_a = end_ts.tz_convert(None) if end_ts.tzinfo is not None else end_ts
        return df.loc[(df.index >= start_a) & (df.index <= end_a)]


