"""Data loading and preprocessing module.

This module provides:
- MarketDataLoader: Low-level loader for tick/OHLC parquet files with caching
- DataHandler: Unified high-level interface for OHLCV and tick data loading
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import re
from typing import List, Tuple, Dict, Optional, Any

from src.data_tools.tick_loader import load_tick_data
from src.data_tools.processors import (
    Processor,
    ProcessorChain,
    Pipeline,
    get_default_processor_chain,
    DataValidator,
    DataQualityReport,
    LookaheadLeakageDetector,
    LookaheadLeakageReport,
    ResearchValidator,
)

TIMEFRAME_CACHE_DIR = Path("cache/timeframes")
TIMEFRAME_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Backward-compatible rename map: old parquet column names -> new names
_CVD_RENAME_MAP = {
    "cvd_short": "cvd_roll20",
    "cvd_medium": "cvd_roll60",
    "cvd_long": "cvd_roll288",
}


class MarketDataLoader:
    """Handles loading of tick-level parquet and resamples to requested timeframe."""

    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path
        self.raw_data: Optional[pd.DataFrame] = None

    def load_data(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        timeframe: str = "15T",
    ) -> pd.DataFrame:
        """
        Load raw market data.

        Args:
            symbol: Trading symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            timeframe: Resampling timeframe (e.g., "15T", "60T", "240T")

        Returns:
            DataFrame with OHLCV data
        """
        if not self.data_path:
            raise ValueError("data_path is required for MarketDataLoader.")

        if not symbol:
            raise ValueError("Symbol must be provided to load tick parquet files.")

        cache_file = self._cache_file_path(symbol, timeframe)
        cache_exists = cache_file.exists()
        if cache_exists:
            df_cached = pd.read_parquet(cache_file)
        else:
            # Build cache for the requested window only (avoid full-history resample cost).
            df_cached = self._build_timeframe_cache(
                symbol, timeframe, start_date, end_date
            )
            df_cached.to_parquet(cache_file)

        df_cached.index = pd.to_datetime(df_cached.index)
        # Handle tz-aware vs tz-naive comparisons robustly
        idx_tz = getattr(df_cached.index, "tz", None)
        if start_date:
            start_ts = (
                pd.to_datetime(start_date, utc=True)
                if idx_tz is not None
                else pd.to_datetime(start_date)
            )
        else:
            start_ts = df_cached.index.min()
        if end_date:
            end_ts = (
                pd.to_datetime(end_date, utc=True)
                if idx_tz is not None
                else pd.to_datetime(end_date)
            )
        else:
            end_ts = df_cached.index.max()

        # If new raw parquet months were downloaded after the cache file was built,
        # the cache may be stale even when the requested window is within cached_min/max.
        # In that case, rebuild the cache.
        rebuild_due_to_new_raw = False
        if cache_exists:
            try:
                data_root = Path(self.data_path)
                pattern = f"{symbol}_*.parquet"
                raw_files = list(data_root.glob(pattern))
                if raw_files:
                    newest_raw_mtime = max(p.stat().st_mtime for p in raw_files)
                    cache_mtime = cache_file.stat().st_mtime
                    if newest_raw_mtime > cache_mtime:
                        rebuild_due_to_new_raw = True
            except Exception:
                # Never fail loading due to cache freshness checks.
                rebuild_due_to_new_raw = False

        # If the timeframe cache exists but doesn't cover the requested window,
        # extend it to incorporate newly downloaded raw parquet months.
        if cache_exists and not df_cached.empty:
            cached_min = df_cached.index.min()
            cached_max = df_cached.index.max()
            if (start_ts is not None and start_ts < cached_min) or (
                end_ts is not None and end_ts > cached_max
            ):
                df_cached = self._extend_timeframe_cache(
                    df_cached=df_cached,
                    symbol=symbol,
                    timeframe=timeframe,
                    start_ts=start_ts,
                    end_ts=end_ts,
                )
                df_cached.to_parquet(cache_file)
                df_cached.index = pd.to_datetime(df_cached.index)
            elif rebuild_due_to_new_raw:
                # Conservative: rebuild only the requested window (fast) rather than full history.
                df_cached = self._build_timeframe_cache(
                    symbol, timeframe, start_date, end_date
                )
                df_cached.to_parquet(cache_file)
                df_cached.index = pd.to_datetime(df_cached.index)

        # Backward-compat: rename old CVD column names from cache/parquet files
        rename_cols = {
            k: v
            for k, v in _CVD_RENAME_MAP.items()
            if k in df_cached.columns and v not in df_cached.columns
        }
        if rename_cols:
            df_cached = df_cached.rename(columns=rename_cols)

        df_subset = df_cached.loc[
            (df_cached.index >= start_ts) & (df_cached.index <= end_ts)
        ].copy()
        df_subset["_symbol"] = symbol
        self.raw_data = df_subset
        print(f"Loaded {len(df_subset)} bars")
        return self.raw_data

    def _cache_file_path(self, symbol: str, timeframe: str) -> Path:
        safe_symbol = symbol.replace("/", "_")
        return TIMEFRAME_CACHE_DIR / f"{safe_symbol}_{timeframe}.parquet"

    def _parse_month_from_raw_filename(self, p: Path) -> Optional[pd.Timestamp]:
        """
        Raw parquet naming convention (monthly):
          <SYMBOL>_YYYY-MM.parquet
        """
        m = re.search(r"_(\d{4}-\d{2})\.parquet$", p.name)
        if not m:
            return None
        try:
            return pd.to_datetime(m.group(1) + "-01", utc=True)
        except Exception:
            return None

    @staticmethod
    def _to_utc_naive(ts: Optional[pd.Timestamp]) -> Optional[pd.Timestamp]:
        """
        Normalize timestamps to UTC tz-naive for safe comparisons.

        We sometimes mix tz-aware (utc=True) and tz-naive timestamps, which breaks
        comparisons like `month_end < start_ts`.
        """
        if ts is None:
            return None
        try:
            t = pd.to_datetime(ts, utc=True)
            return t.tz_convert("UTC").tz_localize(None)
        except Exception:
            try:
                t = pd.to_datetime(ts)
                if getattr(t, "tz", None) is not None:
                    t = t.tz_convert("UTC").tz_localize(None)
                return t
            except Exception:
                return None

    def _select_raw_files_for_range(
        self,
        *,
        symbol: str,
        start_ts: Optional[pd.Timestamp],
        end_ts: Optional[pd.Timestamp],
    ) -> List[Path]:
        data_root = Path(self.data_path)
        pattern = f"{symbol}_*.parquet"
        files = sorted(data_root.glob(pattern))
        if not files:
            return []
        if start_ts is None and end_ts is None:
            return files
        out: List[Path] = []
        start_ts_n = self._to_utc_naive(start_ts)
        end_ts_n = self._to_utc_naive(end_ts)
        for fp in files:
            m = self._parse_month_from_raw_filename(fp)
            if m is None:
                out.append(fp)
                continue
            month_start = self._to_utc_naive(m)
            month_end = self._to_utc_naive((m + pd.offsets.MonthEnd(1)).normalize())
            if (
                start_ts_n is not None
                and month_end is not None
                and month_end < start_ts_n
            ):
                continue
            if (
                end_ts_n is not None
                and month_start is not None
                and month_start > end_ts_n
            ):
                continue
            out.append(fp)
        return out

    def _build_timeframe_cache(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        data_root = Path(self.data_path)
        # Only read raw months needed for requested window (huge speedup vs full-history resample).
        start_ts = pd.to_datetime(start_date, utc=True) if start_date else None
        end_ts = pd.to_datetime(end_date, utc=True) if end_date else None
        files = self._select_raw_files_for_range(
            symbol=symbol, start_ts=start_ts, end_ts=end_ts
        )
        if not files:
            raise FileNotFoundError(
                f"No parquet files found for symbol {symbol} under {data_root}"
            )

        frames = []
        for file_path in files:
            df = pd.read_parquet(file_path)
            if df.empty:
                continue
            if {"open", "high", "low", "close"}.issubset(df.columns):
                frames.append(self._resample_from_ohlc(df, timeframe))
            else:
                frames.append(self._resample_from_ticks(df, timeframe))

        if not frames:
            raise ValueError(f"No usable data after reading {len(files)} files.")

        # Normalize tz: strip timezone info to avoid tz-naive vs tz-aware errors
        for i, f in enumerate(frames):
            try:
                if f.index.tz is not None:
                    frames[i] = f.tz_convert(None)
            except (TypeError, AttributeError):
                pass
        result = pd.concat(frames)
        try:
            if result.index.tz is not None:
                result.index = result.index.tz_convert(None)
        except (TypeError, AttributeError):
            pass
        result = result.sort_index()
        result = result[~result.index.duplicated(keep="last")]
        # Recompute CVD columns from buy_qty/sell_qty across the full concatenated
        # range so that cumsum and rolling windows are continuous across months.
        result = self._recompute_cvd_columns(result)
        return result

    def _extend_timeframe_cache(
        self,
        *,
        df_cached: pd.DataFrame,
        symbol: str,
        timeframe: str,
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
    ) -> pd.DataFrame:
        # Extend forward/backward by resampling only missing raw months, then concat.
        cached_min = df_cached.index.min()
        cached_max = df_cached.index.max()
        need_before = start_ts is not None and start_ts < cached_min
        need_after = end_ts is not None and end_ts > cached_max

        frames = [df_cached]
        if need_before:
            files = self._select_raw_files_for_range(
                symbol=symbol, start_ts=start_ts, end_ts=cached_min
            )
            if files:
                frames.append(self._build_timeframe_cache_for_files(files, timeframe))
        if need_after:
            files = self._select_raw_files_for_range(
                symbol=symbol, start_ts=cached_max, end_ts=end_ts
            )
            if files:
                frames.append(self._build_timeframe_cache_for_files(files, timeframe))

        # Normalize tz: strip timezone info to avoid tz-naive vs tz-aware errors
        for i, f in enumerate(frames):
            try:
                if f.index.tz is not None:
                    frames[i] = f.tz_convert(None)
            except (TypeError, AttributeError):
                pass
        out = pd.concat(frames)
        # Second pass after concat
        try:
            if out.index.tz is not None:
                out.index = out.index.tz_convert(None)
        except (TypeError, AttributeError):
            pass
        out = out.sort_index()
        out = out[~out.index.duplicated(keep="last")]
        # Recompute CVD across the full extended range for continuity.
        out = self._recompute_cvd_columns(out)
        return out

    def _build_timeframe_cache_for_files(
        self, files: List[Path], timeframe: str
    ) -> pd.DataFrame:
        frames = []
        for file_path in files:
            df = pd.read_parquet(file_path)
            if df.empty:
                continue
            if {"open", "high", "low", "close"}.issubset(df.columns):
                frames.append(self._resample_from_ohlc(df, timeframe))
            else:
                frames.append(self._resample_from_ticks(df, timeframe))
        if not frames:
            raise ValueError(f"No usable data after reading {len(files)} files.")
        # Normalize tz: strip timezone info to avoid tz-naive vs tz-aware errors
        for i, f in enumerate(frames):
            try:
                if f.index.tz is not None:
                    frames[i] = f.tz_convert(None)
            except (TypeError, AttributeError):
                pass
        result = pd.concat(frames)
        # Second pass: strip tz after concat as well
        try:
            if result.index.tz is not None:
                result.index = result.index.tz_convert(None)
        except (TypeError, AttributeError):
            pass
        result = result.sort_index()
        result = result[~result.index.duplicated(keep="last")]
        # Recompute CVD columns across full range (continuous across months).
        result = self._recompute_cvd_columns(result)
        return result

    @staticmethod
    def _recompute_cvd_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Recompute all CVD-derived columns from buy_qty / sell_qty.

        This must run AFTER cross-file concatenation so that cumsum and rolling
        windows are continuous across month boundaries.  Previous implementation
        aggregated per-bar cumsum values via 'sum' during resample, which was
        incorrect and caused shd_pct and bpc_cvd_z mismatches.
        """
        if "buy_qty" not in df.columns or "sell_qty" not in df.columns:
            # No orderflow data — fill CVD columns with 0
            for col in [
                "cvd",
                "cvd_change_1",
                "cvd_change_5",
                "cvd_change_20",
                "cvd_roll20",
                "cvd_roll60",
                "cvd_roll288",
                "cvd_normalized",
                "cvd_change_5_normalized",
            ]:
                df[col] = 0.0
            return df

        delta = df["buy_qty"].fillna(0) - df["sell_qty"].fillna(0)
        total_flow = df["buy_qty"].fillna(0) + df["sell_qty"].fillna(0)

        df["cvd_change_1"] = delta
        df["cvd_change_5"] = delta.rolling(window=5, min_periods=1).sum()
        df["cvd_change_20"] = delta.rolling(window=20, min_periods=1).sum()
        df["cvd_roll20"] = delta.rolling(window=20, min_periods=1).sum()
        df["cvd_roll60"] = delta.rolling(window=60, min_periods=1).sum()
        df["cvd_roll288"] = delta.rolling(window=288, min_periods=1).sum()
        df["cvd"] = delta.cumsum()
        df["cvd_normalized"] = (delta / total_flow.replace(0, np.nan)).fillna(0)
        total_flow_5 = total_flow.rolling(window=5, min_periods=1).sum()
        df["cvd_change_5_normalized"] = (
            df["cvd_change_5"] / total_flow_5.replace(0, np.nan)
        ).fillna(0)

        # Recompute taker_buy_ratio from aggregated buy/sell if not already correct
        if "taker_buy_ratio" not in df.columns:
            df["taker_buy_ratio"] = (
                df["buy_qty"] / total_flow.replace(0, np.nan)
            ).fillna(0.5)

        return df

    def _resample_from_ohlc(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        df = df.copy()
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
        df = df.sort_index()

        # Backward-compat: rename old CVD column names from existing parquet files
        rename_cols = {
            k: v
            for k, v in _CVD_RENAME_MAP.items()
            if k in df.columns and v not in df.columns
        }
        if rename_cols:
            df = df.rename(columns=rename_cols)

        agg_dict = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
        # Only aggregate additive orderflow columns (buy_qty, sell_qty).
        # CVD-derived columns (cvd, cvd_change_*, cvd_roll*) must NOT be summed
        # because they are cumulative or rolling — summing sub-bar values is wrong.
        # They will be recomputed after cross-month concatenation in
        # _recompute_cvd_columns().
        additive_of_cols = ["buy_qty", "sell_qty"]
        for col in additive_of_cols:
            if col in df.columns:
                agg_dict[col] = "sum"
        if "taker_buy_ratio" in df.columns:
            agg_dict["taker_buy_ratio"] = "mean"

        resampled = df.resample(timeframe).agg(agg_dict).dropna()
        resampled["trade_count"] = df["close"].resample(timeframe).size()
        return resampled

    def _resample_from_ticks(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        df = df.copy()
        if "timestamp" not in df.columns:
            raise ValueError("Tick parquet must contain 'timestamp' column.")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.dropna(subset=["timestamp", "price", "volume"])
        df = df.sort_values("timestamp").set_index("timestamp")

        price_ohlc = df["price"].resample(timeframe).ohlc().dropna()
        volume = df["volume"].resample(timeframe).sum()
        result = price_ohlc.rename(
            columns={"open": "open", "high": "high", "low": "low", "close": "close"}
        )
        result["volume"] = volume
        result["trade_count"] = df["price"].resample(timeframe).size()

        if "side" in df.columns:
            buy_series = np.where(df["side"] == 1, df["volume"], 0.0)
            sell_series = np.where(df["side"] == -1, df["volume"], 0.0)
            order_flow = pd.DataFrame(
                {"buy_qty": buy_series, "sell_qty": sell_series}, index=df.index
            )
            flow_resampled = order_flow.resample(timeframe).sum()
            result["buy_qty"] = flow_resampled["buy_qty"]
            result["sell_qty"] = flow_resampled["sell_qty"]
            total_flow = result["buy_qty"] + result["sell_qty"]
            result["taker_buy_ratio"] = (
                result["buy_qty"] / total_flow.replace(0, np.nan)
            ).fillna(0.5)
            # NOTE: CVD-derived columns (cvd, cvd_change_*, cvd_roll*) are NOT
            # computed here because per-file cumsum would break at month
            # boundaries after concatenation.  They are recomputed in
            # _recompute_cvd_columns() after cross-file concat.
        else:
            for col in [
                "buy_qty",
                "sell_qty",
                "taker_buy_ratio",
            ]:
                result[col] = 0.0

        return result

    def resample_data(self, timeframe: str) -> pd.DataFrame:
        """
        Resample data to specified timeframe.

        Args:
            timeframe: Target timeframe (e.g., '5min', '15min', '45min')

        Returns:
            Resampled DataFrame
        """
        if self.raw_data is None:
            self.load_data()

        # Ensure raw_data is not None before using it
        if self.raw_data is None:
            raise ValueError("Raw data is None. Call load_data() first.")

        # Convert timeframe format: pandas resample accepts 'T' (minutes) directly
        # Keep 'T' format as pandas prefers it (e.g., '5T', '15T', '240T')
        # Handle different input formats:
        # 1. Pure number (e.g., "15") -> "15T" (15 minutes)
        # 2. "min" format (e.g., "15min") -> "15T"
        # 3. Already in "T" format (e.g., "15T") -> keep as is
        if timeframe.isdigit():
            # Pure number: assume minutes
            timeframe = f"{timeframe}T"
        elif timeframe.endswith("min") and not timeframe.endswith("T"):
            # "min" format: convert to "T"
            timeframe = timeframe.replace("min", "T")
        # If already ends with "T", keep as is

        # Using separate operations for each column to avoid type issues
        resampled_open = self.raw_data["open"].resample(timeframe).first()
        resampled_high = self.raw_data["high"].resample(timeframe).max()
        resampled_low = self.raw_data["low"].resample(timeframe).min()
        resampled_close = self.raw_data["close"].resample(timeframe).last()
        resampled_volume = self.raw_data["volume"].resample(timeframe).sum()

        # Optional microstructure columns propagated if present in raw_data
        have_buy = "buy_qty" in self.raw_data.columns
        have_sell = "sell_qty" in self.raw_data.columns
        have_ratio = "taker_buy_ratio" in self.raw_data.columns
        have_cvd = "cvd" in self.raw_data.columns

        data_dict = {
            "open": resampled_open,
            "high": resampled_high,
            "low": resampled_low,
            "close": resampled_close,
            "volume": resampled_volume,
        }
        if have_buy:
            data_dict["buy_qty"] = self.raw_data["buy_qty"].resample(timeframe).sum()
        if have_sell:
            data_dict["sell_qty"] = self.raw_data["sell_qty"].resample(timeframe).sum()
        if have_ratio:
            # ratio is averaged over the window
            data_dict["taker_buy_ratio"] = (
                self.raw_data["taker_buy_ratio"].resample(timeframe).mean()
            )
        if have_cvd:
            # cvd is cumulative; use last value in window
            data_dict["cvd"] = self.raw_data["cvd"].resample(timeframe).last()

        # Combine into a single DataFrame
        resampled = pd.DataFrame(data_dict).dropna()

        return resampled


class DataHandler:
    """
    Unified data handler for market data loading.

    This class provides a consistent interface for loading OHLCV and tick data,
    ensuring that all entry points (training, backtesting, feature store, live)
    use the same data preprocessing logic.

    Attributes:
        data_path: Path to OHLCV parquet data directory
        tick_data_path: Path to tick data directory (defaults to data_path)
    """

    # Standard base columns that should be present in all OHLCV DataFrames
    BASE_COLUMNS = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "_symbol",
    ]

    # Orderflow base columns (computed from ticks or present in raw data)
    ORDERFLOW_BASE_COLUMNS = [
        "buy_qty",
        "sell_qty",
        "taker_buy_ratio",
        "cvd",
        "cvd_roll20",
        "cvd_roll60",
        "cvd_roll288",
        "cvd_change_1",
        "cvd_change_5",
        "cvd_change_20",
        "cvd_normalized",
        "delta",
    ]

    def __init__(
        self,
        data_path: str,
        tick_data_path: Optional[str] = None,
        processors: Optional[List[Processor]] = None,
        use_default_processors: bool = False,
    ):
        """
        Initialize DataHandler.

        Args:
            data_path: Path to OHLCV parquet data directory
            tick_data_path: Path to tick data directory (defaults to data_path)
            processors: Optional list of processors to apply after loading
            use_default_processors: If True, use default processor chain
        """
        self.data_path = Path(data_path)
        self.tick_data_path = Path(tick_data_path) if tick_data_path else self.data_path
        self._market_loader = MarketDataLoader(str(self.data_path))

        # Setup processor chain
        if processors:
            self._processor_chain = ProcessorChain(processors)
        elif use_default_processors:
            self._processor_chain = get_default_processor_chain()
        else:
            self._processor_chain = None

    def load_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        validate: bool = False,
    ) -> pd.DataFrame:
        """
        Load OHLCV data with base columns.

        This method loads and resamples market data, ensuring a consistent
        base schema across all entry points.

        Args:
            symbol: Trading symbol(s), comma-separated for multi-asset
            timeframe: Resampling timeframe (e.g., "15T", "1H", "240T")
            start_date: Start date (YYYY-MM-DD), optional
            end_date: End date (YYYY-MM-DD), optional
            validate: If True, run data quality validation and print report

        Returns:
            DataFrame with OHLCV + base columns, indexed by datetime

        Raises:
            ValueError: If no data found for symbol(s)
        """
        symbol_list = [s.strip() for s in symbol.split(",") if s.strip()]
        all_dfs = []

        for sym in symbol_list:
            df_single = self._market_loader.load_data(
                symbol=sym,
                start_date=start_date,
                end_date=end_date,
                timeframe=timeframe,
            )

            if df_single is not None and not df_single.empty:
                # Ensure datetime index
                if not isinstance(df_single.index, pd.DatetimeIndex):
                    for col in ("datetime", "timestamp", "date"):
                        if col in df_single.columns:
                            df_single.index = pd.to_datetime(df_single[col])
                            break

                # Resample to ensure consistent aggregation rules
                if isinstance(df_single.index, pd.DatetimeIndex):
                    agg_dict = {
                        "open": "first",
                        "high": "max",
                        "low": "min",
                        "close": "last",
                        "volume": "sum",
                    }

                    # Add orderflow columns if they exist
                    for col in self.ORDERFLOW_BASE_COLUMNS:
                        if col in df_single.columns:
                            if col in ["buy_qty", "sell_qty", "delta"]:
                                agg_dict[col] = "sum"
                            else:
                                agg_dict[col] = "last"

                    # Add other numeric columns (use last as default)
                    for col in df_single.columns:
                        if (
                            col not in agg_dict
                            and pd.api.types.is_numeric_dtype(df_single[col])
                            and col != "_symbol"
                        ):
                            agg_dict[col] = "last"

                    df_single = (
                        df_single.resample(timeframe)
                        .agg(agg_dict)
                        .dropna(subset=["close"])
                    )

                # Ensure _symbol column
                if "_symbol" not in df_single.columns:
                    df_single["_symbol"] = sym

                df_single = df_single.sort_index()
                all_dfs.append(df_single)

        if not all_dfs:
            raise ValueError(f"No data found for symbol(s): {symbol}")

        df = pd.concat(all_dfs, axis=0).sort_index()

        # Remove duplicate indices (keep last)
        if df.index.duplicated().any():
            df = df[~df.index.duplicated(keep="last")]

        # Apply processor chain if configured
        if self._processor_chain:
            df = self._processor_chain.process(df)

        # Optional validation
        if validate:
            validator = DataValidator()
            report = validator.validate(df)
            report.print_report()

        return df

    def load_ticks(
        self,
        symbol: str,
        start_ts: str,
        end_ts: str,
        df_bars: Optional[pd.DataFrame] = None,
        lookback_minutes: int = 60,
    ) -> pd.DataFrame:
        """
        Load tick data aligned to bar timeframe.

        Args:
            symbol: Trading symbol
            start_ts: Start timestamp (YYYY-MM-DD HH:MM:SS or ISO format)
            end_ts: End timestamp (YYYY-MM-DD HH:MM:SS or ISO format)
            df_bars: Optional DataFrame with bar index to align ticks
            lookback_minutes: Additional lookback minutes for tick loading

        Returns:
            DataFrame with tick data (columns: price, volume, side)
        """
        if df_bars is not None and not df_bars.empty:
            if isinstance(df_bars.index, pd.DatetimeIndex):
                start_ts = df_bars.index.min() - pd.Timedelta(minutes=lookback_minutes)
                end_ts = df_bars.index.max() + pd.Timedelta(minutes=lookback_minutes)
                start_ts = start_ts.strftime("%Y-%m-%d %H:%M:%S")
                end_ts = end_ts.strftime("%Y-%m-%d %H:%M:%S")

        return load_tick_data(
            symbol=symbol,
            start_ts=start_ts,
            end_ts=end_ts,
            ticks_dir=str(self.tick_data_path),
            lookback_minutes=lookback_minutes,
        )

    def get_base_schema(self) -> List[str]:
        """Get the list of base columns that should be present in all OHLCV DataFrames."""
        return self.BASE_COLUMNS.copy()

    def get_orderflow_base_schema(self) -> List[str]:
        """Get the list of orderflow base columns."""
        return self.ORDERFLOW_BASE_COLUMNS.copy()

    def ensure_base_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure that base columns are present in the DataFrame.

        Missing columns are filled with appropriate defaults.
        """
        df = df.copy()

        if "_symbol" not in df.columns:
            df["_symbol"] = "UNKNOWN"

        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                df[col] = 0.0

        return df

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the processor chain to a DataFrame.

        Useful for processing data that was loaded separately or
        for applying processors to feature data.

        Args:
            df: Input DataFrame

        Returns:
            Processed DataFrame
        """
        if self._processor_chain:
            return self._processor_chain.process(df)
        return df

    def set_processors(self, processors: List[Processor]) -> None:
        """
        Set a new processor chain.

        Args:
            processors: List of processors to apply
        """
        self._processor_chain = ProcessorChain(processors)

    def pipeline(self, df: pd.DataFrame) -> Pipeline:
        """
        Create a Pipeline for fluent data processing.

        This enables chained operations on DataFrames.

        Usage:
            handler = DataHandler(data_path)
            df = handler.load_ohlcv(symbol, timeframe)

            # Chain processing and validation
            result = (handler.pipeline(df)
                .replace_inf()
                .fill_na()
                .validate(quality=True, leakage=True)
                .downcast()
                .result()
            )

            # Or use the shorthand
            result = (handler.load_and_pipeline(symbol, timeframe)
                .validate(quality=True)
                .result()
            )
        """
        return Pipeline(df)

    def load_and_pipeline(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Pipeline:
        """
        Load OHLCV data and return a Pipeline for fluent processing.

        Combines load_ohlcv() and pipeline() for convenience.

        Usage:
            handler = DataHandler(data_path)
            result = (handler.load_and_pipeline("BTCUSDT", "240T")
                .replace_inf()
                .fill_na()
                .validate(quality=True, leakage=True)
                .downcast()
                .result()
            )
        """
        df = self.load_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            validate=False,  # Validation will be done in the pipeline
        )
        return Pipeline(df)

    # =========================================================================
    # Validation Methods (legacy, prefer using pipeline().validate())
    # =========================================================================

    def validate_quality(
        self,
        df: pd.DataFrame,
        print_report: bool = True,
        **kwargs,
    ) -> DataQualityReport:
        """
        Validate data quality (NaN, inf, outliers, constant columns).

        Args:
            df: DataFrame to validate
            print_report: Whether to print the report
            **kwargs: Additional arguments for DataValidator

        Returns:
            DataQualityReport with detected issues

        Usage:
            handler = DataHandler(data_path)
            df = handler.load_ohlcv(symbol, timeframe)
            report = handler.validate_quality(df)
        """
        validator = DataValidator(**kwargs)
        report = validator.validate(df)
        if print_report:
            report.print_report()
        return report

    def validate_leakage(
        self,
        df: pd.DataFrame,
        target_col: str = "close",
        print_report: bool = True,
        **kwargs,
    ) -> LookaheadLeakageReport:
        """
        Detect potential lookahead (future data) leakage.

        This is critical for research/backtesting to ensure features
        don't accidentally use future information.

        Args:
            df: DataFrame with features to validate
            target_col: Target column for return calculation
            print_report: Whether to print the report
            **kwargs: Additional arguments for LookaheadLeakageDetector

        Returns:
            LookaheadLeakageReport with detected issues

        Usage:
            handler = DataHandler(data_path)
            df_features = compute_features(df)
            report = handler.validate_leakage(df_features)
            if report.has_critical_issues():
                raise ValueError("Lookahead leakage detected!")
        """
        detector = LookaheadLeakageDetector(**kwargs)
        report = detector.detect(df, target_col=target_col)
        if print_report:
            report.print_report()
        return report

    def validate_research(
        self,
        df: pd.DataFrame,
        target_col: str = "close",
        print_report: bool = True,
        fail_on_critical: bool = True,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Run full research validation suite (quality + leakage detection).

        Args:
            df: DataFrame to validate
            target_col: Target column for leakage detection
            print_report: Whether to print reports
            fail_on_critical: Return False if critical issues found

        Returns:
            Tuple of (is_valid, reports_dict)

        Usage:
            handler = DataHandler(data_path)
            df_features = compute_features(df)
            is_valid, reports = handler.validate_research(df_features)
            if not is_valid:
                raise ValueError("Research validation failed!")
        """
        validator = ResearchValidator(fail_on_critical=fail_on_critical)
        return validator.validate(df, target_col=target_col, print_report=print_report)
