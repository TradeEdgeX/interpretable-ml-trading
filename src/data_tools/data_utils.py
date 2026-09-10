"""
Data loading utilities for time series training.

This module provides reusable functions for loading and preparing market data
with features, used by training and evaluation scripts.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import pandas as pd

from src.data_tools.data_handler import MarketDataLoader
from src.features.loader.config_feature_engineer import ConfigFeatureEngineer
from src.time_series_model.pipeline.dimensionality.utils import load_top_factors_list


def load_raw_data(
    data_path: str,
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    timeframe: str = "15T",
) -> pd.DataFrame:
    """
    Load and resample raw market data (without feature engineering).

    This function handles data loading and resampling only, not feature engineering.

    Args:
        data_path: Path to data directory
        symbol: Trading symbol(s), comma-separated for multi-asset
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        timeframe: Resampling timeframe (e.g., "15T", "1H")

    Returns:
        DataFrame with raw OHLCV data (resampled, with _symbol column)
    """
    symbol_list = [s.strip() for s in symbol.split(",") if s.strip()]
    loader = MarketDataLoader(data_path)
    all_dfs = []

    for sym in symbol_list:
        df_single = loader.load_data(
            symbol=sym, start_date=start_date, end_date=end_date, timeframe=timeframe
        )

        if df_single is not None and not df_single.empty:
            # MarketDataLoader already handles resampling, no need to resample again
            # Just add the symbol column and append
            df_single["_symbol"] = sym
            all_dfs.append(df_single)

    if not all_dfs:
        raise ValueError(f"No data found for symbol(s): {symbol}")

    df = pd.concat(all_dfs, axis=0)
    # Normalize index timezone to avoid tz-aware vs tz-naive comparison issues.
    idx = pd.to_datetime(df.index, utc=True, errors="coerce")
    df.index = idx
    df = df[~df.index.isna()]
    df = df.sort_index()
    return df


def load_data(
    data_path: str,
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    timeframe: str = "15T",
    feature_strategy: str = "sr_breakout",
    top_factors: Optional[str] = None,
    engineer: Optional[ConfigFeatureEngineer] = None,
    fit: bool = True,
) -> Tuple[pd.DataFrame, List[str], Optional[ConfigFeatureEngineer]]:
    """
    Load and prepare market data with features.

    This function loads raw data, performs feature engineering, and returns
    a DataFrame with features ready for training or evaluation.

    Args:
        data_path: Path to data directory
        symbol: Trading symbol(s), comma-separated for multi-asset
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        timeframe: Resampling timeframe (e.g., "15T", "1H")
        feature_strategy: Strategy name defined in strategy_features.yaml
        top_factors: Path to top_factors.json file to filter features
        engineer: Pre-fitted ConfigFeatureEngineer (optional)
        fit: Whether to fit the engineer (True) or transform only (False)

    Returns:
        Tuple of (DataFrame with features, feature column list, engineer object)
    """
    print(f"📊 Loading data for {symbol}...")

    # Load raw data
    df = load_raw_data(
        data_path=data_path,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        timeframe=timeframe,
    )
    symbol_list = [s.strip() for s in symbol.split(",") if s.strip()]
    print(f"   ✅ Loaded {len(df)} samples from {len(symbol_list)} asset(s)")

    # Load top factors if specified
    selected_features = None
    if top_factors:
        print(f"📋 Loading top factors from {top_factors}...")
        try:
            top_factors_list = load_top_factors_list(top_factors)
            selected_features = set(top_factors_list)
            print(
                f"   ✅ Loaded {len(selected_features)} features from top_factors.json"
            )
            print(f"   📊 Will only generate these features (others will be skipped)")
        except Exception as e:
            print(f"   ⚠️  Failed to load top factors: {e}")
            print(f"   ⚠️  Will generate all features for strategy {feature_strategy}")

    # Feature engineering
    if engineer is None:
        print(f"🔧 Engineering features (strategy: {feature_strategy})...")
        engineer = ConfigFeatureEngineer(strategy_name=feature_strategy)
        df_features = engineer.engineer_all_features(
            df, fit=fit, required_features=selected_features
        )
    else:
        print(
            f"🔧 Transforming features using pre-fitted engineer (strategy: {feature_strategy})..."
        )
        df_features = engineer.engineer_all_features(
            df, fit=False, required_features=selected_features
        )

    # Keep close price for label preparation
    if "close" not in df_features.columns and "close" in df.columns:
        df_features["close"] = df["close"]

    # Filter out label columns and raw prices
    exclude_exact = {
        "timestamp",
        "open",
        "high",
        "low",
        "volume",
        "signal",
        "binary_signal",
        "future_return",
    }
    exclude_prefixes = ("signal_", "binary_signal_", "future_return_")

    # Get all potential feature columns
    all_potential_features = [
        col
        for col in df_features.columns
        if (col not in exclude_exact)
        and (not any(col.startswith(pfx) for pfx in exclude_prefixes))
        and col != "_symbol"  # Keep _symbol but don't include in features
    ]

    # If selected_features is specified, only keep those features
    if selected_features is not None:
        feature_cols = [
            col for col in all_potential_features if col in selected_features
        ]
        print(
            f"   ✅ Generated {len(all_potential_features)} features, filtered to {len(feature_cols)} features from top_factors.json"
        )
        if len(feature_cols) < len(selected_features):
            missing = selected_features - set(feature_cols)
            print(
                f"   ⚠️  Warning: {len(missing)} features from top_factors.json were not generated:"
            )
            for feat in list(missing)[:10]:  # Show first 10 missing
                print(f"      - {feat}")
            if len(missing) > 10:
                print(f"      ... and {len(missing) - 10} more")
    else:
        feature_cols = all_potential_features
        print(f"   ✅ Generated {len(feature_cols)} features")

    # Keep symbol column and close for multi-asset support and label prep
    keep_cols = [*feature_cols, "close"]
    if "_symbol" in df_features.columns:
        keep_cols.append("_symbol")

    df_features = df_features[keep_cols].copy()

    return df_features, feature_cols, engineer
