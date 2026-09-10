"""
Utilities for Rank IC evaluation and signal generation.

This module provides functions for:
- Rank IC (Spearman correlation) evaluation
- Dynamic prediction quantile calculation
- Confidence score computation
- Signal generation based on quantiles and confidence
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def compute_rank_ic(
    predictions: np.ndarray | pd.Series,
    true_returns: np.ndarray | pd.Series,
    mask: Optional[np.ndarray] = None,
) -> float:
    """
    Compute Rank IC (Spearman correlation) between predictions and true returns.

    Rank IC is the core metric for regression models in quant finance:
    - We care about ranking (who will perform better), not exact values
    - Rank IC > 0.03 (crypto) or > 0.02 (stocks) indicates predictive power

    Args:
        predictions: Model predictions
        true_returns: True future returns
        mask: Optional boolean mask to filter samples

    Returns:
        Rank IC (Spearman correlation coefficient), or 0.0 if insufficient samples
    """
    pred_series = (
        pd.Series(predictions)
        if not isinstance(predictions, pd.Series)
        else predictions
    )
    true_series = (
        pd.Series(true_returns)
        if not isinstance(true_returns, pd.Series)
        else true_returns
    )

    # Apply mask if provided
    if mask is not None:
        pred_series = pred_series[mask]
        true_series = true_series[mask]

    # Filter NaN values
    valid_mask = pred_series.notna() & true_series.notna()
    pred_valid = pred_series[valid_mask]
    true_valid = true_series[valid_mask]

    if len(pred_valid) < 10:
        return 0.0

    # Compute Spearman correlation
    ic, p_value = spearmanr(pred_valid.values, true_valid.values, nan_policy="omit")

    return float(ic) if not np.isnan(ic) else 0.0


def prediction_quantile(
    predictions: pd.Series,
    window: int = 30,
    min_periods: int = 10,
    asset_col: Optional[pd.Series] = None,
) -> pd.Series:
    """
    Compute dynamic prediction quantile: prediction's position in its recent distribution.

    This calculates what percentile the current prediction falls into relative to
    its recent rolling window, which is used for:
    - Signal generation (only trade extreme predictions)
    - Confidence scoring (extreme predictions = higher confidence)

    Args:
        predictions: Model predictions
        window: Rolling window size for quantile calculation
        min_periods: Minimum periods required
        asset_col: Optional asset identifier for multi-asset data

    Returns:
        Prediction quantile series (0-1), where 1 means current prediction is in top percentile
    """

    def _pred_quantile(group: pd.Series) -> pd.Series:
        """Compute quantile for a single asset/series."""
        return group.rolling(window=window, min_periods=min_periods).rank(pct=True)

    if asset_col is not None:
        # Multi-asset: compute quantile within each asset
        result = predictions.groupby(asset_col).apply(_pred_quantile)
        if isinstance(result.index, pd.MultiIndex):
            result = result.droplevel(0)
        result = result.reindex(predictions.index)
    else:
        # Single asset
        result = _pred_quantile(predictions)

    return result.rename("pred_quantile")


def confidence_score(
    pred_quantile: pd.Series,
) -> pd.Series:
    """
    Compute confidence score from prediction quantile.

    Confidence = |pred_quantile - 0.5| * 2

    This maps quantile distance from neutral (0.5) to [0, 1]:
    - pred_quantile = 0.5 (neutral) → confidence = 0
    - pred_quantile = 0.0 or 1.0 (extreme) → confidence = 1

    Args:
        pred_quantile: Prediction quantile series (0-1)

    Returns:
        Confidence score series (0-1), where 1 = highest confidence
    """
    confidence = (pred_quantile - 0.5).abs() * 2
    return confidence.rename("confidence_score")


def generate_trading_signals(
    pred_quantile: pd.Series,
    confidence_score: pd.Series,
    confidence_threshold: float = 0.85,
    long_threshold: float = 0.9,
    short_threshold: float = 0.1,
) -> pd.Series:
    """
    Generate trading signals based on prediction quantile and confidence.

    Signals:
    - Long (1): pred_quantile >= long_threshold AND confidence >= confidence_threshold
    - Short (-1): pred_quantile <= short_threshold AND confidence >= confidence_threshold
    - Hold (0): Otherwise

    This ensures we only trade when:
    1. Prediction is in extreme quantile (strong signal)
    2. Confidence is high (signal is reliable)

    Args:
        pred_quantile: Prediction quantile series (0-1)
        confidence_score: Confidence score series (0-1)
        confidence_threshold: Minimum confidence to trade (default: 0.85)
        long_threshold: Quantile threshold for Long signal (default: 0.9)
        short_threshold: Quantile threshold for Short signal (default: 0.1)

    Returns:
        Signal series: 1 (Long), -1 (Short), 0 (Hold)
    """
    signals = pd.Series(0, index=pred_quantile.index, dtype=int)

    # High confidence Long
    long_mask = (pred_quantile >= long_threshold) & (
        confidence_score >= confidence_threshold
    )
    signals.loc[long_mask] = 1

    # High confidence Short
    short_mask = (pred_quantile <= short_threshold) & (
        confidence_score >= confidence_threshold
    )
    signals.loc[short_mask] = -1

    return signals.rename("signal")
