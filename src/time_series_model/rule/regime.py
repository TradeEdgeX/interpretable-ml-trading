"""
[DEPRECATED] Physics / Regime Classifier

⚠️ DEPRECATED: Regime classification has been migrated to gate rules in execution_archetypes.yaml.
Physical features (path_efficiency_pct, jump_risk_pct, etc.) are now computed in FeatureStore
and checked directly by gate rules. This module is kept for backward compatibility and diagnostics only.

This module implements a minimal Physics/Regime classifier that determines
which execution regime the market is currently in, based on statistical
constraints on price path feasibility.

Regime types:
- TC_REGIME: Trend Continuation regime (low noise, stable)
- TE_REGIME: Trend Expansion regime (volatility expansion, range expansion)
- MEAN_REGIME: Extreme Mean Reversion regime (extreme dislocations only) [PROXY V0 - NOT production-ready]
- ET_REGIME: Exhaustion Turn regime (trend late stage, high volatility, orderflow active) [NEW]
- NO_TRADE: No viable execution regime (microstructure unmodelable zone)

Key principle: Regime determines "feasibility", not "direction" or "profitability".

⚠️ IMPORTANT WARNINGS:
1. dir_conf is only a weak signal in Regime classification, not the main axis
2. MEAN_REGIME is PROXY V0 and should NOT be enabled in production execution
3. This module is DEPRECATED - use gate rules in execution_archetypes.yaml instead
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal
import numpy as np
import pandas as pd


RegimeType = Literal["TC_REGIME", "TE_REGIME", "MEAN_REGIME", "ET_REGIME", "NO_TRADE"]


@dataclass(frozen=True)
class PhysicsRegimeConfig:
    """
    Configuration for Physics/Regime classification.

    Uses existing features + head outputs to determine execution feasibility.
    """

    # Input columns
    pred_dir_prob_col: str = "pred_dir_prob"
    atr_col: str = "atr"
    atr_percentile_col: str = "atr_percentile"
    high_col: str = "high"
    low_col: str = "low"
    close_col: str = "close"

    # Feature columns (optional, computed if not present)
    atr_slope_window: int = 20
    range_expansion_window: int = 10
    jump_risk_window: int = 10

    # TC Regime constraints
    # ⚠️ IMPORTANT: dir_conf is only a weak signal, not the main axis
    # Main conditions should be: atr_slope, jump_risk_percentile, path_length, dir_conf stability
    tc_dir_conf_min: float = 0.0  # Do not require high dir_conf (stability > magnitude)
    tc_dir_conf_std_max_pct: float = 0.6  # Percentile for stability (lower std)
    tc_dir_sign_consistency_min_pct: float = 0.6  # Percentile for direction consistency
    tc_atr_slope_max_pct: float = 0.6  # Percentile max for low vol expansion
    tc_path_length_min_pct: float = 0.4  # Percentile for path length (longer paths)

    # TE Regime constraints
    # dir_conf is only for stability, not primary
    te_dir_conf_std_max_pct: float = 0.7  # Allow more direction variability
    te_dir_sign_consistency_min_pct: float = 0.5
    te_atr_slope_min_pct: float = 0.6  # Percentile min for vol expanding
    te_range_expansion_min_pct: float = 0.6  # Percentile min for range expansion

    # Physics feasibility score threshold (percentile-based)
    physics_score_min_pct: float = 0.9  # Top 10% of physics_score

    # MEAN Regime constraints (optimized for FR/ET mean reversion)
    # ⚠️ PROXY V0: Current definition is not fully physical
    # True MEAN_REGIME requires: price path statistically "unsustainable"
    # TODO: Add distance-to-anchor (z-score), path length > limit, liquidity vacuum
    # NOTE: Optimized based on FR/ET analysis (2026-01-22)
    # Analysis shows profitable FR/ET have:
    # - path_efficiency_pct <= 0.4 (low efficiency, choppy paths)
    # - price_dir_consistency_pct <= 0.5 (unstable direction)
    # - deviation_z_abs_pct >= 0.6 (high deviation from mean)
    # - jump_risk_pct <= 0.3 (low jump risk)
    # ⚠️ IMPORTANT: All thresholds are percentile-based for consistency
    mean_deviation_window: int = 200
    mean_deviation_z_abs_min_pct: float = (
        0.5  # Optimized: top 50% deviation (relaxed from 0.6 to increase samples)
    )
    mean_path_length_min_pct: float = (
        0.5  # Optimized: top 50% path length (was 0.7, too strict)
    )
    mean_dir_sign_consistency_max_pct: float = (
        0.5  # Keep: unstable direction (bottom 50%)
    )
    # NEW: Path efficiency constraint (low efficiency = choppy/mean-reverting)
    mean_path_efficiency_max_pct: float = (
        0.5  # Bottom 50% efficiency (relaxed from 0.4 to increase samples)
    )
    # NEW: Price direction consistency (based on actual returns, not pred_dir_prob)
    mean_price_dir_consistency_max_pct: float = (
        0.5  # Bottom 50% consistency (unstable direction)
    )
    mean_atr_percentile_min: float = 0.5  # Optimized: relaxed from 0.8 (was too strict)
    mean_dir_conf_max: float = 0.4  # Keep: weaker constraint, not primary
    mean_vol_spike_min: float = 2.0  # Simplified proxy
    # NEW: Jump risk constraint (low jump risk for mean reversion)
    mean_jump_risk_max_pct: float = (
        0.4  # Bottom 40% jump risk (relaxed from 0.3 to increase samples)
    )

    # Jump risk percentile bands (relative, not absolute)
    # These define the regime layers instead of an absolute veto.
    jump_risk_no_trade_pct: float = 0.9  # Top 10% -> NO_TRADE
    jump_risk_te_min_pct: float = 0.6
    jump_risk_te_max_pct: float = 0.9
    jump_risk_tc_min_pct: float = 0.3
    jump_risk_tc_max_pct: float = 0.6
    jump_risk_mean_max_pct: float = (
        0.4  # Bottom 40% -> MEAN/IDLE candidate (relaxed from 0.3)
    )

    # ET Regime constraints (Exhaustion Turn: trend late stage)
    # ET occurs at trend late stage, requiring:
    # - Moderate trend strength (ADX: 20-30)
    # - High volatility (atr_percentile > 0.85) - Optimized: higher volatility for better Sharpe
    # - Active orderflow (vpin quantile > 0.5)
    # - Near key levels (sr_distance_normalized < 0.3)
    # - Momentum divergence (cvd_change_5 < 0)
    # - Higher path efficiency (0.55-0.7) - Optimized: higher efficiency for better Sharpe
    # - Lower jump risk (0.2-0.5) - Optimized: lower risk for better Sharpe
    et_adx_min: float = 20.0  # Minimum ADX for trend strength
    et_adx_max: float = 30.0  # Maximum ADX (not too strong)
    et_atr_percentile_min: float = 0.85  # High volatility (optimized from 0.8)
    et_path_efficiency_min_pct: float = (
        0.55  # Minimum path efficiency (optimized from 0.4)
    )
    et_path_efficiency_max_pct: float = (
        0.7  # Maximum path efficiency (optimized from 0.6)
    )
    et_jump_risk_min_pct: float = 0.2  # Minimum jump risk (optimized from 0.3)
    et_jump_risk_max_pct: float = 0.5  # Maximum jump risk (optimized from 0.6)

    # Physics v2 (Recall-first) hard veto
    hard_jump_risk_pct: float = 0.98  # Extreme jump-only veto
    hard_atr_percentile_min: float = 0.2  # Exclude dead/illiquid regime

    # Regime strategy
    # - "score_shape": use physics_score + shape constraints (v1.x)
    # - "simple_band": use jump_risk bands only (v2.1 recall-first)
    regime_strategy: str = "simple_band"

    # Missing handling
    missing_default_false: bool = True


def _dir_conf(p: np.ndarray) -> np.ndarray:
    """Convert pred_dir_prob [0,1] to dir_conf [0,1]."""
    return np.clip(np.abs(p - 0.5) * 2.0, 0.0, 1.0)


def _compute_atr_slope(
    atr: np.ndarray,
    window: int = 20,
) -> np.ndarray:
    """Compute ATR slope (rate of change)."""
    if len(atr) < window:
        return np.full(len(atr), np.nan)

    # Simple linear regression slope over window
    slopes = np.full(len(atr), np.nan)
    x = np.arange(window)

    for i in range(window - 1, len(atr)):
        y = atr[i - window + 1 : i + 1]
        if not np.all(np.isnan(y)):
            valid_mask = ~np.isnan(y)
            if valid_mask.sum() >= 3:
                x_valid = x[valid_mask]
                y_valid = y[valid_mask]
                # Linear regression slope
                n = len(x_valid)
                sum_x = x_valid.sum()
                sum_y = y_valid.sum()
                sum_xy = (x_valid * y_valid).sum()
                sum_x2 = (x_valid**2).sum()

                denominator = n * sum_x2 - sum_x**2
                if abs(denominator) > 1e-9:
                    slope = (n * sum_xy - sum_x * sum_y) / denominator
                    slopes[i] = slope

    return slopes / (atr + 1e-9)  # Normalize by ATR level


def _compute_range_expansion(
    high: np.ndarray,
    low: np.ndarray,
    atr: np.ndarray,
    window: int = 10,
) -> np.ndarray:
    """Compute range expansion ratio (current range / ATR)."""
    if len(high) < window:
        return np.full(len(high), np.nan)

    ranges = (high - low) / (atr + 1e-9)
    expansion = np.full(len(ranges), np.nan)

    for i in range(window - 1, len(ranges)):
        window_ranges = ranges[i - window + 1 : i + 1]
        if not np.all(np.isnan(window_ranges)):
            recent = window_ranges[-3:]  # Last 3 bars
            historical = window_ranges[:-3]  # Earlier bars
            if len(historical) > 0 and not np.all(np.isnan(historical)):
                recent_mean = np.nanmean(recent)
                historical_mean = np.nanmean(historical)
                if historical_mean > 0:
                    expansion[i] = recent_mean / historical_mean

    return expansion


def _compute_jump_risk(
    close: np.ndarray,
    atr: np.ndarray,
    window: int = 10,
) -> np.ndarray:
    """Compute jump risk (max abs return / std of returns)."""
    if len(close) < window:
        return np.full(len(close), np.nan)

    returns = np.diff(close) / (close[:-1] + 1e-9)
    jump_risk = np.full(len(close), np.nan)

    for i in range(window, len(close)):
        window_returns = returns[i - window : i]
        if not np.all(np.isnan(window_returns)):
            max_abs_ret = np.nanmax(np.abs(window_returns))
            std_ret = np.nanstd(window_returns)
            if std_ret > 1e-9:
                jump_risk[i] = max_abs_ret / std_ret

    return jump_risk


def _rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    if len(arr) < window:
        return np.full(len(arr), np.nan)
    s = pd.Series(arr)
    return s.rolling(window=window, min_periods=window).std().to_numpy()


def _compute_price_direction_consistency(close: np.ndarray, window: int) -> np.ndarray:
    """
    Compute rolling consistency of actual price direction (sign of returns).

    This is based on actual price movements, NOT model predictions.
    Returns values in [0,1] where 1 = perfectly consistent direction, 0 = flipping.
    """
    if len(close) < 2:
        return np.full(len(close), np.nan)
    if len(close) < window:
        return np.full(len(close), np.nan)

    # Compute returns
    returns = np.diff(close, prepend=close[0])
    # Get sign of returns
    signs = np.sign(returns)
    s = pd.Series(signs)
    # abs(mean(sign)) = 1 if stable direction, ~0 if flipping
    return s.rolling(window=window, min_periods=window).mean().abs().to_numpy()


def _compute_path_efficiency(
    close: np.ndarray, atr: np.ndarray, window: int
) -> np.ndarray:
    """
    Compute path efficiency: net_displacement / total_path_length.

    Path efficiency measures how efficiently price moves in a direction:
    - High efficiency (close to 1): price moves in one direction with little backtracking
    - Low efficiency (close to 0): price moves a lot but net displacement is small (choppy/mean-reverting)

    This is a pure physical feature based on price trajectory, independent of model predictions.
    """
    if len(close) < window:
        return np.full(len(close), np.nan)

    # Net displacement: absolute change in price over window
    net_displacement = np.full(len(close), np.nan)
    for i in range(window - 1, len(close)):
        start_price = close[i - window + 1]
        end_price = close[i]
        net_displacement[i] = abs(end_price - start_price)

    # Total path length: sum of absolute returns over window
    diffs = np.abs(np.diff(close, prepend=close[0]))
    atr_safe = atr + 1e-9
    path = diffs / atr_safe
    s = pd.Series(path)
    total_path_length = s.rolling(window=window, min_periods=window).sum().to_numpy()

    # Normalize net_displacement by ATR for comparability
    net_displacement_atr = net_displacement / atr_safe

    # Path efficiency: net_displacement / total_path_length
    # Avoid division by zero
    path_efficiency = np.full(len(close), np.nan)
    valid_mask = (
        (total_path_length > 1e-9)
        & np.isfinite(net_displacement_atr)
        & np.isfinite(total_path_length)
    )
    path_efficiency[valid_mask] = (
        net_displacement_atr[valid_mask] / total_path_length[valid_mask]
    )

    return path_efficiency


def _rolling_dir_sign_consistency(p: np.ndarray, window: int) -> np.ndarray:
    """
    DEPRECATED: This function uses pred_dir_prob (model prediction).
    Use _compute_price_direction_consistency instead, which is based on actual price movements.

    Rolling consistency of directional sign in [0,1].
    """
    if len(p) < window:
        return np.full(len(p), np.nan)
    sign = np.sign(p - 0.5)
    s = pd.Series(sign)
    # abs(mean(sign)) = 1 if stable, ~0 if flipping
    return s.rolling(window=window, min_periods=window).mean().abs().to_numpy()


def _percentile_rank(arr: np.ndarray) -> np.ndarray:
    """Percentile rank in [0,1], ignoring NaNs."""
    s = pd.Series(arr)
    return s.rank(pct=True).to_numpy()


def _compute_path_length(
    close: np.ndarray,
    atr: np.ndarray,
    window: int = 20,
) -> np.ndarray:
    """Rolling path length in ATR units (sum of abs returns / ATR)."""
    if len(close) < window:
        return np.full(len(close), np.nan)
    diffs = np.abs(np.diff(close, prepend=close[0]))
    atr_safe = atr + 1e-9
    path = diffs / atr_safe
    s = pd.Series(path)
    return s.rolling(window=window, min_periods=window).sum().to_numpy()


def _compute_deviation_z(
    close: np.ndarray,
    window: int,
) -> np.ndarray:
    """Rolling z-score of close relative to local mean/std."""
    if len(close) < window:
        return np.full(len(close), np.nan)
    s = pd.Series(close)
    mean = s.rolling(window=window, min_periods=window).mean()
    std = s.rolling(window=window, min_periods=window).std()
    z = (s - mean) / (std + 1e-9)
    return z.to_numpy()


def classify_regime(
    df: pd.DataFrame,
    *,
    cfg: PhysicsRegimeConfig = PhysicsRegimeConfig(),
) -> pd.DataFrame:
    """
    [DEPRECATED] Classify Physics/Regime for each timestamp.

    ⚠️ DEPRECATED: Regime classification has been migrated to gate rules.
    Physical features are now computed in FeatureStore and checked directly by gate rules
    in execution_archetypes.yaml. This function is kept for backward compatibility and diagnostics only.

    Returns DataFrame with 'regime' column containing:
    - TC_REGIME: Trend Continuation regime
    - TE_REGIME: Trend Expansion regime
    - MEAN_REGIME: Extreme Mean Reversion regime
    - ET_REGIME: Exhaustion Turn regime (trend late stage)
    - NO_TRADE: No viable execution regime

    Args:
        df: DataFrame with required features and head outputs
        cfg: Configuration for regime classification

    Returns:
        DataFrame with 'regime' column added
    """
    out = pd.DataFrame(index=df.index)

    # Extract required columns
    def _get_col(col: str, default: Optional[float] = None) -> np.ndarray:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        if cfg.missing_default_false and default is None:
            return np.full(len(df), np.nan)
        return np.full(len(df), default if default is not None else 0.0)

    def _has_col(col: str) -> bool:
        """Check if column exists in DataFrame."""
        return col in df.columns

    pred_dir_prob = _get_col(cfg.pred_dir_prob_col)
    atr = _get_col(cfg.atr_col, default=1.0)
    atr_percentile = _get_col(cfg.atr_percentile_col)
    high = _get_col(cfg.high_col)
    low = _get_col(cfg.low_col)
    close = _get_col(cfg.close_col)

    # Compute dir_conf from pred_dir_prob
    pred_dir_prob_clipped = np.clip(pred_dir_prob, 0.0, 1.0)
    dir_conf = _dir_conf(pred_dir_prob_clipped)

    # Compute physics features
    # Prefer features from FeatureStore if available (computed in feature_dependencies.yaml)
    # Only compute if not present in input DataFrame
    if _has_col("atr_slope"):
        atr_slope = _get_col("atr_slope")
    else:
        atr_slope = _compute_atr_slope(atr, window=cfg.atr_slope_window)
    if _has_col("atr_slope_pct"):
        atr_slope_pct = _get_col("atr_slope_pct")
    else:
        atr_slope_pct = _percentile_rank(atr_slope)

    if _has_col("range_expansion"):
        range_expansion = _get_col("range_expansion")
    else:
        range_expansion = _compute_range_expansion(
            high, low, atr, window=cfg.range_expansion_window
        )
    if _has_col("range_expansion_pct"):
        range_expansion_pct = _get_col("range_expansion_pct")
    else:
        range_expansion_pct = _percentile_rank(range_expansion)

    if _has_col("jump_risk"):
        jump_risk = _get_col("jump_risk")
    else:
        jump_risk = _compute_jump_risk(close, atr, window=cfg.jump_risk_window)
    if _has_col("jump_risk_pct"):
        jump_risk_pct = _get_col("jump_risk_pct")
    else:
        jump_risk_pct = _percentile_rank(jump_risk)

    if _has_col("dir_conf_std"):
        dir_conf_std = _get_col("dir_conf_std")
    else:
        dir_conf_std = _rolling_std(dir_conf, window=cfg.atr_slope_window)
    if _has_col("dir_conf_std_pct"):
        dir_conf_std_pct = _get_col("dir_conf_std_pct")
    else:
        dir_conf_std_pct = _percentile_rank(dir_conf_std)

    if _has_col("dir_sign_consistency"):
        dir_sign_consistency = _get_col("dir_sign_consistency")
    else:
        dir_sign_consistency = _rolling_dir_sign_consistency(
            pred_dir_prob_clipped, window=cfg.atr_slope_window
        )
    if _has_col("dir_sign_consistency_pct"):
        dir_sign_consistency_pct = _get_col("dir_sign_consistency_pct")
    else:
        dir_sign_consistency_pct = _percentile_rank(dir_sign_consistency)

    if _has_col("path_length"):
        path_length = _get_col("path_length")
    else:
        path_length = _compute_path_length(close, atr, window=cfg.atr_slope_window)
    if _has_col("path_length_pct"):
        path_length_pct = _get_col("path_length_pct")
    else:
        path_length_pct = _percentile_rank(path_length)

    # NEW: Path efficiency (net_displacement / total_path_length)
    # Prefer from FeatureStore if available
    if _has_col("path_efficiency"):
        path_efficiency = _get_col("path_efficiency")
    else:
        path_efficiency = _compute_path_efficiency(
            close, atr, window=cfg.atr_slope_window
        )
    if _has_col("path_efficiency_pct"):
        path_efficiency_pct = _get_col("path_efficiency_pct")
    else:
        path_efficiency_pct = _percentile_rank(path_efficiency)

    # NEW: Price direction consistency (based on actual returns, NOT pred_dir_prob)
    # Prefer from FeatureStore if available
    if _has_col("price_dir_consistency"):
        price_dir_consistency = _get_col("price_dir_consistency")
    else:
        price_dir_consistency = _compute_price_direction_consistency(
            close, window=cfg.atr_slope_window
        )
    if _has_col("price_dir_consistency_pct"):
        price_dir_consistency_pct = _get_col("price_dir_consistency_pct")
    else:
        price_dir_consistency_pct = _percentile_rank(price_dir_consistency)

    if _has_col("deviation_z"):
        deviation_z = _get_col("deviation_z")
    else:
        deviation_z = _compute_deviation_z(close, window=cfg.mean_deviation_window)
    deviation_z_abs = np.abs(deviation_z)
    if _has_col("deviation_z_abs_pct"):
        deviation_z_abs_pct = _get_col("deviation_z_abs_pct")
    else:
        deviation_z_abs_pct = _percentile_rank(
            deviation_z_abs
        )  # Convert to percentile for consistency

    # Compute percentile versions of order flow and volume features for gate rules
    # Prefer from FeatureStore if available (computed in feature_dependencies.yaml)
    # These features are used in gate rules but are not percentile-based by default
    cvd_change_5 = _get_col("cvd_change_5", default=np.nan)
    if _has_col("cvd_change_5_pct"):
        cvd_change_5_pct = _get_col("cvd_change_5_pct")
    else:
        cvd_change_5_pct = (
            _percentile_rank(cvd_change_5)
            if not np.all(np.isnan(cvd_change_5))
            else np.full(len(df), np.nan)
        )

    volume_ratio = _get_col("volume_ratio", default=np.nan)
    if _has_col("volume_ratio_pct"):
        volume_ratio_pct = _get_col("volume_ratio_pct")
    else:
        volume_ratio_pct = (
            _percentile_rank(volume_ratio)
            if not np.all(np.isnan(volume_ratio))
            else np.full(len(df), np.nan)
        )

    bb_width_normalized = _get_col("bb_width_normalized", default=np.nan)
    if _has_col("bb_width_normalized_pct"):
        bb_width_normalized_pct = _get_col("bb_width_normalized_pct")
    else:
        bb_width_normalized_pct = (
            _percentile_rank(bb_width_normalized)
            if not np.all(np.isnan(bb_width_normalized))
            else np.full(len(df), np.nan)
        )

    cvd_change_5_normalized = _get_col("cvd_change_5_normalized", default=np.nan)
    if _has_col("cvd_change_5_normalized_pct"):
        cvd_change_5_normalized_pct = _get_col("cvd_change_5_normalized_pct")
    else:
        cvd_change_5_normalized_pct = (
            _percentile_rank(cvd_change_5_normalized)
            if not np.all(np.isnan(cvd_change_5_normalized))
            else np.full(len(df), np.nan)
        )

    # Initialize regime as NO_TRADE
    regime = np.full(len(df), "NO_TRADE", dtype=object)

    # Physics v2 hard veto (recall-first: only kill extremes)
    hard_veto = (
        ~np.isnan(jump_risk_pct) & (jump_risk_pct >= cfg.hard_jump_risk_pct)
    ) | (~np.isnan(atr_percentile) & (atr_percentile < cfg.hard_atr_percentile_min))

    # Relative jump risk bands (percentile-based)
    no_trade_band = ~np.isnan(jump_risk_pct) & (
        jump_risk_pct >= cfg.jump_risk_no_trade_pct
    )
    te_band = (
        ~np.isnan(jump_risk_pct)
        & (jump_risk_pct >= cfg.jump_risk_te_min_pct)
        & (jump_risk_pct < cfg.jump_risk_te_max_pct)
    )
    tc_band = (
        ~np.isnan(jump_risk_pct)
        & (jump_risk_pct >= cfg.jump_risk_tc_min_pct)
        & (jump_risk_pct < cfg.jump_risk_tc_max_pct)
    )
    mean_band = ~np.isnan(jump_risk_pct) & (jump_risk_pct < cfg.jump_risk_mean_max_pct)

    # Physics feasibility score (soft-min proxy)
    # UPDATED: Use physical features only (path_efficiency, price_dir_consistency)
    # instead of pred_dir_prob-based features (dir_conf_std, dir_sign_consistency)
    physics_inputs = np.stack(
        [
            1.0 - jump_risk_pct,  # Low jump risk
            1.0 - atr_slope_pct,  # Low volatility expansion
            path_length_pct,  # Long path
            path_efficiency_pct,  # High path efficiency (NEW: replaces dir_conf_std)
            price_dir_consistency_pct,  # Stable price direction (NEW: replaces dir_sign_consistency)
        ],
        axis=0,
    )
    physics_score = np.full(len(df), np.nan)
    valid_mask = ~np.all(np.isnan(physics_inputs), axis=0)
    if valid_mask.any():
        physics_score[valid_mask] = np.nanmin(physics_inputs[:, valid_mask], axis=0)
    physics_score_pct = _percentile_rank(physics_score)

    # ET Regime: Exhaustion Turn (trend late stage)
    # Check ET_REGIME BEFORE TC/TE to ensure ET-specific conditions take priority
    # ET occurs when:
    # - Lower jump risk (0.2-0.5) - Optimized: lower risk for better Sharpe
    # - Higher volatility (atr_percentile >= 0.85) - Optimized: higher volatility for better Sharpe
    # - Higher path efficiency (0.55-0.7) - Optimized: higher efficiency for better Sharpe
    # - Path length sufficient (>= 0.6) - Optimized: higher path length requirement
    # Note: ADX, vpin, sr_distance, cvd_change_5 are checked in gate rules, not here
    et_band = (
        ~np.isnan(jump_risk_pct)
        & (jump_risk_pct >= cfg.et_jump_risk_min_pct)
        & (jump_risk_pct < cfg.et_jump_risk_max_pct)
    )
    et_physical_ok = (
        (~np.isnan(atr_percentile) & (atr_percentile >= cfg.et_atr_percentile_min))
        & (
            ~np.isnan(path_efficiency_pct)
            & (path_efficiency_pct >= cfg.et_path_efficiency_min_pct)
            & (path_efficiency_pct <= cfg.et_path_efficiency_max_pct)
        )
        & (
            ~np.isnan(path_length_pct)
            & (
                path_length_pct >= 0.6
            )  # Optimized: higher path length requirement (0.5 -> 0.6)
        )
    )
    # ET_REGIME should be checked before TC_REGIME (more specific conditions)
    # Priority: TE > ET > TC > MEAN
    et_mask = (
        et_physical_ok
        & et_band
        & (regime == "NO_TRADE")  # Only assign if not already TE
        & (~hard_veto)
    )
    regime[et_mask] = "ET_REGIME"

    if cfg.regime_strategy == "simple_band":
        # v2.1 recall-first: regime by jump_risk bands only
        # But exclude samples already classified as ET_REGIME
        tc_mask = tc_band & (~hard_veto) & (regime != "ET_REGIME")
        regime[tc_mask] = "TC_REGIME"
        regime[te_band & (~hard_veto)] = "TE_REGIME"
    else:
        # score_shape strategy (v1.x)
        tc_shape_ok = (
            ~np.isnan(atr_slope_pct) & (atr_slope_pct < cfg.tc_atr_slope_max_pct)
        ) & (
            ~np.isnan(path_length_pct) & (path_length_pct >= cfg.tc_path_length_min_pct)
        )
        tc_mask = (
            (
                ~np.isnan(physics_score_pct)
                & (physics_score_pct >= cfg.physics_score_min_pct)
            )
            & tc_shape_ok
            & tc_band
            & (regime == "NO_TRADE")
            & (~hard_veto)
        )
        # Exclude samples already classified as ET_REGIME
        tc_mask = tc_mask & (regime != "ET_REGIME")
        regime[tc_mask] = "TC_REGIME"

        te_feasible_inputs = np.stack(
            [
                1.0 - atr_slope_pct,
                range_expansion_pct,
                path_length_pct,
                1.0 - dir_conf_std_pct,
                dir_sign_consistency_pct,
            ],
            axis=0,
        )
        te_score = np.full(len(df), np.nan)
        te_valid_mask = ~np.all(np.isnan(te_feasible_inputs), axis=0)
        if te_valid_mask.any():
            te_score[te_valid_mask] = np.nanmin(
                te_feasible_inputs[:, te_valid_mask], axis=0
            )
        te_score_pct = _percentile_rank(te_score)

        te_shape_ok = (
            ~np.isnan(atr_slope_pct) & (atr_slope_pct >= cfg.te_atr_slope_min_pct)
        ) & (
            ~np.isnan(range_expansion_pct)
            & (range_expansion_pct >= cfg.te_range_expansion_min_pct)
        )
        te_mask = (
            (~np.isnan(te_score_pct) & (te_score_pct >= cfg.physics_score_min_pct))
            & te_shape_ok
            & te_band
            & (regime == "NO_TRADE")
            & (~hard_veto)
        )
        regime[te_mask] = "TE_REGIME"

    # MEAN Regime: Extreme dislocations only
    # ⚠️ PROXY V0: Current definition is not fully physical
    # True MEAN_REGIME requires:
    # - distance-to-anchor extreme (z-score) - TODO
    # - unidirectional path_length > reasonable limit - TODO
    # - local liquidity vacuum + quick refill - TODO
    # Current implementation is a simplified proxy
    # NOTE: This should NOT be enabled in production execution
    # ⚠️ IMPORTANT: All thresholds are percentile-based for consistency with TC/TE
    # Optimized conditions based on profitable FR/ET analysis:
    # 1. High deviation from mean (deviation_z_abs_pct >= 0.6)
    # 2. Low path efficiency (path_efficiency_pct <= 0.4) - choppy/mean-reverting
    # 3. Unstable direction (price_dir_consistency_pct <= 0.5)
    # 4. Low jump risk (jump_risk_pct <= 0.3)
    # 5. Moderate path length and ATR (relaxed constraints)
    mean_physical_ok = (
        (
            ~np.isnan(deviation_z_abs_pct)
            & (deviation_z_abs_pct >= cfg.mean_deviation_z_abs_min_pct)
        )
        & (
            ~np.isnan(path_length_pct)
            & (path_length_pct >= cfg.mean_path_length_min_pct)
        )
        & (
            ~np.isnan(price_dir_consistency_pct)
            & (price_dir_consistency_pct <= cfg.mean_price_dir_consistency_max_pct)
        )
        & (~np.isnan(atr_percentile) & (atr_percentile >= cfg.mean_atr_percentile_min))
        & (
            ~np.isnan(path_efficiency_pct)
            & (path_efficiency_pct <= cfg.mean_path_efficiency_max_pct)
        )
        & (~np.isnan(jump_risk_pct) & (jump_risk_pct <= cfg.mean_jump_risk_max_pct))
    )
    # ET Regime: Exhaustion Turn (trend late stage)
    # ET occurs when:
    # - Moderate jump risk (0.3-0.6) - not too low (TC) or too high (TE/NO_TRADE)
    # - High volatility (atr_percentile > 0.8)
    # - Moderate path efficiency (0.4-0.6) - not too high (TC) or too low (MEAN)
    # - Path length sufficient (>= 0.5)
    # Note: ADX, vpin, sr_distance, cvd_change_5 are checked in gate rules, not here
    # Priority: ET should be checked BEFORE TC, because ET has more specific conditions
    et_band = (
        ~np.isnan(jump_risk_pct)
        & (jump_risk_pct >= cfg.et_jump_risk_min_pct)
        & (jump_risk_pct < cfg.et_jump_risk_max_pct)
    )
    et_physical_ok = (
        (~np.isnan(atr_percentile) & (atr_percentile >= cfg.et_atr_percentile_min))
        & (
            ~np.isnan(path_efficiency_pct)
            & (path_efficiency_pct >= cfg.et_path_efficiency_min_pct)
            & (path_efficiency_pct <= cfg.et_path_efficiency_max_pct)
        )
        & (
            ~np.isnan(path_length_pct)
            & (path_length_pct >= 0.5)  # Sufficient path length
        )
    )
    # ET_REGIME should be checked before TC_REGIME (more specific conditions)
    # Priority: TE > ET > TC > MEAN
    et_mask = (
        et_physical_ok
        & et_band
        & (regime == "NO_TRADE")  # Only assign if not already TE
        & (~hard_veto)
    )
    regime[et_mask] = "ET_REGIME"

    mean_mask = mean_physical_ok & mean_band & (regime == "NO_TRADE") & (~hard_veto)
    regime[mean_mask] = "MEAN_REGIME"

    out["regime"] = regime

    # Also output intermediate features for diagnostics
    # NEW: Physical features (based on actual price movements)
    out["path_efficiency"] = path_efficiency
    out["path_efficiency_pct"] = path_efficiency_pct
    out["price_dir_consistency"] = price_dir_consistency
    out["price_dir_consistency_pct"] = price_dir_consistency_pct

    # Percentile versions of order flow and volume features for gate rules
    out["cvd_change_5_pct"] = cvd_change_5_pct
    out["volume_ratio_pct"] = volume_ratio_pct
    out["bb_width_normalized_pct"] = bb_width_normalized_pct
    out["cvd_change_5_normalized_pct"] = cvd_change_5_normalized_pct

    # DEPRECATED: Keep for backward compatibility but should not be used
    out["dir_conf"] = dir_conf
    out["dir_conf_std"] = dir_conf_std
    out["dir_conf_std_pct"] = dir_conf_std_pct
    out["dir_sign_consistency"] = dir_sign_consistency
    out["dir_sign_consistency_pct"] = dir_sign_consistency_pct
    out["atr_slope"] = atr_slope
    out["atr_slope_pct"] = atr_slope_pct
    out["range_expansion"] = range_expansion
    out["range_expansion_pct"] = range_expansion_pct
    out["jump_risk"] = jump_risk
    out["jump_risk_pct"] = jump_risk_pct
    out["path_length"] = path_length
    out["path_length_pct"] = path_length_pct
    out["physics_score"] = physics_score
    out["physics_score_pct"] = physics_score_pct
    if cfg.regime_strategy == "simple_band":
        out["te_score"] = np.full(len(df), np.nan)
        out["te_score_pct"] = np.full(len(df), np.nan)
    else:
        out["te_score"] = te_score
        out["te_score_pct"] = te_score_pct
    out["deviation_z"] = deviation_z
    out["deviation_z_abs"] = deviation_z_abs
    out["deviation_z_abs_pct"] = deviation_z_abs_pct
    out["hard_veto"] = hard_veto

    # Semantic scores (do NOT affect regime assignment; consumed by Gate)
    tc_semantic_inputs = np.stack(
        [
            1.0 - atr_slope_pct,
            path_length_pct,
            1.0 - dir_conf_std_pct,
            dir_sign_consistency_pct,
        ],
        axis=0,
    )
    tc_semantic_score = np.full(len(df), np.nan)
    tc_sem_valid = ~np.all(np.isnan(tc_semantic_inputs), axis=0)
    if tc_sem_valid.any():
        tc_semantic_score[tc_sem_valid] = np.nanmin(
            tc_semantic_inputs[:, tc_sem_valid], axis=0
        )

    te_semantic_inputs = np.stack(
        [
            atr_slope_pct,  # High volatility expansion
            range_expansion_pct,  # Range expansion
            path_length_pct,  # Long path
            path_efficiency_pct,  # High path efficiency (NEW: replaces dir_conf_std)
            price_dir_consistency_pct,  # Stable price direction (NEW: replaces dir_sign_consistency)
        ],
        axis=0,
    )
    te_semantic_score = np.full(len(df), np.nan)
    te_sem_valid = ~np.all(np.isnan(te_semantic_inputs), axis=0)
    if te_sem_valid.any():
        te_semantic_score[te_sem_valid] = np.nanmin(
            te_semantic_inputs[:, te_sem_valid], axis=0
        )

    out["tc_semantic_score"] = tc_semantic_score
    out["te_semantic_score"] = te_semantic_score

    # FR/ET semantic scores (for MEAN_REGIME archetypes)
    # FR: Failure Reversion - extreme deviation + unstable direction + overextended path
    fr_semantic_inputs = np.stack(
        [
            np.clip(deviation_z_abs / 5.0, 0.0, 1.0),  # Normalize z-score to [0,1]
            1.0
            - price_dir_consistency_pct,  # Direction instability (lower consistency = better for FR)
            path_length_pct,
            atr_percentile,
        ],
        axis=0,
    )
    fr_semantic_score = np.full(len(df), np.nan)
    fr_sem_valid = ~np.all(np.isnan(fr_semantic_inputs), axis=0)
    if fr_sem_valid.any():
        fr_semantic_score[fr_sem_valid] = np.nanmin(
            fr_semantic_inputs[:, fr_sem_valid], axis=0
        )

    # ET: Exhaustion Turn - volatility spike + overextended path + unstable direction
    et_semantic_inputs = np.stack(
        [
            atr_percentile,
            path_length_pct,
            1.0 - price_dir_consistency_pct,  # Direction instability
            np.clip(deviation_z_abs / 5.0, 0.0, 1.0),  # Normalize z-score to [0,1]
        ],
        axis=0,
    )
    et_semantic_score = np.full(len(df), np.nan)
    et_sem_valid = ~np.all(np.isnan(et_semantic_inputs), axis=0)
    if et_sem_valid.any():
        et_semantic_score[et_sem_valid] = np.nanmin(
            et_semantic_inputs[:, et_sem_valid], axis=0
        )

    out["fr_semantic_score"] = fr_semantic_score
    out["et_semantic_score"] = et_semantic_score

    return out
