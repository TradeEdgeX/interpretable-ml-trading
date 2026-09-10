"""
扩展波动率特征提取器
添加历史波动率、滞后特征、趋势特征等，提高波动率预测准确性

优化版本：
- 避免重复计算 vol_base，提升效率
- 使用高效斜率计算替代 np.polyfit
- 优化 percentile_rank 计算
- 改进特征命名规范
- 增强数值稳定性（价格清洗、异常值处理）

与其他模块的关系：
- 本模块：特征提取器，提取扩展波动率特征（extract_extended_volatility_features）
- volatility_model_config.py：波动率模型配置，内部调用本模块来获取基础波动率特征
- 关系：本模块被波动率模型训练流程使用，通过配置文件选择特征

WPT 波动率增强特征：
- enhance_wpt_vol_features()：从 WPT 特征中提取波动率相关衍生特征
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict

from src.features.registry import register_feature

# 导入 Volume Profile 相关类型
try:
    from scipy.stats import skew
except ImportError:
    # Fallback if scipy is not available
    def skew(x):
        return 0.0

from src.features.time_series.utils_volume_profile import (
    VolumeProfileResult,
    compute_wpt_volume_profile,
)


@register_feature("extract_extended_volatility_features", category="volatility")
def extract_extended_volatility_features(
    df: pd.DataFrame,
    price_col: str = "close",
    atr_col: str = "atr",
    window: int = 20,
    lag_periods: List[int] = None,
) -> pd.DataFrame:
    """
    提取扩展的波动率特征（优化版）
    
    Args:
        df: DataFrame with price data
        price_col: Column name for price
        atr_col: Column name for ATR
        window: Rolling window size for base volatility calculation
        lag_periods: List of lag periods for lag features (default: [1, 2, 3, 5, 10])
    
    Returns:
        DataFrame with extended volatility features:
        - vol_raw_*: Multi-scale historical volatility (rolling std of returns)
        - vol_atr_*: ATR-derived features (normalized, ratios, statistics)
        - vol_lag_*: Lag features of base volatility
        - vol_slope_*: Trend features (simple slope approximation, optimized)
        - vol_ma_*, vol_ema_*: Moving averages of base volatility
        - vol_zscore: Z-score of volatility (regime indicator)
        - vol_percentile_approx: Approximate percentile rank (fast approximation)
        - vol_range_*: Range and position features
        - vol_mom_*: Momentum features
    """
    if lag_periods is None:
        lag_periods = [1, 2, 3, 5, 10]

    # IMPORTANT: do NOT copy/mutate a potentially wide input df here.
    # The feature pipeline already passes a slim df when configured, and this
    # function should be read-only to avoid “wide table” memory blow-ups.
    if price_col not in df.columns:
        raise ValueError(f"Price column '{price_col}' not found")
    
    # Clean price data: clip to avoid zero/negative prices
    price = df[price_col].clip(lower=1e-8)
    
    # Calculate returns with fillna to handle edge cases
    returns = price.pct_change().fillna(0.0)
    
    # Initialize result DataFrame (only feature outputs live here)
    result = pd.DataFrame(index=df.index)
    
    # 1. Multi-scale historical volatility (raw volatility at different windows)
    for w in [5, 10, 20, 60]:
        result[f"vol_raw_{w}"] = returns.rolling(
            window=w, min_periods=max(1, w // 2)
        ).std()
    
    # 2. Base volatility (compute once, reuse throughout - KEY OPTIMIZATION)
    vol_base = returns.rolling(
        window=window, min_periods=max(1, window // 2)
    ).std()
    
    # 3. ATR-derived features
    if atr_col in df.columns:
        # More robust ATR handling: fill initial NaN with rolling mean, then ffill/bfill
        atr_raw = df[atr_col]
        # First, use rolling mean to fill initial NaN (ATR needs 14-day initialization)
        atr_filled = atr_raw.fillna(atr_raw.rolling(window=14, min_periods=1).mean())
        # Then forward fill and backward fill to handle any remaining NaN
        atr = atr_filled.ffill().bfill().clip(lower=1e-8)
        
        # ATR / Price ratio (normalized volatility)
        result["vol_atr_norm"] = atr / price
        
        # ATR rolling statistics
        for w in [5, 10, 20]:
            result[f"vol_atr_ma_{w}"] = atr.rolling(
                window=w, min_periods=max(1, w // 2)
            ).mean()
            result[f"vol_atr_std_{w}"] = atr.rolling(
                window=w, min_periods=max(1, w // 2)
            ).std()
            result[f"vol_atr_max_{w}"] = atr.rolling(
                window=w, min_periods=max(1, w // 2)
            ).max()
            result[f"vol_atr_min_{w}"] = atr.rolling(
                window=w, min_periods=max(1, w // 2)
            ).min()
        
        # ATR ratio (current / historical mean)
        atr_ma_20 = atr.rolling(window=20, min_periods=10).mean()
        result["vol_atr_ratio_20"] = atr / (atr_ma_20 + 1e-8)
        
        # ATR change rate
        result["vol_atr_change"] = atr.pct_change().fillna(0.0)
        result["vol_atr_change_abs"] = atr.diff().abs().fillna(0.0)
    
    # 4. Lag features of base volatility
    for lag in lag_periods:
        result[f"vol_lag_{lag}"] = vol_base.shift(lag)
    
    # 5. Trend features (simple slope approximation - much faster than polyfit)
    # OPTIMIZATION: Replaced np.polyfit with simple slope calculation
    # Simple slope = (current - past) / time_diff
    for w in [5, 10, 20]:
        # Approximate slope: (vol_base - vol_base.shift(w-1)) / (w-1)
        result[f"vol_slope_{w}"] = (
            vol_base - vol_base.shift(w - 1)
        ) / (w - 1 + 1e-8)
    
    # Volatility acceleration (derivative of slope)
    # Note: vol_slope_5 is always computed above (w=5 in [5,10,20]), so no need for if check
    result["vol_accel"] = result["vol_slope_5"].diff().fillna(0.0)
    
    # 6. Moving averages of base volatility
    for w in [5, 10, 20]:
        result[f"vol_ma_{w}"] = vol_base.rolling(
            window=w, min_periods=max(1, w // 2)
        ).mean()
        result[f"vol_ema_{w}"] = vol_base.ewm(
            span=w, min_periods=max(1, w // 2)
        ).mean()
    
    # 7. Volatility regime features
    vol_ma_20 = vol_base.rolling(window=20, min_periods=10).mean()
    vol_std_20 = vol_base.rolling(window=20, min_periods=10).std()
    
    # Z-score of volatility (regime indicator)
    # Note: baseline_features has volatility_zscore_* (based on volatility column),
    # but this vol_zscore is based on vol_base, so no conflict
    result["vol_zscore"] = (vol_base - vol_ma_20) / (vol_std_20 + 1e-8)
    
    # Approximate percentile rank (using z-score transformation for efficiency)
    # OPTIMIZATION: Replaced O(n²) percentile calculation with fast approximation
    # Note: Actual volatility distribution is right-skewed (fat tail, high peak)
    # Using piecewise linear approximation for better accuracy while maintaining O(1) speed
    vol_zscore_clipped = result["vol_zscore"].clip(-4, 4)
    # Piecewise linear: [-4,-2]→[0.01,0.1], [-2,0]→[0.1,0.5], [0,2]→[0.5,0.9], [2,4]→[0.9,0.99]
    # Alternative: Use sigmoid for smoother transition (1 / (1 + exp(-1.5 * z)))
    result["vol_percentile_approx"] = np.where(
        vol_zscore_clipped < -2,
        0.01 + 0.09 * (vol_zscore_clipped + 4) / 2,  # [-4,-2] → [0.01, 0.1]
        np.where(
            vol_zscore_clipped < 0,
            0.1 + 0.4 * (vol_zscore_clipped + 2) / 2,  # [-2,0] → [0.1, 0.5]
            np.where(
                vol_zscore_clipped < 2,
                0.5 + 0.4 * vol_zscore_clipped / 2,  # [0,2] → [0.5, 0.9]
                0.9 + 0.09 * (vol_zscore_clipped - 2) / 2  # [2,4] → [0.9, 0.99]
            )
        )
    )
    
    # Alternative: true percentile rank (slower but more accurate, use for offline training)
    # Uncomment if exact percentile is needed:
    # result["vol_percentile_rank"] = vol_base.rolling(window=60, min_periods=30).apply(
    #     lambda x: (x.iloc[-1] > x).sum() / len(x) if len(x) > 0 else 0.5,
    #     raw=False
    # )
    
    # 8. Volatility range features
    for w in [10, 20]:
        vol_roll = vol_base.rolling(window=w, min_periods=max(1, w // 2))
        vol_max = vol_roll.max()
        vol_min = vol_roll.min()
        vol_range = vol_max - vol_min
        result[f"vol_range_{w}"] = vol_range
        result[f"vol_range_pos_{w}"] = (vol_base - vol_min) / (vol_range + 1e-8)
    
    # 9. Volatility momentum features
    for w in [3, 5, 10]:
        result[f"vol_mom_{w}"] = vol_base / (vol_base.shift(w) + 1e-8) - 1.0
    
    # Final fill: forward fill then fill remaining NaN with 0
    result = result.ffill().fillna(0.0)
    
    return result


# =============================================================================
# Narrow-IO sub-features for Feature DAG (extended_volatility_features拆分)
# =============================================================================


def _compute_returns_from_close(close: pd.Series) -> pd.Series:
    close = pd.to_numeric(close, errors="coerce").astype(float)
    close = close.clip(lower=1e-8)
    return close.pct_change().fillna(0.0)


def _compute_vol_base(returns: pd.Series, window: int) -> pd.Series:
    return returns.rolling(window=window, min_periods=max(1, window // 2)).std()


@register_feature("compute_vol_raw_features_from_series", category="volatility")
def compute_vol_raw_features_from_series(
    *,
    close: pd.Series,
    windows: List[int] | None = None,
) -> pd.DataFrame:
    """vol_raw_*: multi-scale rolling std of returns."""
    if windows is None:
        windows = [5, 10, 20, 60]
    returns = _compute_returns_from_close(close)
    out = pd.DataFrame(index=close.index)
    for w in windows:
        out[f"vol_raw_{w}"] = returns.rolling(window=w, min_periods=max(1, w // 2)).std()
    return out


@register_feature("compute_vol_atr_features_from_series", category="volatility")
def compute_vol_atr_features_from_series(
    *,
    close: pd.Series,
    atr: pd.Series,
) -> pd.DataFrame:
    """vol_atr_*: ATR-derived volatility features."""
    price = pd.to_numeric(close, errors="coerce").astype(float).clip(lower=1e-8)
    atr_raw = pd.to_numeric(atr, errors="coerce").astype(float)
    atr_filled = atr_raw.fillna(atr_raw.rolling(window=14, min_periods=1).mean())
    atr_s = atr_filled.ffill().bfill().clip(lower=1e-8)

    out = pd.DataFrame(index=price.index)
    out["vol_atr_norm"] = atr_s / price

    for w in [5, 10, 20]:
        roll = atr_s.rolling(window=w, min_periods=max(1, w // 2))
        out[f"vol_atr_ma_{w}"] = roll.mean()
        out[f"vol_atr_std_{w}"] = roll.std()
        out[f"vol_atr_max_{w}"] = roll.max()
        out[f"vol_atr_min_{w}"] = roll.min()

    atr_ma_20 = atr_s.rolling(window=20, min_periods=10).mean()
    out["vol_atr_ratio_20"] = atr_s / (atr_ma_20 + 1e-8)
    out["vol_atr_change"] = atr_s.pct_change().fillna(0.0)
    out["vol_atr_change_abs"] = atr_s.diff().abs().fillna(0.0)
    return out


@register_feature("compute_vol_lag_features_from_series", category="volatility")
def compute_vol_lag_features_from_series(
    *,
    close: pd.Series,
    window: int = 20,
    lag_periods: List[int] | None = None,
) -> pd.DataFrame:
    """vol_lag_*: lags of vol_base."""
    if lag_periods is None:
        lag_periods = [1, 2, 3]
    returns = _compute_returns_from_close(close)
    vol_base = _compute_vol_base(returns, window=window)
    out = pd.DataFrame(index=close.index)
    for lag in lag_periods:
        out[f"vol_lag_{lag}"] = vol_base.shift(lag)
    return out


@register_feature("compute_vol_trend_features_from_series", category="volatility")
def compute_vol_trend_features_from_series(
    *,
    close: pd.Series,
    window: int = 20,
) -> pd.DataFrame:
    """vol_slope_* + vol_accel: trend/acceleration of vol_base."""
    returns = _compute_returns_from_close(close)
    vol_base = _compute_vol_base(returns, window=window)
    out = pd.DataFrame(index=close.index)
    for w in [5, 10, 20]:
        out[f"vol_slope_{w}"] = (vol_base - vol_base.shift(w - 1)) / (w - 1 + 1e-8)
    out["vol_accel"] = out["vol_slope_5"].diff().fillna(0.0)
    return out


@register_feature("compute_vol_ma_features_from_series", category="volatility")
def compute_vol_ma_features_from_series(
    *,
    close: pd.Series,
    window: int = 20,
) -> pd.DataFrame:
    """vol_ma_* + vol_ema_*: moving averages of vol_base."""
    returns = _compute_returns_from_close(close)
    vol_base = _compute_vol_base(returns, window=window)
    out = pd.DataFrame(index=close.index)
    for w in [5, 10, 20]:
        out[f"vol_ma_{w}"] = vol_base.rolling(window=w, min_periods=max(1, w // 2)).mean()
        out[f"vol_ema_{w}"] = vol_base.ewm(span=w, min_periods=max(1, w // 2)).mean()
    return out


@register_feature("compute_vol_regime_features_from_series", category="volatility")
def compute_vol_regime_features_from_series(
    *,
    close: pd.Series,
    window: int = 20,
) -> pd.DataFrame:
    """vol_zscore + vol_percentile_approx."""
    returns = _compute_returns_from_close(close)
    vol_base = _compute_vol_base(returns, window=window)
    out = pd.DataFrame(index=close.index)
    vol_ma_20 = vol_base.rolling(window=20, min_periods=10).mean()
    vol_std_20 = vol_base.rolling(window=20, min_periods=10).std()
    out["vol_zscore"] = (vol_base - vol_ma_20) / (vol_std_20 + 1e-8)

    z = out["vol_zscore"].clip(-4, 4)
    out["vol_percentile_approx"] = np.where(
        z < -2,
        0.01 + 0.09 * (z + 4) / 2,
        np.where(
            z < 0,
            0.1 + 0.4 * (z + 2) / 2,
            np.where(
                z < 2,
                0.5 + 0.4 * z / 2,
                0.9 + 0.09 * (z - 2) / 2,
            ),
        ),
    )
    return out


@register_feature("compute_vol_range_features_from_series", category="volatility")
def compute_vol_range_features_from_series(
    *,
    close: pd.Series,
    window: int = 20,
) -> pd.DataFrame:
    """vol_range_* + vol_range_pos_* from vol_base."""
    returns = _compute_returns_from_close(close)
    vol_base = _compute_vol_base(returns, window=window)
    out = pd.DataFrame(index=close.index)
    for w in [10, 20]:
        vol_roll = vol_base.rolling(window=w, min_periods=max(1, w // 2))
        vol_max = vol_roll.max()
        vol_min = vol_roll.min()
        vol_range = vol_max - vol_min
        out[f"vol_range_{w}"] = vol_range
        out[f"vol_range_pos_{w}"] = (
            (vol_base - vol_min) / (vol_range.replace(0, np.nan) + 1e-8)
        ).replace([np.inf, -np.inf], np.nan).fillna(0.5)
    return out


@register_feature("compute_vol_mom_features_from_series", category="volatility")
def compute_vol_mom_features_from_series(
    *,
    close: pd.Series,
    window: int = 20,
) -> pd.DataFrame:
    """vol_mom_*: momentum of vol_base."""
    returns = _compute_returns_from_close(close)
    vol_base = _compute_vol_base(returns, window=window)
    out = pd.DataFrame(index=close.index)
    for p in [3, 5, 10]:
        out[f"vol_mom_{p}"] = vol_base.pct_change(p).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


@register_feature("select_extended_volatility_features", category="volatility")
def select_extended_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Composite node for Feature DAG.
    Dependencies compute the individual sub-feature blocks; this node just passes them through.
    """
    return df


def extract_volatility_features_from_vp(
    vp: Optional[VolumeProfileResult],
    current_price: Optional[float] = None,
) -> Dict[str, float]:
    """
    从 VolumeProfileResult 提取波动率预测相关特征
    
    Args:
        vp: VolumeProfileResult 对象（可能为 None）
        current_price: 当前价格（用于计算 POC 偏离度，可选）
    
    Returns:
        dict of scalar features (all float)
    """
    if vp is None or len(vp.hist) == 0:
        return {
            "vp_width_ratio": 0.0,
            "vp_poc_deviation": 0.0,
            "vp_skewness": 0.0,
            "vp_entropy": 0.0,
            "vp_lv_ratio": 0.0,
            "vp_hv_ratio": 0.0,
        }
    
    hist = vp.hist
    centers = vp.centers
    total_vol = np.sum(hist)
    
    if total_vol <= 1e-8:
        return {
            "vp_width_ratio": 0.0,
            "vp_poc_deviation": 0.0,
            "vp_skewness": 0.0,
            "vp_entropy": 0.0,
            "vp_lv_ratio": 0.0,
            "vp_hv_ratio": 0.0,
        }
    
    price_range = vp.price_max - vp.price_min
    
    # 1. Value Area Width / Full Range → 衡量市场共识强度
    if price_range <= 1e-10:
        width_ratio = 0.0
    else:
        # 计算加权均值和标准差（近似 Value Area）
        weighted_mean = np.average(centers, weights=hist)
        weighted_std = np.sqrt(np.average((centers - weighted_mean) ** 2, weights=hist))
        va_width = 2 * weighted_std  # ≈ 68% 区间（正态假设）
        width_ratio = va_width / price_range
        width_ratio = np.clip(width_ratio, 0.0, 1.0)
    
    # 2. 找 POC（最大成交量对应的价格中心）
    poc_idx = np.argmax(hist)
    poc_price = centers[poc_idx]
    
    # 计算当前价格 vs POC 的标准化偏离
    if current_price is not None and price_range > 1e-10:
        poc_deviation = (current_price - poc_price) / price_range
        poc_deviation = np.clip(poc_deviation, -1.0, 1.0)
    else:
        poc_deviation = 0.0
    
    # 3. 成交量分布偏度（衡量趋势倾向）
    try:
        vp_skew = float(skew(hist))
    except Exception:
        vp_skew = 0.0
    # Normalize skewness to a stable, bounded range for NN / cross-asset comparability.
    # This is per-window (no lookahead) and keeps semantics (sign + magnitude) while
    # preventing extreme outliers.
    vp_skew = float(np.tanh(vp_skew))
    
    # 4. 信息熵（衡量不确定性）
    prob = hist / total_vol
    prob = prob[prob > 0]  # 避免除零
    if len(prob) > 0:
        entropy = -np.sum(prob * np.log(prob + 1e-8))
        # 归一化到 [0,1]
        max_entropy = np.log(len(prob))
        norm_entropy = entropy / (max_entropy + 1e-8) if max_entropy > 0 else 0.0
    else:
        norm_entropy = 0.0
    
    # 5. 低成交量区域比例（LVN）
    mean_vol = total_vol / len(hist)
    lv_mask = hist < (mean_vol * 0.3)  # 低于均值30%视为低量
    lv_ratio = np.sum(lv_mask) / len(hist)
    
    # 6. 高成交量区域比例（HVN）
    hv_mask = hist > (mean_vol * 1.5)  # 高于均值150%视为高量
    hv_ratio = np.sum(hv_mask) / len(hist)
    
    return {
        "vp_width_ratio": float(width_ratio),  # 越小 → 共识强 → 波动可能低
        "vp_poc_deviation": float(poc_deviation),  # 绝对值大 → 远离价值中枢 → 波动可能高
        "vp_skewness": float(vp_skew),  # bounded to [-1,1]
        "vp_entropy": float(norm_entropy),  # 越大 → 分歧大 → 波动可能高
        "vp_lv_ratio": float(lv_ratio),  # 越大 → 薄弱区多 → 波动易放大
        "vp_hv_ratio": float(hv_ratio),  # 越大 → 支撑/阻力密集 → 波动可能受限
    }


@register_feature("extract_volume_profile_volatility_features_from_series", category="volatility")
def extract_volume_profile_volatility_features_from_series(
    *,
    close: pd.Series,
    volume: pd.Series,
    window: int = 100,
    wavelet: str = "db4",
    level: int = 4,
) -> pd.DataFrame:
    """
    Narrow-IO entrypoint for Volume Profile volatility features.

    Returns only the vp_* output columns.
    """
    close_s = pd.to_numeric(close, errors="coerce").astype(float)
    vol_s = pd.to_numeric(volume, errors="coerce").astype(float)
    idx = close_s.index
    n = len(close_s)

    keep = [
        "vp_width_ratio",
        "vp_poc_deviation",
        "vp_skewness",
        "vp_entropy",
        "vp_lv_ratio",
        "vp_hv_ratio",
    ]

    # Initialize arrays
    out_arrs = {c: np.full(n, np.nan, dtype=float) for c in keep}

    price_values = close_s.values
    volume_values = vol_s.values
    min_length = max(window, 2**level)

    for i in range(min_length, n):
        price_window = price_values[i - window : i]
        volume_window = volume_values[i - window : i]
        current_price = price_values[i]

        if len(price_window) < 2**level:
            continue

        vp = compute_wpt_volume_profile(
            price_window=price_window,
            volume_window=volume_window,
            bins="auto",
            wavelet=wavelet,
            level=level,
            drop_high_freq=True,
        )
        if vp is None:
            continue

        feats = extract_volatility_features_from_vp(vp, current_price=current_price)
        for c in keep:
            if c in feats:
                out_arrs[c][i] = float(feats[c])

    out = pd.DataFrame(out_arrs, index=idx)
    out = out.ffill().fillna(0.0)
    return out


def enhance_wpt_vol_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    从 WPT 结果中提取波动率相关衍生特征
    
    Args:
        df: DataFrame with WPT features
    
    Returns:
        DataFrame with enhanced WPT volatility features
    """
    df = df.copy()
    
    # 首先处理重复列名：如果有重复列，保留第一个
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]
    
    # 1. 高频能量占比 → 噪声强度
    if "wpt_price_energy_high_ratio" in df.columns:
        # 确保是 Series，不是 DataFrame（处理重复列名的情况）
        energy_high = df["wpt_price_energy_high_ratio"]
        if isinstance(energy_high, pd.DataFrame):
            # 如果是 DataFrame（可能因为重复列名），取第一列
            if len(energy_high.columns) > 0:
                energy_high = energy_high.iloc[:, 0]
            else:
                energy_high = pd.Series(0.0, index=df.index)
        df["wpt_price_high_energy_ratio"] = energy_high
    else:
        # fallback: 手动计算
        df["wpt_price_high_energy_ratio"] = 0.0
    
    # 2. 波动信号的 L1/L2 范数比 → 尖峰程度（衡量极端波动概率）
    if "wpt_price_fluctuation" in df.columns:
        fluct = df["wpt_price_fluctuation"]
        
        # 确保 fluct 是 Series，不是 DataFrame
        if isinstance(fluct, pd.DataFrame):
            # 如果是 DataFrame，取第一列
            if len(fluct.columns) > 0:
                fluct = fluct.iloc[:, 0]
            else:
                df["wpt_price_fluct_l1_l2_ratio"] = 0.0
                return df
        elif not isinstance(fluct, pd.Series):
            # 如果是其他类型，尝试转换为 Series
            try:
                fluct = pd.Series(fluct, index=df.index)
            except Exception:
                df["wpt_price_fluct_l1_l2_ratio"] = 0.0
                return df
        
        # 现在 fluct 应该是 Series，进行计算
        # 确保 fluct 是 Series（再次检查，以防万一）
        if not isinstance(fluct, pd.Series):
            # 如果仍然不是 Series，尝试强制转换
            try:
                if isinstance(fluct, pd.DataFrame):
                    fluct = fluct.iloc[:, 0] if len(fluct.columns) > 0 else pd.Series(0.0, index=df.index)
                else:
                    fluct = pd.Series(fluct, index=df.index)
            except Exception:
                df["wpt_price_fluct_l1_l2_ratio"] = 0.0
                return df
        
        # 使用 .values 确保得到 numpy 数组，避免 DataFrame 问题
        fluct_values = fluct.values if isinstance(fluct, pd.Series) else np.asarray(fluct).flatten()
        
        # 进行计算（使用 numpy 数组）
        l1 = np.abs(fluct_values)
        l2 = fluct_values ** 2
        ratio_values = (l1 + 1e-8) / (np.sqrt(l2) + 1e-8)
        
        # 确保结果是 Series（使用原始索引）
        df["wpt_price_fluct_l1_l2_ratio"] = pd.Series(ratio_values, index=df.index)
    else:
        df["wpt_price_fluct_l1_l2_ratio"] = 0.0
    
    # 3. 体积-价格高频同步性
    if "wpt_volume_energy_high_ratio" in df.columns and "wpt_price_energy_high_ratio" in df.columns:
        # 确保是 Series，不是 DataFrame（处理重复列名的情况）
        vol_energy_high = df["wpt_volume_energy_high_ratio"]
        price_energy_high = df["wpt_price_energy_high_ratio"]
        
        if isinstance(vol_energy_high, pd.DataFrame):
            if len(vol_energy_high.columns) > 0:
                vol_energy_high = vol_energy_high.iloc[:, 0]
            else:
                vol_energy_high = pd.Series(0.0, index=df.index)
        
        if isinstance(price_energy_high, pd.DataFrame):
            if len(price_energy_high.columns) > 0:
                price_energy_high = price_energy_high.iloc[:, 0]
            else:
                price_energy_high = pd.Series(0.0, index=df.index)
        
        df["wpt_vhph_sync"] = vol_energy_high * price_energy_high
    else:
        df["wpt_vhph_sync"] = 0.0
    
    # 4. 低频趋势稳定性（trend 的 std）
    # 注意：当前 extract_wpt_features 只保留每个窗口的最后一个点，无法计算 trend 的标准差
    # 若需此特征，建议在 WPT 模块中增加选项 return_full_trend_series=False
    # 暂略，可用 vol_slope_* 替代
    
    return df
