"""
流动性真空区与 WPT+Volume 能量协同分析特征

核心功能：
1. 流动性真空区识别（Liquidity Void / Gap Detection）
2. WPT + Volume 能量协同分析（VPER、能量下移、真假突破判断）
3. WPT 降噪的 VPVR（Volume Profile Visible Range）

关键认知：
- 流动性真空区 ≠ 历史低成交量区域，而是当前订单簿深度缺失
- 价格快速通过真空区 ≠ 真突破，需结合多尺度能量验证
- WPT 降噪后的 VPVR 能更清晰地识别关键支撑/阻力位
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

import pywt
from scipy import stats

from src.features.registry import register_feature

from .utils_volume_profile import (
    VolumeProfileResult,
    compute_wpt_volume_profile,
)


def _get_wpt_frequency_center(path: str, level: int) -> float:
    """Calculate approximate frequency center for a WPT node path.

    For a Daubechies-4 wavelet at level N:
    - Path with more 'a' characters represents lower frequencies
    - Path with more 'd' characters represents higher frequencies

    Frequency center is estimated as: a_count / level
    This gives a value in [0, 1] where 0 is lowest frequency, 1 is highest.

    Args:
        path: WPT node path string (e.g., "aada" for level 4)
        level: WPT decomposition level

    Returns:
        Frequency center in [0, 1] range
    """
    a_count = path.count("a")
    # Normalize by level to get frequency center
    freq_center = a_count / level if level > 0 else 0.5
    return float(freq_center)


def _classify_freq_bands_by_center(
    price_energy: Dict[str, float], level: int
) -> Tuple[List[str], List[str], List[str]]:
    """Classify WPT subbands into low/mid/high frequency based on frequency center.

    More robust than path string matching, works correctly for any level.

    Args:
        price_energy: Dictionary mapping WPT paths to energy values
        level: WPT decomposition level

    Returns:
        Tuple of (low_freq_paths, mid_freq_paths, high_freq_paths)
    """
    freq_centers = {
        path: _get_wpt_frequency_center(path, level) for path in price_energy.keys()
    }

    low_freq_paths = [p for p, fc in freq_centers.items() if fc < 0.25]
    mid_freq_paths = [p for p, fc in freq_centers.items() if 0.25 <= fc < 0.75]
    high_freq_paths = [p for p, fc in freq_centers.items() if fc >= 0.75]

    # Fallback to path-based classification if no paths match
    if not low_freq_paths and not mid_freq_paths and not high_freq_paths:
        # Original path-based method as fallback
        low_freq_paths = ["a" * level]
        mid_freq_paths = [
            k for k in price_energy.keys() if k.startswith("aa") and k != "a" * level
        ]
        high_freq_paths = [k for k in price_energy.keys() if not k.startswith("aa")]

    return low_freq_paths, mid_freq_paths, high_freq_paths


def _adaptive_wpt_window(
    price: pd.Series,
    atr: Optional[pd.Series] = None,
    min_window: int = 10,
    max_window: int = 50,
) -> pd.Series:
    """Calculate adaptive window based on ATR for WPT computation.

    Args:
        price: Price series
        atr: ATR series (optional, if None will estimate from price)
        min_window: Minimum window size
        max_window: Maximum window size

    Returns:
        Series of adaptive window sizes
    """
    if atr is None:
        # Estimate ATR from price changes if not provided
        atr_est = price.rolling(window=14).std()
    else:
        atr_est = atr

    price_ma = price.rolling(window=14).mean()

    # Fill NaN values before division to avoid inf
    price_ma = price_ma.fillna(price.mean() if price.mean() > 0 else 1.0)
    atr_est = atr_est.fillna(atr_est.mean() if atr_est.mean() > 0 else 0.1)

    # Calculate adaptive window: window = max(min_window, int(14 * atr_ma / price_ma))
    # This adapts to volatility: higher volatility -> larger window
    adaptive_window = (14 * atr_est / (price_ma + 1e-8)).clip(
        lower=min_window, upper=max_window
    )
    # Replace any remaining inf or NaN with default window
    adaptive_window = adaptive_window.replace([np.inf, -np.inf], min_window).fillna(
        min_window
    )
    return adaptive_window.astype(int)


# build_wpt_denoised_vpvr 已删除，现在使用 utils_volume_profile.compute_unified_volume_profile_features


def compute_liquidity_void_features(
    df: pd.DataFrame,
    price_col: str = "close",
    volume_col: str = "volume",
    high_col: str = "high",
    low_col: str = "low",
    atr_col: str = "atr",
    lookback_window: int = 20,
    speed_threshold_multiplier: float = 2.0,
    volume_quantile: float = 0.2,
) -> pd.DataFrame:
    """
    识别流动性真空区（Liquidity Void / Gap）

    核心逻辑：
    1. 价格快速穿越某区域（速度 > 阈值）
    2. 该区域成交量显著低于前后区间（VPVR 低量节点）
    3. 突破后 1~3 根 K 线内迅速回撤 >50%

    注意：流动性真空区 ≠ 历史低成交量区域
    - 历史低成交量：过去交易少（可能因无人关注）
    - 流动性真空：当前挂单少（即使过去交易多，也可能因订单撤走而变真空）

    Args:
        df: DataFrame with OHLCV data
        price_col: Price column name
        volume_col: Volume column name
        high_col: High column name
        low_col: Low column name
        atr_col: ATR column name
        lookback_window: Window for calculating price speed and volume reference
        speed_threshold_multiplier: Multiplier for price speed threshold
        volume_quantile: Quantile threshold for low volume detection

    Returns:
        DataFrame with liquidity void features:
        - liquidity_void_detected: 1.0 if void detected, 0.0 otherwise
        - liquidity_void_speed: Price speed (normalized by ATR)
        - liquidity_void_volume_ratio: Volume ratio (current vs. reference)
        - liquidity_void_retracement: Retracement within 3 bars (if applicable)
        - liquidity_void_false_breakout_risk: Risk score (0-1) for false breakout

    Note: price_impact is only available in narrow-IO version (compute_liquidity_void_features_from_series)
    """
    df = df.copy()

    # 初始化输出列
    df["liquidity_void_detected"] = 0.0
    df["liquidity_void_speed"] = 0.0
    df["liquidity_void_volume_ratio"] = 1.0
    df["liquidity_void_retracement"] = 0.0
    df["liquidity_void_false_breakout_risk"] = 0.0

    if atr_col not in df.columns:
        # 如果没有 ATR，使用价格变化的标准差作为替代
        df[atr_col] = df[price_col].rolling(window=14).std()

    # 计算价格速度（归一化）
    price_change = df[price_col].pct_change()
    price_speed = price_change.rolling(window=3).mean()  # 3-bar average speed
    price_speed_normalized = price_speed / (df[atr_col] / df[price_col] + 1e-8)

    # 计算成交量比率
    volume_ma = df[volume_col].rolling(window=lookback_window).mean()
    volume_ratio = df[volume_col] / (volume_ma + 1e-8)

    # 计算成交量分位数（用于识别低量区）
    volume_quantile_value = (
        df[volume_col].rolling(window=lookback_window).quantile(volume_quantile)
    )
    is_low_volume = df[volume_col] < volume_quantile_value

    # 计算价格速度阈值
    speed_threshold = (
        price_speed_normalized.rolling(window=lookback_window).mean()
        * speed_threshold_multiplier
    )

    # 检测流动性真空区
    for i in range(lookback_window, len(df)):
        # 条件1: 价格速度异常高
        speed_high = price_speed_normalized.iloc[i] > speed_threshold.iloc[i]

        # 条件2: 成交量低（相对于历史）
        volume_low = is_low_volume.iloc[i] or volume_ratio.iloc[i] < 0.8

        if speed_high and volume_low:
            df.iloc[i, df.columns.get_loc("liquidity_void_detected")] = 1.0
            df.iloc[i, df.columns.get_loc("liquidity_void_speed")] = (
                price_speed_normalized.iloc[i]
            )
            df.iloc[i, df.columns.get_loc("liquidity_void_volume_ratio")] = (
                volume_ratio.iloc[i]
            )

            # 条件3: 检查后续回撤（如果有足够数据）
            if i + 3 < len(df):
                current_price = df[price_col].iloc[i]
                future_prices = df[price_col].iloc[i + 1 : i + 4]

                if len(future_prices) > 0:
                    max_future_price = future_prices.max()
                    min_future_price = future_prices.min()

                    # 计算回撤（相对于当前价格）
                    if current_price > 0:
                        retracement_up = (
                            max_future_price - current_price
                        ) / current_price
                        retracement_down = (
                            current_price - min_future_price
                        ) / current_price
                        retracement = max(retracement_up, retracement_down)

                        df.iloc[i, df.columns.get_loc("liquidity_void_retracement")] = (
                            retracement
                        )

                        # 如果回撤 > 50%，增加假突破风险
                        if retracement > 0.5:
                            df.iloc[
                                i,
                                df.columns.get_loc(
                                    "liquidity_void_false_breakout_risk"
                                ),
                            ] = 0.8
                        elif retracement > 0.3:
                            df.iloc[
                                i,
                                df.columns.get_loc(
                                    "liquidity_void_false_breakout_risk"
                                ),
                            ] = 0.5
                        else:
                            df.iloc[
                                i,
                                df.columns.get_loc(
                                    "liquidity_void_false_breakout_risk"
                                ),
                            ] = 0.2

    return df


def compute_wpt_volume_energy_features(
    df: pd.DataFrame,
    price_col: str = "close",
    volume_col: str = "volume",
    wavelet: str = "db4",
    level: int = 4,
    lookback_window: int = 20,
    adaptive_window: bool = False,
    atr_col: Optional[str] = None,
    use_log_returns: bool = False,
    detrend_price: bool = False,
) -> pd.DataFrame:
    """
    计算 WPT + Volume 能量协同分析特征

    核心指标：
    1. VPER (Volume-Price Energy Ratio): 量价能量比
    2. 能量下移（Energy Cascade）: 高频能量向中低频转移
    3. 多尺度一致性验证: 至少两个中低频子带同时出现能量上升
    4. 真假突破评分: 基于多尺度能量和 VPER 的综合评分

    Args:
        df: DataFrame with OHLCV data
        price_col: Price column name
        volume_col: Volume column name
        wavelet: Wavelet function
        level: WPT decomposition level
        lookback_window: Window for calculating reference values (or base window if adaptive_window=True)
        adaptive_window: If True, use ATR-based adaptive window instead of fixed lookback_window
        atr_col: ATR column name (required if adaptive_window=True)
        use_log_returns: If True, apply WPT to log returns instead of raw price (recommended for non-stationary data)
        detrend_price: If True, detrend price before WPT (alternative to log returns)

    Returns:
        DataFrame with WPT+Volume energy features:
        - wpt_vper_low: VPER for low-frequency subband
        - wpt_vper_mid: VPER for mid-frequency subbands
        - wpt_vper_high: VPER for high-frequency subband
        - wpt_energy_cascade: Energy cascade indicator (high -> mid -> low)
        - wpt_multi_scale_consistency: Multi-scale consistency score (0-1)
        - wpt_breakout_confidence: Breakout confidence score (0-1)
        - wpt_false_breakout_risk: False breakout risk score (0-1)
    """
    df = df.copy()

    # 初始化输出列
    df["wpt_vper_low"] = 0.0
    df["wpt_vper_mid"] = 0.0
    df["wpt_vper_high"] = 0.0
    df["wpt_energy_cascade"] = 0.0
    df["wpt_multi_scale_consistency"] = 0.0
    df["wpt_breakout_confidence"] = 0.0
    df["wpt_false_breakout_risk"] = 0.0

    # 提取价格和成交量
    price_series = df[price_col]
    price = price_series.values
    volume = df[volume_col].values

    # 计算自适应窗口（如果需要）
    if adaptive_window:
        if atr_col is None or atr_col not in df.columns:
            # Try to use "atr" column if available
            if "atr" in df.columns:
                atr_series = df["atr"]
            else:
                # Estimate ATR from price
                atr_series = None
        else:
            atr_series = df[atr_col]

        adaptive_windows = _adaptive_wpt_window(
            price_series,
            atr=atr_series,
            min_window=max(10, lookback_window // 2),
            max_window=lookback_window * 3,
        )
    else:
        adaptive_windows = None

    # 滚动窗口计算
    for i in range(lookback_window, len(df)):
        # Use adaptive window if enabled, otherwise use fixed window
        if adaptive_window and adaptive_windows is not None:
            current_window = int(adaptive_windows.iloc[i])
        else:
            current_window = lookback_window

        window_start = max(0, i - current_window)
        price_window_raw = price[window_start : i + 1]
        volume_window = volume[window_start : i + 1]

        if len(price_window_raw) < 10:
            continue

        # Preprocess price: log returns or detrend
        if use_log_returns:
            # Use log returns for WPT (stationary transformation)
            price_window = np.diff(np.log(price_window_raw))
            if len(price_window) < level * 2:  # Need minimum data points
                continue
        elif detrend_price:
            # Detrend price before WPT
            from scipy import signal

            price_window = signal.detrend(price_window_raw)
        else:
            price_window = price_window_raw

        try:
            # Step 1: 对价格和成交量分别做 WPT 分解
            wp_price = pywt.WaveletPacket(
                data=price_window, wavelet=wavelet, mode="symmetric", maxlevel=level
            )
            wp_volume = pywt.WaveletPacket(
                data=volume_window, wavelet=wavelet, mode="symmetric", maxlevel=level
            )

            # Step 2: 提取各子带能量
            price_energy = {}
            volume_energy = {}

            for node in wp_price.get_level(level, "natural"):
                price_energy[node.path] = np.sum(node.data**2)

            for node in wp_volume.get_level(level, "natural"):
                volume_energy[node.path] = np.sum(node.data**2)

            # Step 3: 分类子带（低频、中频、高频）- 使用频率中心方法
            low_freq_paths, mid_freq_paths, high_freq_paths = (
                _classify_freq_bands_by_center(price_energy, level)
            )

            # Step 4: 计算各频带的能量和 VPER
            price_energy_low = sum(price_energy.get(p, 0) for p in low_freq_paths)
            price_energy_mid = sum(price_energy.get(p, 0) for p in mid_freq_paths)
            price_energy_high = sum(price_energy.get(p, 0) for p in high_freq_paths)

            volume_energy_low = sum(volume_energy.get(p, 0) for p in low_freq_paths)
            volume_energy_mid = sum(volume_energy.get(p, 0) for p in mid_freq_paths)
            volume_energy_high = sum(volume_energy.get(p, 0) for p in high_freq_paths)

            # VPER = Volume Energy / Price Energy
            eps = 1e-8
            vper_low = volume_energy_low / (price_energy_low + eps)
            vper_mid = volume_energy_mid / (price_energy_mid + eps)
            vper_high = volume_energy_high / (price_energy_high + eps)

            df.iloc[i, df.columns.get_loc("wpt_vper_low")] = vper_low
            df.iloc[i, df.columns.get_loc("wpt_vper_mid")] = vper_mid
            df.iloc[i, df.columns.get_loc("wpt_vper_high")] = vper_high

            # Step 5: 计算能量下移（Energy Cascade）
            # 理想情况：高频能量下降，中低频能量上升
            total_price_energy = price_energy_low + price_energy_mid + price_energy_high
            if total_price_energy > 0:
                energy_ratio_low = price_energy_low / total_price_energy
                energy_ratio_mid = price_energy_mid / total_price_energy
                energy_ratio_high = price_energy_high / total_price_energy

                # 能量下移指标：中低频能量占比增加
                energy_cascade = (
                    energy_ratio_low + energy_ratio_mid
                ) - energy_ratio_high
                df.iloc[i, df.columns.get_loc("wpt_energy_cascade")] = energy_cascade

            # Step 6: 多尺度一致性验证
            # 检查至少两个中低频子带是否同时出现能量上升
            if i > 0:
                # Use same window size as current computation for consistency
                prev_window_start = max(0, i - current_window - 1)
                prev_price_window_raw = price[prev_window_start:i]
                if len(prev_price_window_raw) >= 10:
                    # Use same preprocessing as current window
                    if use_log_returns:
                        prev_price_window = np.diff(np.log(prev_price_window_raw))
                        if len(prev_price_window) < level * 2:
                            continue
                    elif detrend_price:
                        from scipy import signal

                        prev_price_window = signal.detrend(prev_price_window_raw)
                    else:
                        prev_price_window = prev_price_window_raw

                    try:
                        wp_price_prev = pywt.WaveletPacket(
                            data=prev_price_window,
                            wavelet=wavelet,
                            mode="symmetric",
                            maxlevel=level,
                        )

                        prev_price_energy = {}
                        for node in wp_price_prev.get_level(level, "natural"):
                            prev_price_energy[node.path] = np.sum(node.data**2)

                        # 计算中低频子带的能量变化
                        mid_freq_energy_changes = []
                        for path in mid_freq_paths:
                            curr_energy = price_energy.get(path, 0)
                            prev_energy = prev_price_energy.get(path, 0)
                            if prev_energy > 0:
                                energy_change = (
                                    curr_energy - prev_energy
                                ) / prev_energy
                                mid_freq_energy_changes.append(energy_change)

                        # 如果至少两个中低频子带能量上升，则一致性高
                        positive_changes = sum(
                            1 for chg in mid_freq_energy_changes if chg > 0.1
                        )
                        consistency_score = min(
                            positive_changes / max(len(mid_freq_paths), 1), 1.0
                        )
                        df.iloc[
                            i, df.columns.get_loc("wpt_multi_scale_consistency")
                        ] = consistency_score
                    except Exception:
                        pass

            # Step 7: 真假突破评分
            # 真突破条件：
            # 1. 中低频能量上升
            # 2. VPER 中频处于高位
            # 3. 多尺度一致性高

            # 计算 VPER 中频的历史分位数
            if i >= lookback_window * 2:
                vper_mid_history = df["wpt_vper_mid"].iloc[i - lookback_window : i]
                if len(vper_mid_history) > 0 and vper_mid_history.notna().any():
                    vper_mid_quantile = (
                        stats.percentileofscore(vper_mid_history.dropna(), vper_mid)
                        / 100.0
                    )
                else:
                    vper_mid_quantile = 0.5
            else:
                vper_mid_quantile = 0.5

            # 突破置信度 = (能量下移 + VPER分位数 + 多尺度一致性) / 3
            # Note: VPVR features (vpvr_poc_deviation, vpvr_hvn) should be provided as separate features
            # to the model, not hardcoded here. Let the model learn the relationship.
            consistency = df.iloc[i, df.columns.get_loc("wpt_multi_scale_consistency")]
            breakout_confidence = np.clip(
                (energy_cascade + 1) / 2 * 0.4
                + vper_mid_quantile * 0.3
                + consistency * 0.3,
                0.0,
                1.0,
            )
            df.iloc[i, df.columns.get_loc("wpt_breakout_confidence")] = (
                breakout_confidence
            )

            # 假突破风险 = 1 - 突破置信度，但如果高频能量突增而中频无增长，风险更高
            if price_energy_high > price_energy_mid * 2 and vper_mid < 0.5:
                false_breakout_risk = min(breakout_confidence + 0.3, 1.0)
            else:
                false_breakout_risk = 1.0 - breakout_confidence

            df.iloc[i, df.columns.get_loc("wpt_false_breakout_risk")] = (
                false_breakout_risk
            )

        except Exception as e:
            # WPT 计算失败时跳过
            continue

    # 前向填充缺失值
    for col in [
        "wpt_vper_low",
        "wpt_vper_mid",
        "wpt_vper_high",
        "wpt_energy_cascade",
        "wpt_multi_scale_consistency",
        "wpt_breakout_confidence",
        "wpt_false_breakout_risk",
    ]:
        df[col] = df[col].ffill().fillna(0.0)

    return df


@register_feature("extract_liquidity_features", category="liquidity")
def extract_liquidity_features(
    df: pd.DataFrame,
    price_col: str = "close",
    volume_col: str = "volume",
    high_col: str = "high",
    low_col: str = "low",
    atr_col: str = "atr",
    wavelet: str = "db4",
    level: int = 4,
    feature_type: str = "all",
) -> pd.DataFrame:
    """
    提取完整的流动性相关特征（整合所有子功能）

    Args:
        df: DataFrame with OHLCV data
        price_col: Price column name
        volume_col: Volume column name
        high_col: High column name
        low_col: Low column name
        atr_col: ATR column name
        wavelet: Wavelet function
        level: WPT decomposition level
        feature_type: Feature type to extract ("vpvr", "void", "energy", "all")

    Returns:
        DataFrame with all liquidity features added
    """
    df = df.copy()

    # 根据 feature_type 选择要提取的特征
    if feature_type in ["vpvr", "all"]:
        # 1. WPT 降噪的 VPVR（使用统一实现）
        from .utils_volume_profile import compute_unified_volume_profile_features

        df = compute_unified_volume_profile_features(
            df,
            price_col=price_col,
            volume_col=volume_col,
            high_col=high_col,
            low_col=low_col,
            window=100,  # VPVR 默认窗口
            use_typical_price=True,  # VPVR 使用典型价格
            wavelet=wavelet,
            level=level,
        )

    if feature_type in ["void", "all"]:
        # 2. 流动性真空区识别
        df = compute_liquidity_void_features(
            df,
            price_col=price_col,
            volume_col=volume_col,
            high_col=high_col,
            low_col=low_col,
            atr_col=atr_col,
        )

    if feature_type in ["energy", "all"]:
        # 3. WPT + Volume 能量协同分析
        df = compute_wpt_volume_energy_features(
            df,
            price_col=price_col,
            volume_col=volume_col,
            wavelet=wavelet,
            level=level,
        )

    return df


@register_feature("compute_liquidity_void_features_from_series", category="liquidity")
def compute_liquidity_void_features_from_series(
    *,
    close: pd.Series,
    volume: pd.Series,
    high: Optional[pd.Series] = None,
    low: Optional[pd.Series] = None,
    atr: Optional[pd.Series] = None,
    lookback_window: int = 20,
    speed_threshold_multiplier: float = 2.0,
    volume_quantile: float = 0.2,
) -> pd.DataFrame:
    """
    Narrow-IO liquidity void features (no wide DF mutation).

    Matches `compute_liquidity_void_features` semantics.
    """
    close_s = pd.to_numeric(close, errors="coerce").astype(float)
    vol_s = pd.to_numeric(volume, errors="coerce").astype(float)
    idx = close_s.index
    n = len(close_s)

    # Calculate price range for price_impact
    if high is not None and low is not None:
        high_s = pd.to_numeric(high, errors="coerce").astype(float).reindex(idx)
        low_s = pd.to_numeric(low, errors="coerce").astype(float).reindex(idx)
        price_range = high_s - low_s
    else:
        # Estimate range from close price changes if high/low not provided
        # Use rolling max/min of close changes as proxy
        price_range = (
            close_s.rolling(window=3)
            .apply(lambda x: x.max() - x.min(), raw=True)
            .fillna(0)
        )

    if atr is None:
        atr_s = close_s.rolling(window=14).std()
    else:
        atr_s = pd.to_numeric(atr, errors="coerce").astype(float).reindex(idx)
    atr_s = atr_s.clip(lower=1e-8)

    # price speed normalized by ATR ratio
    price_change = close_s.pct_change()
    price_speed = price_change.rolling(window=3).mean()
    price_speed_normalized = price_speed / (atr_s / close_s + 1e-8)

    # volume ratios / low-volume detection
    volume_ma = vol_s.rolling(window=lookback_window).mean()
    volume_ratio = vol_s / (volume_ma + 1e-8)
    volume_quantile_value = vol_s.rolling(window=lookback_window).quantile(
        volume_quantile
    )
    is_low_volume = vol_s < volume_quantile_value

    speed_threshold = price_speed_normalized.rolling(
        window=lookback_window
    ).mean() * float(speed_threshold_multiplier)

    # Calculate price impact
    price_impact = price_range / (vol_s + 1e-8)

    detected = np.zeros(n, dtype=float)
    void_speed = np.zeros(n, dtype=float)
    void_volume_ratio = np.ones(n, dtype=float)
    void_price_impact = np.zeros(n, dtype=float)
    retracement_arr = np.zeros(n, dtype=float)
    risk_arr = np.zeros(n, dtype=float)

    close_vals = close_s.values
    psn_vals = price_speed_normalized.values
    st_vals = speed_threshold.values
    vr_vals = volume_ratio.values
    low_vals = is_low_volume.values

    for i in range(lookback_window, n):
        speed_high = (
            bool(psn_vals[i] > st_vals[i]) if np.isfinite(st_vals[i]) else False
        )
        volume_low = bool(low_vals[i]) or (
            vr_vals[i] < 0.8 if np.isfinite(vr_vals[i]) else False
        )

        if speed_high and volume_low:
            detected[i] = 1.0
            void_speed[i] = float(psn_vals[i]) if np.isfinite(psn_vals[i]) else 0.0
            void_volume_ratio[i] = float(vr_vals[i]) if np.isfinite(vr_vals[i]) else 1.0
            void_price_impact[i] = (
                float(price_impact.iloc[i])
                if np.isfinite(price_impact.iloc[i])
                else 0.0
            )

            if i + 3 < n:
                current_price = float(close_vals[i])
                if current_price > 0:
                    future_prices = close_vals[i + 1 : i + 4]
                    if future_prices.size > 0:
                        max_future_price = float(np.nanmax(future_prices))
                        min_future_price = float(np.nanmin(future_prices))
                        retracement_up = (
                            max_future_price - current_price
                        ) / current_price
                        retracement_down = (
                            current_price - min_future_price
                        ) / current_price
                        retracement = float(max(retracement_up, retracement_down))
                        retracement_arr[i] = retracement

                        if retracement > 0.5:
                            risk_arr[i] = 0.8
                        elif retracement > 0.3:
                            risk_arr[i] = 0.5
                        else:
                            risk_arr[i] = 0.2

    return pd.DataFrame(
        {
            "liquidity_void_detected": detected,
            "liquidity_void_speed": void_speed,
            "liquidity_void_volume_ratio": void_volume_ratio,
            "liquidity_void_price_impact": void_price_impact,
            "liquidity_void_retracement": retracement_arr,
            "liquidity_void_false_breakout_risk": risk_arr,
        },
        index=idx,
    )


@register_feature(
    "compute_wpt_volume_energy_features_from_series", category="liquidity"
)
def compute_wpt_volume_energy_features_from_series(
    *,
    close: pd.Series,
    volume: pd.Series,
    wavelet: str = "db4",
    level: int = 4,
    lookback_window: int = 20,
    use_log_returns: bool = False,
    detrend_price: bool = False,
) -> pd.DataFrame:
    """
    Narrow-IO WPT + Volume energy features (no wide DF mutation).

    Matches `compute_wpt_volume_energy_features` semantics.
    """
    close_s = pd.to_numeric(close, errors="coerce").astype(float)
    vol_s = pd.to_numeric(volume, errors="coerce").astype(float)
    idx = close_s.index
    n = len(close_s)

    out = {
        "wpt_vper_low": np.zeros(n, dtype=float),
        "wpt_vper_mid": np.zeros(n, dtype=float),
        "wpt_vper_high": np.zeros(n, dtype=float),
        "wpt_energy_cascade": np.zeros(n, dtype=float),
        "wpt_multi_scale_consistency": np.zeros(n, dtype=float),
        "wpt_breakout_confidence": np.zeros(n, dtype=float),
        "wpt_false_breakout_risk": np.zeros(n, dtype=float),
    }

    price = close_s.values
    vol = vol_s.values

    for i in range(lookback_window, n):
        window_start = max(0, i - lookback_window)
        price_window_raw = price[window_start : i + 1]
        volume_window = vol[window_start : i + 1]

        if len(price_window_raw) < 10:
            continue

        # Preprocess price: log returns or detrend
        if use_log_returns:
            price_window = np.diff(np.log(price_window_raw))
            if len(price_window) < level * 2:
                continue
        elif detrend_price:
            from scipy import signal

            price_window = signal.detrend(price_window_raw)
        else:
            price_window = price_window_raw

        try:
            wp_price = pywt.WaveletPacket(
                data=price_window, wavelet=wavelet, mode="symmetric", maxlevel=level
            )
            wp_volume = pywt.WaveletPacket(
                data=volume_window, wavelet=wavelet, mode="symmetric", maxlevel=level
            )

            price_energy: Dict[str, float] = {}
            volume_energy: Dict[str, float] = {}

            for node in wp_price.get_level(level, "natural"):
                price_energy[node.path] = float(np.sum(node.data**2))

            for node in wp_volume.get_level(level, "natural"):
                volume_energy[node.path] = float(np.sum(node.data**2))

            # Use frequency center-based classification
            low_freq_paths, mid_freq_paths, high_freq_paths = (
                _classify_freq_bands_by_center(price_energy, level)
            )

            price_energy_low = sum(price_energy.get(p, 0.0) for p in low_freq_paths)
            price_energy_mid = sum(price_energy.get(p, 0.0) for p in mid_freq_paths)
            price_energy_high = sum(price_energy.get(p, 0.0) for p in high_freq_paths)

            volume_energy_low = sum(volume_energy.get(p, 0.0) for p in low_freq_paths)
            volume_energy_mid = sum(volume_energy.get(p, 0.0) for p in mid_freq_paths)
            volume_energy_high = sum(volume_energy.get(p, 0.0) for p in high_freq_paths)

            eps = 1e-8
            vper_low = volume_energy_low / (price_energy_low + eps)
            vper_mid = volume_energy_mid / (price_energy_mid + eps)
            vper_high = volume_energy_high / (price_energy_high + eps)

            out["wpt_vper_low"][i] = vper_low
            out["wpt_vper_mid"][i] = vper_mid
            out["wpt_vper_high"][i] = vper_high

            energy_cascade = 0.0
            total_price_energy = price_energy_low + price_energy_mid + price_energy_high
            if total_price_energy > 0:
                energy_ratio_low = price_energy_low / total_price_energy
                energy_ratio_mid = price_energy_mid / total_price_energy
                energy_ratio_high = price_energy_high / total_price_energy
                energy_cascade = (
                    energy_ratio_low + energy_ratio_mid
                ) - energy_ratio_high
                out["wpt_energy_cascade"][i] = energy_cascade

            # multi-scale consistency
            if i > 0:
                prev_price_window_raw = price[max(0, i - lookback_window - 1) : i]
                if len(prev_price_window_raw) >= 10:
                    # Use same preprocessing as current window
                    if use_log_returns:
                        prev_price_window = np.diff(np.log(prev_price_window_raw))
                        if len(prev_price_window) < level * 2:
                            continue
                    elif detrend_price:
                        from scipy import signal

                        prev_price_window = signal.detrend(prev_price_window_raw)
                    else:
                        prev_price_window = prev_price_window_raw

                    try:
                        wp_price_prev = pywt.WaveletPacket(
                            data=prev_price_window,
                            wavelet=wavelet,
                            mode="symmetric",
                            maxlevel=level,
                        )
                        prev_price_energy: Dict[str, float] = {}
                        for node in wp_price_prev.get_level(level, "natural"):
                            prev_price_energy[node.path] = float(np.sum(node.data**2))

                        mid_freq_energy_changes: List[float] = []
                        for path in mid_freq_paths:
                            curr_energy = float(price_energy.get(path, 0.0))
                            prev_energy = float(prev_price_energy.get(path, 0.0))
                            if prev_energy > 0:
                                mid_freq_energy_changes.append(
                                    (curr_energy - prev_energy) / prev_energy
                                )

                        positive_changes = sum(
                            1 for chg in mid_freq_energy_changes if chg > 0.1
                        )
                        consistency_score = min(
                            positive_changes / max(len(mid_freq_paths), 1), 1.0
                        )
                        out["wpt_multi_scale_consistency"][i] = consistency_score
                    except Exception:
                        pass

            # breakout score
            if i >= lookback_window * 2:
                vper_mid_history = out["wpt_vper_mid"][i - lookback_window : i]
                if vper_mid_history.size > 0 and np.isfinite(vper_mid_history).any():
                    vper_mid_quantile = (
                        stats.percentileofscore(
                            vper_mid_history[np.isfinite(vper_mid_history)], vper_mid
                        )
                        / 100.0
                    )
                else:
                    vper_mid_quantile = 0.5
            else:
                vper_mid_quantile = 0.5

            consistency = out["wpt_multi_scale_consistency"][i]
            breakout_confidence = np.clip(
                (energy_cascade + 1) / 2 * 0.4
                + vper_mid_quantile * 0.3
                + consistency * 0.3,
                0.0,
                1.0,
            )
            out["wpt_breakout_confidence"][i] = breakout_confidence

            if price_energy_high > price_energy_mid * 2 and vper_mid < 0.5:
                false_breakout_risk = min(breakout_confidence + 0.3, 1.0)
            else:
                false_breakout_risk = 1.0 - breakout_confidence
            out["wpt_false_breakout_risk"][i] = false_breakout_risk

        except Exception:
            continue

    result = pd.DataFrame(out, index=idx)
    for col in result.columns:
        result[col] = result[col].ffill().fillna(0.0)
    return result
