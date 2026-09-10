"""
统一的 Volume Profile 特征计算

整合所有 Volume Profile 相关功能：
1. WPT 降噪的 Volume Profile 基础计算
2. POC/HAL 特征（价值区间）
3. HVN/LVN 特征（流动性节点）
4. 衍生特征（无量纲特征）

所有功能统一在一个文件中，避免代码分散。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd

from src.features.registry import register_feature

# pywt is absent from lean CI (requirements-ci.txt). Import on use so OMS /
# OrderFlowListener import paths do not require the research stack.


def _pywt():
    import pywt

    return pywt


def freedman_diaconis_bins(data: np.ndarray, min_bins: int = 10, max_bins: int = 100) -> int:
    """
    使用 Freedman–Diaconis rule 自动计算直方图 bin 数量。
    
    该方法特别适合处理非正态分布或含异常值的数据，比传统的 Sturges 公式更鲁棒。
    
    公式：
        bin_width = 2 * IQR(x) / n^(1/3)
        bins = ceil((max(x) - min(x)) / bin_width)
    
    Args:
        data: 数据序列
        min_bins: 最小 bin 数量（默认 10）
        max_bins: 最大 bin 数量（默认 100）
    
    Returns:
        计算得到的 bin 数量（限制在 [min_bins, max_bins] 范围内）
    
    Examples:
        >>> x = np.random.normal(100, 5, 1000)
        >>> bins = freedman_diaconis_bins(x)
        >>> assert 20 <= bins <= 40
    """
    data = np.asarray(data)
    n = len(data)
    
    if n < 2:
        return min_bins
    
    # 计算 IQR（四分位距）
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    
    # 如果 IQR = 0（所有值相同或几乎相同），使用 fallback
    if iqr <= 0:
        # fallback: 使用数据范围，假设需要 10 个 bins
        data_range = np.ptp(data)  # peak-to-peak (max - min)
        if data_range <= 1e-10:
            # 所有值完全相同，使用默认值
            return min_bins
        # 使用固定比例：假设每个 bin 占数据范围的 1/10
        bin_width = data_range / 10.0
    else:
        # 标准 FD rule：bin_width = 2 * IQR / n^(1/3)
        bin_width = 2 * iqr / (n ** (1/3))
    
    if bin_width <= 0:
        return min_bins
    
    # 计算 bins 数量：ceil((max - min) / bin_width)
    data_range = np.max(data) - np.min(data)
    bins = int(np.ceil(data_range / bin_width))
    
    # 限制在合理范围内
    return int(np.clip(bins, min_bins, max_bins))


@dataclass
class VolumeProfileResult:
    """Volume Profile 计算结果"""
    hist: np.ndarray
    edges: np.ndarray
    centers: np.ndarray
    price_min: float
    price_max: float
    price_denoised: Optional[np.ndarray] = None
    hal_high: Optional[float] = None
    hal_low: Optional[float] = None


def compute_raw_volume_profile(
    price_window: np.ndarray,
    volume_window: np.ndarray,
    bins: int | str = "auto",
    value_area_ratio: float = 0.7,
) -> Optional[VolumeProfileResult]:
    """
    使用原始价格序列（不使用 WPT）构建 volume profile 直方图。
    
    Args:
        price_window: 价格序列窗口（原始价格，不降噪）
        volume_window: 成交量序列窗口（与 price_window 等长）
        bins: 直方图 bins 数量。如果为 "auto"（默认），则使用 Freedman-Diaconis rule 自动计算
        value_area_ratio: Value Area ratio (default: 0.7, i.e., 70%)
    
    Returns:
        VolumeProfileResult 对象，包含：
        - hist: 成交量直方图
        - edges: bins 边界
        - centers: bins 中心价格
        - price_min: 价格最小值
        - price_max: 价格最大值
        - hal_high: HAL high (Value Area upper bound)
        - hal_low: HAL low (Value Area lower bound)
    """
    if price_window is None or volume_window is None:
        return None
    
    if len(price_window) != len(volume_window) or len(price_window) < 10:
        return None
    
    price_window = np.asarray(price_window, dtype=float)
    volume_window = np.asarray(volume_window, dtype=float)
    
    # 过滤无效值
    valid_mask = (
        np.isfinite(price_window)
        & np.isfinite(volume_window)
        & (volume_window > 0)
    )
    
    if not np.any(valid_mask):
        return None
    
    price_valid = price_window[valid_mask]
    volume_valid = volume_window[valid_mask]
    
    if len(price_valid) < 10:
        return None
    
    price_min = price_valid.min()
    price_max = price_valid.max()
    price_range = price_max - price_min
    
    if not np.isfinite(price_min) or not np.isfinite(price_max) or price_range <= 1e-10:
        return None
    
    # 动态 bins 计算
    if isinstance(bins, str) and bins == "auto":
        bins = freedman_diaconis_bins(price_valid, min_bins=10, max_bins=100)
    
    bins = min(bins, len(price_valid))
    if bins < 1:
        bins = 1
    
    # 构建 volume profile
    hist, edges = np.histogram(
        price_valid,
        bins=bins,
        range=(price_min, price_max),
        weights=volume_valid,
    )
    
    if np.sum(hist) <= 0:
        return None
    
    centers = (edges[:-1] + edges[1:]) / 2
    
    # 计算 HAL (Value Area)
    total_volume = hist.sum()
    target_volume = total_volume * value_area_ratio
    max_vol_idx = int(np.argmax(hist))
    accumulated_volume = hist[max_vol_idx]
    upper_idx = max_vol_idx
    lower_idx = max_vol_idx
    
    while accumulated_volume < target_volume:
        upper_vol = hist[upper_idx + 1] if upper_idx + 1 < len(hist) else 0.0
        lower_vol = hist[lower_idx - 1] if lower_idx - 1 >= 0 else 0.0
        
        if (upper_vol >= lower_vol and upper_idx + 1 < len(hist)) or lower_idx == 0:
            if upper_idx + 1 < len(hist):
                upper_idx += 1
                accumulated_volume += hist[upper_idx]
            else:
                break
        elif lower_idx - 1 >= 0:
            lower_idx -= 1
            accumulated_volume += hist[lower_idx]
        else:
            break
    
    # HAL 上下界对应价格档的边界
    hal_high_edge_idx = min(upper_idx + 1, len(edges) - 1)
    hal_low_edge_idx = max(lower_idx, 0)
    hal_high = edges[hal_high_edge_idx]
    hal_low = edges[hal_low_edge_idx]
    
    return VolumeProfileResult(
        hist=hist,
        edges=edges,
        centers=centers,
        price_min=float(price_min),
        price_max=float(price_max),
        price_denoised=None,
        hal_high=float(hal_high),
        hal_low=float(hal_low),
    )


def compute_wpt_volume_profile(
    price_window: np.ndarray,
    volume_window: np.ndarray,
    bins: int | str = "auto",
    wavelet: str = "db4",
    level: int = 4,
    drop_high_freq: bool = True,
) -> Optional[VolumeProfileResult]:
    """
    使用 WPT 降噪后的价格序列构建 volume profile 直方图。
    
    Args:
        price_window: 价格序列窗口
        volume_window: 成交量序列窗口（与 price_window 等长）
        bins: 直方图 bins 数量。如果为 "auto"（默认），则使用 Freedman-Diaconis rule 自动计算
        wavelet: 小波函数名称（默认 "db4"）
        level: WPT 分解层数（默认 4）
        drop_high_freq: 是否剔除高频子带进行降噪（默认 True）
            - 如果为 True，将按频率排序剔除最高频的 25% 子带
    
    Returns:
        VolumeProfileResult 对象，包含：
        - hist: 成交量直方图
        - edges: bins 边界
        - centers: bins 中心价格
        - price_min: 价格最小值
        - price_max: 价格最大值
        - price_denoised: 降噪后的价格序列（可选，用于上层特征扩展）
    """
    if price_window is None or volume_window is None:
        return None
    
    if len(price_window) != len(volume_window) or len(price_window) < 10:
        return None
    
    price_window = np.asarray(price_window, dtype=float)
    volume_window = np.asarray(volume_window, dtype=float)
    
    # Step 1: WPT 降噪（改进版：按频率排序剔除高频子带）
    try:
        pywt = _pywt()
        _ = pywt.Wavelet(wavelet)
        max_level = pywt.dwt_max_level(len(price_window), wavelet)
        actual_level = min(level, max_level) if max_level > 0 else 1
    except (ValueError, RuntimeError, TypeError, ImportError):
        price_denoised = price_window
        actual_level = 0
    
    if actual_level < 1:
        price_denoised = price_window
    else:
        try:
            pywt = _pywt()
            wp = pywt.WaveletPacket(
                data=price_window, wavelet=wavelet, mode="symmetric", maxlevel=actual_level
            )
            if drop_high_freq:
                freq_nodes = wp.get_level(actual_level, "freq")
                if len(freq_nodes) > 0:
                    n_drop = max(1, len(freq_nodes) // 4)
                    for node in freq_nodes[-n_drop:]:
                        wp[node.path].data = np.zeros_like(node.data)
            price_denoised = wp.reconstruct(update=True)
            
            if len(price_denoised) != len(price_window):
                if len(price_denoised) > len(price_window):
                    price_denoised = price_denoised[:len(price_window)]
                else:
                    price_denoised = price_window
        except (ValueError, RuntimeError, TypeError):
            price_denoised = price_window
    
    # Step 2: 过滤无效值
    valid_mask = (
        np.isfinite(price_denoised)
        & np.isfinite(volume_window)
        & (volume_window > 0)
    )
    
    if not np.any(valid_mask):
        return None
    
    price_valid = price_denoised[valid_mask]
    volume_valid = volume_window[valid_mask]
    
    if len(price_valid) < 10:
        return None
    
    price_min = price_valid.min()
    price_max = price_valid.max()
    price_range = price_max - price_min
    
    if not np.isfinite(price_min) or not np.isfinite(price_max) or price_range <= 1e-10:
        return None
    
    # Step 3: 动态 bins 计算
    if isinstance(bins, str) and bins == "auto":
        bins = freedman_diaconis_bins(price_valid, min_bins=10, max_bins=100)
    
    bins = min(bins, len(price_valid))
    if bins < 1:
        bins = 1
    
    # Step 4: 构建 volume profile
    hist, edges = np.histogram(
        price_valid,
        bins=bins,
        range=(price_min, price_max),
        weights=volume_valid,
    )
    
    if np.sum(hist) <= 0:
        return None
    
    centers = (edges[:-1] + edges[1:]) / 2
    
    # 计算 HAL (Value Area) - 与 raw VP 保持一致的计算逻辑
    total_volume = hist.sum()
    value_area_ratio = 0.7  # 默认值，与 compute_unified_volume_profile_features 保持一致
    target_volume = total_volume * value_area_ratio
    max_vol_idx = int(np.argmax(hist))
    accumulated_volume = hist[max_vol_idx]
    upper_idx = max_vol_idx
    lower_idx = max_vol_idx
    
    while accumulated_volume < target_volume:
        upper_vol = hist[upper_idx + 1] if upper_idx + 1 < len(hist) else 0.0
        lower_vol = hist[lower_idx - 1] if lower_idx - 1 >= 0 else 0.0
        
        if (upper_vol >= lower_vol and upper_idx + 1 < len(hist)) or lower_idx == 0:
            if upper_idx + 1 < len(hist):
                upper_idx += 1
                accumulated_volume += hist[upper_idx]
            else:
                break
        elif lower_idx - 1 >= 0:
            lower_idx -= 1
            accumulated_volume += hist[lower_idx]
        else:
            break
    
    # HAL 上下界对应价格档的边界
    hal_high_edge_idx = min(upper_idx + 1, len(edges) - 1)
    hal_low_edge_idx = max(lower_idx, 0)
    hal_high = edges[hal_high_edge_idx]
    hal_low = edges[hal_low_edge_idx]
    
    return VolumeProfileResult(
        hist=hist,
        edges=edges,
        centers=centers,
        price_min=float(price_min),
        price_max=float(price_max),
        price_denoised=price_denoised,
        hal_high=float(hal_high),
        hal_low=float(hal_low),
    )


def compute_unified_volume_profile_features(
    df: pd.DataFrame,
    price_col: str = "close",
    volume_col: str = "volume",
    high_col: str = "high",
    low_col: str = "low",
    window: int = 160,
    bins: int | str = "auto",
    value_area_ratio: float = 0.7,
    wavelet: str = "db4",
    level: int = 4,
    drop_high_freq: bool = True,
    use_typical_price: bool = False,
    price_series: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    统一的 Volume Profile 特征计算（合并 POC/HAL 和 VPVR）
    
    在一次计算中同时输出：
    1. POC/HAL 特征（价值区间）
    2. HVN/LVN 特征（流动性节点）
    3. 边界稳定性分数（raw VP vs WPT VP 的一致性）
    
    Args:
        df: DataFrame with OHLCV data
        price_col: Price column name (default: "close")
        volume_col: Volume column name (default: "volume")
        high_col: High column name (default: "high")
        low_col: Low column name (default: "low")
        window: Rolling window size (default: 160)
        bins: Number of price bins. If "auto" (default), uses Freedman-Diaconis rule
        value_area_ratio: Value Area ratio (default: 0.7, i.e., 70%)
        wavelet: Wavelet function (default: "db4")
        level: WPT decomposition level (default: 4)
        drop_high_freq: Whether to drop highest frequency subband (default: True)
        use_typical_price: If True, use (H+L+C)/3; else use price_col or price_series
        price_series: Optional price series (e.g., WPT reconstructed price)
    
    Returns:
        DataFrame with unified volume profile features:
        
        # POC/HAL 特征
        - vp_poc: Point of Control (price with highest volume)
        - vp_poc_volume_ratio: Volume ratio at POC
        - vp_hal_high: HAL high (Value Area upper bound)
        - vp_hal_low: HAL low (Value Area lower bound)
        - vp_hal_mid: HAL mid point
        
        # HVN/LVN 特征
        - vp_hvn_count: High Volume Node count
        - vp_lvn_count: Low Volume Node count
        - vp_lvn_distance: Distance to nearest LVN (normalized)
        - vp_volume_density: Volume density at current price
        - vp_price_in_lvn: Whether current price is in LVN (1.0/0.0)
        
        # 边界稳定性特征
        - vp_boundary_stability_score: Raw VP vs WPT VP 的边界一致性分数 (0-1)
          - 1.0 = 完美一致（结构稳定）
          - 接近 0 = 结构不一致（噪音 / 过渡）
    """
    df = df.copy()
    
    # 确定使用的价格序列
    if price_series is not None:
        price_data = price_series
    elif use_typical_price and high_col in df.columns and low_col in df.columns:
        price_data = (df[high_col] + df[low_col] + df[price_col]) / 3.0
    else:
        price_data = df[price_col]
    
    # 初始化输出列
    # POC/HAL 特征
    df["vp_poc"] = np.nan
    df["vp_poc_volume_ratio"] = np.nan
    df["vp_hal_high"] = np.nan
    df["vp_hal_low"] = np.nan
    df["vp_hal_mid"] = np.nan
    
    # HVN/LVN 特征
    df["vp_hvn_count"] = 0.0
    df["vp_lvn_count"] = 0.0
    df["vp_lvn_distance"] = np.nan
    df["vp_volume_density"] = 0.0
    df["vp_price_in_lvn"] = 0.0
    
    # 边界稳定性分数（raw VP vs WPT VP）
    df["vp_boundary_stability_score"] = np.nan
    
    # 滚动窗口计算
    for i in range(window, len(df)):
        price_window = price_data.iloc[i - window : i].values
        volume_window = df[volume_col].iloc[i - window : i].values
        
        # 同时计算 raw VP 和 WPT VP
        raw_vp_result: Optional[VolumeProfileResult] = compute_raw_volume_profile(
            price_window=price_window,
            volume_window=volume_window,
            bins=bins,
            value_area_ratio=value_area_ratio,
        )
        
        wpt_vp_result: Optional[VolumeProfileResult] = compute_wpt_volume_profile(
            price_window=price_window,
            volume_window=volume_window,
            bins=bins,
            wavelet=wavelet,
            level=level,
            drop_high_freq=drop_high_freq,
        )
        
        # 计算边界稳定性分数
        if raw_vp_result is not None and wpt_vp_result is not None:
            stability_score = compute_boundary_stability_score(
                raw_vp=raw_vp_result,
                wpt_vp=wpt_vp_result,
                tau=0.075,
            )
            df.iloc[i, df.columns.get_loc("vp_boundary_stability_score")] = stability_score
        
        # 使用 WPT VP 作为主要输出（保持向后兼容）
        vp_result = wpt_vp_result
        
        if vp_result is None:
            continue
        
        hist = vp_result.hist
        edges = vp_result.edges
        centers = vp_result.centers
        total_volume = hist.sum()
        
        if total_volume <= 0:
            continue
        
        current_price = price_data.iloc[i]
        price_range = vp_result.price_max - vp_result.price_min
        
        # ========== POC/HAL 计算 ==========
        # 1. POC (Point of Control)
        max_vol_idx = int(np.argmax(hist))
        poc_price = centers[max_vol_idx]
        poc_volume_ratio = hist[max_vol_idx] / total_volume
        
        df.iloc[i, df.columns.get_loc("vp_poc")] = poc_price
        df.iloc[i, df.columns.get_loc("vp_poc_volume_ratio")] = poc_volume_ratio
        
        # 2. HAL (Value Area 70%)
        target_volume = total_volume * value_area_ratio
        accumulated_volume = hist[max_vol_idx]
        upper_idx = max_vol_idx
        lower_idx = max_vol_idx
        
        while accumulated_volume < target_volume:
            upper_vol = hist[upper_idx + 1] if upper_idx + 1 < len(hist) else 0.0
            lower_vol = hist[lower_idx - 1] if lower_idx - 1 >= 0 else 0.0
            
            if (upper_vol >= lower_vol and upper_idx + 1 < len(hist)) or lower_idx == 0:
                if upper_idx + 1 < len(hist):
                    upper_idx += 1
                    accumulated_volume += hist[upper_idx]
                else:
                    break
            elif lower_idx - 1 >= 0:
                lower_idx -= 1
                accumulated_volume += hist[lower_idx]
            else:
                break
        
        # HAL 上下界对应价格档的边界
        hal_high_edge_idx = min(upper_idx + 1, len(edges) - 1)
        hal_low_edge_idx = max(lower_idx, 0)
        hal_high = edges[hal_high_edge_idx]
        hal_low = edges[hal_low_edge_idx]
        hal_mid = (hal_high + hal_low) / 2.0
        
        df.iloc[i, df.columns.get_loc("vp_hal_high")] = hal_high
        df.iloc[i, df.columns.get_loc("vp_hal_low")] = hal_low
        df.iloc[i, df.columns.get_loc("vp_hal_mid")] = hal_mid
        
        # ========== HVN/LVN 计算 ==========
        # 1. 计算 HVN 和 LVN
        positive_mask = hist > 0
        volume_mean = np.mean(hist[positive_mask]) if np.any(positive_mask) else 0
        volume_std = np.std(hist[positive_mask]) if np.any(positive_mask) else 0
        
        if volume_std > 0:
            # HVN: 成交量 > mean + 0.5*std
            hvn_mask = hist > (volume_mean + 0.5 * volume_std)
            df.iloc[i, df.columns.get_loc("vp_hvn_count")] = np.sum(hvn_mask)
            
            # LVN: 成交量 < mean - 0.5*std
            lvn_mask = hist < (volume_mean - 0.5 * volume_std)
            df.iloc[i, df.columns.get_loc("vp_lvn_count")] = np.sum(lvn_mask)
            
            # 2. 计算当前价格到最近 LVN 的距离
            lvn_prices = centers[lvn_mask]
            
            if len(lvn_prices) > 0:
                lvn_distances = np.abs(lvn_prices - current_price)
                nearest_lvn_idx = np.argmin(lvn_distances)
                nearest_lvn_price = lvn_prices[nearest_lvn_idx]
                nearest_lvn_distance = lvn_distances[nearest_lvn_idx]
                
                # 归一化距离（相对于价格范围）
                if price_range > 0:
                    df.iloc[i, df.columns.get_loc("vp_lvn_distance")] = (
                        nearest_lvn_distance / price_range
                    )
                
                # 判断当前价格是否在 LVN 中
                if isinstance(bins, int):
                    bin_width = price_range / bins
                else:
                    bin_width = price_range / len(centers) if len(centers) > 0 else 0
                
                if bin_width > 0 and np.abs(current_price - nearest_lvn_price) < bin_width:
                    df.iloc[i, df.columns.get_loc("vp_price_in_lvn")] = 1.0
        
        # 3. 计算当前价格的成交量密度
        current_price_bin = np.digitize(current_price, edges) - 1
        current_price_bin = np.clip(current_price_bin, 0, len(hist) - 1)
        
        if current_price_bin < len(hist):
            current_volume = hist[current_price_bin]
            max_volume = np.max(hist) if len(hist) > 0 else 1.0
            df.iloc[i, df.columns.get_loc("vp_volume_density")] = (
                current_volume / max_volume if max_volume > 0 else 0.0
            )
    
    # 前向填充缺失值
    df["vp_poc"] = df["vp_poc"].ffill()
    df["vp_hal_high"] = df["vp_hal_high"].ffill()
    df["vp_hal_low"] = df["vp_hal_low"].ffill()
    df["vp_hal_mid"] = df["vp_hal_mid"].ffill()
    df["vp_lvn_distance"] = df["vp_lvn_distance"].fillna(0.0)
    df["vp_boundary_stability_score"] = df["vp_boundary_stability_score"].ffill().fillna(0.0)
    
    return df


def compute_boundary_stability_score(
    raw_vp: VolumeProfileResult,
    wpt_vp: VolumeProfileResult,
    tau: float = 0.075,
) -> float:
    """
    计算 raw VP 和 WPT VP 的边界一致性分数。
    
    比较 raw VP 和 WPT VP 的 HAL 边界差异，用一致性来判断：
    当前市场是"结构主导"还是"噪音 / 过渡主导"。
    
    Args:
        raw_vp: Raw volume profile result (不使用 WPT)
        wpt_vp: WPT volume profile result (使用 WPT 降噪)
        tau: Temperature parameter (default: 0.075, 建议范围 0.05 ~ 0.1)
    
    Returns:
        Stability score in (0, 1], where:
        - 1.0 = perfect consistency (结构一致)
        - 接近 0 = 结构不一致 (噪音 / 过渡)
    
    Formula:
        Δ_high = |HAL_high_raw − HAL_high_wpt|
        Δ_low  = |HAL_low_raw  − HAL_low_wpt|
        range = max(price_max_raw, price_max_wpt) - min(price_min_raw, price_min_wpt)
        d = (Δ_high + Δ_low) / (2 * range)
        score = exp(− d / τ)
    """
    if raw_vp is None or wpt_vp is None:
        return 0.0
    
    if raw_vp.hal_high is None or raw_vp.hal_low is None:
        return 0.0
    
    if wpt_vp.hal_high is None or wpt_vp.hal_low is None:
        return 0.0
    
    # 计算 HAL 差异
    delta_high = abs(raw_vp.hal_high - wpt_vp.hal_high)
    delta_low = abs(raw_vp.hal_low - wpt_vp.hal_low)
    
    # 归一化：使用两个 VP 的联合价格范围
    price_min_union = min(raw_vp.price_min, wpt_vp.price_min)
    price_max_union = max(raw_vp.price_max, wpt_vp.price_max)
    price_range = price_max_union - price_min_union
    
    if price_range <= 0:
        return 0.0
    
    d = (delta_high + delta_low) / (2.0 * price_range)
    
    # 映射到稳定性分数
    score = np.exp(-d / tau)
    return float(np.clip(score, 0.0, 1.0))


def compute_unified_volume_profile_derived_features(
    df: pd.DataFrame,
    price_col: str = "close",
) -> pd.DataFrame:
    """
    计算统一的 Volume Profile 衍生特征（无量纲特征）
    
    基于 compute_unified_volume_profile_features 的输出，计算相对距离等衍生特征。
    
    Args:
        df: DataFrame with volume profile features (from compute_unified_volume_profile_features)
        price_col: Price column name (default: "close")
    
    Returns:
        DataFrame with derived features added:
        
        # POC 衍生特征
        - vp_price_to_poc_pct: Current price to POC relative distance
        - vp_poc_position_ratio: POC position in price range (0-1)
        
        # HAL 衍生特征
        - vp_price_to_hal_high_pct: Current price to HAL high relative distance
        - vp_price_to_hal_low_pct: Current price to HAL low relative distance
        - vp_price_to_hal_mid_pct: Current price to HAL mid relative distance
        - vp_hal_bandwidth_pct: HAL bandwidth (relative)
    """
    df = df.copy()
    
    close = df[price_col].replace(0, np.nan)
    
    # POC 衍生特征
    if "vp_poc" in df.columns:
        poc = df["vp_poc"]
        
        # 当前价格到 POC 的相对距离
        if "vp_price_to_poc_pct" not in df.columns:
            df["vp_price_to_poc_pct"] = ((poc - close) / close).replace(
                [np.inf, -np.inf], np.nan
            )
        
        # POC 在价格区间中的位置（需要 high/low）
        if "high" in df.columns and "low" in df.columns:
            price_range = df["high"] - df["low"]
            if "vp_poc_position_ratio" not in df.columns:
                df["vp_poc_position_ratio"] = (
                    ((poc - df["low"]) / price_range).replace([np.inf, -np.inf], np.nan)
                )
    
    # HAL 衍生特征
    if "vp_hal_high" in df.columns and "vp_hal_low" in df.columns:
        hal_high = df["vp_hal_high"]
        hal_low = df["vp_hal_low"]
        hal_mid = df.get("vp_hal_mid", (hal_high + hal_low) / 2.0)
        
        # 当前价格到 HAL 的相对距离
        if "vp_price_to_hal_high_pct" not in df.columns:
            df["vp_price_to_hal_high_pct"] = (
                ((hal_high - close) / close).replace([np.inf, -np.inf], np.nan)
            )
        
        if "vp_price_to_hal_low_pct" not in df.columns:
            df["vp_price_to_hal_low_pct"] = (
                ((hal_low - close) / close).replace([np.inf, -np.inf], np.nan)
            )
        
        if "vp_price_to_hal_mid_pct" not in df.columns:
            df["vp_price_to_hal_mid_pct"] = (
                ((hal_mid - close) / close).replace([np.inf, -np.inf], np.nan)
            )
        
        # HAL 带宽（相对）
        if "vp_hal_bandwidth_pct" not in df.columns:
            hal_bandwidth = hal_high - hal_low
            df["vp_hal_bandwidth_pct"] = (
                (hal_bandwidth / close).replace([np.inf, -np.inf], np.nan)
            )
    
    return df


@register_feature("compute_wpt_vpvr_from_series", category="volume_profile")
def compute_wpt_vpvr_from_series(
    *,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
    window: int = 100,
    bins: int | str = 50,
    wavelet: str = "db4",
    level: int = 4,
    drop_high_freq: bool = True,
) -> pd.DataFrame:
    """
    Narrow-IO VPVR (Visible Range Volume Profile) features based on WPT-denoised volume profile.

    Outputs the strategy-facing `vpvr_*` columns:
    - vpvr_pvp (≈ POC)
    - vpvr_hvn_count / vpvr_lvn_count
    - vpvr_lvn_distance
    - vpvr_volume_density
    - vpvr_price_in_lvn
    """
    close_s = pd.to_numeric(close, errors="coerce").astype(float)
    high_s = pd.to_numeric(high, errors="coerce").astype(float)
    low_s = pd.to_numeric(low, errors="coerce").astype(float)
    vol_s = pd.to_numeric(volume, errors="coerce").astype(float)
    idx = close_s.index
    n = len(close_s)

    # Typical price (VPVR uses typical price)
    price_data = (high_s + low_s + close_s) / 3.0

    out_cols = [
        "vpvr_pvp",
        "vpvr_hvn_count",
        "vpvr_lvn_count",
        "vpvr_lvn_distance",
        "vpvr_volume_density",
        "vpvr_price_in_lvn",
    ]
    arr = {
        "vpvr_pvp": np.full(n, np.nan, dtype=float),
        "vpvr_hvn_count": np.zeros(n, dtype=float),
        "vpvr_lvn_count": np.zeros(n, dtype=float),
        "vpvr_lvn_distance": np.full(n, np.nan, dtype=float),
        "vpvr_volume_density": np.zeros(n, dtype=float),
        "vpvr_price_in_lvn": np.zeros(n, dtype=float),
    }

    min_length = max(window, 2**level)
    price_vals = price_data.values
    vol_vals = vol_s.values

    for i in range(window, n):
        if i < min_length:
            continue
        price_window = price_vals[i - window : i]
        volume_window = vol_vals[i - window : i]
        current_price = price_vals[i]

        vp_result = compute_wpt_volume_profile(
            price_window=price_window,
            volume_window=volume_window,
            bins=bins,
            wavelet=wavelet,
            level=level,
            drop_high_freq=drop_high_freq,
        )
        if vp_result is None:
            continue

        hist = vp_result.hist
        edges = vp_result.edges
        centers = vp_result.centers
        total_volume = float(np.sum(hist))
        if total_volume <= 0:
            continue

        # PVP (POC)
        max_vol_idx = int(np.argmax(hist))
        pvp_price = float(centers[max_vol_idx])
        arr["vpvr_pvp"][i] = pvp_price

        # HVN/LVN masks
        positive_mask = hist > 0
        volume_mean = float(np.mean(hist[positive_mask])) if np.any(positive_mask) else 0.0
        volume_std = float(np.std(hist[positive_mask])) if np.any(positive_mask) else 0.0

        price_range = float(vp_result.price_max - vp_result.price_min)
        if volume_std > 0:
            hvn_mask = hist > (volume_mean + 0.5 * volume_std)
            lvn_mask = hist < (volume_mean - 0.5 * volume_std)
            arr["vpvr_hvn_count"][i] = float(np.sum(hvn_mask))
            arr["vpvr_lvn_count"][i] = float(np.sum(lvn_mask))

            lvn_prices = centers[lvn_mask]
            if len(lvn_prices) > 0 and price_range > 0:
                lvn_distances = np.abs(lvn_prices - current_price)
                nearest_idx = int(np.argmin(lvn_distances))
                nearest_price = float(lvn_prices[nearest_idx])
                nearest_dist = float(lvn_distances[nearest_idx])
                arr["vpvr_lvn_distance"][i] = nearest_dist / price_range

                if isinstance(bins, int):
                    bin_width = price_range / float(bins) if bins > 0 else 0.0
                else:
                    bin_width = price_range / float(len(centers)) if len(centers) > 0 else 0.0
                if bin_width > 0 and abs(current_price - nearest_price) < bin_width:
                    arr["vpvr_price_in_lvn"][i] = 1.0

        # Volume density at current price
        current_price_bin = int(np.digitize(current_price, edges) - 1)
        current_price_bin = int(np.clip(current_price_bin, 0, len(hist) - 1))
        current_volume = float(hist[current_price_bin]) if 0 <= current_price_bin < len(hist) else 0.0
        max_volume = float(np.max(hist)) if len(hist) > 0 else 1.0
        arr["vpvr_volume_density"][i] = current_volume / max_volume if max_volume > 0 else 0.0

    out = pd.DataFrame(arr, index=idx)
    out["vpvr_pvp"] = out["vpvr_pvp"].ffill()
    out["vpvr_lvn_distance"] = out["vpvr_lvn_distance"].fillna(0.0)
    out[out_cols] = out[out_cols].fillna(0.0)
    return out[out_cols]


@register_feature("compute_volume_profile_vpvr_from_series", category="volume_profile")
def compute_volume_profile_vpvr_from_series(
    *,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
    window: int = 100,
    bins: int | str = 50,
    wavelet: str = "db4",
    level: int = 4,
    drop_high_freq: bool = True,
) -> pd.DataFrame:
    """
    Canonical VPVR/volume-profile feature entrypoint (alias of compute_wpt_vpvr_from_series).
    This exists to avoid having both 'wpt_vpvr' and 'volume profile' concepts in configs.
    """
    return compute_wpt_vpvr_from_series(
        close=close,
        high=high,
        low=low,
        volume=volume,
        window=window,
        bins=bins,
        wavelet=wavelet,
        level=level,
        drop_high_freq=drop_high_freq,
    )
