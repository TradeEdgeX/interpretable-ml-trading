"""
频谱分析特征工程

核心功能（5个稳健特征）：
1. Spectral Flatness - 频谱平坦度（信号稀疏度/压缩度）
2. High-Freq Energy Ratio - 高频能量占比（噪声强度/流动性碎片化）
3. Low-Freq Energy Ratio - 低频能量占比（趋势/慢速资金主导）
4. Spectral Entropy - 谱熵（系统有序性）
5. Spectral Centroid - 频谱重心（能量集中在低频还是高频）

策略分配建议：
- Strategy 1 (Noise-Adaptive): High-Freq Energy Ratio, Spectral Entropy
- Strategy 2 (Regime Detection): Spectral Flatness, Low-Freq Energy Ratio
- Strategy 3 (Volatility Forecasting): High-Freq Energy Ratio, Spectral Centroid
- Strategy 4 (ML Quality Control): Spectral Flatness, Spectral Entropy

设计理念：
- 金融收益率通常无显著主频，直接提取"周期"容易误导
- 频谱的分布形态（平坦度、能量分布、熵）更能反映市场状态
- 适用于：噪声水平评估、趋势强度代理、极端事件预警、市场状态识别
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from scipy import signal as sp_signal

from src.features.registry import register_feature
import os


def _rolling_quantile_normalize(arr: np.ndarray, idx: pd.Index, window: int = 252) -> np.ndarray:
    """
    滚动分位数归一化：将每个值映射为其在历史 window 中的分位数 [0, 1]
    
    用于 centroid 等未归一化的特征，使其跨品种/跨时间框架可比。
    """
    def _quantile_rank(x: np.ndarray) -> float:
        if len(x) < 10 or np.all(np.isnan(x)):
            return np.nan
        return float((x <= x[-1]).mean())
    
    s = pd.Series(arr, index=idx)
    result = s.rolling(window=window, min_periods=10).apply(_quantile_rank, raw=True)
    return result.to_numpy()


def compute_spectrum_features(
    x: np.ndarray,
    fs: float = 1.0,
    nperseg: Optional[int] = None,
) -> Dict[str, float]:
    """
    计算频谱特征（专注于稳健的频谱统计量，而非误导性的"主频"）
    
    Args:
        signal: 输入信号（建议使用收益率、差分等平稳序列）
        fs: 采样频率
        nperseg: 分段长度（默认根据信号长度动态设置）
    
    Returns:
        Dict with spectrum features:
        - has_dominant_freq: 是否存在显著主频（布尔，0/1），而非主频值本身
        - spectral_flatness: 频谱平坦度（0-1），越低表示能量越集中（趋势/共振）
        - high_freq_energy_ratio: 高频能量占比（0-1），越高表示噪声/流动性碎片化越强
        - low_freq_energy_ratio: 低频能量占比（0-1），越高表示长期驱动/宏观因素主导
        - spectral_entropy: 谱熵（0-1），越低表示系统有序性越高（如闪崩前的同步）
        - spectral_centroid: 频谱重心（Hz），能量集中在低频还是高频
    """
    # 提高最小长度要求，确保 Welch 方法有意义
    if len(x) < 8:
        return {
            "has_dominant_freq": 0.0,
            "spectral_flatness": 1.0,
            "high_freq_energy_ratio": 0.0,
            "low_freq_energy_ratio": 0.0,
            "spectral_entropy": 1.0,
            "spectral_centroid": 0.0,
        }
    
    # 动态设置 nperseg，确保在合理范围内
    # 要求：8 <= nperseg <= min(len(signal), 64)
    if nperseg is None:
        nperseg = min(max(8, len(x) // 2), 64)
    
    # 确保 nperseg 不超过信号长度，且至少为 4（Welch 最小要求）
    nperseg = min(nperseg, len(x))
    if nperseg < 4:
        # 信号太短，返回默认值
        return {
            "has_dominant_freq": 0.0,
            "spectral_flatness": 1.0,
            "high_freq_energy_ratio": 0.0,
            "low_freq_energy_ratio": 0.0,
            "spectral_entropy": 1.0,
            "spectral_centroid": 0.0,
        }
    
    # Welch's method (更稳健的功率谱估计)
    try:
        freqs, psd = sp_signal.welch(x, fs=fs, nperseg=nperseg, scaling="density")
    except Exception:
        # 异常时返回默认值
        return {
            "has_dominant_freq": 0.0,
            "spectral_flatness": 1.0,
            "high_freq_energy_ratio": 0.0,
            "low_freq_energy_ratio": 0.0,
            "spectral_entropy": 1.0,
            "spectral_centroid": 0.0,
        }
    
    # 主频显著性检查（布尔特征，而非主频值）
    # 仅当主频处 PSD 显著高于均值时才认为存在显著主频
    psd_mean = np.mean(psd)
    psd_std = np.std(psd)
    dominant_freq_idx = np.argmax(psd)
    has_dominant_freq = float(psd[dominant_freq_idx] > (psd_mean + 2 * psd_std))
    
    # 频谱平坦度（越低越压缩，表示存在短暂趋势或共振）
    # 几何平均 / 算术平均
    psd_positive = psd[psd > 0]
    if len(psd_positive) > 0:
        geometric_mean = np.exp(np.mean(np.log(psd_positive)))
        arithmetic_mean = np.mean(psd)
        spectral_flatness = (
            geometric_mean / arithmetic_mean if arithmetic_mean > 0 else 1.0
        )
    else:
        spectral_flatness = 1.0
    
    # 频率分段：低频（0 ~ fs/8）、中频（fs/8 ~ fs/4）、高频（fs/4 ~ fs/2）
    nyquist = fs / 2
    low_freq_threshold = nyquist / 4  # fs/8
    mid_freq_threshold = nyquist / 2  # fs/4
    
    low_freq_mask = freqs <= low_freq_threshold
    high_freq_mask = freqs > mid_freq_threshold
    
    low_freq_energy = np.sum(psd[low_freq_mask])
    high_freq_energy = np.sum(psd[high_freq_mask])
    total_energy = np.sum(psd)
    
    low_freq_energy_ratio = (
        low_freq_energy / total_energy if total_energy > 0 else 0.0
    )
    high_freq_energy_ratio = (
        high_freq_energy / total_energy if total_energy > 0 else 0.0
    )
    
    # 谱熵（Spectral Entropy）：衡量频谱能量分布的均匀性
    # 越低表示能量越集中（系统有序性高），越高表示能量越分散（随机性强）
    # 公式：H = -Σ(p_i * log(p_i))，其中 p_i = PSD[i] / Σ(PSD)
    psd_normalized = psd / (total_energy + 1e-12)  # 归一化，避免除零
    # 只对非零值计算熵
    psd_nonzero = psd_normalized[psd_normalized > 1e-12]
    if len(psd_nonzero) > 0:
        spectral_entropy = -np.sum(psd_nonzero * np.log(psd_nonzero + 1e-12))
        # 归一化到 [0, 1]：除以最大可能熵 log(N)
        max_entropy = np.log(len(psd_normalized) + 1e-12)
        spectral_entropy = spectral_entropy / max_entropy if max_entropy > 0 else 1.0
    else:
        spectral_entropy = 1.0
    
    # 频谱重心（Spectral Centroid）：能量集中在低频还是高频
    # 公式：C = Σ(f_i * PSD_i) / Σ(PSD_i)
    # 值越大表示能量越集中在高频（噪声/冲击），值越小表示能量越集中在低频（趋势/慢速）
    if total_energy > 1e-12:
        spectral_centroid = np.sum(freqs * psd) / total_energy
    else:
        spectral_centroid = 0.0
    
    return {
        "has_dominant_freq": has_dominant_freq,
        # Clamp all bounded/statistical outputs defensively for NN stability.
        "spectral_flatness": float(np.clip(spectral_flatness, 0.0, 1.0)),
        "high_freq_energy_ratio": float(np.clip(high_freq_energy_ratio, 0.0, 1.0)),
        "low_freq_energy_ratio": float(np.clip(low_freq_energy_ratio, 0.0, 1.0)),
        "spectral_entropy": float(np.clip(spectral_entropy, 0.0, 1.0)),
        # centroid is in Hz and depends on fs; keep as-is (unitless in bar-time scale only if fs fixed).
        "spectral_centroid": float(np.clip(spectral_centroid, 0.0, fs / 2 if fs > 0 else 0.5)),
    }


# 标准化签名的Spectrum特征计算函数
@register_feature("compute_spectrum_features", category="spectrum")
def compute_spectrum_features_main(data, is_streaming=False, state=None, **params):
    """
    Spectrum特征计算函数 - 符合标准化签名，支持流式计算
    
    Args:
        data: 输入数据 (DataFrame 或单行数据)
        is_streaming: 是否为流式模式
        state: 计算状态，用于维护滑动窗口历史数据和中间结果
        **params: 额外参数，包括:
            - close: 价格序列
            - volume: 成交量序列 (可选)
            - cvd: CVD序列 (可选)
            - rolling_window: 滚动窗口大小 (default: 64)
            - step: 计算步长 (default: 1)
    
    Returns:
        在批处理模式下返回DataFrame，在流式模式下返回单行特征值和更新的状态
    """
    # 从params中获取参数
    close = params.get('close', None)
    volume = params.get('volume', None)
    cvd = params.get('cvd', None)
    rolling_window = params.get('rolling_window', 64)
    step = params.get('step', 1)
    
    if is_streaming:
        return _compute_spectrum_features_streaming(data, state, close=close, volume=volume, cvd=cvd,
                                               rolling_window=rolling_window, step=step)
    else:
        # 批处理模式，保持原有逻辑
        return _compute_spectrum_features_batch(close=close, volume=volume, cvd=cvd,
                                           rolling_window=rolling_window, step=step)


def _compute_spectrum_features_streaming(new_data, state, **kwargs):
    """
    流式计算Spectrum特征
    
    Args:
        new_data: 新到达的数据点 (单行DataFrame或Series)
        state: 包含历史数据和中间计算结果的状态
        **kwargs: 与批处理相同的参数
        
    Returns:
        tuple: (特征值, 更新后的state)
    """
    # 从kwargs获取参数
    close = kwargs.get('close', None)
    volume = kwargs.get('volume', None)
    cvd = kwargs.get('cvd', None)
    rolling_window = kwargs.get('rolling_window', 64)
    step = kwargs.get('step', 1)
    
    # 初始化状态
    if state is None:
        state = {
            'close_history': [],  # 价格历史
            'volume_history': [],  # 成交量历史
            'cvd_history': [],  # CVD历史
            'price_returns_history': [],  # 价格收益率历史
            'volume_diff_history': [],  # 成交量差分历史
            'cvd_diff_history': [],  # CVD差分历史
            'spectrum_cache': {},  # Spectrum计算结果缓存
        }
    
    # 将新数据添加到历史记录
    if isinstance(new_data, pd.DataFrame):
        if len(new_data) == 1:
            new_row = new_data.iloc[0]
        else:
            raise ValueError("流式模式下new_data应为单行数据")
    elif isinstance(new_data, pd.Series):
        new_row = new_data
    else:
        raise ValueError("流式模式下new_data应为DataFrame单行或Series")
    
    # 更新历史数据
    if 'close' in new_row:
        state['close_history'].append(new_row['close'])
    if volume is not None and 'volume' in new_row:
        state['volume_history'].append(new_row['volume'])
    if cvd is not None and 'cvd' in new_row:
        state['cvd_history'].append(new_row['cvd'])
    
    # 计算收益率和差分
    if len(state['close_history']) > 1:
        # 计算价格收益率
        prev_close = state['close_history'][-2]
        curr_close = state['close_history'][-1]
        if prev_close != 0:
            ret = (curr_close - prev_close) / prev_close
        else:
            ret = 0.0
        state['price_returns_history'].append(ret)
        
        # 确保历史长度不超过窗口
        if len(state['price_returns_history']) > rolling_window * 2:
            state['price_returns_history'] = state['price_returns_history'][-rolling_window*2:]
    
    if volume is not None and len(state['volume_history']) > 1:
        # 计算成交量差分
        prev_vol = state['volume_history'][-2]
        curr_vol = state['volume_history'][-1]
        diff = curr_vol - prev_vol
        state['volume_diff_history'].append(diff)
        
        if len(state['volume_diff_history']) > rolling_window * 2:
            state['volume_diff_history'] = state['volume_diff_history'][-rolling_window*2:]
    
    if cvd is not None and len(state['cvd_history']) > 1:
        # 计算CVD差分
        prev_cvd = state['cvd_history'][-2]
        curr_cvd = state['cvd_history'][-1]
        diff = curr_cvd - prev_cvd
        state['cvd_diff_history'].append(diff)
        
        if len(state['cvd_diff_history']) > rolling_window * 2:
            state['cvd_diff_history'] = state['cvd_diff_history'][-rolling_window*2:]
    
    # 计算特征值
    features = {}
    
    # 计算价格频谱特征
    if len(state['price_returns_history']) >= rolling_window:
        window_returns = np.array(state['price_returns_history'][-rolling_window:])
        spec = compute_spectrum_features(window_returns)
        features['spectrum_price_has_dominant_freq'] = spec["has_dominant_freq"]
        features['spectrum_price_flatness'] = spec["spectral_flatness"]
        features['spectrum_price_high_freq_ratio'] = spec["high_freq_energy_ratio"]
        features['spectrum_price_low_freq_ratio'] = spec["low_freq_energy_ratio"]
        features['spectrum_price_entropy'] = spec["spectral_entropy"]
        features['spectrum_price_centroid'] = spec["spectral_centroid"]
    else:
        features['spectrum_price_has_dominant_freq'] = np.nan
        features['spectrum_price_flatness'] = np.nan
        features['spectrum_price_high_freq_ratio'] = np.nan
        features['spectrum_price_low_freq_ratio'] = np.nan
        features['spectrum_price_entropy'] = np.nan
        features['spectrum_price_centroid'] = np.nan
    
    # 计算成交量频谱特征
    if volume is not None and len(state['volume_diff_history']) >= rolling_window:
        window_diff = np.array(state['volume_diff_history'][-rolling_window:])
        spec = compute_spectrum_features(window_diff)
        features['spectrum_volume_flatness'] = spec["spectral_flatness"]
        features['spectrum_volume_high_freq_ratio'] = spec["high_freq_energy_ratio"]
        features['spectrum_volume_low_freq_ratio'] = spec["low_freq_energy_ratio"]
        features['spectrum_volume_entropy'] = spec["spectral_entropy"]
        features['spectrum_volume_centroid'] = spec["spectral_centroid"]
    else:
        features['spectrum_volume_flatness'] = np.nan
        features['spectrum_volume_high_freq_ratio'] = np.nan
        features['spectrum_volume_low_freq_ratio'] = np.nan
        features['spectrum_volume_entropy'] = np.nan
        features['spectrum_volume_centroid'] = np.nan
    
    # 计算CVD频谱特征
    if cvd is not None and len(state['cvd_diff_history']) >= rolling_window:
        window_diff = np.array(state['cvd_diff_history'][-rolling_window:])
        spec = compute_spectrum_features(window_diff)
        features['spectrum_cvd_flatness'] = spec["spectral_flatness"]
        features['spectrum_cvd_high_freq_ratio'] = spec["high_freq_energy_ratio"]
        features['spectrum_cvd_low_freq_ratio'] = spec["low_freq_energy_ratio"]
        features['spectrum_cvd_entropy'] = spec["spectral_entropy"]
        features['spectrum_cvd_centroid'] = spec["spectral_centroid"]
    else:
        features['spectrum_cvd_flatness'] = np.nan
        features['spectrum_cvd_high_freq_ratio'] = np.nan
        features['spectrum_cvd_low_freq_ratio'] = np.nan
        features['spectrum_cvd_entropy'] = np.nan
        features['spectrum_cvd_centroid'] = np.nan
    
    return features, state


def _compute_spectrum_features_batch(**kwargs):
    """
    批处理模式计算Spectrum特征（保持原有逻辑）
    """
    # 从kwargs获取参数
    close = kwargs.get('close', None)
    volume = kwargs.get('volume', None)
    cvd = kwargs.get('cvd', None)
    rolling_window = kwargs.get('rolling_window', 64)
    step = kwargs.get('step', 1)
    
    close = pd.to_numeric(close, errors="coerce").astype(float)
    n = len(close)
    idx = close.index

    # Fast mode: compute spectrum less frequently (and forward-fill) to save a lot of time during search.
    fast_mode = str(os.getenv("FEATURE_FAST_MODE", "")).strip() in {
        "1",
        "true",
        "True",
        "yes",
        "YES",
    }
    try:
        step_i = int(step)
    except Exception:
        step_i = 1
    if step_i < 1:
        step_i = 1
    if fast_mode and step_i == 1:
        step_i = 4

    # Allocate outputs (match extract_spectrum_features defaults)
    price_has_dom = np.zeros(n, dtype=float)
    price_flat = np.ones(n, dtype=float)
    price_high = np.zeros(n, dtype=float)
    price_low = np.zeros(n, dtype=float)
    price_ent = np.ones(n, dtype=float)
    price_cent = np.zeros(n, dtype=float)

    vol_flat = np.full(n, np.nan, dtype=float)
    vol_high = np.full(n, np.nan, dtype=float)
    vol_low = np.full(n, np.nan, dtype=float)
    vol_ent = np.full(n, np.nan, dtype=float)
    vol_cent = np.full(n, np.nan, dtype=float)

    cvd_flat = np.full(n, np.nan, dtype=float)
    cvd_high = np.full(n, np.nan, dtype=float)
    cvd_low = np.full(n, np.nan, dtype=float)
    cvd_ent = np.full(n, np.nan, dtype=float)
    cvd_cent = np.full(n, np.nan, dtype=float)

    # Price rolling spectrum (optionally downsampled by step_i; results forward-filled)
    price_returns = close.pct_change().fillna(0.0).values
    for i in range(rolling_window, n):
        if step_i > 1 and (i % step_i) != 0:
            continue
        window_returns = price_returns[i - rolling_window : i]
        spec = compute_spectrum_features(window_returns)
        price_has_dom[i] = spec["has_dominant_freq"]
        price_flat[i] = spec["spectral_flatness"]
        price_high[i] = spec["high_freq_energy_ratio"]
        price_low[i] = spec["low_freq_energy_ratio"]
        price_ent[i] = spec["spectral_entropy"]
        price_cent[i] = spec["spectral_centroid"]

    # Optional: volume rolling spectrum (diff)
    if volume is not None:
        volume = pd.to_numeric(volume, errors="coerce").astype(float)
        v = volume.values
        v_diff = np.diff(v, prepend=v[0] if len(v) else 0.0)
        for i in range(rolling_window, n):
            if step_i > 1 and (i % step_i) != 0:
                continue
            spec = compute_spectrum_features(v_diff[i - rolling_window : i])
            vol_flat[i] = spec["spectral_flatness"]
            vol_high[i] = spec["high_freq_energy_ratio"]
            vol_low[i] = spec["low_freq_energy_ratio"]
            vol_ent[i] = spec["spectral_entropy"]
            vol_cent[i] = spec["spectral_centroid"]

    # Optional: cvd rolling spectrum (diff)
    if cvd is not None:
        cvd = pd.to_numeric(cvd, errors="coerce").astype(float)
        c = cvd.values
        c_diff = np.diff(c, prepend=c[0] if len(c) else 0.0)
        for i in range(rolling_window, n):
            if step_i > 1 and (i % step_i) != 0:
                continue
            spec = compute_spectrum_features(c_diff[i - rolling_window : i])
            cvd_flat[i] = spec["spectral_flatness"]
            cvd_high[i] = spec["high_freq_energy_ratio"]
            cvd_low[i] = spec["low_freq_energy_ratio"]
            cvd_ent[i] = spec["spectral_entropy"]
            cvd_cent[i] = spec["spectral_centroid"]

    # Forward-fill downsampled outputs if step_i > 1 (keep initial warmup as defaults/NaNs)
    if step_i > 1 and n > 0:
        def _ffill(a: np.ndarray) -> np.ndarray:
            s = pd.Series(a, index=idx)
            s = s.replace([np.inf, -np.inf], np.nan).ffill()
            return s.to_numpy()

        price_has_dom = _ffill(price_has_dom)
        price_flat = _ffill(price_flat)
        price_high = _ffill(price_high)
        price_low = _ffill(price_low)
        price_ent = _ffill(price_ent)
        price_cent = _ffill(price_cent)

        if volume is not None:
            vol_flat = _ffill(vol_flat)
            vol_high = _ffill(vol_high)
            vol_low = _ffill(vol_low)
            vol_ent = _ffill(vol_ent)
            vol_cent = _ffill(vol_cent)

        if cvd is not None:
            cvd_flat = _ffill(cvd_flat)
            cvd_high = _ffill(cvd_high)
            cvd_low = _ffill(cvd_low)
            cvd_ent = _ffill(cvd_ent)
            cvd_cent = _ffill(cvd_cent)

    # 对 centroid 应用滚动分位数归一化，使其跨品种/跨时间框架可比 [0, 1]
    price_cent = _rolling_quantile_normalize(price_cent, idx, window=252)
    if volume is not None:
        vol_cent = _rolling_quantile_normalize(vol_cent, idx, window=252)
    if cvd is not None:
        cvd_cent = _rolling_quantile_normalize(cvd_cent, idx, window=252)

    return pd.DataFrame(
        {
            "spectrum_price_has_dominant_freq": price_has_dom,
            "spectrum_price_flatness": price_flat,
            "spectrum_price_high_freq_ratio": price_high,
            "spectrum_price_low_freq_ratio": price_low,
            "spectrum_price_entropy": price_ent,
            "spectrum_price_centroid": price_cent,
            "spectrum_volume_flatness": vol_flat,
            "spectrum_volume_high_freq_ratio": vol_high,
            "spectrum_volume_low_freq_ratio": vol_low,
            "spectrum_volume_entropy": vol_ent,
            "spectrum_volume_centroid": vol_cent,
            "spectrum_cvd_flatness": cvd_flat,
            "spectrum_cvd_high_freq_ratio": cvd_high,
            "spectrum_cvd_low_freq_ratio": cvd_low,
            "spectrum_cvd_entropy": cvd_ent,
            "spectrum_cvd_centroid": cvd_cent,
        },
        index=idx,
    )


# 保持原有的函数名作为别名，以便向后兼容
@register_feature("extract_spectrum_features_from_series", category="spectrum")
def extract_spectrum_features_from_series(
    *,
    close: pd.Series,
    volume: Optional[pd.Series] = None,
    cvd: Optional[pd.Series] = None,
    rolling_window: int = 64,
    step: int = 1,
) -> pd.DataFrame:
    """
    旧版函数，保持向后兼容性
    """
    return _compute_spectrum_features_batch(close=close, volume=volume, cvd=cvd,
                                        rolling_window=rolling_window, step=step)


@register_feature("extract_spectrum_price_features_from_series", category="spectrum")
def extract_spectrum_price_features_from_series(
    *,
    close: pd.Series,
    rolling_window: int = 64,
    step: int = 1,
) -> pd.DataFrame:
    """Price-only spectrum block (cheaper than spectrum_features_f: no volume/cvd)."""
    df = extract_spectrum_features_from_series(
        close=close, volume=None, cvd=None, rolling_window=rolling_window, step=step
    )
    cols = [
        "spectrum_price_has_dominant_freq",
        "spectrum_price_flatness",
        "spectrum_price_high_freq_ratio",
        "spectrum_price_low_freq_ratio",
        "spectrum_price_entropy",
        "spectrum_price_centroid",
    ]
    return df[cols]


@register_feature("extract_spectrum_volume_features_from_series", category="spectrum")
def extract_spectrum_volume_features_from_series(
    *,
    volume: pd.Series,
    rolling_window: int = 64,
    step: int = 1,
) -> pd.DataFrame:
    """Volume-only spectrum block (diff spectrum)."""
    # Reuse the shared implementation by passing a dummy close series for index alignment.
    # Price outputs are ignored.
    dummy_close = pd.Series(0.0, index=volume.index)
    df = extract_spectrum_features_from_series(
        close=dummy_close, volume=volume, cvd=None, rolling_window=rolling_window, step=step
    )
    cols = [
        "spectrum_volume_flatness",
        "spectrum_volume_high_freq_ratio",
        "spectrum_volume_low_freq_ratio",
        "spectrum_volume_entropy",
        "spectrum_volume_centroid",
    ]
    return df[cols]


@register_feature("extract_spectrum_cvd_features_from_series", category="spectrum")
def extract_spectrum_cvd_features_from_series(
    *,
    cvd: pd.Series,
    rolling_window: int = 64,
    step: int = 1,
) -> pd.DataFrame:
    """CVD-only spectrum block (diff spectrum)."""
    dummy_close = pd.Series(0.0, index=cvd.index)
    df = extract_spectrum_features_from_series(
        close=dummy_close, volume=None, cvd=cvd, rolling_window=rolling_window, step=step
    )
    cols = [
        "spectrum_cvd_flatness",
        "spectrum_cvd_high_freq_ratio",
        "spectrum_cvd_low_freq_ratio",
        "spectrum_cvd_entropy",
        "spectrum_cvd_centroid",
    ]
    return df[cols]


def add_spectrum_derived_features(
    df: pd.DataFrame,
    prefix: str = "spectrum_price",
    zscore_window: int = 50,
    diff_periods: List[int] = [1, 5, 10],
) -> pd.DataFrame:
    """
    为频谱特征添加派生特征（滚动z-score、变化率等），用于策略特定需求
    
    Args:
        df: DataFrame with spectrum features
        prefix: Prefix of spectrum features (e.g., "spectrum_price", "spectrum_volume")
        zscore_window: Window size for rolling z-score normalization
        diff_periods: List of periods for difference features
    
    Returns:
        DataFrame with derived features added:
        - {prefix}_flatness_zscore: Rolling z-score of flatness
        - {prefix}_high_freq_ratio_diff_{period}: Difference of high-freq ratio
        - etc.
    """
    df = df.copy()
    
    # 核心特征列表
    core_features = ["flatness", "high_freq_ratio", "low_freq_ratio", "entropy", "centroid"]
    
    for feature in core_features:
        col_name = f"{prefix}_{feature}"
        if col_name not in df.columns:
            continue
        
        # 1. 滚动 z-score（用于策略2和4：regime detection）
        zscore_col = f"{col_name}_zscore"
        rolling_mean = df[col_name].rolling(window=zscore_window, min_periods=zscore_window//2).mean()
        rolling_std = df[col_name].rolling(window=zscore_window, min_periods=zscore_window//2).std()
        df[zscore_col] = (df[col_name] - rolling_mean) / (rolling_std + 1e-8)
        
        # 2. 变化率特征（用于策略3：volatility forecasting）
        for period in diff_periods:
            diff_col = f"{col_name}_diff_{period}"
            df[diff_col] = df[col_name].diff(period)
        
        # 3. 滚动变化率（用于检测突变）
        change_col = f"{col_name}_change"
        df[change_col] = df[col_name].pct_change()
    
    return df


def get_strategy_spectrum_features(
    df: pd.DataFrame,
    strategy: str,
    prefix: str = "spectrum_price",
) -> pd.DataFrame:
    """
    根据策略类型返回相关的频谱特征
    
    Args:
        df: DataFrame with spectrum features
        strategy: Strategy name ("noise_adaptive", "regime_detection", 
                 "volatility_forecasting", "ml_quality_control")
        prefix: Prefix of spectrum features
    
    Returns:
        DataFrame with selected features for the strategy
    """
    strategy_features = {
        "noise_adaptive": [
            f"{prefix}_high_freq_ratio",
            f"{prefix}_entropy",
            f"{prefix}_high_freq_ratio_zscore",
            f"{prefix}_entropy_zscore",
        ],
        "regime_detection": [
            f"{prefix}_flatness",
            f"{prefix}_low_freq_ratio",
            f"{prefix}_flatness_zscore",
            f"{prefix}_low_freq_ratio_zscore",
        ],
        "volatility_forecasting": [
            f"{prefix}_high_freq_ratio",
            f"{prefix}_centroid",
            f"{prefix}_high_freq_ratio_diff_5",
            f"{prefix}_centroid_diff_5",
        ],
        "ml_quality_control": [
            f"{prefix}_flatness",
            f"{prefix}_entropy",
            f"{prefix}_flatness_zscore",
            f"{prefix}_entropy_zscore",
        ],
    }
    
    if strategy not in strategy_features:
        raise ValueError(
            f"Unknown strategy: {strategy}. "
            f"Must be one of: {list(strategy_features.keys())}"
        )
    
    # 先添加派生特征
    df = add_spectrum_derived_features(df, prefix=prefix)
    
    # 返回策略相关的特征
    available_features = [f for f in strategy_features[strategy] if f in df.columns]
    return df[available_features]
