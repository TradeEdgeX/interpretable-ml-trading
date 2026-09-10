"""
Hilbert 变换特征工程（改进版 + 高级功能）

核心原则：
1. 只提取 Hilbert 包络（envelope）作为特征，放弃瞬时相位和频率
2. 使用滚动窗口计算，避免未来信息泄露
3. EMA 平滑包络序列，提升稳健性
4. 提取实用衍生特征（包络比值、斜率等）

高级功能：
- 自适应窗口：基于局部周期估计，动态匹配市场波动周期
- 分位数标准化：跨品种可比，支持多品种统一模型
- 成交量融合：结合价格、资金流、成交量，识别背离信号

原因：
- 金融波动信号非窄带，相位/频率噪声大、不可靠
- 包络可作为瞬时波动强度的有效代理
- EMA 是因果、高效、金融友好的平滑器
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional
from scipy.signal import hilbert

from src.features.registry import register_feature


def compute_hilbert_envelope(
    signal: np.ndarray,
) -> np.ndarray:
    """
    计算信号的 Hilbert 包络
    
    Args:
        signal: 输入信号（应已去趋势，如 WPT 波动分量）
    
    Returns:
        包络序列（envelope = |analytic_signal|）
    """
    # Hilbert 变换
    analytic = hilbert(signal)
    
    # 包络
    envelope = np.abs(analytic)
    
    return envelope


def estimate_local_period(
    signal: np.ndarray,
    min_period: int = 8,
    max_period: int = 128,
) -> int:
    """
    基于零穿越率（Zero-Crossing Rate）估计局部周期
    
    对去趋势信号，相邻零点间距 ≈ 半周期 → 可估计局部周期
    
    Args:
        signal: 输入信号（应已去趋势）
        min_period: 最小周期长度
        max_period: 最大周期长度
    
    Returns:
        估计的周期长度（整数）
    """
    # 内部 NaN 过滤：防止外部传入含 NaN 的数组导致符号判断错误
    signal = signal[~np.isnan(signal)]
    
    if len(signal) < 20:
        return min(max_period, len(signal))
    
    # 符号序列（0 视为正）
    signs = np.sign(signal)
    signs[signs == 0] = 1
    
    # 零穿越位置（符号变化处）
    # 注意：仅使用 signal[:-1] 判断穿越，避免用最后一个点
    crossings = np.where(np.diff(signs) != 0)[0] + 1  # +1 因为 diff 缩短了
    
    if len(crossings) < 2:
        return min_period
    
    # 计算最近几个半周期长度（取后 M 个）
    half_periods = np.diff(crossings)[-5:]  # 最近5个半周期
    if len(half_periods) == 0:
        return min_period
    
    avg_half_period = np.median(half_periods)
    estimated_period = int(2 * avg_half_period)
    
    # 限制范围
    estimated_period = np.clip(estimated_period, min_period, max_period)
    return int(estimated_period)


def rolling_quantile_normalize(
    series: pd.Series,
    window: int = 252,
) -> pd.Series:
    """
    因果滚动分位数标准化：将每个值映射为其在历史 window 中的分位数
    
    优势：
    - 消除量纲差异，支持跨品种可比
    - 自适应市场状态（牛市/熊市波动率不同）
    - 保留相对强弱信息
    
    Args:
        series: 输入序列
        window: 滚动窗口大小（建议：252=日线年化，7*24*60=分钟线周化）
    
    Returns:
        标准化后的序列（值域 [0, 1]）
    """
    def _quantile_rank(x: np.ndarray) -> float:
        """计算当前值在窗口中的分位排名"""
        if len(x) < 10:
            return np.nan
        return (x <= x[-1]).mean()  # 当前值在窗口中的分位
    
    return series.rolling(window=window, min_periods=10).apply(
        _quantile_rank, raw=True
    )


# 标准化签名的Hilbert特征计算函数
@register_feature("compute_hilbert_features", category="hilbert")
def compute_hilbert_features(data, is_streaming=False, state=None, **params):
    """
    Hilbert特征计算函数 - 符合标准化签名，支持流式计算
    
    Args:
        data: 输入数据 (DataFrame 或单行数据)
        is_streaming: 是否为流式模式
        state: 计算状态，用于维护滑动窗口历史数据和中间结果
        **params: 额外参数，包括:
            - price_fluctuation_col: 价格波动列名 (default: "wpt_price_fluctuation")
            - cvd_fluctuation_col: CVD波动列名 (default: "wpt_cvd_fluctuation")
            - price_col: 价格列名 (default: "close")
            - volume_col: 成交量列名 (default: "volume")
            - window: 滚动窗口大小 (default: 64)
            - ema_span: EMA平滑跨度 (default: 10)
            - use_adaptive_window: 是否使用自适应窗口 (default: False)
            - base_window_min: 自适应窗口最小值 (default: 32)
            - base_window_max: 自适应窗口最大值 (default: 128)
            - period_lookback: 用于估计周期的历史窗口大小 (default: 64)
            - use_quantile_normalize: 是否使用分位数标准化 (default: False)
            - quantile_window: 分位数标准化的滚动窗口 (default: 252)
            - use_volume_fusion: 是否融合成交量包络特征 (default: False)
            - vol_detrend_window: 成交量去趋势的滚动窗口 (default: 20)
            - replace_env_with_qnorm: 是否用分位数标准化替换基础包络 (default: False)
    
    Returns:
        在批处理模式下返回DataFrame，在流式模式下返回单行特征值和更新的状态
    """
    # 从params中获取参数
    price_fluctuation_col = params.get('price_fluctuation_col', 'wpt_price_fluctuation')
    cvd_fluctuation_col = params.get('cvd_fluctuation_col', 'wpt_cvd_fluctuation')
    price_col = params.get('price_col', 'close')
    volume_col = params.get('volume_col', 'volume')
    window = params.get('window', 64)
    ema_span = params.get('ema_span', 10)
    use_adaptive_window = params.get('use_adaptive_window', False)
    base_window_min = params.get('base_window_min', 32)
    base_window_max = params.get('base_window_max', 128)
    period_lookback = params.get('period_lookback', 64)
    use_quantile_normalize = params.get('use_quantile_normalize', False)
    quantile_window = params.get('quantile_window', 252)
    use_volume_fusion = params.get('use_volume_fusion', False)
    vol_detrend_window = params.get('vol_detrend_window', 20)
    replace_env_with_qnorm = params.get('replace_env_with_qnorm', False)
    
    if is_streaming:
        return _compute_hilbert_features_streaming(data, state, price_fluctuation_col=price_fluctuation_col,
                                                cvd_fluctuation_col=cvd_fluctuation_col, price_col=price_col,
                                                volume_col=volume_col, window=window, ema_span=ema_span,
                                                use_adaptive_window=use_adaptive_window,
                                                base_window_min=base_window_min, base_window_max=base_window_max,
                                                period_lookback=period_lookback,
                                                use_quantile_normalize=use_quantile_normalize,
                                                quantile_window=quantile_window, use_volume_fusion=use_volume_fusion,
                                                vol_detrend_window=vol_detrend_window,
                                                replace_env_with_qnorm=replace_env_with_qnorm)
    else:
        # 批处理模式，保持原有逻辑
        return _compute_hilbert_features_batch(data, price_fluctuation_col=price_fluctuation_col,
                                             cvd_fluctuation_col=cvd_fluctuation_col, price_col=price_col,
                                             volume_col=volume_col, window=window, ema_span=ema_span,
                                             use_adaptive_window=use_adaptive_window,
                                             base_window_min=base_window_min, base_window_max=base_window_max,
                                             period_lookback=period_lookback,
                                             use_quantile_normalize=use_quantile_normalize,
                                             quantile_window=quantile_window, use_volume_fusion=use_volume_fusion,
                                             vol_detrend_window=vol_detrend_window,
                                             replace_env_with_qnorm=replace_env_with_qnorm)


def _compute_hilbert_features_streaming(new_data, state, **kwargs):
    """
    流式计算Hilbert特征
    
    Args:
        new_data: 新到达的数据点 (单行DataFrame或Series)
        state: 包含历史数据和中间计算结果的状态
        **kwargs: 与批处理相同的参数
        
    Returns:
        tuple: (特征值, 更新后的state)
    """
    # 从kwargs获取参数
    price_fluctuation_col = kwargs.get('price_fluctuation_col', 'wpt_price_fluctuation')
    cvd_fluctuation_col = kwargs.get('cvd_fluctuation_col', 'wpt_cvd_fluctuation')
    price_col = kwargs.get('price_col', 'close')
    volume_col = kwargs.get('volume_col', 'volume')
    window = kwargs.get('window', 64)
    ema_span = kwargs.get('ema_span', 10)
    use_adaptive_window = kwargs.get('use_adaptive_window', False)
    base_window_min = kwargs.get('base_window_min', 32)
    base_window_max = kwargs.get('base_window_max', 128)
    period_lookback = kwargs.get('period_lookback', 64)
    use_quantile_normalize = kwargs.get('use_quantile_normalize', False)
    quantile_window = kwargs.get('quantile_window', 252)
    use_volume_fusion = kwargs.get('use_volume_fusion', False)
    vol_detrend_window = kwargs.get('vol_detrend_window', 20)
    replace_env_with_qnorm = kwargs.get('replace_env_with_qnorm', False)
    
    # 初始化状态
    if state is None:
        state = {
            'history': [],  # 历史数据点
            'price_fluc_history': [],  # 价格波动历史
            'cvd_fluc_history': [],  # CVD波动历史
            'volume_history': [],  # 成交量历史
            'adaptive_windows': [],  # 自适应窗口历史
            'envelope_cache': {},  # 包络计算结果缓存
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
    state['history'].append(new_row)
    if price_fluctuation_col in new_row:
        state['price_fluc_history'].append(new_row[price_fluctuation_col])
    if cvd_fluctuation_col and cvd_fluctuation_col in new_row:
        state['cvd_fluc_history'].append(new_row[cvd_fluctuation_col])
    if volume_col and volume_col in new_row:
        state['volume_history'].append(new_row[volume_col])
    
    # 保持历史数据在窗口长度内
    max_history_len = window * 2  # 使用稍大的缓冲区
    if len(state['history']) > max_history_len:
        state['history'] = state['history'][-max_history_len:]
        state['price_fluc_history'] = state['price_fluc_history'][-max_history_len:]
        state['cvd_fluc_history'] = state['cvd_fluc_history'][-max_history_len:]
        state['volume_history'] = state['volume_history'][-max_history_len:]
    
    # 计算特征值
    features = {}
    
    # 确保有足够的数据进行计算
    if len(state['price_fluc_history']) < window:
        # 返回空特征
        for col in ['hilbert_price_env', 'hilbert_cvd_env', 'hilbert_cvd_price_env_ratio', 
                   'hilbert_price_env_slope', 'hilbert_cvd_env_slope']:
            features[col] = np.nan
        return features, state
    
    # 获取当前窗口的数据
    price_fluc_window = state['price_fluc_history'][-window:]
    cvd_fluc_window = state['cvd_fluc_history'][-window:] if len(state['cvd_fluc_history']) >= window else []
    
    # 检查数据有效性
    price_fluc_valid = np.array([x for x in price_fluc_window if pd.notna(x)])
    if len(price_fluc_valid) >= max(10, window // 2):
        # 使用前向填充处理NaN
        price_series = pd.Series(price_fluc_window)
        price_series = price_series.ffill().fillna(method='bfill')
        if not price_series.isna().any():
            try:
                # 计算价格包络
                price_envelope = compute_hilbert_envelope(price_series.values)
                if len(price_envelope) > 0:
                    features['hilbert_price_env'] = price_envelope[-1]
                else:
                    features['hilbert_price_env'] = np.nan
            except Exception:
                features['hilbert_price_env'] = np.nan
        else:
            features['hilbert_price_env'] = np.nan
    else:
        features['hilbert_price_env'] = np.nan
    
    # 计算CVD包络
    if len(cvd_fluc_window) > 0:
        cvd_fluc_valid = np.array([x for x in cvd_fluc_window if pd.notna(x)])
        if len(cvd_fluc_valid) >= max(10, window // 2):
            cvd_series = pd.Series(cvd_fluc_window)
            cvd_series = cvd_series.ffill().fillna(method='bfill')
            if not cvd_series.isna().any():
                try:
                    cvd_envelope = compute_hilbert_envelope(cvd_series.values)
                    if len(cvd_envelope) > 0:
                        features['hilbert_cvd_env'] = cvd_envelope[-1]
                    else:
                        features['hilbert_cvd_env'] = np.nan
                except Exception:
                    features['hilbert_cvd_env'] = np.nan
            else:
                features['hilbert_cvd_env'] = np.nan
        else:
            features['hilbert_cvd_env'] = np.nan
    else:
        features['hilbert_cvd_env'] = np.nan
    
    # 计算包络比率
    if 'hilbert_price_env' in features and 'hilbert_cvd_env' in features:
        price_env = features['hilbert_price_env']
        cvd_env = features['hilbert_cvd_env']
        if not (np.isnan(price_env) or np.isnan(cvd_env) or price_env == 0):
            features['hilbert_cvd_price_env_ratio'] = cvd_env / price_env
        else:
            features['hilbert_cvd_price_env_ratio'] = np.nan
    else:
        features['hilbert_cvd_price_env_ratio'] = np.nan
    
    # 为简单起见，斜率特征在流式模式下暂设为NaN，实际应用中可以使用历史值计算
    features['hilbert_price_env_slope'] = np.nan
    features['hilbert_cvd_env_slope'] = np.nan
    
    return features, state


def _compute_hilbert_features_batch(df, **kwargs):
    """
    批处理模式计算Hilbert特征（保持原有逻辑）
    """
    # 从kwargs获取参数
    price_fluctuation_col = kwargs.get('price_fluctuation_col', 'wpt_price_fluctuation')
    cvd_fluctuation_col = kwargs.get('cvd_fluctuation_col', 'wpt_cvd_fluctuation')
    price_col = kwargs.get('price_col', 'close')
    volume_col = kwargs.get('volume_col', 'volume')
    window = kwargs.get('window', 64)
    ema_span = kwargs.get('ema_span', 10)
    use_adaptive_window = kwargs.get('use_adaptive_window', False)
    base_window_min = kwargs.get('base_window_min', 32)
    base_window_max = kwargs.get('base_window_max', 128)
    period_lookback = kwargs.get('period_lookback', 64)
    use_quantile_normalize = kwargs.get('use_quantile_normalize', False)
    quantile_window = kwargs.get('quantile_window', 252)
    use_volume_fusion = kwargs.get('use_volume_fusion', False)
    vol_detrend_window = kwargs.get('vol_detrend_window', 20)
    replace_env_with_qnorm = kwargs.get('replace_env_with_qnorm', False)
    
    df = df.copy()
    
    # 初始化基础特征列
    hilbert_cols = [
        "hilbert_price_env",
        "hilbert_cvd_env",
        "hilbert_cvd_price_env_ratio",
        "hilbert_price_env_slope",
        "hilbert_cvd_env_slope",
    ]
    
    # 添加高级功能特征列
    if use_adaptive_window:
        hilbert_cols.append("hilbert_adaptive_window")
    
    if use_quantile_normalize:
        hilbert_cols.extend([
            "hilbert_price_env_qnorm",
            "hilbert_cvd_env_qnorm",
        ])
    
    if use_volume_fusion:
        hilbert_cols.extend([
            "hilbert_volume_env",
            "hilbert_env_price_vol_ratio",
            "hilbert_triple_divergence",
        ])
    
    for col in hilbert_cols:
        df[col] = np.nan
    
    # 存储原始包络序列（用于 EMA 平滑）
    price_envelope_raw = []
    cvd_envelope_raw = []
    price_envelope_indices = []
    cvd_envelope_indices = []
    
    # 获取输入数据（如果存在）
    price_fluc = df[price_fluctuation_col].values if price_fluctuation_col in df.columns else None
    cvd_fluc = (
        df[cvd_fluctuation_col].values
        if (cvd_fluctuation_col and cvd_fluctuation_col in df.columns)
        else None
    )
    
    # ========== 自适应窗口：估计局部周期 ==========
    adaptive_windows = None
    if use_adaptive_window and price_fluc is not None:
        adaptive_windows = []
        for i in range(len(df)):
            if i < period_lookback:
                adaptive_windows.append(base_window_min)
            else:
                # 使用历史窗口估计周期（因果！）
                window_data = price_fluc[i - period_lookback : i]
                # 检查有效数据
                if np.any(~np.isnan(window_data)) and len(window_data) >= 20:
                    period = estimate_local_period(
                        window_data[~np.isnan(window_data)],
                        min_period=base_window_min // 2,
                        max_period=base_window_max // 2,
                    )
                    # window ≈ 3×周期，确保覆盖完整周期
                    adaptive_windows.append(min(period * 3, base_window_max))
                else:
                    adaptive_windows.append(base_window_min)
        
        # 保存自适应窗口序列
        df["hilbert_adaptive_window"] = adaptive_windows
    
    # 合并循环：同时处理 Price 和 CVD，提升性能
    min_valid_points = max(10, window // 2)  # 最小有效数据点要求
    
    # 确定实际使用的窗口（固定或自适应）
    def get_window(i: int) -> int:
        if adaptive_windows is not None:
            return adaptive_windows[i]
        return window
    
    for i in range(window, len(df)):
        current_window = get_window(i)
        # 确保窗口不超过可用数据
        current_window = min(current_window, i)
        if current_window < min_valid_points:
            continue
        # ========== 处理价格波动 ==========
        if price_fluc is not None:
            # 使用历史窗口数据 [i-current_window, i)
            window_data = price_fluc[i - current_window : i]
            
            # 检查窗口长度
            if window_data.size < min_valid_points:
                continue
            
            # 检查有效数据点数量（非 NaN）
            valid_mask = ~np.isnan(window_data)
            valid_count = valid_mask.sum()
            
            if valid_count < min_valid_points:
                continue
            
            # 填充 NaN（仅前向填充，避免未来信息泄露）
            # 注意：在滚动窗口 [t-W, t) 中，只能用更早的历史值填充
            # 禁用 bfill()，因为它会用右侧值填左侧 → 引入未来信息
            window_series = pd.Series(window_data)
            window_series = window_series.ffill()  # 仅前向填充（因果！）
            
            # 如果开头仍是 NaN（无历史有效值），跳过
            if window_series.isna().any():
                continue
            
            window_data_clean = window_series.values
            
            try:
                # 计算 Hilbert 包络
                envelope = compute_hilbert_envelope(window_data_clean)
                
                # 只使用最后一个点的值（当前时刻的特征）
                if len(envelope) > 0:
                    price_envelope_raw.append(envelope[-1])
                    price_envelope_indices.append(i)
            except Exception:
                # 如果计算失败，跳过
                continue
        
        # ========== 处理 CVD 波动 ==========
        if cvd_fluc is not None:
            window_data = cvd_fluc[i - current_window : i]
            
            if window_data.size < min_valid_points:
                continue
            
            valid_mask = ~np.isnan(window_data)
            valid_count = valid_mask.sum()
            
            if valid_count < min_valid_points:
                continue
            
            # 仅前向填充（因果！）
            window_series = pd.Series(window_data)
            window_series = window_series.ffill()
            
            if window_series.isna().any():
                continue
            
            window_data_clean = window_series.values
            
            try:
                envelope = compute_hilbert_envelope(window_data_clean)
                
                if len(envelope) > 0:
                    cvd_envelope_raw.append(envelope[-1])
                    cvd_envelope_indices.append(i)
            except Exception:
                continue
    
    # EMA 平滑包络序列
    if len(price_envelope_raw) > 0 and len(price_envelope_indices) > 0:
        # 创建临时 Series 用于 EMA
        price_env_series = pd.Series(price_envelope_raw, index=price_envelope_indices)
        # adjust=False: 使用递归 EMA 公式 y[t] = α·x[t] + (1−α)·y[t−1]，确保完全因果
        # 这是必须的，因为 adjust=True 会使用非递归公式，可能引入轻微的未来信息
        price_env_smoothed = price_env_series.ewm(span=ema_span, adjust=False).mean()
        
        # 将平滑后的值写回 DataFrame
        for idx, val in price_env_smoothed.items():
            if idx < len(df):
                df.iloc[idx, df.columns.get_loc("hilbert_price_env")] = val
    
    if len(cvd_envelope_raw) > 0 and len(cvd_envelope_indices) > 0:
        # 确保长度匹配
        if len(cvd_envelope_raw) == len(cvd_envelope_indices):
            cvd_env_series = pd.Series(cvd_envelope_raw, index=cvd_envelope_indices)
            # adjust=False: 使用递归 EMA 公式，确保完全因果
            cvd_env_smoothed = cvd_env_series.ewm(span=ema_span, adjust=False).mean()
            
            for idx, val in cvd_env_smoothed.items():
                if idx < len(df):
                    df.iloc[idx, df.columns.get_loc("hilbert_cvd_env")] = val
    
    # 计算衍生特征
    # 1. 包络比值（CVD / Price）
    if "hilbert_cvd_env" in df.columns and "hilbert_price_env" in df.columns:
        price_env = df["hilbert_price_env"]
        cvd_env = df["hilbert_cvd_env"]
        
        # 避免除零，使用安全除法
        ratio = cvd_env / price_env.replace(0, np.nan)
        # Guard against inf/-inf from degenerate envelopes (flat price / tiny denominators)
        ratio = ratio.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        df["hilbert_cvd_price_env_ratio"] = ratio
    
    # 2. 包络斜率（波动加速/减速）
    if "hilbert_price_env" in df.columns:
        price_env = df["hilbert_price_env"]
        # 使用 diff 计算斜率（因果操作）
        df["hilbert_price_env_slope"] = price_env.diff()
    
    if "hilbert_cvd_env" in df.columns:
        cvd_env = df["hilbert_cvd_env"]
        df["hilbert_cvd_env_slope"] = cvd_env.diff()

    # Final safety: never output inf values for downstream consumers/tests
    for col in [
        "hilbert_price_env",
        "hilbert_cvd_env",
        "hilbert_cvd_price_env_ratio",
        "hilbert_price_env_slope",
        "hilbert_cvd_env_slope",
    ]:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    
    # ========== 分位数标准化（跨品种可比）==========
    if use_quantile_normalize:
        if "hilbert_price_env" in df.columns:
            df["hilbert_price_env_qnorm"] = rolling_quantile_normalize(
                df["hilbert_price_env"], window=quantile_window
            )
        
        if "hilbert_cvd_env" in df.columns:
            df["hilbert_cvd_env_qnorm"] = rolling_quantile_normalize(
                df["hilbert_cvd_env"], window=quantile_window
            )

    # Optional: overwrite base envelope outputs with their quantile-normalized versions.
    # This keeps column names stable for downstream strategy configs while enforcing
    # cross-asset comparability (rank transformation).
    if bool(replace_env_with_qnorm):
        if "hilbert_price_env_qnorm" in df.columns:
            df["hilbert_price_env"] = df["hilbert_price_env_qnorm"]
        if "hilbert_cvd_env_qnorm" in df.columns:
            df["hilbert_cvd_env"] = df["hilbert_cvd_env_qnorm"]
    
    # ========== 成交量包络融合 ==========
    if use_volume_fusion and volume_col and volume_col in df.columns:
        # 先对成交量做去趋势（简单滚动均值去趋势）
        # 注意：vol_detrend_window 可配置，适应不同品种的成交量趋势周期
        volume = df[volume_col].values
        volume_trend = pd.Series(volume).rolling(window=vol_detrend_window, min_periods=1).mean()
        volume_fluc = volume - volume_trend.values
        
        # 直接计算成交量包络（避免递归调用，使用内联逻辑）
        vol_window = 32  # 成交量使用较小窗口，因为成交量波动更频繁
        vol_ema_span = 5
        vol_min_valid_points = max(10, vol_window // 2)
        
        vol_envelope_raw = []
        vol_envelope_indices = []
        
        for i in range(vol_window, len(df)):
            window_data = volume_fluc[i - vol_window : i]
            
            if window_data.size < vol_min_valid_points:
                continue
            
            valid_mask = ~np.isnan(window_data)
            valid_count = valid_mask.sum()
            
            if valid_count < vol_min_valid_points:
                continue
            
            # 仅前向填充（因果！）
            window_series = pd.Series(window_data)
            window_series = window_series.ffill()
            
            if window_series.isna().any():
                continue
            
            window_data_clean = window_series.values
            
            try:
                envelope = compute_hilbert_envelope(window_data_clean)
                if len(envelope) > 0:
                    vol_envelope_raw.append(envelope[-1])
                    vol_envelope_indices.append(i)
            except Exception:
                continue
        
        # EMA 平滑成交量包络
        if len(vol_envelope_raw) > 0 and len(vol_envelope_indices) > 0:
            vol_env_series = pd.Series(vol_envelope_raw, index=vol_envelope_indices)
            vol_env_smoothed = vol_env_series.ewm(span=vol_ema_span, adjust=False).mean()
            
            for idx, val in vol_env_smoothed.items():
                if idx < len(df):
                    df.iloc[idx, df.columns.get_loc("hilbert_volume_env")] = val

            # If we are enforcing rank-based normalization, quantile-normalize volume envelope too.
            if bool(use_quantile_normalize):
                df["hilbert_volume_env_qnorm"] = rolling_quantile_normalize(
                    df["hilbert_volume_env"], window=quantile_window
                )
                if bool(replace_env_with_qnorm):
                    df["hilbert_volume_env"] = df["hilbert_volume_env_qnorm"]
            
            # 计算价格/成交量包络比
            if "hilbert_price_env" in df.columns:
                price_env = df["hilbert_price_env"]
                vol_env = df["hilbert_volume_env"]
                # 避免除零
                df["hilbert_env_price_vol_ratio"] = (
                    price_env / (vol_env + 1e-8)
                )
                
                # 三元背离信号：价格包络新高，但 CVD 包络未新高，且成交量包络下降
                if "hilbert_cvd_env" in df.columns:
                    cvd_env = df["hilbert_cvd_env"]
                    
                    # 价格包络创新高（滚动20期）
                    price_new_high = (
                        price_env > price_env.rolling(window=20, min_periods=1).max().shift(1)
                    )
                    
                    # CVD 包络未创新高
                    cvd_not_high = (
                        cvd_env < cvd_env.rolling(window=20, min_periods=1).max()
                    )
                    
                    # 成交量包络下降
                    vol_declining = vol_env < vol_env.shift(1)
                    
                    # 三元背离信号
                    df["hilbert_triple_divergence"] = (
                        (price_new_high & cvd_not_high & vol_declining).astype(float)
                    )

    # ------------------------------------------------------------------
    # If we replaced envelopes with quantile-normalized values, refresh the
    # derived features so they are consistent with the final envelope series.
    # Also apply safe transforms for ratio-like outputs to avoid extreme spikes.
    # ------------------------------------------------------------------
    if bool(replace_env_with_qnorm):
        eps = 1e-8

        def _robust_rolling_z(v: pd.Series, window: int, min_periods: int = 10) -> pd.Series:
            v = pd.to_numeric(v, errors="coerce").astype(float)
            med = v.rolling(window=window, min_periods=min_periods).median()
            q25 = v.rolling(window=window, min_periods=min_periods).quantile(0.25)
            q75 = v.rolling(window=window, min_periods=min_periods).quantile(0.75)
            iqr = (q75 - q25).replace(0, np.nan)
            z = (v - med) / (iqr + eps)
            return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Ratio (CVD/Price env): use log-ratio then robust rolling scaling.
        if "hilbert_cvd_env" in df.columns and "hilbert_price_env" in df.columns:
            price_env = pd.to_numeric(df["hilbert_price_env"], errors="coerce").astype(float).fillna(0.0)
            cvd_env = pd.to_numeric(df["hilbert_cvd_env"], errors="coerce").astype(float).fillna(0.0)
            log_ratio = np.log((cvd_env + eps) / (price_env + eps))
            df["hilbert_cvd_price_env_ratio"] = _robust_rolling_z(log_ratio, window=int(quantile_window))

            # Slopes on normalized envelope (bounded diffs)
            df["hilbert_price_env_slope"] = price_env.diff().clip(-1.0, 1.0)
            df["hilbert_cvd_env_slope"] = cvd_env.diff().clip(-1.0, 1.0)

        # Price/Volume env ratio: log-ratio then robust scaling (only when fusion exists)
        if "hilbert_env_price_vol_ratio" in df.columns and "hilbert_volume_env" in df.columns and "hilbert_price_env" in df.columns:
            price_env = pd.to_numeric(df["hilbert_price_env"], errors="coerce").astype(float).fillna(0.0)
            vol_env = pd.to_numeric(df["hilbert_volume_env"], errors="coerce").astype(float).fillna(0.0)
            log_ratio = np.log((price_env + eps) / (vol_env + eps))
            df["hilbert_env_price_vol_ratio"] = _robust_rolling_z(log_ratio, window=int(quantile_window))
    
    # 使用 shift(1) 确保时间对齐，只使用历史信息
    # 注意：shift(1) 后，NaN 表示历史数据不足，不应 fillna(0.0)
    for col in hilbert_cols:
        if col in df.columns:
            df[col] = df[col].shift(1)
    
    # 不填充 NaN：0 波动 ≠ 未知，保留 NaN 让模型知道这是缺失的历史信息
    
    return df


# 保持原有的函数名作为别名，以便向后兼容
@register_feature("extract_hilbert_features", category="hilbert")
def extract_hilbert_features(
    df: pd.DataFrame,
    price_fluctuation_col: str = "wpt_price_fluctuation",
    cvd_fluctuation_col: Optional[str] = "wpt_cvd_fluctuation",
    price_col: Optional[str] = "close",
    volume_col: Optional[str] = "volume",
    window: int = 64,
    ema_span: int = 10,
    # 高级功能参数
    use_adaptive_window: bool = False,
    base_window_min: int = 32,
    base_window_max: int = 128,
    period_lookback: int = 64,
    use_quantile_normalize: bool = False,
    quantile_window: int = 252,
    use_volume_fusion: bool = False,
    vol_detrend_window: int = 20,
    # Contract-focused output normalization:
    # If enabled, we will compute rolling quantile normalization for envelope-like columns
    # and overwrite the base envelope outputs to make them cross-asset comparable.
    replace_env_with_qnorm: bool = False,
) -> pd.DataFrame:
    """
    旧版函数，保持向后兼容性
    """
    return _compute_hilbert_features_batch(df, price_fluctuation_col=price_fluctuation_col,
                                         cvd_fluctuation_col=cvd_fluctuation_col, price_col=price_col,
                                         volume_col=volume_col, window=window, ema_span=ema_span,
                                         use_adaptive_window=use_adaptive_window,
                                         base_window_min=base_window_min, base_window_max=base_window_max,
                                         period_lookback=period_lookback,
                                         use_quantile_normalize=use_quantile_normalize,
                                         quantile_window=quantile_window, use_volume_fusion=use_volume_fusion,
                                         vol_detrend_window=vol_detrend_window,
                                         replace_env_with_qnorm=replace_env_with_qnorm)
