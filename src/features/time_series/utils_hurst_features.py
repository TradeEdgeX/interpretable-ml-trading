"""
Hurst 指数特征工程（生产优化版）

核心功能：
1. 滚动 Hurst（捕捉市场状态切换）
2. 支持价格、CVD、成交量三个信号维度

改进点：
1. 严格因果：滚动窗口仅使用 [t-W, t-1] 数据
2. 高效计算：支持 update_freq 控制更新频率
3. 保留 NaN：不填充 0.5，由模型处理缺失
4. 统一使用 DFA（更稳健）
5. 明确信号定义：价格→收益率，CVD→单期变化，Volume→成交量收益率

为什么 DFA 更好？
R/S 方法对非平稳序列（如带趋势的价格）敏感，而 DFA 通过局部去趋势，
更适合金融时间序列。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Union

from src.features.registry import register_feature


def compute_hurst_dfa(
    series: np.ndarray,
    min_window: int = 4,
    max_window: Optional[int] = None,
    eps: float = 1e-9,
) -> float:
    """
    使用 DFA 计算 Hurst 指数（仅适用于增量序列，如收益率）
    
    Args:
        series: 增量序列（如收益率、差分等）
        min_window: 最小窗口大小
        max_window: 最大窗口大小（默认 len(series)//4，上限100）
    
    Returns:
        Hurst 指数（0-1之间），如果数据不足则返回 np.nan
    """
    if len(series) < 20 or np.allclose(series, 0):
        return np.nan
    
    if max_window is None:
        max_window = min(len(series) // 4, 100)  # 上限避免过拟合
    
    max_window = min(max_window, len(series) // 2)
    if max_window < min_window:
        return np.nan
    
    # 累积和（去均值）
    y = np.cumsum(series - np.nanmean(series))
    
    # 生成对数均匀分布的窗口
    n_windows = 8
    windows = np.unique(
        np.logspace(np.log10(min_window), np.log10(max_window), n_windows).astype(int)
    )
    windows = windows[windows >= min_window]
    
    fluctuations = []
    valid_windows = []
    
    for w in windows:
        n_segs = len(y) // w
        if n_segs < 2:
            continue
        
        f_list = []
        for i in range(n_segs):
            seg_y = y[i * w : (i + 1) * w]
            if len(seg_y) < 2:
                continue
            
            x = np.arange(len(seg_y))
            # 检查输入数据是否有效
            if not np.all(np.isfinite(seg_y)):
                continue  # 跳过包含 inf/NaN 的段
            # 快速线性去趋势
            x_var = np.var(x)
            if x_var < 1e-12 or not np.isfinite(x_var):
                continue  # 跳过方差过小或无效的段
            slope = (np.mean(x * seg_y) - np.mean(x) * np.mean(seg_y)) / x_var
            # 检查 slope 是否有效
            if not np.isfinite(slope):
                continue  # 跳过产生 inf 的段
            intercept = np.mean(seg_y) - slope * np.mean(x)
            trend = slope * x + intercept
            detrended = seg_y - trend
            # 检查 detrended 是否有效
            if not np.all(np.isfinite(detrended)):
                continue  # 跳过包含 inf/NaN 的去趋势结果
            
            f = np.sqrt(np.nanmean(detrended ** 2))
            if not np.isnan(f):
                f_list.append(f)
        
        if f_list:
            fluctuations.append(np.nanmean(f_list))
            valid_windows.append(w)
    
    # 至少需要 3 个有效尺度点才能进行稳健的线性回归
    # 2 个点会过拟合（斜率由两点唯一确定），3 点更稳健
    if len(fluctuations) < 3:
        return np.nan
    
    log_w = np.log(valid_windows)
    # 防止波动为 0 导致 log(0) -> -inf
    log_f = np.log(np.maximum(fluctuations, eps))
    # 检查是否有 inf 值
    if np.any(~np.isfinite(log_f)) or np.any(~np.isfinite(log_w)):
        print(f"   ⚠️  Hurst: log_f or log_w contains inf, log_f inf count: {np.sum(~np.isfinite(log_f))}, log_w inf count: {np.sum(~np.isfinite(log_w))}")
        return np.nan
    
    # 简单线性回归（忽略 NaN）
    mask = ~(np.isnan(log_w) | np.isnan(log_f))
    if mask.sum() < 3:
        return np.nan
    
    try:
        hurst = np.polyfit(log_w[mask], log_f[mask], 1)[0]
        # 检查结果是否有效
        if not np.isfinite(hurst):
            return np.nan
        return np.clip(hurst, 0.0, 1.0)
    except (np.linalg.LinAlgError, ValueError):
        # 线性回归失败（如所有点共线、数值不稳定等）
        return np.nan


def _infer_data_frequency(df: pd.DataFrame) -> Optional[str]:
    """
    从 DataFrame 的索引推断数据频率
    
    Returns:
        频率字符串（如 '1T', '5T', '15T', '1H', '4H', '1D'）或 None
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        return None
    
    # 尝试从索引推断频率
    inferred_freq = df.index.inferred_freq
    if inferred_freq and inferred_freq.strip():
        return inferred_freq
    
    # 如果推断失败，从时间差计算
    if len(df.index) < 2:
        return None
    
    time_diffs = df.index[1:10] - df.index[0:9]
    median_diff = time_diffs.median()
    
    # 转换为 pandas 频率字符串
    total_seconds = median_diff.total_seconds()
    
    if total_seconds < 60:
        return f"{int(total_seconds)}S"
    elif total_seconds < 3600:
        minutes = int(total_seconds / 60)
        return f"{minutes}T" if minutes > 0 else None
    elif total_seconds < 86400:
        hours = int(total_seconds / 3600)
        return f"{hours}H" if hours > 0 else None
    else:
        days = int(total_seconds / 86400)
        return f"{days}D" if days > 0 else None


def _estimate_volatility_characteristics(
    returns: pd.Series,
    window: int = 100,
) -> dict:
    """
    估计品种的波动特性
    
    Returns:
        dict with keys:
        - avg_volatility: 平均波动率
        - volatility_percentile: 波动率百分位数（相对于历史）
        - max_daily_return: 最大单日收益率
        - extreme_return_ratio: 极端收益率比例（>10%）
    """
    if len(returns) < window:
        window = len(returns)
    
    # 计算滚动波动率
    rolling_vol = returns.rolling(window=window, min_periods=10).std()
    avg_volatility = rolling_vol.mean()
    
    # 计算波动率百分位数（相对于历史）
    if len(rolling_vol.dropna()) > 0:
        current_vol = rolling_vol.iloc[-1] if not np.isnan(rolling_vol.iloc[-1]) else avg_volatility
        volatility_percentile = (rolling_vol < current_vol).sum() / len(rolling_vol.dropna())
    else:
        volatility_percentile = 0.5
    
    # 最大单日收益率
    max_daily_return = returns.abs().max()
    
    # 极端收益率比例（>10%）
    extreme_threshold = 0.10
    extreme_return_ratio = (returns.abs() > extreme_threshold).sum() / len(returns)
    
    return {
        "avg_volatility": avg_volatility,
        "volatility_percentile": volatility_percentile,
        "max_daily_return": max_daily_return,
        "extreme_return_ratio": extreme_return_ratio,
    }


def _auto_adjust_update_freq(
    df: pd.DataFrame,
    default_update_freq: int = 1,
) -> int:
    """
    根据数据频率自动调整 update_freq
    
    规则：
    - 高频数据（< 15分钟）：update_freq = 5
    - 中频数据（15分钟-4小时）：update_freq = 3
    - 低频数据（>= 4小时）：update_freq = 1
    """
    if default_update_freq != 1:
        # 如果用户明确指定，使用用户值
        return default_update_freq
    
    freq_str = _infer_data_frequency(df)
    if freq_str is None or not freq_str or len(freq_str) < 2:
        return default_update_freq
    
    # 解析频率字符串
    try:
        if freq_str.endswith('S'):  # 秒级
            return 5
        elif freq_str.endswith('T'):  # 分钟级
            minutes_str = freq_str[:-1]
            if not minutes_str:
                return default_update_freq
            minutes = int(minutes_str)
            if minutes < 15:
                return 5  # 高频：每5根K线更新
            elif minutes < 60:
                return 3  # 中频：每3根K线更新
            else:
                return 1  # 低频：每根K线更新
        elif freq_str.endswith('H'):  # 小时级
            hours_str = freq_str[:-1]
            if not hours_str:
                return default_update_freq
            hours = int(hours_str)
            if hours < 4:
                return 1  # 4小时以下：每根K线更新
            else:
                return 1  # 4小时以上：每根K线更新
        else:  # 日级及以上
            return 1
    except (ValueError, IndexError):
        # 如果解析失败，返回默认值
        return default_update_freq


def _auto_adjust_clip_pct(
    df: pd.DataFrame,
    price_col: str,
    default_clip_pct: Optional[float] = 0.5,
) -> Optional[float]:
    """
    根据品种波动特性自动调整 clip_pct
    
    规则：
    - 高波动品种（波动率 > 3%）：clip_pct = 1.0（放宽到100%）
    - 中波动品种（波动率 1-3%）：clip_pct = 0.5（默认50%）
    - 低波动品种（波动率 < 1%）：clip_pct = 0.3（收紧到30%）
    """
    if default_clip_pct is None:
        # 如果用户明确禁用裁剪，保持禁用
        return None
    
    if price_col not in df.columns:
        return default_clip_pct
    
    # 计算收益率
    returns = df[price_col].pct_change().dropna()
    if len(returns) < 20:
        return default_clip_pct
    
    # 估计波动特性
    vol_chars = _estimate_volatility_characteristics(returns, window=100)
    avg_vol = vol_chars["avg_volatility"]
    max_return = vol_chars["max_daily_return"]
    extreme_ratio = vol_chars["extreme_return_ratio"]
    
    # 根据波动率调整
    if avg_vol > 0.03:  # 高波动（>3%）
        # 进一步检查是否有极端事件
        if max_return > 0.5 or extreme_ratio > 0.05:
            clip_pct = 1.0  # 放宽到100%
        else:
            clip_pct = 0.8  # 放宽到80%
    elif avg_vol > 0.01:  # 中波动（1-3%）
        if max_return > 0.3 or extreme_ratio > 0.02:
            clip_pct = 0.7  # 放宽到70%
        else:
            clip_pct = 0.5  # 默认50%
    else:  # 低波动（<1%）
        clip_pct = 0.3  # 收紧到30%
    
    return clip_pct


# 标准化签名的Hurst特征计算函数
@register_feature("compute_hurst_features", category="hurst")
def compute_hurst_features(data, is_streaming=False, state=None, **params):
    """
    Hurst特征计算函数 - 符合标准化签名，支持流式计算
    
    Args:
        data: 输入数据 (DataFrame 或单行数据)
        is_streaming: 是否为流式模式
        state: 计算状态，用于维护滑动窗口历史数据和中间结果
        **params: 额外参数，包括:
            - price_col: 价格列名 (default: "close")
            - cvd_col: CVD列名 (default: None)
            - volume_col: 成交量列名 (default: None)
            - rolling_window: 滚动窗口大小 (default: 50)
            - update_freq: 更新频率 (default: "auto")
            - clip_pct: 极值裁剪百分比 (default: "auto")
    
    Returns:
        在批处理模式下返回DataFrame，在流式模式下返回单行特征值和更新的状态
    """
    # 从params中获取参数
    price_col = params.get('price_col', 'close')
    cvd_col = params.get('cvd_col', None)
    volume_col = params.get('volume_col', None)
    rolling_window = params.get('rolling_window', 50)
    update_freq_param = params.get('update_freq', 'auto')
    clip_pct_param = params.get('clip_pct', 'auto')
    
    if is_streaming:
        return _compute_hurst_features_streaming(data, state, price_col=price_col, cvd_col=cvd_col,
                                            volume_col=volume_col, rolling_window=rolling_window,
                                            update_freq=update_freq_param, clip_pct=clip_pct_param)
    else:
        # 批处理模式，保持原有逻辑
        return _compute_hurst_features_batch(data, price_col=price_col, cvd_col=cvd_col,
                                          volume_col=volume_col, rolling_window=rolling_window,
                                          update_freq=update_freq_param, clip_pct=clip_pct_param)


def _compute_hurst_features_streaming(new_data, state, **kwargs):
    """
    流式计算Hurst特征
    
    Args:
        new_data: 新到达的数据点 (单行DataFrame或Series)
        state: 包含历史数据和中间计算结果的状态
        **kwargs: 与批处理相同的参数
        
    Returns:
        tuple: (特征值, 更新后的state)
    """
    # 从kwargs获取参数
    price_col = kwargs.get('price_col', 'close')
    cvd_col = kwargs.get('cvd_col', None)
    volume_col = kwargs.get('volume_col', None)
    rolling_window = kwargs.get('rolling_window', 50)
    update_freq_param = kwargs.get('update_freq', 'auto')
    clip_pct_param = kwargs.get('clip_pct', 'auto')
    
    # 初始化状态
    if state is None:
        state = {
            'history': [],  # 历史数据点
            'price_history': [],  # 价格历史
            'cvd_history': [],  # CVD历史
            'volume_history': [],  # 成交量历史
            'price_returns_history': [],  # 价格收益率历史
            'cvd_diff_history': [],  # CVD差分历史
            'volume_returns_history': [],  # 成交量收益率历史
            'hurst_cache': {},  # Hurst指数计算结果缓存
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
    if price_col in new_row:
        state['price_history'].append(new_row[price_col])
    if cvd_col and cvd_col in new_row:
        state['cvd_history'].append(new_row[cvd_col])
    if volume_col and volume_col in new_row:
        state['volume_history'].append(new_row[volume_col])
    
    # 保持历史数据在窗口长度内
    max_history_len = rolling_window * 2  # 使用稍大的缓冲区
    if len(state['history']) > max_history_len:
        state['history'] = state['history'][-max_history_len:]
        state['price_history'] = state['price_history'][-max_history_len:]
        state['cvd_history'] = state['cvd_history'][-max_history_len:]
        state['volume_history'] = state['volume_history'][-max_history_len:]
    
    # 计算收益率和差分
    price_returns = []
    cvd_diff = []
    volume_returns = []
    
    if len(state['price_history']) > 1:
        # 计算价格收益率
        prev_price = state['price_history'][-2]
        curr_price = state['price_history'][-1]
        if prev_price != 0:
            ret = (curr_price - prev_price) / prev_price
            state['price_returns_history'].append(ret)
        else:
            state['price_returns_history'].append(0.0)
        
        # 确保收益率历史长度不超过窗口
        if len(state['price_returns_history']) > max_history_len:
            state['price_returns_history'] = state['price_returns_history'][-max_history_len:]
    
    if len(state['cvd_history']) > 1:
        # 计算CVD差分
        prev_cvd = state['cvd_history'][-2]
        curr_cvd = state['cvd_history'][-1]
        diff = curr_cvd - prev_cvd
        state['cvd_diff_history'].append(diff)
        
        if len(state['cvd_diff_history']) > max_history_len:
            state['cvd_diff_history'] = state['cvd_diff_history'][-max_history_len:]
    
    if len(state['volume_history']) > 1:
        # 计算成交量收益率
        prev_vol = state['volume_history'][-2]
        curr_vol = state['volume_history'][-1]
        if prev_vol != 0:
            ret = (curr_vol - prev_vol) / prev_vol
            state['volume_returns_history'].append(ret)
        else:
            state['volume_returns_history'].append(0.0)
        
        if len(state['volume_returns_history']) > max_history_len:
            state['volume_returns_history'] = state['volume_returns_history'][-max_history_len:]
    
    # 计算特征值
    features = {}
    
    # 计算价格Hurst指数
    if len(state['price_returns_history']) >= rolling_window:
        window_returns = np.array(state['price_returns_history'][-rolling_window:])
        hurst_price = compute_hurst_dfa(window_returns)
        features['hurst_price_rolling'] = hurst_price
    else:
        features['hurst_price_rolling'] = np.nan
    
    # 计算CVD Hurst指数
    if cvd_col and len(state['cvd_diff_history']) >= rolling_window:
        window_diff = np.array(state['cvd_diff_history'][-rolling_window:])
        hurst_cvd = compute_hurst_dfa(window_diff)
        features['hurst_cvd_rolling'] = hurst_cvd
    else:
        features['hurst_cvd_rolling'] = np.nan
    
    # 计算成交量Hurst指数
    if volume_col and len(state['volume_returns_history']) >= rolling_window:
        window_returns = np.array(state['volume_returns_history'][-rolling_window:])
        hurst_volume = compute_hurst_dfa(window_returns)
        features['hurst_volume_rolling'] = hurst_volume
    else:
        features['hurst_volume_rolling'] = np.nan
    
    return features, state


def _compute_hurst_features_batch(df, **kwargs):
    """
    批处理模式计算Hurst特征（保持原有逻辑）
    """
    # 从kwargs获取参数
    price_col = kwargs.get('price_col', 'close')
    cvd_col = kwargs.get('cvd_col', None)
    volume_col = kwargs.get('volume_col', None)
    rolling_window = kwargs.get('rolling_window', 50)
    update_freq_param = kwargs.get('update_freq', 'auto')
    clip_pct_param = kwargs.get('clip_pct', 'auto')
    
    df = df.copy()
    
    # === 自动调整参数 ===
    # 1. 自动调整 update_freq
    if update_freq_param == "auto":
        update_freq = _auto_adjust_update_freq(df, default_update_freq=1)
        # 只在非测试环境打印（避免测试输出过多）
        import os
        if os.getenv("PYTEST_CURRENT_TEST") is None:
            print(f"  📊 自动检测数据频率，设置 update_freq={update_freq}")
    elif isinstance(update_freq_param, str):
        # 如果不是 "auto"，尝试转换为 int
        try:
            update_freq = int(update_freq_param)
        except ValueError:
            raise ValueError(f"update_freq 必须是 'auto' 或整数，当前值: {update_freq_param}")
    else:
        update_freq = update_freq_param
    
    # 2. 自动调整 clip_pct
    # 注意：如果只计算 CVD 或 Volume Hurst（没有价格数据），使用默认值
    if clip_pct_param == "auto":
        if price_col in df.columns:
            clip_pct = _auto_adjust_clip_pct(df, price_col, default_clip_pct=0.5)
            # 只在非测试环境打印
            import os
            if os.getenv("PYTEST_CURRENT_TEST") is None:
                print(f"  📊 自动检测波动特性，设置 clip_pct={clip_pct}")
        else:
            # 没有价格数据时，使用默认值（主要用于 CVD/Volume，它们不需要价格数据）
            clip_pct = 0.5
            # 只在非测试环境打印
            import os
            if os.getenv("PYTEST_CURRENT_TEST") is None:
                print(f"  📊 无价格数据，使用默认 clip_pct={clip_pct}")
    elif isinstance(clip_pct_param, str):
        # 如果不是 "auto"，尝试转换
        if clip_pct_param.lower() == "none":
            clip_pct = None
        else:
            try:
                clip_pct = float(clip_pct_param)
            except ValueError:
                raise ValueError(f"clip_pct 必须是 'auto'、数值或 None，当前值: {clip_pct_param}")
    else:
        clip_pct = clip_pct_param
    
    # 初始化列
    df["hurst_price_rolling"] = np.nan
    if cvd_col:
        df["hurst_cvd_rolling"] = np.nan
    if volume_col:
        df["hurst_volume_rolling"] = np.nan
    
    # === 1. 价格收益率（t 时刻收益 = (P_t / P_{t-1}) - 1）===
    if price_col in df.columns:
        # 监控：检查输入数据质量
        try:
            from src.features.utils.data_monitor import check_data_quality
            check_data_quality(
                df[[price_col]],
                data_source="HURST_FEATURES",
                stage="before_price_returns_calc",
                raise_on_inf=False,
            )
        except Exception:
            pass
        
        # 首先检查输入数据是否包含 inf/NaN，如果有，先清理
        price_series = df[price_col].replace([np.inf, -np.inf], np.nan)
        # 如果价格序列包含 inf/NaN，pct_change 可能产生 inf
        # 在计算 pct_change 前，确保没有 inf 值
        price_returns = price_series.pct_change()
        # 处理 inf 值（可能由除权、价格归零等导致）
        price_returns = price_returns.replace([np.inf, -np.inf], np.nan)
        # 裁剪极端值（防止除权、闪崩等异常情况）
        if clip_pct is not None:
            price_returns = price_returns.clip(-clip_pct, clip_pct)
        price_returns = price_returns.values  # t=0 为 NaN
        
        # 滚动计算（从 rolling_window 开始）
        for i in range(rolling_window, len(df)):
            if (i - rolling_window) % update_freq != 0:
                continue  # 跳过非更新点
            
            # 关键：只用 [i - rolling_window, i - 1] 的收益（共 rolling_window 个点）
            window_ret = price_returns[i - rolling_window : i]
            if len(window_ret) < rolling_window:
                continue
            
            # 跳过包含 NaN 的窗口
            if np.any(np.isnan(window_ret)):
                continue
            
            h = compute_hurst_dfa(window_ret)
            df.iloc[i, df.columns.get_loc("hurst_price_rolling")] = h
    
    # === 2. CVD 单期变化 ===
    if cvd_col and cvd_col in df.columns:
        # 首先检查输入数据是否包含 inf/NaN，如果有，先清理
        cvd_series = df[cvd_col].replace([np.inf, -np.inf], np.nan)
        # 在计算 diff 前，确保没有 inf 值
        cvd_diff = cvd_series.diff().replace([np.inf, -np.inf], np.nan).values  # t=0 为 NaN
        
        for i in range(rolling_window, len(df)):
            if (i - rolling_window) % update_freq != 0:
                continue
            
            window_diff = cvd_diff[i - rolling_window : i]
            if len(window_diff) < rolling_window:
                continue
            
            # 跳过包含 NaN 的窗口
            if np.any(np.isnan(window_diff)):
                continue
            
            h = compute_hurst_dfa(window_diff)
            df.iloc[i, df.columns.get_loc("hurst_cvd_rolling")] = h
    
    # === 3. 成交量收益率 ===
    if volume_col and volume_col in df.columns:
        # 首先检查输入数据是否包含 inf/NaN，如果有，先清理
        vol_series = df[volume_col].replace([np.inf, -np.inf], np.nan)
        # 如果成交量序列包含 inf/NaN，pct_change 可能产生 inf
        # 在计算 pct_change 前，确保没有 inf 值
        vol_returns = vol_series.pct_change()
        # 处理 inf 值（可能由成交量归零等导致）
        vol_returns = vol_returns.replace([np.inf, -np.inf], np.nan)
        # 裁剪极端值（成交量偶尔会有异常波动）
        if clip_pct is not None:
            vol_returns = vol_returns.clip(-clip_pct, clip_pct)
        vol_returns = vol_returns.values  # t=0 为 NaN
        
        for i in range(rolling_window, len(df)):
            if (i - rolling_window) % update_freq != 0:
                continue
            
            window_ret = vol_returns[i - rolling_window : i]
            if len(window_ret) < rolling_window:
                continue
            
            # 跳过包含 NaN 的窗口
            if np.any(np.isnan(window_ret)):
                continue
            
            h = compute_hurst_dfa(window_ret)
            df.iloc[i, df.columns.get_loc("hurst_volume_rolling")] = h
    
    # === 时间对齐：shift(1) 确保特征在 t 时刻仅依赖历史 ===
    hurst_cols = [col for col in df.columns if col.startswith("hurst_")]
    for col in hurst_cols:
        df[col] = df[col].shift(1)
    
    # 注意：不再 fillna(0.5)，保留 NaN 表示"不可用"
    # 下游模型（如 LightGBM）可以自行处理缺失值
    
    return df


# 保持原有的函数名作为别名，以便向后兼容
@register_feature("extract_hurst_features", category="hurst")
def extract_hurst_features(
    df: pd.DataFrame,
    price_col: str = "close",
    cvd_col: Optional[str] = None,
    volume_col: Optional[str] = None,
    rolling_window: int = 50,
    update_freq: Union[int, str] = "auto",  # "auto" 或具体数值
    clip_pct: Union[Optional[float], str] = "auto",  # "auto" 或具体数值或 None
) -> pd.DataFrame:
    """
    旧版函数，保持向后兼容性
    """
    return _compute_hurst_features_batch(df, price_col=price_col, cvd_col=cvd_col,
                                       volume_col=volume_col, rolling_window=rolling_window,
                                       update_freq=update_freq, clip_pct=clip_pct)
