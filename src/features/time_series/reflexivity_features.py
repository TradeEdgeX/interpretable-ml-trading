"""
反身性监测特征（Reflexivity Monitoring Features）

实现反身性监测指标，用于识别市场反身性风险：
- OFCI (Order Flow Consensus Index): 订单流一致性指数
- SHD (Strategy Homogeneity Detector): 策略同质化探测器

注意：LFI (Liquidity Fragility Index) 需要订单簿数据，暂不实现。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from src.features.registry import register_feature
from src.features.time_series.baseline_features import compute_percentile_rank_from_series


@register_feature("compute_ofci_from_trades", category="reflexivity")
def compute_ofci_from_trades(
    trades: Optional[pd.DataFrame] = None,
    window: int = 100,
    ticks_loader_json: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    计算订单流一致性指数（Order Flow Consensus Index, OFCI）
    
    定义：衡量全市场"方向性共识强度"——越高越危险
    
    Args:
        trades: 逐笔成交数据，必须包含'side'列（+1=buy, -1=sell）。如果为None，将从ticks_loader_json加载
        window: 滚动窗口大小（建议100，对应10-30秒）
        ticks_loader_json: JSON字符串，用于加载tick数据（如果trades为None）
        df: OHLCV DataFrame（用于获取时间范围，如果使用ticks_loader_json）
    
    Returns:
        DataFrame with column 'ofci': [-1, 1] 的对称指标
        - |OFCI| > 0.7 → 高度一致（警惕反身性踩踏或追涨）
        - |OFCI| < 0.3 → 方向分散（相对安全）
    """
    # 如果没有提供trades，尝试从ticks_loader_json加载
    if trades is None or len(trades) == 0:
        if ticks_loader_json:
            from src.data_tools.tick_loader import (
                deserialize_tick_loader_params,
                load_tick_data,
            )
            loader_params = deserialize_tick_loader_params(ticks_loader_json)
            symbol = str(loader_params["symbol"]).upper()
            start_ts = loader_params.get("start_ts")
            end_ts = loader_params.get("end_ts")
            ticks_dir = loader_params.get("ticks_dir", "data/parquet_data")
            
            # 如果df提供了时间范围，使用df的范围
            if df is not None and isinstance(df.index, pd.DatetimeIndex) and len(df) > 0:
                start_ts = str(df.index.min())
                end_ts = str(df.index.max())
            
            # 加载tick数据
            # 确保start_ts和end_ts是字符串格式
            if start_ts is None:
                if df is not None and isinstance(df.index, pd.DatetimeIndex) and len(df) > 0:
                    start_ts = str(df.index.min())
                else:
                    start_ts = "2024-01-01 00:00:00"
            if end_ts is None:
                if df is not None and isinstance(df.index, pd.DatetimeIndex) and len(df) > 0:
                    end_ts = str(df.index.max())
                else:
                    end_ts = "2024-12-31 23:59:59"
            
            lookback_minutes = loader_params.get("lookback_minutes", 60)
            trades = load_tick_data(
                symbol=symbol,
                start_ts=start_ts,
                end_ts=end_ts,
                ticks_dir=ticks_dir,
                lookback_minutes=lookback_minutes,
            )
        else:
            # 如果没有trades也没有ticks_loader_json，返回空DataFrame
            if df is not None and isinstance(df.index, pd.DatetimeIndex):
                return pd.DataFrame(index=df.index, columns=["ofci"], dtype=float).fillna(0.0)
            return pd.DataFrame(columns=["ofci"], dtype=float)
    
    if len(trades) == 0:
        if df is not None and isinstance(df.index, pd.DatetimeIndex):
            return pd.DataFrame(index=df.index, columns=["ofci"], dtype=float).fillna(0.0)
        return pd.DataFrame(columns=["ofci"], dtype=float)
    
    # 检查是否有side列
    if "side" in trades.columns:
        # 从tick数据计算
        directions = trades["side"].copy()
        # 标准化side值（确保是+1/-1）
        if directions.dtype == "object":
            directions = directions.map({"buy": 1, "sell": -1, "BUY": 1, "SELL": -1, 1: 1, -1: -1})
        else:
            directions = directions.astype(float)
        
        # 过滤无效值
        valid_mask = directions.isin([1, -1])
        if not valid_mask.any():
            return pd.DataFrame(index=trades.index, columns=["ofci"], dtype=float).fillna(0.0)
        
        # 使用 volume 加权计算 buy_ratio（与 VPIN signed_imbalance 保持一致）
        if "volume" in trades.columns:
            volume = trades["volume"].copy()
            buy_volume = np.where(directions == 1, volume, 0.0)
            sell_volume = np.where(directions == -1, volume, 0.0)
            
            buy_vol_rolling = pd.Series(buy_volume, index=trades.index).rolling(
                window=window, min_periods=window
            ).sum()
            sell_vol_rolling = pd.Series(sell_volume, index=trades.index).rolling(
                window=window, min_periods=window
            ).sum()
            total_vol_rolling = buy_vol_rolling + sell_vol_rolling
            
            # buy_ratio = buy_volume / total_volume
            buy_ratio = buy_vol_rolling / total_vol_rolling.replace(0, np.nan)
        else:
            # 回退到 tick 数量计算（不推荐，聚合数据场景下会失真）
            buy_count = (directions == 1).rolling(window=window, min_periods=window).sum()
            total_count = valid_mask.rolling(window=window, min_periods=window).sum()
            buy_ratio = buy_count / total_count.replace(0, np.nan)
        
        # 转换为[-1, 1]对称指标
        ofci = 2 * buy_ratio - 1
        ofci = ofci.fillna(0.0).clip(-1.0, 1.0)
        
        return pd.DataFrame({"ofci": ofci}, index=trades.index)
    
    else:
        # 如果没有side列，返回0（需要tick数据）
        return pd.DataFrame(index=trades.index, columns=["ofci"], dtype=float).fillna(0.0)


@register_feature("compute_ofci_pct_from_series", category="reflexivity")
def compute_ofci_pct_from_series(
    *,
    ofci: Optional[pd.Series] = None,  # 直接传入 OFCI 序列（用于测试）
    ticks_loader_json: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
    ofci_window: int = 100,
    percentile_window: int = 540,
    window: Optional[int] = None,  # 别名，兼容旧测试
    shift: int = 1,
) -> pd.DataFrame:
    """
    自包含版本：内部计算 OFCI 再转百分位
    
    1. OFCI = rolling_mean(trade_side) from tick data
    2. ofci_pct = percentile_rank(abs(OFCI))  # 反映极端一致性
    
    Args:
        ofci: 直接传入的 OFCI 序列（可选，用于测试场景）
        ticks_loader_json: JSON字符串，用于加载tick数据
        df: OHLCV DataFrame（用于获取时间范围）
        ofci_window: OFCI计算的滚动窗口
        percentile_window: 百分位计算的滚动窗口
        window: percentile_window 的别名（兼容旧测试）
        shift: 滞后 shift
    
    Returns:
        DataFrame with column 'ofci_pct': [0, 1] 的percentile值
        - ofci_pct > 0.9 表示极端一致性
    """
    # 兼容旧测试：window 是 percentile_window 的别名
    if window is not None:
        percentile_window = window
    
    # 如果直接传入 ofci 序列，跳过内部计算
    if ofci is not None:
        ofci_abs = ofci.abs()
        return compute_percentile_rank_from_series(
            series=ofci_abs,
            window=percentile_window,
            shift=shift,
            output_name="ofci_pct",
        )
    
    # Step 1: 计算 OFCI
    ofci_df = compute_ofci_from_trades(
        trades=None,
        window=ofci_window,
        ticks_loader_json=ticks_loader_json,
        df=df,
    )
    
    if ofci_df.empty or "ofci" not in ofci_df.columns:
        # 如果没有tick数据，返回默认值
        if df is not None and isinstance(df.index, pd.DatetimeIndex):
            return pd.DataFrame(index=df.index, columns=["ofci_pct"], dtype=float).fillna(0.5)
        return pd.DataFrame(columns=["ofci_pct"], dtype=float)
    
    # Step 2: 使用绝对值计算percentile rank
    ofci_abs = ofci_df["ofci"].abs()
    return compute_percentile_rank_from_series(
        series=ofci_abs,
        window=percentile_window,
        shift=shift,
        output_name="ofci_pct",
    )


@register_feature("compute_shd_from_series", category="reflexivity")
def compute_shd_from_series(
    *,
    cvd_series: pd.Series,
    price_returns: pd.Series,
    window: int = 60,
) -> pd.DataFrame:
    """
    计算策略同质化探测器（Strategy Homogeneity Detector, SHD）
    
    定义：检测"是否太多人在用类似逻辑交易"
    原理：如果CVD和价格变动高度同步，说明大量人用订单流策略
    
    Args:
        cvd_series: 累计成交量差额序列（CVD）
        price_returns: 价格收益率序列（log return或pct_change）
        window: 滚动窗口大小（建议60，对应1-5分钟）
    
    Returns:
        DataFrame with column 'shd': [0, 1] 的指标
        - SHD > 0.6 → 多数量化在用相似信号 → 反身性风险高
        - SHD接近0 → 多种策略在博弈（健康）
        - SHD接近1 → "同一类人推动价格"（危险）
    """
    if len(cvd_series) == 0 or len(price_returns) == 0:
        return pd.DataFrame(columns=["shd"], dtype=float)
    
    # 确保索引对齐
    common_index = cvd_series.index.intersection(price_returns.index)
    if len(common_index) == 0:
        return pd.DataFrame(columns=["shd"], dtype=float)
    
    cvd = cvd_series.loc[common_index].copy()
    ret = price_returns.loc[common_index].copy()
    
    # 计算CVD的变化（ΔCVD）
    # 根据文档：纯成交流版本使用 rolling_corr(ΔCVD, return)
    d_cvd = cvd.diff().fillna(0.0)
    
    # 使用pandas的rolling.corr计算滚动相关系数（更高效）
    # 根据文档：SHD = abs(rolling_corr(d_cvd, ret))
    rolling_corr = d_cvd.rolling(window=window, min_periods=max(window // 2, 10)).corr(ret)
    
    # 取绝对值作为SHD（符合文档要求）
    shd_series = rolling_corr.abs().fillna(0.0).clip(0.0, 1.0)
    
    return pd.DataFrame({"shd": shd_series}, index=common_index)


@register_feature("compute_shd_pct_from_series", category="reflexivity")
def compute_shd_pct_from_series(
    *,
    close: pd.Series,
    cvd: pd.Series,
    shd_window: int = 60,
    percentile_window: int = 540,
    shift: int = 1,
) -> pd.DataFrame:
    """
    自包含版本：内部计算 SHD 再转百分位
    
    1. SHD = abs(rolling_corr(ΔCVD, price_returns))
    2. shd_pct = percentile_rank(SHD)
    
    Args:
        close: 收盘价序列
        cvd: CVD序列（从 orderflow features 中获取）
        shd_window: SHD计算的滚动窗口
        percentile_window: 百分位计算的滚动窗口
        shift: 滞后 shift
    
    Returns:
        DataFrame with column 'shd_pct': [0, 1] 的percentile值
    """
    # Step 1: 计算 SHD
    price_returns = np.log(close / close.shift(1)).fillna(0.0)
    shd_df = compute_shd_from_series(
        cvd_series=cvd,
        price_returns=price_returns,
        window=shd_window,
    )
    shd_series = shd_df["shd"] if "shd" in shd_df.columns else pd.Series(0.0, index=close.index)
    
    # Step 2: 转百分位
    return compute_percentile_rank_from_series(
        series=shd_series,
        window=percentile_window,
        shift=shift,
        output_name="shd_pct",
    )


@register_feature("compute_shd_from_ohlcv", category="reflexivity")
def compute_shd_from_ohlcv(
    *,
    close: pd.Series,
    cvd: pd.Series,
    window: int = 60,
) -> pd.DataFrame:
    """
    从OHLCV数据计算SHD（不需要tick数据）
    
    这是SHD的简化版本，使用价格收益率和CVD序列计算。
    
    Args:
        close: 收盘价序列
        cvd: CVD序列（从orderflow features中获取）
        window: 滚动窗口大小
    
    Returns:
        DataFrame with column 'shd': [0, 1] 的指标
    """
    # 计算价格收益率
    price_returns = np.log(close / close.shift(1)).fillna(0.0)
    
    # 使用标准SHD计算
    return compute_shd_from_series(
        cvd_series=cvd,
        price_returns=price_returns,
        window=window,
    )
