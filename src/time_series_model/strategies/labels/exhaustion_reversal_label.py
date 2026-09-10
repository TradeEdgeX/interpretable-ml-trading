"""
AER (AuctionExhaustionReversal) 策略标签

语义：趋势末端衰竭（量/波动极值）后反转
在 SR 反转的基础上增加"衰竭"条件过滤：
- atr_percentile 处于高位（趋势末端波动放大）
- path_efficiency 降低（价格效率下降，衰竭信号）
- 可选：volume 出现极值
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.time_series_model.pipeline.training.label_utils import (
    _ensure_atr,
)
from src.time_series_model.pipeline.training.label_utils import compute_rr_label


def compute_exhaustion_reversal_label(
    df: pd.DataFrame,
    price_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    atr_col: str = "atr",
    atr_window: int = 14,
    max_holding_bars: int = 50,
    stop_loss_r: float = 1.0,
    take_profit_r: float = 2.0,
    # 衰竭条件参数
    atr_percentile_col: str = "atr_percentile",
    atr_percentile_min: float = 0.7,  # ATR 百分位需 >= 此值（高波动）
    path_efficiency_col: str = "path_efficiency_pct",
    path_efficiency_max: Optional[float] = 0.5,  # 路径效率需 <= 此值（效率下降）
    # 可选：SR 过滤
    dist_to_sr_col: Optional[str] = "dist_to_nearest_sr",
    dist_atr_mult: Optional[float] = 1.5,
    # 方向
    combine_mode: str = "long_only",
) -> pd.Series:
    """
    计算衰竭反转标签：在 SR 反转基础上增加衰竭条件过滤。

    衰竭条件：
    1. ATR 百分位处于高位（atr_percentile >= atr_percentile_min）
    2. 路径效率下降（path_efficiency <= path_efficiency_max，可选）

    Args:
        df: DataFrame with features
        atr_percentile_col: ATR 百分位列名
        atr_percentile_min: ATR 百分位最小阈值（高波动 = 衰竭信号）
        path_efficiency_col: 路径效率列名
        path_efficiency_max: 路径效率最大阈值（效率低 = 衰竭信号）
        dist_to_sr_col: 距离 SR 列名（用于 SR 过滤）
        dist_atr_mult: SR 距离阈值（ATR 倍数）
        combine_mode: "long_only", "short_only", "any_success"

    Returns:
        pd.Series: 连续 RR 标签，非衰竭区域为 NaN
    """
    work_df = df.copy()
    atr_series = _ensure_atr(work_df, atr_col, price_col, high_col, low_col, atr_window)
    work_df[atr_col] = atr_series

    # 构建衰竭掩码
    exhaustion_mask = pd.Series(True, index=work_df.index)

    # 条件 1：ATR 百分位高（波动放大）
    if atr_percentile_col in work_df.columns:
        atr_pct = work_df[atr_percentile_col].fillna(0.5)
        exhaustion_mask &= atr_pct >= atr_percentile_min

    # 条件 2：路径效率低（可选）
    if path_efficiency_max is not None and path_efficiency_col in work_df.columns:
        path_eff = work_df[path_efficiency_col].fillna(0.5)
        exhaustion_mask &= path_eff <= path_efficiency_max

    # SR 过滤（可选）
    sr_mask = pd.Series(True, index=work_df.index)
    if dist_to_sr_col is not None and dist_atr_mult is not None:
        if dist_to_sr_col in work_df.columns:
            price_series = work_df[price_col]
            dist_pct = work_df[dist_to_sr_col].abs()
            abs_distance = dist_pct * price_series
            dist_normalized = abs_distance / (atr_series + 1e-8)
            sr_mask = dist_normalized <= dist_atr_mult
            sr_mask = sr_mask.fillna(False)

    # 最终掩码：衰竭 + SR 附近
    final_mask = exhaustion_mask & sr_mask

    # 计算 RR 标签
    if combine_mode == "long_only":
        work_df["__signal"] = 1.0
    elif combine_mode == "short_only":
        work_df["__signal"] = -1.0
    else:
        work_df["__signal"] = 1.0  # 默认做多，后面再合并

    rr_series = compute_rr_label(
        work_df,
        signal_col="__signal",
        price_col=price_col,
        atr_col=atr_col,
        atr_window=atr_window,
        rr_ratio=take_profit_r / stop_loss_r if stop_loss_r != 0 else take_profit_r,
        max_holding_bars=max_holding_bars,
        stop_loss_r=stop_loss_r,
        take_profit_r=take_profit_r,
        use_continuous_label=True,
        entry_price_col="open",
        entry_offset=1,
    )

    # 应用衰竭掩码
    rr_series = rr_series.where(final_mask)
    rr_series.name = "rr_label"

    return rr_series
