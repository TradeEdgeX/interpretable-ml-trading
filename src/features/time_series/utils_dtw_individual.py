"""
独立的DTW特征提取器
每个DTW模板一个独立的函数，支持按需加载和多窗口特征
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Union

from src.features.registry import register_feature
from .utils_dtw_features import (
    create_dtw_templates,
    extract_dtw_features,
    normalize_series,
    smooth_template,
)

try:
    from dtaidistance import dtw
    DTW_AVAILABLE = True
except ImportError:
    DTW_AVAILABLE = False


# 定义每个DTW模板推荐的窗口范围
DTW_TEMPLATE_WINDOWS = {
    # 单K线形态（窗口较小）
    "hammer": [10, 15, 20],  # 锤子线：10-20
    "shooting_star": [10, 15, 20],  # 射击之星：10-20
    "bullish_engulfing": [10, 15, 20],  # 看涨吞没：10-20
    "bearish_engulfing": [10, 15, 20],  # 看跌吞没：10-20
    
    # 简单形态（窗口中等）
    "double_bottom": [20, 30, 40],  # 双底：20-40
    "double_top": [20, 30, 40],  # 双顶：20-40
    
    # 复杂形态（窗口较大）
    "head_shoulder_bottom": [30, 40, 50, 60],  # 头肩底：30-60
    "head_shoulder_top": [30, 40, 50, 60],  # 头肩顶：30-60
    
    # 中继形态（窗口较大）
    "bull_flag": [20, 30, 40],  # 上升旗形：20-40
    "bear_flag": [20, 30, 40],  # 下降旗形：20-40
    "triangle": [30, 40, 50, 60],  # 三角收敛：30-60
    "decline_consolidation": [30, 40, 50],  # 下跌后横盘：30-50
}


def _extract_single_dtw_template(
    df: pd.DataFrame,
    template_name: str,
    price_col: str = "close",
    windows: Union[int, List[int]] = None,
    compute_only_near_sr: bool = False,
    sr_dist_col: Optional[str] = None,
    sr_threshold: float = 1.5,
) -> pd.DataFrame:
    """
    提取单个DTW模板的特征（支持多窗口）
    
    Args:
        df: DataFrame with price data
        template_name: Name of the template (e.g., "hammer", "head_shoulder_bottom")
        price_col: Column name for price
        windows: Window size(s) for DTW matching. Can be:
            - int: Single window (e.g., 20)
            - List[int]: Multiple windows (e.g., [10, 20, 30])
            - None: Use recommended windows from DTW_TEMPLATE_WINDOWS
        compute_only_near_sr: Only compute near SR levels
        sr_dist_col: Column name for distance to nearest SR
        sr_threshold: Threshold for "near SR" (in ATR units)
    
    Returns:
        DataFrame with DTW distance feature(s):
        - If single window: dtw_{template_name}_dist
        - If multiple windows: dtw_{template_name}_w{window}_dist
    """
    if not DTW_AVAILABLE:
        # Return empty features if dtaidistance is not available
        if windows is None:
            windows = DTW_TEMPLATE_WINDOWS.get(template_name, [20])
        if isinstance(windows, int):
            windows = [windows]
        
        cols = [f"dtw_{template_name}_w{w}_dist" for w in windows]
        return pd.DataFrame(
            index=df.index,
            columns=cols,
        ).fillna(1.0)
    
    # Get the specific template
    all_templates = create_dtw_templates()
    if template_name not in all_templates:
        raise ValueError(f"Unknown template: {template_name}")
    
    template = {template_name: all_templates[template_name]}
    
    # Determine windows to use
    if windows is None:
        # Use recommended windows for this template
        windows = DTW_TEMPLATE_WINDOWS.get(template_name, [20])
    elif isinstance(windows, int):
        # Single window
        windows = [windows]
    
    # Extract DTW features for each window
    result = pd.DataFrame(index=df.index)
    
    for window in windows:
        # Extract DTW features for this window
        window_result = extract_dtw_features(
            df=df,
            price_col=price_col,
            window=window,
            templates=template,
            compute_only_near_sr=compute_only_near_sr,
            sr_dist_col=sr_dist_col,
            sr_threshold=sr_threshold,
        )
        
        # Get the distance column for this template
        output_col = f"dtw_{template_name}_dist"
        if output_col in window_result.columns:
            # Rename to include window size
            if len(windows) > 1:
                new_col = f"dtw_{template_name}_w{window}_dist"
            else:
                # Single window: use original name for backward compatibility
                new_col = f"dtw_{template_name}_dist"
            result[new_col] = window_result[output_col]
    
    # Fill NaN values
    result = result.fillna(1.0)
    
    return result


# 反转相关DTW特征（适合SR Reversal策略）
@register_feature("extract_dtw_hammer", category="dtw")
def extract_dtw_hammer(
    df: pd.DataFrame,
    price_col: str = "close",
    windows: Union[int, List[int]] = None,  # None = use recommended [10, 15, 20]
    compute_only_near_sr: bool = False,
    sr_dist_col: Optional[str] = None,
    sr_threshold: float = 1.5,
) -> pd.DataFrame:
    """
    提取DTW锤子线特征（看涨反转）
    
    推荐窗口: [10, 15, 20] - 单K线形态，窗口较小
    """
    return _extract_single_dtw_template(
        df, "hammer", price_col, windows, compute_only_near_sr, sr_dist_col, sr_threshold
    )


@register_feature("extract_dtw_head_shoulder_bottom", category="dtw")
def extract_dtw_head_shoulder_bottom(
    df: pd.DataFrame,
    price_col: str = "close",
    windows: Union[int, List[int]] = None,  # None = use recommended [30, 40, 50, 60]
    compute_only_near_sr: bool = False,
    sr_dist_col: Optional[str] = None,
    sr_threshold: float = 1.5,
) -> pd.DataFrame:
    """
    提取DTW头肩底特征（看涨反转）
    
    推荐窗口: [30, 40, 50, 60] - 复杂形态，需要较大窗口
    """
    return _extract_single_dtw_template(
        df, "head_shoulder_bottom", price_col, windows, compute_only_near_sr, sr_dist_col, sr_threshold
    )


@register_feature("extract_dtw_double_bottom", category="dtw")
def extract_dtw_double_bottom(
    df: pd.DataFrame,
    price_col: str = "close",
    windows: Union[int, List[int]] = None,  # None = use recommended [20, 30, 40]
    compute_only_near_sr: bool = False,
    sr_dist_col: Optional[str] = None,
    sr_threshold: float = 1.5,
) -> pd.DataFrame:
    """
    提取DTW双底特征（看涨反转）
    
    推荐窗口: [20, 30, 40] - 简单形态，中等窗口
    """
    return _extract_single_dtw_template(
        df, "double_bottom", price_col, windows, compute_only_near_sr, sr_dist_col, sr_threshold
    )


@register_feature("extract_dtw_bullish_engulfing", category="dtw")
def extract_dtw_bullish_engulfing(
    df: pd.DataFrame,
    price_col: str = "close",
    windows: Union[int, List[int]] = None,  # None = use recommended [10, 15, 20]
    compute_only_near_sr: bool = False,
    sr_dist_col: Optional[str] = None,
    sr_threshold: float = 1.5,
) -> pd.DataFrame:
    """
    提取DTW看涨吞没特征（看涨反转）
    
    推荐窗口: [10, 15, 20] - 单K线形态，窗口较小
    """
    return _extract_single_dtw_template(
        df, "bullish_engulfing", price_col, windows, compute_only_near_sr, sr_dist_col, sr_threshold
    )


@register_feature("extract_dtw_shooting_star", category="dtw")
def extract_dtw_shooting_star(
    df: pd.DataFrame,
    price_col: str = "close",
    windows: Union[int, List[int]] = None,  # None = use recommended [10, 15, 20]
    compute_only_near_sr: bool = False,
    sr_dist_col: Optional[str] = None,
    sr_threshold: float = 1.5,
) -> pd.DataFrame:
    """
    提取DTW射击之星特征（看跌反转）
    
    推荐窗口: [10, 15, 20] - 单K线形态，窗口较小
    """
    return _extract_single_dtw_template(
        df, "shooting_star", price_col, windows, compute_only_near_sr, sr_dist_col, sr_threshold
    )


@register_feature("extract_dtw_head_shoulder_top", category="dtw")
def extract_dtw_head_shoulder_top(
    df: pd.DataFrame,
    price_col: str = "close",
    windows: Union[int, List[int]] = None,  # None = use recommended [30, 40, 50, 60]
    compute_only_near_sr: bool = False,
    sr_dist_col: Optional[str] = None,
    sr_threshold: float = 1.5,
) -> pd.DataFrame:
    """
    提取DTW头肩顶特征（看跌反转）
    
    推荐窗口: [30, 40, 50, 60] - 复杂形态，需要较大窗口
    """
    return _extract_single_dtw_template(
        df, "head_shoulder_top", price_col, windows, compute_only_near_sr, sr_dist_col, sr_threshold
    )


@register_feature("extract_dtw_double_top", category="dtw")
def extract_dtw_double_top(
    df: pd.DataFrame,
    price_col: str = "close",
    windows: Union[int, List[int]] = None,  # None = use recommended [20, 30, 40]
    compute_only_near_sr: bool = False,
    sr_dist_col: Optional[str] = None,
    sr_threshold: float = 1.5,
) -> pd.DataFrame:
    """
    提取DTW双顶特征（看跌反转）
    
    推荐窗口: [20, 30, 40] - 简单形态，中等窗口
    """
    return _extract_single_dtw_template(
        df, "double_top", price_col, windows, compute_only_near_sr, sr_dist_col, sr_threshold
    )


@register_feature("extract_dtw_bearish_engulfing", category="dtw")
def extract_dtw_bearish_engulfing(
    df: pd.DataFrame,
    price_col: str = "close",
    windows: Union[int, List[int]] = None,  # None = use recommended [10, 15, 20]
    compute_only_near_sr: bool = False,
    sr_dist_col: Optional[str] = None,
    sr_threshold: float = 1.5,
) -> pd.DataFrame:
    """
    提取DTW看跌吞没特征（看跌反转）
    
    推荐窗口: [10, 15, 20] - 单K线形态，窗口较小
    """
    return _extract_single_dtw_template(
        df, "bearish_engulfing", price_col, windows, compute_only_near_sr, sr_dist_col, sr_threshold
    )


# 中继形态DTW特征（适合趋势/压缩突破策略）
@register_feature("extract_dtw_bull_flag", category="dtw")
def extract_dtw_bull_flag(
    df: pd.DataFrame,
    price_col: str = "close",
    windows: Union[int, List[int]] = None,  # None = use recommended [20, 30, 40]
    compute_only_near_sr: bool = False,
    sr_dist_col: Optional[str] = None,
    sr_threshold: float = 1.5,
) -> pd.DataFrame:
    """
    提取DTW上升旗形特征（中继）
    
    推荐窗口: [20, 30, 40] - 中继形态，中等窗口
    """
    return _extract_single_dtw_template(
        df, "bull_flag", price_col, windows, compute_only_near_sr, sr_dist_col, sr_threshold
    )


@register_feature("extract_dtw_bear_flag", category="dtw")
def extract_dtw_bear_flag(
    df: pd.DataFrame,
    price_col: str = "close",
    windows: Union[int, List[int]] = None,  # None = use recommended [20, 30, 40]
    compute_only_near_sr: bool = False,
    sr_dist_col: Optional[str] = None,
    sr_threshold: float = 1.5,
) -> pd.DataFrame:
    """
    提取DTW下降旗形特征（中继）
    
    推荐窗口: [20, 30, 40] - 中继形态，中等窗口
    """
    return _extract_single_dtw_template(
        df, "bear_flag", price_col, windows, compute_only_near_sr, sr_dist_col, sr_threshold
    )


@register_feature("extract_dtw_triangle", category="dtw")
def extract_dtw_triangle(
    df: pd.DataFrame,
    price_col: str = "close",
    windows: Union[int, List[int]] = None,  # None = use recommended [30, 40, 50, 60]
    compute_only_near_sr: bool = False,
    sr_dist_col: Optional[str] = None,
    sr_threshold: float = 1.5,
) -> pd.DataFrame:
    """
    提取DTW三角收敛特征（中继）
    
    推荐窗口: [30, 40, 50, 60] - 复杂中继形态，需要较大窗口
    """
    return _extract_single_dtw_template(
        df, "triangle", price_col, windows, compute_only_near_sr, sr_dist_col, sr_threshold
    )


@register_feature("extract_dtw_decline_consolidation", category="dtw")
def extract_dtw_decline_consolidation(
    df: pd.DataFrame,
    price_col: str = "close",
    windows: Union[int, List[int]] = None,  # None = use recommended [30, 40, 50]
    compute_only_near_sr: bool = False,
    sr_dist_col: Optional[str] = None,
    sr_threshold: float = 1.5,
) -> pd.DataFrame:
    """
    提取DTW下跌后横盘特征（中继）
    
    推荐窗口: [30, 40, 50] - 中继形态，中等偏大窗口
    """
    return _extract_single_dtw_template(
        df, "decline_consolidation", price_col, windows, compute_only_near_sr, sr_dist_col, sr_threshold
    )

