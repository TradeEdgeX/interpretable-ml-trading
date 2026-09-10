"""
DTW (Dynamic Time Warping) 特征提取器
用于形态匹配和模式识别

改进版本特性：
1. ✅ 支持多尺度模板（短/中/长周期）- 通过 multi_scale=True 参数
2. ✅ 添加反向模板和随机模板作为负样本对比 - 通过 include_inverse=True, include_random=True
3. ✅ DTW距离归一化（除以sqrt(窗口长度)）- 通过 normalize_distance=True 参数
4. ✅ 支持warping window constraint（限制扭曲）- 通过 warping_window 参数
5. ✅ 支持灵活的窗口范围参数 - window 可以是 int, List[int], 或 Tuple[int, int, int]

使用示例：
    # 基础用法：单个窗口
    df_features = extract_dtw_features(df, window=20)
    
    # 多窗口：自动生成特征
    df_features = extract_dtw_features(df, window=[20, 30, 40])
    
    # 窗口范围：自动遍历
    df_features = extract_dtw_features(df, window=(20, 50, 10))  # 20, 30, 40, 50
    
    # 包含反向和随机模板（用于负样本对比）
    templates = create_dtw_templates(include_inverse=True, include_random=True)
    df_features = extract_dtw_features(df, templates=templates, window=30)
    
    # 使用warping window constraint（限制扭曲范围）
    df_features = extract_dtw_features(df, window=30, warping_window=0.1)  # 10%约束
    
    # 距离归一化（使不同窗口的距离可比）
    df_features = extract_dtw_features(df, window=[20, 30, 40], normalize_distance=True)
    
    # 多尺度模板
    templates = create_dtw_templates(multi_scale=True)
    df_features = extract_dtw_features(df, templates=templates, window=30)

注意事项：
    - 反向模板和随机模板的距离特征也要传入LightGBM，用于对比学习
    - 归一化距离使得不同窗口大小的特征可以公平比较
    - warping_window 参数可以防止过度扭曲，提高匹配质量
    - 多窗口特征会自动添加 _w{window} 后缀区分
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple, Union
import warnings
import os

from src.features.registry import register_feature

warnings.filterwarnings("ignore")

try:
    from dtaidistance import dtw
    DTW_AVAILABLE = True
except ImportError:
    DTW_AVAILABLE = False
    print("⚠️ dtaidistance package not available. DTW features will be disabled.")

# 使用公共归一化模块
from src.features.time_series.utils_normalization import normalize_series

# 导入配置管理模块（可选）
try:
    from .utils_dtw_config import get_dtw_config
except ImportError:
    get_dtw_config = None


def smooth_template(template: np.ndarray, window: int = 3) -> np.ndarray:
    """Simple moving average smoothing to avoid jagged edges"""
    if len(template) < window:
        return template
    return np.convolve(template, np.ones(window) / window, mode="same")


def resize_template(template: np.ndarray, target_length: int) -> np.ndarray:
    """
    调整模板长度（用于生成多尺度模板）
    
    Args:
        template: 原始模板
        target_length: 目标长度
    
    Returns:
        调整后的模板
    """
    if len(template) == target_length:
        return template
    
    # 使用线性插值调整长度
    original_indices = np.linspace(0, len(template) - 1, len(template))
    target_indices = np.linspace(0, len(template) - 1, target_length)
    resized = np.interp(target_indices, original_indices, template)
    return resized


def create_inverse_template(template: np.ndarray) -> np.ndarray:
    """
    创建反向模板（用于负样本对比）
    
    Args:
        template: 原始模板
    
    Returns:
        反向模板（上下翻转）
    """
    # 归一化到[0,1]范围
    min_val = template.min()
    max_val = template.max()
    if max_val - min_val < 1e-8:
        return template
    
    normalized = (template - min_val) / (max_val - min_val + 1e-8)
    # 翻转：1 - normalized
    inverse_normalized = 1.0 - normalized
    # 还原到原始范围
    inverse = inverse_normalized * (max_val - min_val) + min_val
    return inverse


def create_random_template(length: int, seed: Optional[int] = 42) -> np.ndarray:
    """
    创建随机模板（用于负样本对比）
    
    Args:
        length: 模板长度
        seed: 随机种子（固定种子保证可重复性）
    
    Returns:
        随机模板
    """
    if seed is not None:
        np.random.seed(seed)
    # 生成随机游走
    random_walk = np.cumsum(np.random.randn(length) * 0.1)
    # 归一化到[0,1]范围
    min_val = random_walk.min()
    max_val = random_walk.max()
    if max_val - min_val < 1e-8:
        return np.full(length, 0.5)
    return (random_walk - min_val) / (max_val - min_val + 1e-8)


def create_dtw_templates(
    include_inverse: bool = True,
    include_random: bool = True,
    multi_scale: bool = False,
) -> Dict[str, np.ndarray]:
    """
    创建DTW模板库（看涨、看跌、中继形态）
    
    Args:
        include_inverse: 是否包含反向模板（用于负样本对比）
        include_random: 是否包含随机模板（用于负样本对比）
        multi_scale: 是否生成多尺度模板（短/中/长周期）
    
    Returns:
        Dictionary of template names and normalized arrays
    """
    templates = {}
    
    # === 看涨形态 ===
    # 1. Hammer (锤子线)
    hammer = np.concatenate([
        np.linspace(1.0, 0.4, 15),
        [0.2, 0.35, 0.5, 0.6, 0.7]
    ])
    templates["hammer"] = smooth_template(hammer)
    
    # 2. 头肩底
    head_shoulders_bottom = np.array([
        0.8, 0.4, 0.6,  # 左肩
        0.3, 0.1, 0.4, 0.6,  # 头部
        0.5, 0.3, 0.5,  # 右肩
        0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0  # 突破
    ])
    templates["head_shoulder_bottom"] = smooth_template(head_shoulders_bottom)
    
    # 3. 双底 (W底)
    double_bottom = np.concatenate([
        np.linspace(1.0, 0.3, 6),  # 第一次下跌
        np.linspace(0.3, 0.7, 4),  # 反弹
        np.linspace(0.7, 0.25, 5),  # 第二次下跌
        np.linspace(0.25, 1.0, 5)  # 突破
    ])
    templates["double_bottom"] = smooth_template(double_bottom)
    
    # 4. 看涨吞没
    bullish_engulfing = np.concatenate([
        np.full(16, 0.8),
        [0.7, 0.6, 0.9, 1.0]
    ])
    templates["bullish_engulfing"] = bullish_engulfing
    
    # === 看跌形态 ===
    # 1. Shooting Star (射击之星)
    shooting_star = np.concatenate([
        np.linspace(0.2, 0.8, 15),
        [1.0, 0.85, 0.7, 0.6, 0.5]
    ])
    templates["shooting_star"] = smooth_template(shooting_star)
    
    # 2. 头肩顶
    head_shoulders_top = np.array([
        0.2, 0.6, 0.4,  # 左肩
        0.7, 1.0, 0.6, 0.4,  # 头部
        0.5, 0.7, 0.5,  # 右肩
        0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # 跌破
    ])
    templates["head_shoulder_top"] = smooth_template(head_shoulders_top)
    
    # 3. 双顶 (M顶)
    double_top = np.concatenate([
        np.linspace(0.2, 0.9, 6),  # 第一次上涨
        np.linspace(0.9, 0.5, 4),  # 回调
        np.linspace(0.5, 0.95, 5),  # 第二次上涨
        np.linspace(0.95, 0.0, 5)  # 跌破
    ])
    templates["double_top"] = smooth_template(double_top)
    
    # 4. 看跌吞没
    bearish_engulfing = np.concatenate([
        np.full(16, 0.3),
        [0.4, 0.5, 0.2, 0.1]
    ])
    templates["bearish_engulfing"] = bearish_engulfing
    
    # === 中继形态 ===
    # 1. 上升旗形
    bull_flag = np.concatenate([
        np.linspace(0.0, 1.0, 5),  # 旗杆
        np.linspace(0.95, 0.8, 15)  # 旗面
    ])
    templates["bull_flag"] = bull_flag
    
    # 2. 下降旗形
    bear_flag = np.concatenate([
        np.linspace(1.0, 0.0, 5),  # 旗杆
        np.linspace(0.05, 0.2, 15)  # 旗面
    ])
    templates["bear_flag"] = bear_flag
    
    # 3. 对称三角收敛
    triangle = np.concatenate([
        np.linspace(0.8, 0.2, 10),
        np.linspace(0.2, 0.8, 10)
    ])
    templates["triangle"] = triangle
    
    # 4. 下跌后横盘（压缩起点）
    decline_consolidation = np.concatenate([
        np.linspace(1.0, 0.3, 8),  # 快速下跌
        np.full(12, 0.3)  # 横盘
    ])
    templates["decline_consolidation"] = decline_consolidation
    
    # === 多尺度模板（短/中/长周期） ===
    if multi_scale:
        base_templates = templates.copy()
        for name, template in base_templates.items():
            base_len = len(template)
            # 短周期（0.7倍）
            short_len = max(int(base_len * 0.7), 5)
            templates[f"{name}_short"] = smooth_template(resize_template(template, short_len))
            # 长周期（1.5倍）
            long_len = int(base_len * 1.5)
            templates[f"{name}_long"] = smooth_template(resize_template(template, long_len))
    
    # === 反向模板（用于负样本对比） ===
    if include_inverse:
        base_templates = {k: v for k, v in templates.items() if not k.endswith("_short") and not k.endswith("_long")}
        for name, template in base_templates.items():
            # 跳过已经是对称的形态（如triangle）
            if name not in ["triangle", "bull_flag", "bear_flag"]:
                inverse_template = create_inverse_template(template)
                templates[f"{name}_inverse"] = smooth_template(inverse_template)
    
    # === 随机模板（用于负样本对比） ===
    if include_random:
        # 创建几个不同长度的随机模板
        for length in [15, 20, 30]:
            templates[f"random_{length}"] = create_random_template(length, seed=42 + length)
    
    return templates


@register_feature("extract_dtw_features", category="dtw")
def extract_dtw_features(
    df: pd.DataFrame,
    price_col: str = "close",
    window: Union[int, List[int], Tuple[int, int, int]] = 20,
    templates: Optional[Dict[str, np.ndarray]] = None,
    template_filter: Optional[List[str]] = None,
    include_inverse: Optional[bool] = None,
    include_random: Optional[bool] = None,
    compute_only_near_sr: bool = False,
    sr_dist_col: Optional[str] = None,
    sr_threshold: float = 1.0,
    normalize_distance: bool = True,
    warping_window: Optional[float] = None,
    use_c: bool = True,
) -> pd.DataFrame:
    """
    提取DTW形态匹配特征（改进版）
    
    Args:
        df: DataFrame with price data
        price_col: Column name for price
        window: Window size(s) for DTW matching. Can be:
            - int: Single window (e.g., 20)
            - List[int]: Multiple windows (e.g., [20, 30, 40])
            - Tuple[int, int, int]: Window range (min, max, step) for auto-generation
        templates: Custom templates dict (if None, uses default templates)
        template_filter: Optional list of template names to include. If provided, only these templates
            and their inverse/random counterparts will be used. Useful for strategy-specific filtering.
            Example: ["hammer", "head_shoulder_bottom"] will include these templates plus their inverses.
        compute_only_near_sr: Only compute DTW near SR levels (for efficiency)
        sr_dist_col: Column name for distance to nearest SR (if compute_only_near_sr=True)
        sr_threshold: Threshold for "near SR" (in ATR units)
        normalize_distance: If True, normalize DTW distance by sqrt(window) to make it comparable across windows
        warping_window: Warping window constraint (Sakoe-Chiba band). 
            If None, no constraint. If float (0-1), relative to sequence length.
            Example: 0.1 means 10% of sequence length.
        use_c: Use C implementation for faster computation (if available)
    
    Returns:
        DataFrame with DTW distance features:
        - dtw_{template_name}_dist or dtw_{template_name}_w{window}_dist: DTW distance to each template
        - dtw_min_dist or dtw_min_dist_w{window}: Minimum distance across all templates
        - dtw_best_match or dtw_best_match_w{window}: Name of best matching template
    """
    if not DTW_AVAILABLE:
        # Return empty features if dtaidistance is not available
        default_templates = create_dtw_templates()
        # Handle window parameter
        if isinstance(window, tuple):
            windows = list(range(window[0], window[1] + 1, window[2]))
        elif isinstance(window, list):
            windows = window
        else:
            windows = [window]
        
        # Keep naming consistent with DTW_AVAILABLE=True path:
        # - dtw_{template}_dist_w{window} (multi-window) / dtw_{template}_dist (single window)
        # - dtw_min_dist_w{window}, dtw_best_match_w{window} (multi-window) / dtw_min_dist, dtw_best_match (single window)
        out = pd.DataFrame(index=df.index)
        for w in windows:
            suffix = f"_w{w}" if len(windows) > 1 else ""
            for name in default_templates.keys():
                out[f"dtw_{name}_dist{suffix}"] = 1.0
            out[f"dtw_min_dist{suffix}"] = 1.0
            out[f"dtw_best_match{suffix}"] = "none"
        return out
    
    df = df.copy()
    
    if price_col not in df.columns:
        raise ValueError(f"Price column '{price_col}' not found")
    
    if templates is None:
        # Fast mode: if FEATURE_FAST_MODE=1, disable random templates unless explicitly enabled.
        fast_mode = str(os.getenv("FEATURE_FAST_MODE", "")).strip() in {
            "1",
            "true",
            "True",
            "yes",
            "YES",
        }
        inv = True if include_inverse is None else bool(include_inverse)
        rnd = True if include_random is None else bool(include_random)
        if fast_mode and include_random is None:
            rnd = False
        templates = create_dtw_templates(include_inverse=inv, include_random=rnd)
    
    # Filter templates if template_filter is provided
    if template_filter is not None:
        filtered_templates = {}
        for name in template_filter:
            if name in templates:
                filtered_templates[name] = templates[name]
            # Also include inverse templates if they exist
            inverse_name = f"{name}_inverse"
            if inverse_name in templates:
                filtered_templates[inverse_name] = templates[inverse_name]
        # Optionally include random templates for contrastive learning
        if bool(include_random) or (
            include_random is None
            and str(os.getenv("FEATURE_FAST_MODE", "")).strip()
            not in {"1", "true", "True", "yes", "YES"}
        ):
            for name, template in templates.items():
                if name.startswith("random_"):
                    filtered_templates[name] = template
        templates = filtered_templates
        if not templates:
            raise ValueError(f"No templates found matching filter: {template_filter}")
    
    # Parse window parameter
    if isinstance(window, tuple):
        # Window range: (min, max, step)
        windows = list(range(window[0], window[1] + 1, window[2]))
    elif isinstance(window, list):
        windows = window
    else:
        windows = [window]
    
    # Initialize result DataFrame
    result = pd.DataFrame(index=df.index)
    
    prices = df[price_col].values
    n = len(df)
    
    # Process each window
    for window_size in windows:
        # Determine which indices to compute
        if compute_only_near_sr and sr_dist_col and sr_dist_col in df.columns:
            # Only compute near SR levels
            sr_dist = df[sr_dist_col].abs()
            
            # If ATR column exists, normalize by ATR
            if "atr" in df.columns:
                atr = df["atr"].fillna(df["atr"].median())
                sr_dist_normalized = sr_dist / (atr + 1e-8)
                compute_mask = sr_dist_normalized <= sr_threshold
            else:
                # Fallback: use absolute distance (less accurate)
                compute_mask = sr_dist <= sr_threshold
            
            compute_indices = np.where(compute_mask)[0]
            if len(compute_indices) == 0:
                # If no indices match, compute for all (fallback)
                compute_indices = np.arange(window_size, n)
        else:
            # Compute for all indices
            compute_indices = np.arange(window_size, n)
        
        # Initialize columns for this window
        window_suffix = f"_w{window_size}" if len(windows) > 1 else ""
        
        # Compute DTW distances for this window
        for i in compute_indices:
            if i < window_size:
                continue
            
            # Get price window
            price_window = prices[i - window_size : i]
            
            # Skip if too many NaN or constant values
            if np.sum(np.isnan(price_window)) > window_size * 0.1:
                continue
            if np.std(price_window) < 1e-8:
                continue
            
            # Normalize price window
            norm_price = normalize_series(price_window)
            
            # Compute DTW distance to each template
            min_dist = np.inf
            best_match = None
            
            for template_name, template in templates.items():
                try:
                    # Normalize template
                    norm_template = normalize_series(template)
                    
                    # Calculate warping window size if specified
                    # dtaidistance uses Sakoe-Chiba band: window parameter is the maximum
                    # allowed shift from the diagonal (in number of steps)
                    warping_window_size = None
                    if warping_window is not None:
                        # Convert relative window (0-1) to absolute size
                        # For Sakoe-Chiba band, we use the maximum sequence length
                        max_len = max(len(norm_price), len(norm_template))
                        warping_window_size = max(1, int(max_len * warping_window))
                    
                    # Compute DTW distance with optional warping window constraint
                    try:
                        if warping_window_size is not None:
                            # Use warping path with constraint (Sakoe-Chiba band)
                            distance = dtw.distance(
                                norm_price, 
                                norm_template,
                                window=warping_window_size,
                                use_c=use_c
                            )
                        else:
                            # Standard DTW without constraint
                            distance = dtw.distance(
                                norm_price, 
                                norm_template,
                                use_c=use_c
                            )
                    except TypeError:
                        # Fallback if window parameter is not supported in this version
                        distance = dtw.distance(
                            norm_price, 
                            norm_template,
                            use_c=use_c
                        )
                    
                    # Normalize distance by sqrt(window) to make it comparable across windows
                    if normalize_distance:
                        distance = distance / np.sqrt(window_size)
                    
                    # Convert distance to similarity score [0, 1] using exp(-dist/scale)
                    # This makes the feature more interpretable:
                    # - similarity = 1.0: perfect match
                    # - similarity = 0.0: no match
                    # The scale parameter controls the decay rate
                    similarity_scale = 0.5  # Can be tuned
                    similarity = np.exp(-distance / similarity_scale)
                    
                    # Store similarity score (not raw distance)
                    # This is the normalized, cross-asset comparable version
                    col_name = f"dtw_{template_name}_dist{window_suffix}"
                    if col_name not in result.columns:
                        result[col_name] = np.nan
                    result.loc[df.index[i], col_name] = similarity
                    
                    # Track minimum
                    if distance < min_dist:
                        min_dist = distance
                        best_match = template_name
                        
                except Exception:
                    # Skip failed DTW computations
                    continue
            
            # Store minimum distance (as similarity score) and best match
            if min_dist < np.inf:
                # Convert min_dist to similarity score as well
                min_similarity = np.exp(-min_dist / 0.5)  # Same scale as individual patterns
                
                min_col = f"dtw_min_dist{window_suffix}"
                match_col = f"dtw_best_match{window_suffix}"
                if min_col not in result.columns:
                    result[min_col] = np.nan
                    result[match_col] = None
                result.loc[df.index[i], min_col] = min_similarity
                result.loc[df.index[i], match_col] = best_match
        
        # NOTE:
        # Do NOT forward-fill DTW distances across time.
        # DTW is window-based; when there is insufficient history (e.g., at the start of a chunk
        # in streaming inference), the distance should remain NaN so callers can drop/ignore it.
        # Forward-filling makes streaming-vs-batch comparisons inconsistent.

    # Fill only categorical match columns. Keep numeric distance columns as NaN if not computed.
    for col in result.columns:
        if "best_match" in col:
            result[col] = result[col].fillna("none")
    
    return result


@register_feature("extract_dtw_features_from_series", category="dtw")
def extract_dtw_features_from_series(
    *,
    close: pd.Series,
    dist_to_nearest_sr: Optional[pd.Series] = None,
    atr: Optional[pd.Series] = None,
    window: Union[int, List[int], Tuple[int, int, int]] = 20,
    templates: Optional[Dict[str, np.ndarray]] = None,
    template_filter: Optional[List[str]] = None,
    include_inverse: Optional[bool] = None,
    include_random: Optional[bool] = None,
    compute_only_near_sr: bool = False,
    sr_dist_col: Optional[str] = None,
    sr_threshold: float = 1.0,
    normalize_distance: bool = True,
    warping_window: Optional[float] = None,
    use_c: bool = True,
) -> pd.DataFrame:
    """
    Narrow-IO wrapper for DTW features (Series-in, DataFrame-out).
    This avoids passing a wide DataFrame through the pipeline while reusing the
    legacy DTW implementation.
    """
    df = pd.DataFrame({"close": close})
    if dist_to_nearest_sr is not None:
        df["dist_to_nearest_sr"] = dist_to_nearest_sr
    if atr is not None:
        df["atr"] = atr
    return extract_dtw_features(
        df,
        price_col="close",
        window=window,
        templates=templates,
        template_filter=template_filter,
        include_inverse=include_inverse,
        include_random=include_random,
        compute_only_near_sr=compute_only_near_sr,
        sr_dist_col=sr_dist_col,
        sr_threshold=sr_threshold,
        normalize_distance=normalize_distance,
        warping_window=warping_window,
        use_c=use_c,
    )
