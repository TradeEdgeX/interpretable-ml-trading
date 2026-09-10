"""
DTW特征配置管理模块
为不同交易策略提供优化的DTW特征配置
"""

from typing import Dict, List, Optional, Any


# DTW策略配置字典
DTW_STRATEGY_CONFIG = {
    "reversal": {
        "templates": [
            "hammer",
            "head_shoulder_bottom",
            "double_bottom",
            "bullish_engulfing",
            "shooting_star",
            "head_shoulder_top",
            "double_top",
            "bearish_engulfing",
        ],
        "windows": [15, 20, 25],  # 反转形态通常在1-5根K线内完成，关键反转点集中在最后10-20根K线
        "add_inverse": True,  # 自动添加反向模板用于对比学习
        "add_random": True,  # 自动添加随机模板用于负样本对比
        "compute_only_near_sr": True,
        "sr_dist_col": "dist_to_nearest_sr",
        "sr_threshold": 1.5,
        "normalize_distance": True,
        "warping_window": 0.1,
        "use_c": True,
        "description": "SR反转策略：反转形态（锤子线、头肩底/顶、双底/顶、吞没形态）",
    },
    "breakout": {
        "templates": [
            "head_shoulder_bottom",
            "head_shoulder_top",
            "double_bottom",
            "double_top",
        ],
        "windows": [30, 40, 50, 60],  # 头肩形态通常需30-60根K线，双底也常在20-50根K线完成
        "add_inverse": True,
        "add_random": True,
        "compute_only_near_sr": False,  # 突破可能发生在远离SR的趋势中
        "sr_dist_col": "dist_to_nearest_sr",
        "sr_threshold": 2.0,  # 放宽阈值，捕捉更多突破机会
        "normalize_distance": True,
        "warping_window": 0.1,
        "use_c": True,
        "description": "SR突破策略：突破形态（头肩底/顶、双底/顶）",
    },
    "compression": {
        "templates": [
            "triangle",
            "bull_flag",
            "bear_flag",
            "decline_consolidation",
        ],
        "windows": [20, 30, 40, 50],  # 对称三角形20-50根K线，旗形10-30根，压缩阶段通常<30根（去掉60避免包含突破后走势）
        "add_inverse": True,
        "add_random": True,
        "compute_only_near_sr": False,  # 压缩区可能不在SR附近
        "normalize_distance": True,
        "warping_window": 0.1,
        "use_c": True,
        "description": "压缩区突破策略：中继形态（三角收敛、旗形、横盘整理）",
    },
    "trend": {
        "templates": [
            "bull_flag",
            "bear_flag",
            "triangle",
        ],
        "windows": [25, 35, 45],  # 旗形理想窗口15-30（旗杆5-10+旗面10-20），三角形可到40，25-45更平衡
        "add_inverse": True,
        "add_random": True,
        "compute_only_near_sr": False,
        "normalize_distance": True,
        "warping_window": 0.1,
        "use_c": True,
        "description": "趋势跟踪策略：中继形态（旗形、三角收敛）",
    },
}


def get_dtw_config(
    strategy_name: str,
    override_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    获取指定策略的DTW特征配置
    
    Args:
        strategy_name: 策略名称，支持：
            - "reversal" 或 "sr_reversal": SR反转策略
            - "breakout" 或 "sr_breakout": SR突破策略
            - "compression" 或 "compression_breakout": 压缩区突破策略
            - "trend" 或 "trend_following": 趋势跟踪策略
        override_params: 可选的参数字典，用于覆盖默认配置
    
    Returns:
        DTW特征配置字典，包含：
        - templates: 模板列表
        - windows: 窗口列表
        - template_filter: 模板过滤列表（用于extract_dtw_features）
        - 其他extract_dtw_features函数参数
    
    Examples:
        >>> config = get_dtw_config("reversal")
        >>> print(config["windows"])  # [15, 20, 25]
        
        >>> # 自定义窗口
        >>> config = get_dtw_config("reversal", {"windows": [10, 15, 20]})
        >>> print(config["windows"])  # [10, 15, 20]
    """
    # 策略名称映射
    strategy_map = {
        "reversal": "reversal",
        "sr_reversal": "reversal",
        "breakout": "breakout",
        "sr_breakout": "breakout",
        "compression": "compression",
        "compression_breakout": "compression",
        "trend": "trend",
        "trend_following": "trend",
    }
    
    # 规范化策略名称
    normalized_name = strategy_map.get(strategy_name.lower())
    if normalized_name is None:
        raise ValueError(
            f"Unknown strategy: {strategy_name}. "
            f"Supported strategies: {list(strategy_map.keys())}"
        )
    
    # 获取基础配置
    base_config = DTW_STRATEGY_CONFIG[normalized_name].copy()
    
    # 应用覆盖参数
    if override_params:
        base_config.update(override_params)
    
    # 构建extract_dtw_features函数所需的参数字典
    config = {
        "template_filter": base_config["templates"],
        "window": base_config["windows"],
        "normalize_distance": base_config["normalize_distance"],
        "warping_window": base_config["warping_window"],
        "use_c": base_config["use_c"],
    }
    
    # 添加SR相关参数（如果适用）
    if base_config.get("compute_only_near_sr", False):
        config["compute_only_near_sr"] = True
        config["sr_dist_col"] = base_config.get("sr_dist_col", "dist_to_nearest_sr")
        config["sr_threshold"] = base_config.get("sr_threshold", 1.5)
    else:
        config["compute_only_near_sr"] = False
    
    # 保留原始配置信息（用于文档/调试）
    config["_meta"] = {
        "strategy": normalized_name,
        "description": base_config.get("description", ""),
        "templates": base_config["templates"],
        "windows": base_config["windows"],
    }
    
    return config


def list_available_strategies() -> List[str]:
    """
    列出所有可用的策略配置
    
    Returns:
        策略名称列表
    """
    return list(DTW_STRATEGY_CONFIG.keys())


def get_strategy_summary() -> Dict[str, str]:
    """
    获取所有策略的简要说明
    
    Returns:
        策略名称到描述的映射
    """
    return {
        name: config.get("description", "")
        for name, config in DTW_STRATEGY_CONFIG.items()
    }


if __name__ == "__main__":
    # 示例用法
    print("=" * 80)
    print("DTW策略配置示例")
    print("=" * 80)
    
    for strategy in list_available_strategies():
        config = get_dtw_config(strategy)
        print(f"\n策略: {strategy}")
        print(f"  描述: {config['_meta']['description']}")
        print(f"  模板: {config['_meta']['templates']}")
        print(f"  窗口: {config['_meta']['windows']}")
        print(f"  仅在SR附近计算: {config.get('compute_only_near_sr', False)}")

