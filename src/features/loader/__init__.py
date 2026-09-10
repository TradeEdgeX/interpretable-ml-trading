"""
特征加载器模块

提供基于配置文件的特征加载、并行计算和缓存功能
"""

from src.features.registry import (
    get_compute_func,
    get_feature_func,
    ensure_features_registered,
)
from src.features.loader.strategy_feature_loader import StrategyFeatureLoader
from src.features.loader.feature_computer import FeatureComputer
from src.features.loader.feature_computer import analyze_dependency_levels

__all__ = [
    "get_compute_func",
    "get_feature_func",
    "ensure_features_registered",
    "StrategyFeatureLoader",
    "FeatureComputer",
    "analyze_dependency_levels",
]
