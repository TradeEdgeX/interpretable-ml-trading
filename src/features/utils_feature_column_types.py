"""
特征列类型工具函数

从配置文件中读取特征列的类型信息（numeric/categorical），用于模型训练时正确处理特征。
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional, Set
import logging

logger = logging.getLogger(__name__)


def load_column_types_config(
    config_path: str = "config/feature_column_types.yaml"
) -> Dict[str, str]:
    """
    加载特征列类型配置
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        Dictionary mapping column names to types ("numeric" or "categorical")
    """
    config_file = Path(config_path)
    if not config_file.exists():
        logger.warning(f"Column types config file not found: {config_path}. Using defaults.")
        return {}
    
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        
        column_types = config.get("column_types", {})
        logger.info(f"Loaded {len(column_types)} column type definitions from {config_path}")
        return column_types
    except Exception as e:
        logger.warning(f"Failed to load column types config from {config_path}: {e}. Using defaults.")
        return {}


def get_categorical_columns(
    feature_columns: List[str],
    column_types_config: Optional[Dict[str, str]] = None,
    config_path: Optional[str] = None,
) -> List[str]:
    """
    从特征列列表中识别分类特征列
    
    Args:
        feature_columns: 特征列名列表
        column_types_config: 列类型配置字典（可选，如果不提供则从配置文件加载）
        config_path: 配置文件路径（仅在 column_types_config 为 None 时使用）
    
    Returns:
        分类特征列名列表
    """
    if column_types_config is None:
        if config_path is None:
            config_path = "config/feature_column_types.yaml"
        column_types_config = load_column_types_config(config_path)
    
    categorical_columns = []
    for col in feature_columns:
        col_type = column_types_config.get(col)
        if col_type == "categorical":
            categorical_columns.append(col)
    
    if categorical_columns:
        logger.info(f"Identified {len(categorical_columns)} categorical columns: {categorical_columns}")
    
    return categorical_columns


def get_categorical_column_patterns(
    column_types_config: Optional[Dict[str, str]] = None,
    config_path: Optional[str] = None,
) -> List[str]:
    """
    从配置中提取分类列的模式（用于匹配列名）
    
    Args:
        column_types_config: 列类型配置字典（可选）
        config_path: 配置文件路径（可选）
    
    Returns:
        分类列的模式列表（例如 ["dtw_best_match", "_symbol"]）
    """
    if column_types_config is None:
        if config_path is None:
            config_path = "config/feature_column_types.yaml"
        column_types_config = load_column_types_config(config_path)
    
    patterns = set()
    for col, col_type in column_types_config.items():
        if col_type == "categorical":
            # Extract pattern: for "dtw_best_match_w20", use "dtw_best_match" as pattern
            if "_w" in col:
                pattern = col.split("_w")[0]  # e.g., "dtw_best_match" from "dtw_best_match_w20"
                patterns.add(pattern)
            else:
                patterns.add(col)
    
    return sorted(list(patterns))

