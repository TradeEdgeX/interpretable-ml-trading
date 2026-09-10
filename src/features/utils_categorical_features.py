"""
分类特征工具函数

用于自动检测和设置分类特征（如 _symbol），支持多资产训练
"""

import pandas as pd
from typing import List, Optional, Set


def detect_categorical_features(
    df: pd.DataFrame,
    known_categorical: Optional[List[str]] = None,
    min_unique_values: int = 2,
    max_unique_values: int = 1000,
) -> List[str]:
    """
    自动检测分类特征
    
    Args:
        df: DataFrame with features
        known_categorical: 已知的分类特征列表（如 ["_symbol"]）
        min_unique_values: 分类特征的最小唯一值数量（避免常数特征）
        max_unique_values: 分类特征的最大唯一值数量（避免高基数特征）
    
    Returns:
        分类特征列表
    """
    categorical_features = []
    
    # 添加已知的分类特征
    if known_categorical:
        for col in known_categorical:
            if col in df.columns:
                unique_count = df[col].nunique()
                if unique_count >= min_unique_values:
                    if unique_count <= max_unique_values:
                        categorical_features.append(col)
                    else:
                        print(
                            f"   ⚠️  Skipping '{col}' as categorical (too many unique values: {unique_count} > {max_unique_values})"
                        )
    
    # 自动检测其他分类特征
    # 1. 字符串类型的列
    for col in df.columns:
        if col in categorical_features:
            continue
        
        if df[col].dtype == "object" or df[col].dtype.name == "category":
            unique_count = df[col].nunique()
            if min_unique_values <= unique_count <= max_unique_values:
                categorical_features.append(col)
                print(
                    f"   ✅ Auto-detected '{col}' as categorical ({unique_count} unique values)"
                )
    
    # 2. 整数类型但唯一值较少的列（可能是编码的分类特征）
    for col in df.columns:
        if col in categorical_features:
            continue
        
        if pd.api.types.is_integer_dtype(df[col]):
            unique_count = df[col].nunique()
            # 如果唯一值数量较少（< 50），且占总样本的比例较高（> 5%），可能是分类特征
            if (
                min_unique_values <= unique_count <= 50
                and (df[col].value_counts().min() / len(df)) > 0.05
            ):
                # 进一步检查：值是否连续（连续值更可能是数值特征）
                sorted_values = sorted(df[col].dropna().unique())
                is_continuous = all(
                    sorted_values[i + 1] - sorted_values[i] == 1
                    for i in range(len(sorted_values) - 1)
                )
                if not is_continuous:
                    categorical_features.append(col)
                    print(
                        f"   ✅ Auto-detected '{col}' as categorical (integer with {unique_count} unique values)"
                    )
    
    return categorical_features


def prepare_categorical_features_for_lightgbm(
    df: pd.DataFrame,
    feature_names: List[str],
    known_categorical: Optional[List[str]] = None,
) -> tuple[List[str], List[int]]:
    """
    为LightGBM准备分类特征
    
    Args:
        df: DataFrame with features
        feature_names: 特征名称列表（用于训练的特征）
        known_categorical: 已知的分类特征列表（如 ["_symbol"]）
    
    Returns:
        (categorical_feature_names, categorical_feature_indices)
        - categorical_feature_names: 分类特征名称列表
        - categorical_feature_indices: 分类特征在feature_names中的索引列表
    """
    # 检测分类特征
    categorical_features = detect_categorical_features(
        df, known_categorical=known_categorical or ["_symbol"]
    )
    
    # 过滤：只保留在feature_names中的分类特征
    categorical_feature_names = [
        f for f in categorical_features if f in feature_names
    ]
    
    # 获取索引
    categorical_feature_indices = [
        i for i, name in enumerate(feature_names) if name in categorical_feature_names
    ]
    
    if categorical_feature_names:
        print(
            f"   📊 Categorical features ({len(categorical_feature_names)}): {categorical_feature_names}"
        )
        print(f"      Indices: {categorical_feature_indices}")
    else:
        print("   📊 No categorical features detected")
    
    return categorical_feature_names, categorical_feature_indices


def validate_categorical_features(
    df: pd.DataFrame,
    categorical_features: List[str],
) -> bool:
    """
    验证分类特征是否有效
    
    Args:
        df: DataFrame with features
        categorical_features: 分类特征列表
    
    Returns:
        True if all categorical features are valid
    """
    valid = True
    
    for col in categorical_features:
        if col not in df.columns:
            print(f"   ❌ Categorical feature '{col}' not found in DataFrame")
            valid = False
            continue
        
        unique_count = df[col].nunique()
        missing_count = df[col].isna().sum()
        
        if unique_count < 2:
            print(
                f"   ⚠️  Categorical feature '{col}' has only {unique_count} unique value(s) (constant)"
            )
        elif missing_count > len(df) * 0.5:
            print(
                f"   ⚠️  Categorical feature '{col}' has {missing_count}/{len(df)} missing values (>50%)"
            )
        else:
            print(
                f"   ✅ Categorical feature '{col}': {unique_count} unique values, {missing_count} missing"
            )
    
    return valid

