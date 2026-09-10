"""
数据监控工具：检测和报告 inf/NaN 异常值

用于追踪 inf 值的来源，区分是源数据错误还是计算错误
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path


def check_data_quality(
    df: pd.DataFrame,
    data_source: str,
    stage: str,
    raise_on_inf: bool = False,
    raise_on_nan: bool = False,
) -> Dict[str, any]:
    """
    检查数据质量，检测 inf/NaN 异常值
    
    Args:
        df: 要检查的 DataFrame
        data_source: 数据来源描述（如 "raw_data", "feature_calculation"）
        stage: 检查阶段（如 "after_load", "after_feature_calc"）
        raise_on_inf: 如果发现 inf 是否抛出异常
        raise_on_nan: 如果发现 NaN 是否抛出异常（通常 False，因为 NaN 可能是正常的）
    
    Returns:
        包含检查结果的字典
    """
    if df.empty:
        return {
            "has_inf": False,
            "has_nan": False,
            "inf_columns": [],
            "nan_columns": [],
            "inf_details": {},
            "nan_details": {},
        }
    
    result = {
        "has_inf": False,
        "has_nan": False,
        "inf_columns": [],
        "nan_columns": [],
        "inf_details": {},
        "nan_details": {},
    }
    
    # 检查 inf 值（严格区分 inf 和 NaN，避免误报）
    for col in df.columns:
        if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            inf_mask = np.isinf(df[col])
            nan_mask = df[col].isna()
            if inf_mask.any():
                result["has_inf"] = True
                result["inf_columns"].append(col)
                
                # 收集详细信息
                inf_indices = df.index[inf_mask]
                inf_values = df.loc[inf_indices, col]
                
                result["inf_details"][col] = {
                    "count": int(inf_mask.sum()),
                    "percentage": float(100.0 * inf_mask.sum() / len(df)),
                    "indices": inf_indices.tolist()[:10],  # 只保存前10个
                    "values": inf_values.tolist()[:10],
                    "min": float(df[col].min()) if not np.isinf(df[col].min()) else None,
                    "max": float(df[col].max()) if not np.isinf(df[col].max()) else None,
                    "mean": float(df[col].mean()) if np.isfinite(df[col].mean()) else None,
                }
                
                # 打印详细信息
                print(f"\n   ⚠️  [DATA MONITOR] {data_source} @ {stage}")
                print(f"      Column '{col}' contains {inf_mask.sum()} inf values ({100.0 * inf_mask.sum() / len(df):.2f}%)")
                print(f"      First few inf indices: {inf_indices[:5].tolist()}")
                print(f"      First few inf values: {inf_values[:5].tolist()}")
                
                # 打印上下文数据（前后各3行）
                if len(inf_indices) > 0:
                    first_inf_idx = inf_indices[0]
                    try:
                        idx_pos = df.index.get_loc(first_inf_idx)
                        context_start = max(0, idx_pos - 3)
                        context_end = min(len(df), idx_pos + 4)
                        context_df = df.iloc[context_start:context_end]
                        print(f"      Context around first inf (row {idx_pos}):")
                        print(f"         {context_df[[col]].to_string()}")
                        
                        # 检查是否是源数据问题（检查原始列）
                        if col in ["open", "high", "low", "close", "volume"]:
                            print(f"      ⚠️  WARNING: Inf found in source column '{col}'!")
                            print(f"         This suggests a problem with the raw data, not feature calculation.")
                            # 检查相邻列
                            for other_col in ["open", "high", "low", "close", "volume"]:
                                if other_col != col and other_col in df.columns:
                                    other_val = df.loc[first_inf_idx, other_col]
                                    if not np.isfinite(other_val):
                                        print(f"         Adjacent column '{other_col}' also has inf: {other_val}")
                                    else:
                                        print(f"         Adjacent column '{other_col}' is OK: {other_val}")
                    except Exception as e:
                        print(f"      Could not get context: {e}")
                
                if raise_on_inf:
                    raise ValueError(f"Inf values found in column '{col}' at stage '{stage}'")
            # NaN 单独报告（仅当 NaN 比例较高时警告，不算 inf）
            elif nan_mask.any():
                nan_count = int(nan_mask.sum())
                nan_pct = 100.0 * nan_count / len(df)
                if nan_pct > 50:  # 超过 50% NaN 才报警
                    print(f"\n   ⚠️  [DATA MONITOR] {data_source} @ {stage}")
                    print(f"      Column '{col}' contains {nan_count} NaN values ({nan_pct:.2f}%) — not inf")
    
    # 检查 NaN 值（仅报告，不抛出异常，因为 NaN 可能是正常的）
    for col in df.columns:
        if df[col].dtype in [np.float64, np.float32]:
            nan_mask = df[col].isna()
            if nan_mask.any():
                result["has_nan"] = True
                if col not in result["nan_columns"]:
                    result["nan_columns"].append(col)
                
                nan_count = int(nan_mask.sum())
                nan_pct = float(100.0 * nan_count / len(df))
                
                if col not in result["nan_details"]:
                    result["nan_details"][col] = {
                        "count": nan_count,
                        "percentage": nan_pct,
                    }
    
    return result


def monitor_feature_calculation(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    feature_name: str,
    stage: str = "feature_calculation",
) -> Dict[str, any]:
    """
    监控特征计算前后的数据变化，检测新产生的 inf 值
    
    Args:
        df_before: 计算前的 DataFrame
        df_after: 计算后的 DataFrame
        feature_name: 特征名称
        stage: 计算阶段
    
    Returns:
        包含监控结果的字典
    """
    result = {
        "new_inf_columns": [],
        "new_inf_details": {},
    }
    
    # 检查新产生的 inf 值
    for col in df_after.columns:
        if col not in df_before.columns:
            # 新列，检查是否有 inf
            inf_mask = ~np.isfinite(df_after[col])
            if inf_mask.any():
                result["new_inf_columns"].append(col)
                result["new_inf_details"][col] = {
                    "count": int(inf_mask.sum()),
                    "percentage": float(100.0 * inf_mask.sum() / len(df_after)),
                }
                print(f"\n   ⚠️  [FEATURE MONITOR] {feature_name} @ {stage}")
                print(f"      New column '{col}' contains {inf_mask.sum()} inf values")
        else:
            # 现有列，检查是否有新的 inf 值
            before_inf = ~np.isfinite(df_before[col])
            after_inf = ~np.isfinite(df_after[col])
            new_inf = after_inf & ~before_inf
            if new_inf.any():
                result["new_inf_columns"].append(col)
                result["new_inf_details"][col] = {
                    "count": int(new_inf.sum()),
                    "percentage": float(100.0 * new_inf.sum() / len(df_after)),
                }
                print(f"\n   ⚠️  [FEATURE MONITOR] {feature_name} @ {stage}")
                print(f"      Column '{col}' gained {new_inf.sum()} new inf values")
    
    return result


def check_source_data_quality(df: pd.DataFrame, data_path: str) -> Dict[str, any]:
    """
    检查源数据质量（在数据加载后立即调用）
    
    Args:
        df: 加载的原始数据
        data_path: 数据路径（用于日志）
    
    Returns:
        检查结果
    """
    print(f"\n   🔍 [DATA MONITOR] Checking source data quality...")
    print(f"      Data path: {data_path}")
    print(f"      Shape: {df.shape}")
    print(f"      Columns: {list(df.columns)}")
    
    result = check_data_quality(
        df,
        data_source="SOURCE_DATA",
        stage="after_load",
        raise_on_inf=False,  # 不抛出异常，只报告
        raise_on_nan=False,
    )
    
    if result["has_inf"]:
        print(f"\n   ❌ [DATA MONITOR] Source data contains inf values!")
        print(f"      This is a CRITICAL issue - source data should not contain inf.")
        print(f"      Please check the data loading process and source files.")
    else:
        print(f"   ✅ [DATA MONITOR] Source data quality check passed (no inf values)")
    
    return result

