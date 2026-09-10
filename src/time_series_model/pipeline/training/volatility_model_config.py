"""
波动率模型配置加载器和特征选择工具
"""

from __future__ import annotations

import re
import gc
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import yaml
import pandas as pd
import numpy as np

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


def load_volatility_model_config(
    config_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """
    加载波动率模型配置文件

    Args:
        config_path: 配置文件路径，如果为None，使用默认路径

    Returns:
        配置字典
    """
    if config_path is None:
        # 默认路径：项目根目录下的 config/volatility_model.yaml
        project_root = Path(__file__).resolve().parents[4]
        config_path = project_root / "config" / "volatility_model.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Volatility model config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    return config


def create_vpin_volatility_features(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    动态创建VPIN volatility衍生特征（如果不存在）

    根据参考文档，创建以下特征：
    - vpin_vol_ratio: vpin_volatility_10 / vpin_volatility_20
    - vpin_vol_zscore: VPIN volatility的Z-score
    - vpin_spike: VPIN volatility spike flag

    Args:
        df: 包含VPIN特征的DataFrame
        config: 波动率模型配置

    Returns:
        添加了衍生特征的DataFrame（如果原始特征存在）
    """
    if config is None:
        config = load_volatility_model_config()

    feature_engineering = config.get("feature_engineering", {})
    df = df.copy()

    # 检查基础VPIN volatility特征是否存在
    vpin_vol_10_col = "vpin_volatility_10"
    vpin_vol_20_col = "vpin_volatility_20"

    has_vpin_vol_10 = vpin_vol_10_col in df.columns
    has_vpin_vol_20 = vpin_vol_20_col in df.columns

    if not (has_vpin_vol_10 and has_vpin_vol_20):
        # 如果没有基础特征，无法创建衍生特征
        return df

    # 1. 创建 vpin_vol_ratio
    if feature_engineering.get("create_vpin_vol_ratio", True):
        ratio_params = feature_engineering.get("vpin_vol_ratio_params", {})
        epsilon = float(ratio_params.get("epsilon", 1e-8))

        if "vpin_vol_ratio" not in df.columns:
            df["vpin_vol_ratio"] = df[vpin_vol_10_col] / (df[vpin_vol_20_col] + epsilon)

    # 2. 创建 vpin_vol_zscore 和 vpin_spike
    if feature_engineering.get("create_vpin_vol_zscore", True):
        zscore_params = feature_engineering.get("vpin_vol_zscore_params", {})
        window = int(zscore_params.get("window", 50))
        threshold = float(zscore_params.get("threshold", 2.0))
        epsilon = (
            float(ratio_params.get("epsilon", 1e-8))
            if "ratio_params" in locals()
            else 1e-8
        )

        if "vpin_vol_zscore" not in df.columns:
            rolling_mean = (
                df[vpin_vol_10_col].rolling(window=window, min_periods=1).mean()
            )
            rolling_std = (
                df[vpin_vol_10_col].rolling(window=window, min_periods=1).std()
            )
            df["vpin_vol_zscore"] = (df[vpin_vol_10_col] - rolling_mean) / (
                rolling_std + epsilon
            )

        if "vpin_spike" not in df.columns:
            df["vpin_spike"] = (df["vpin_vol_zscore"] > threshold).astype(int)

    return df


def get_volatility_model_params(
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    获取波动率模型的训练参数

    Args:
        config: 波动率模型配置

    Returns:
        模型参数字典
    """
    if config is None:
        config = load_volatility_model_config()

    trainer_config = config.get("trainer", {})
    model_params = trainer_config.get("model_params", {})

    return model_params.copy()


def get_categorical_features(
    X: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> Optional[List[str]]:
    """
    获取分类特征列表

    Args:
        X: 特征DataFrame
        config: 波动率模型配置

    Returns:
        分类特征列表，如果不存在则返回None
    """
    if config is None:
        config = load_volatility_model_config()

    feature_selection = config.get("feature_selection", {})
    categorical_features = feature_selection.get("categorical_features", [])

    # 检查特征是否存在且有多值
    available_categorical = []
    for cat_col in categorical_features:
        if cat_col in X.columns and X[cat_col].nunique() > 1:
            available_categorical.append(cat_col)

    return available_categorical if available_categorical else None


def prepare_volatility_model_data(
    X: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    feature_loader: Optional[Any] = None,
    original_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, List[str], Optional[List[str]]]:
    """
    准备波动率模型训练数据：确保必需特征存在 + 特征选择 + 特征工程

    Args:
        X: 原始特征DataFrame（可能不包含所有波动率模型需要的特征）
        config: 波动率模型配置
        feature_loader: 特征加载器（可选，用于计算缺失的特征）
        original_df: 原始数据DataFrame（包含基础OHLCV列，可选）

    Returns:
        (处理后的特征DataFrame, 选中的特征列表, 分类特征列表)
    """
    if config is None:
        config = load_volatility_model_config()

    X_processed = X.copy()
    feature_config = config.get("volatility_features", {})
    feature_groups = feature_config.get("groups", [])
    selected_columns: List[str] = []

    # 存储原始输入以便后续访问基础列
    # 优先使用传入的 original_df，否则使用 X
    original_X = original_df.copy() if original_df is not None else X.copy()

    # ✅ 确保基础 OHLCV 列存在于 X_processed 中（特征计算需要这些列）
    base_columns = ["open", "high", "low", "close", "volume"]
    for col in base_columns:
        if col not in X_processed.columns and col in original_X.columns:
            # 确保索引对齐
            if len(X_processed) > 0:
                try:
                    X_processed[col] = original_X[col].reindex(X_processed.index)
                except Exception:
                    if len(original_X) == len(X_processed):
                        X_processed[col] = original_X[col].values
            else:
                X_processed[col] = original_X[col]

    def _compute_feature(feature_name: str) -> None:
        nonlocal X_processed
        if not feature_loader or not feature_name:
            return
        try:
            # ✅ 首先确保基础 OHLCV 列存在于 X_processed 中
            # 这些列是特征计算的基础，即使不在 required_columns 中也需要
            base_columns = ["open", "high", "low", "close", "volume"]
            for col in base_columns:
                if col not in X_processed.columns and col in original_X.columns:
                    # 确保索引对齐
                    if len(X_processed) > 0:
                        try:
                            X_processed[col] = original_X[col].reindex(
                                X_processed.index
                            )
                        except Exception:
                            if len(original_X) == len(X_processed):
                                X_processed[col] = original_X[col].values
                    else:
                        X_processed[col] = original_X[col]

            # 确保 required_columns 存在于 DataFrame 中
            # 从 feature_dependencies.yaml 中获取 required_columns
            features_cfg = feature_loader.feature_deps.get("features", {})
            if feature_name in features_cfg:
                feature_info = features_cfg[feature_name]
                required_columns = feature_info.get("required_columns", [])
                if required_columns:
                    missing_cols = [
                        col
                        for col in required_columns
                        if col not in X_processed.columns
                    ]
                    if missing_cols:
                        # 尝试从原始输入 X 中获取缺失的列
                        for col in missing_cols:
                            if col in original_X.columns:
                                # 确保索引对齐
                                if len(X_processed) > 0:
                                    # 使用 reindex 确保索引对齐，如果索引不匹配则使用 NaN
                                    try:
                                        X_processed[col] = original_X[col].reindex(
                                            X_processed.index
                                        )
                                    except Exception:
                                        # 如果 reindex 失败，尝试直接赋值（假设索引相同）
                                        if len(original_X) == len(X_processed):
                                            X_processed[col] = original_X[col].values
                                        else:
                                            print(
                                                f"   ⚠️  Warning: Cannot align column '{col}' - index mismatch"
                                            )
                                else:
                                    X_processed[col] = original_X[col]
                            else:
                                print(
                                    f"   ⚠️  Warning: Required column '{col}' not found in input data for {feature_name}"
                                )
                                print(
                                    f"      Available columns in original_X: {list(original_X.columns)[:20]}..."
                                )
                                print(
                                    f"      Available columns in X_processed: {list(X_processed.columns)[:20]}..."
                                )

            # 确保 feature_loader 的配置（如 tick_loader_json）被保留
            # 注意：feature_loader 应该是从外部传入的，已经配置好的实例
            # 在调用 load_features_from_requested 之前，确保所有 required_columns 都在 X_processed 中
            # 这样 load_features_from_requested 中的检查也能找到它们
            result_df = feature_loader.load_features_from_requested(
                X_processed,
                requested_features=[feature_name],
                fit=True,
            )
            # ✅ 确保新计算的列被添加到 X_processed 中
            # 使用 merge 或直接赋值来添加新列
            new_cols = [
                col for col in result_df.columns if col not in X_processed.columns
            ]
            if new_cols:
                # 确保索引对齐
                for col in new_cols:
                    if col in result_df.columns:
                        X_processed[col] = result_df[col].reindex(X_processed.index)
            # 如果 result_df 有更多列，也更新现有列（以防计算过程中值发生变化）
            common_cols = [
                col for col in result_df.columns if col in X_processed.columns
            ]
            if common_cols:
                for col in common_cols:
                    X_processed[col] = result_df[col].reindex(X_processed.index)

            print(
                f"   ✅ Computed missing feature: {feature_name}"
                + (f" (added columns: {new_cols})" if new_cols else "")
            )

            # 清理内存（每个特征计算后）
            gc.collect()
            if PSUTIL_AVAILABLE:
                try:
                    process = psutil.Process()
                    mem_mb = process.memory_info().rss / (1024**2)
                    if mem_mb > 10000:  # 如果内存使用超过 10GB，打印警告
                        print(
                            f"   ⚠️  Memory usage after {feature_name}: {mem_mb/1024:.2f}GB"
                        )
                except Exception:
                    pass
        except Exception as exc:
            print(f"   ⚠️ Failed to compute feature '{feature_name}': {exc}")
            import traceback

            print(f"   Traceback: {traceback.format_exc()}")

    for group_idx, group in enumerate(feature_groups):
        feature_name = group.get("feature_name")
        required = bool(group.get("required", False))
        columns = group.get("columns", [])
        if not columns:
            continue

        missing_cols = [col for col in columns if col not in X_processed.columns]
        # 只有当 feature_name 存在且不为 None 时才尝试计算特征
        if missing_cols and feature_name:
            _compute_feature(feature_name)
            missing_cols = [col for col in columns if col not in X_processed.columns]

            # 每计算 1 个特征组后清理一次内存（更频繁的清理，防止 OOM）
            gc.collect()
            if PSUTIL_AVAILABLE:
                try:
                    process = psutil.Process()
                    mem_gb = process.memory_info().rss / (1024**3)
                    if (group_idx + 1) % 3 == 0:  # 每3个特征组打印一次
                        print(
                            f"   📊 Memory after {group_idx + 1} feature groups: {mem_gb:.2f}GB"
                        )
                    # 如果内存使用超过 10GB，打印警告
                    if mem_gb > 10:
                        print(
                            f"   ⚠️  High memory usage after {group_idx + 1} feature groups: {mem_gb:.2f}GB"
                        )
                except Exception:
                    pass
        elif missing_cols and not feature_name:
            # feature_name 为 null，说明这些列可能来自多个特征或通过其他方式生成
            # 尝试根据列名推断并计算相关特征

            # 检查是否是 ATR 相关列
            if "atr" in missing_cols and "atr_ratio" in missing_cols:
                # 尝试计算 atr 和 atr_ratio
                print(
                    f"   🔧 Attempting to compute atr and atr_ratio features for missing columns..."
                )
                try:
                    _compute_feature("atr")
                    _compute_feature("atr_ratio")  # atr_ratio 依赖于 atr，会自动处理
                    missing_cols = [
                        col for col in columns if col not in X_processed.columns
                    ]
                    if missing_cols:
                        print(
                            f"   ✅ Computed atr/atr_ratio, {len(missing_cols)} columns still missing"
                        )
                    else:
                        print(f"   ✅ All columns computed successfully")
                except Exception as e:
                    print(f"   ⚠️  Failed to compute atr/atr_ratio: {e}")

            # 如果还有缺失列，提供提示信息
            if missing_cols:
                missing_cols_str = ", ".join(missing_cols[:5])
                if len(missing_cols) > 5:
                    missing_cols_str += f", ... (and {len(missing_cols) - 5} more)"

                # 根据列名推断可能来自的特征（用于提示）
                possible_features = []
                if any("wpt_price" in col for col in missing_cols):
                    possible_features.append(
                        "wpt_price_reconstructed or wpt_volatility_features"
                    )
                if any("wpt_volume" in col for col in missing_cols):
                    possible_features.append("wpt_volatility_features (if available)")
                if any("wpt_cvd" in col for col in missing_cols):
                    possible_features.append("wpt_volatility_features (if available)")
                if any("atr" in col for col in missing_cols):
                    possible_features.append("atr or atr_ratio")

                feature_hint = ""
                if possible_features:
                    feature_hint = f" (may come from: {', '.join(possible_features)})"

                print(
                    f"   ℹ️  Feature group '{group.get('name')}' has {len(missing_cols)} missing columns{feature_hint}"
                )
                if len(missing_cols) <= 10:
                    print(f"      Missing: {missing_cols_str}")
                # 非 WPT 相关列，只显示警告
                missing_cols_str = ", ".join(missing_cols[:5])
                if len(missing_cols) > 5:
                    missing_cols_str += f", ... (and {len(missing_cols) - 5} more)"

                possible_features = []
                if any("wpt_price" in col for col in missing_cols):
                    possible_features.append(
                        "wpt_price_reconstructed or wpt_price_fluctuation"
                    )
                if any("wpt_volume" in col for col in missing_cols):
                    possible_features.append("wpt_volume_features (if available)")
                if any("wpt_cvd" in col for col in missing_cols):
                    possible_features.append("wpt_cvd_features (if available)")

                feature_hint = ""
                if possible_features:
                    feature_hint = f" (may come from: {', '.join(possible_features)})"

                print(
                    f"   ℹ️  Feature group '{group.get('name')}' has {len(missing_cols)} missing columns{feature_hint}"
                )
                if len(missing_cols) <= 10:
                    print(f"      Missing: {missing_cols_str}")

        if missing_cols and required:
            print(
                f"   ⚠️ Required feature group '{group.get('name')}' missing columns: {missing_cols}"
            )

        existing_cols = [col for col in columns if col in X_processed.columns]
        if existing_cols:
            selected_columns.extend(existing_cols)

    # 创建VPIN衍生特征
    X_processed = create_vpin_volatility_features(X_processed, config)

    # 增强WPT特征（如果存在WPT特征）
    # 注意：增强的特征应该在配置文件的 wpt_volatility 组中列出
    from src.features.time_series.utils_volatility_features import (
        enhance_wpt_vol_features,
    )

    wpt_cols = [col for col in X_processed.columns if col.startswith("wpt_")]
    if wpt_cols:
        X_processed = enhance_wpt_vol_features(X_processed)
        # 清理内存（WPT 特征增强后）
        gc.collect()

    # 注意：Volume Profile 波动率特征现在通过 yaml 配置统一管理
    # 如果配置中定义了 volume_profile_volatility_features，会在上面的循环中通过 feature_loader 计算
    # 不再需要这里的手动计算，避免重复计算

    # 去重并保持顺序
    seen = set()
    ordered_columns = []
    for col in selected_columns:
        if col in X_processed.columns and col not in seen:
            ordered_columns.append(col)
            seen.add(col)

    if not ordered_columns:
        ordered_columns = [
            col
            for col in X_processed.columns
            if col not in {"open", "high", "low", "close", "volume", "signal", "label"}
        ]

    categorical_features = get_categorical_features(
        X_processed[ordered_columns], config
    )

    X_vol = X_processed[ordered_columns].copy()

    return X_vol, ordered_columns, categorical_features
