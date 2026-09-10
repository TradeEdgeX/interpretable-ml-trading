"""
特征归一化工具函数（公共模块）

用于多资产训练的特征归一化，确保不同价格水平的资产可以一起训练。

核心特性：
- 支持滚动窗口归一化（防止未来信息泄露）✅ 推荐用于时序数据
- 显式处理NaN值
- 提供robust min-max选项（基于分位数，对异常值鲁棒）
- 性能优化（向量化操作）

⚠️ 重要警告：全局归一化（window=None）在时序数据中的问题

全局归一化使用整个数据集计算统计量（mean/std），在时序场景下会导致：

1. **未来信息泄露（Look-ahead Bias）**：
   - 训练时：使用全样本（包括测试集）计算统计量 → 模型"看到"未来数据
   - 测试时：使用全样本统计量 → 实际部署时无法使用（需要未来数据）

2. **性能虚高**：
   - 模型在训练/验证集上表现好，但实际部署时性能大幅下降

3. **无法在线部署**：
   - 在线预测时无法获取未来数据的统计量

✅ 解决方案：使用滚动归一化（window参数）

- 日频数据：window=252（1年）
- 周频数据：window=52（1年）  
- 小时数据：window=168（1周）

滚动归一化只使用历史窗口数据计算统计量，确保：
- 训练时只使用历史数据
- 测试时只使用历史数据
- 在线部署时可以使用

📝 全局归一化的适用场景（仅限）：
- EDA（探索性数据分析）
- 非时序场景（横截面数据）
- 严格按时间分割的训练/测试集，且只在训练集上计算统计量
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, List, Dict, Literal, Tuple
import pickle
import warnings


def normalize_series(
    x: Union[np.ndarray, pd.Series],
    method: str = "zscore",
    robust: bool = False,
) -> np.ndarray:
    """
    单序列归一化，显式处理NaN值
    
    Args:
        x: 输入序列（numpy array 或 pandas Series）
        method: 归一化方法 ("zscore", "minmax", "robust_minmax")
        robust: 是否使用robust模式（仅对minmax有效，基于分位数）
    
    Returns:
        归一化后的序列（numpy array）
    
    Note:
        - NaN值会被保留在输出中（或填充为0，取决于后续处理）
        - 对于常数序列（std=0或max=min），返回全零
    """
    x = np.array(x, dtype=float)
    mask = np.isfinite(x)  # 过滤NaN和Inf
    
    if not np.any(mask):
        # 全部为NaN或Inf，返回全零
        return np.zeros_like(x)
    
    if method == "zscore":
        mean = np.mean(x[mask])
        std = np.std(x[mask])
        if std < 1e-8:
            result = np.zeros_like(x)
        else:
            result = (x - mean) / (std + 1e-8)
        result[~mask] = np.nan  # 保留原始NaN位置
        return result
    
    elif method in ["minmax", "robust_minmax"]:
        if robust or method == "robust_minmax":
            # 基于分位数的robust min-max（对异常值鲁棒）
            q_low = np.percentile(x[mask], 1)
            q_high = np.percentile(x[mask], 99)
            min_val = q_low
            max_val = q_high
        else:
            min_val = np.min(x[mask])
            max_val = np.max(x[mask])
        
        if max_val - min_val < 1e-8:
            result = np.zeros_like(x)
        else:
            result = (x - min_val) / (max_val - min_val + 1e-8)
            result = np.clip(result, 0, 1)  # minmax默认裁剪到[0,1]
        
        result[~mask] = np.nan
        return result
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'zscore', 'minmax', or 'robust_minmax'")


def normalize_by_group(
    df: pd.DataFrame,
    value_col: str,
    group_col: str = "_symbol",
    method: str = "zscore",
    window: Optional[int] = None,
    clip: bool = True,
    fillna: bool = True,
    warn_global: bool = True,
) -> pd.Series:
    """
    按组归一化（用于多资产数据），支持滚动窗口归一化
    
    Args:
        df: DataFrame with data
        value_col: Column to normalize
        group_col: Group column (e.g., "_symbol" for multi-asset)
        method: Normalization method ("zscore", "minmax", "robust_minmax")
        window: Rolling window size (None = global normalization)
                ⚠️ WARNING: 全局归一化（window=None）在时序数据中会导致未来信息泄露！
                - 训练时：使用全样本统计量（包括测试集）→ 引入未来信息
                - 测试时：使用全样本统计量 → 实际部署时无法使用
                ✅ 强烈建议使用滚动归一化：
                - 日频数据：window=252（1年）
                - 周频数据：window=52（1年）
                - 小时数据：window=168（1周）
        clip: Whether to clip after normalization
              - zscore: clip to [-3, 3]
              - minmax/robust_minmax: clip to [0, 1] (already done in normalize_series)
        fillna: Whether to fill NaN values with 0 after normalization
        warn_global: Whether to warn when using global normalization (window=None)
    
    Returns:
        Normalized Series with same index as df
    
    Note:
        - 滚动归一化是时序数据的标准做法，防止未来信息泄露
        - 前 window-1 行会因窗口不足而可能为NaN（滚动场景）
        - 全局归一化仅适用于：
          * EDA（探索性数据分析）
          * 非时序场景（如横截面数据）
          * 严格按时间分割的训练/测试集，且只在训练集上计算统计量
    """
    import warnings
    
    if value_col not in df.columns:
        raise ValueError(f"Column '{value_col}' not found in DataFrame")
    
    # 警告：全局归一化在时序数据中的风险
    if window is None and warn_global:
        warnings.warn(
            "⚠️ 使用全局归一化（window=None）可能导致未来信息泄露！\n"
            "在时序数据中，全局归一化会使用整个数据集（包括未来数据）计算统计量，\n"
            "这会导致：\n"
            "1. 训练时引入测试集信息 → 模型性能虚高\n"
            "2. 测试/部署时无法使用（需要未来数据）\n"
            "✅ 建议使用滚动归一化：window=252（日频）或 window=52（周频）\n"
            "如果确实需要全局归一化（如EDA），请设置 warn_global=False 以关闭此警告。",
            UserWarning,
            stacklevel=2
        )
    
    # 检查是否有分组列
    has_group = group_col in df.columns
    
    # 滚动归一化
    if window is not None and window > 0:
        if has_group:
            # 多资产滚动归一化
            grouped = df.groupby(group_col)[value_col]
            
            if method == "zscore":
                rolling_mean = grouped.rolling(window=window, min_periods=1).mean()
                rolling_std = grouped.rolling(window=window, min_periods=1).std()
                # 处理分组后的索引
                rolling_mean = rolling_mean.reset_index(level=0, drop=True)
                rolling_std = rolling_std.reset_index(level=0, drop=True)
                normalized = (df[value_col] - rolling_mean) / (rolling_std + 1e-8)
                
            elif method in ["minmax", "robust_minmax"]:
                if method == "robust_minmax":
                    # 滚动分位数计算（使用quantile更高效）
                    rolling_q_low = grouped.rolling(window=window, min_periods=1).quantile(0.01)
                    rolling_q_high = grouped.rolling(window=window, min_periods=1).quantile(0.99)
                    rolling_q_low = rolling_q_low.reset_index(level=0, drop=True)
                    rolling_q_high = rolling_q_high.reset_index(level=0, drop=True)
                    diff = rolling_q_high - rolling_q_low
                    normalized = (df[value_col] - rolling_q_low) / (diff + 1e-8)
                    normalized = np.clip(normalized, 0, 1)
                    # 处理常数序列
                    normalized = normalized.where(diff >= 1e-8, 0.0)
                else:
                    # 标准minmax滚动
                    rolling_min = grouped.rolling(window=window, min_periods=1).min()
                    rolling_max = grouped.rolling(window=window, min_periods=1).max()
                    rolling_min = rolling_min.reset_index(level=0, drop=True)
                    rolling_max = rolling_max.reset_index(level=0, drop=True)
                    normalized = (df[value_col] - rolling_min) / (rolling_max - rolling_min + 1e-8)
                    normalized = np.clip(normalized, 0, 1)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # 前window-1行可能为NaN（窗口不足），填充为0
            if fillna:
                normalized = normalized.fillna(0.0)
        else:
            # 单资产滚动归一化
            if method == "zscore":
                rolling_mean = df[value_col].rolling(window=window, min_periods=1).mean()
                rolling_std = df[value_col].rolling(window=window, min_periods=1).std()
                normalized = (df[value_col] - rolling_mean) / (rolling_std + 1e-8)
            elif method in ["minmax", "robust_minmax"]:
                if method == "robust_minmax":
                    # 使用quantile更高效
                    rolling_q_low = df[value_col].rolling(window=window, min_periods=1).quantile(0.01)
                    rolling_q_high = df[value_col].rolling(window=window, min_periods=1).quantile(0.99)
                    diff = rolling_q_high - rolling_q_low
                    normalized = (df[value_col] - rolling_q_low) / (diff + 1e-8)
                    normalized = np.clip(normalized, 0, 1)
                    # 处理常数序列
                    normalized = normalized.where(diff >= 1e-8, 0.0)
                else:
                    rolling_min = df[value_col].rolling(window=window, min_periods=1).min()
                    rolling_max = df[value_col].rolling(window=window, min_periods=1).max()
                    normalized = (df[value_col] - rolling_min) / (rolling_max - rolling_min + 1e-8)
                    normalized = np.clip(normalized, 0, 1)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            if fillna:
                normalized = normalized.fillna(0.0)
    
    else:
        # 全局归一化（非滚动）
        if not has_group:
            # 单资产全局归一化
            normalized_values = normalize_series(
                df[value_col].values,
                method=method,
                robust=(method == "robust_minmax")
            )
            normalized = pd.Series(normalized_values, index=df.index)
        else:
            # 多资产全局归一化（使用向量化优化）
            if method == "zscore":
                normalized = df.groupby(group_col)[value_col].transform(
                    lambda x: normalize_series(x.values, method="zscore")
                )
            elif method == "minmax":
                normalized = df.groupby(group_col)[value_col].transform(
                    lambda x: normalize_series(x.values, method="minmax", robust=False)
                )
            elif method == "robust_minmax":
                normalized = df.groupby(group_col)[value_col].transform(
                    lambda x: normalize_series(x.values, method="robust_minmax", robust=True)
                )
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # 确保索引对齐
            normalized = pd.Series(normalized.values, index=df.index)
        
        # 填充NaN
        if fillna:
            normalized = normalized.fillna(0.0)
    
    # 裁剪（zscore裁剪到[-3,3]，minmax已在normalize_series中裁剪到[0,1]）
    if clip:
        if method == "zscore":
            normalized = normalized.clip(-3, 3)
        # minmax已经在normalize_series中裁剪，无需再次裁剪
    
    return normalized


def normalize_dataframe(
    df: pd.DataFrame,
    value_cols: List[str],
    group_col: str = "_symbol",
    method: str = "zscore",
    window: Optional[int] = None,
    clip: bool = True,
    fillna: bool = True,
    suffix: Optional[str] = None,
    warn_global: bool = True,
) -> pd.DataFrame:
    """
    批量归一化DataFrame的多个列
    
    Args:
        df: DataFrame with data
        value_cols: List of columns to normalize
        group_col: Group column (e.g., "_symbol" for multi-asset)
        method: Normalization method ("zscore", "minmax", "robust_minmax")
        window: Rolling window size (None = global normalization)
                ⚠️ WARNING: 全局归一化在时序数据中会导致未来信息泄露！
                强烈建议使用滚动归一化：window=252（日频）或 window=52（周频）
        clip: Whether to clip after normalization
        fillna: Whether to fill NaN values with 0 after normalization
        suffix: Suffix to add to normalized column names (e.g., "_zscore")
                If None, original columns are replaced
        warn_global: Whether to warn when using global normalization (window=None)
    
    Returns:
        DataFrame with normalized columns
    
    Example:
        >>> # ✅ 推荐：使用滚动归一化
        >>> df_norm = normalize_dataframe(
        ...     df, 
        ...     value_cols=["feature1", "feature2"],
        ...     method="zscore",
        ...     window=252,  # 滚动窗口，防止未来信息泄露
        ...     suffix="_zscore"
        ... )
        >>> # Creates: df["feature1_zscore"], df["feature2_zscore"]
        
        >>> # ⚠️ 不推荐：全局归一化（仅用于EDA或非时序场景）
        >>> df_norm = normalize_dataframe(
        ...     df,
        ...     value_cols=["feature1", "feature2"],
        ...     window=None,  # 全局归一化，会有警告
        ...     warn_global=False  # 关闭警告（如果确实需要）
        ... )
    """
    df_result = df.copy()
    
    for col in value_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
        
        normalized = normalize_by_group(
            df=df,
            value_col=col,
            group_col=group_col,
            method=method,
            window=window,
            clip=clip,
            fillna=fillna,
            warn_global=warn_global,
        )
        
        if suffix:
            new_col = f"{col}{suffix}"
        else:
            new_col = col
        
        df_result[new_col] = normalized
    
    return df_result


# ============================================================================
# UnifiedNormalizer 类 - 统一的归一化接口（基于 normalize_by_group）
# ============================================================================

class UnifiedNormalizer:
    """
    统一的归一化器（基于 normalize_by_group 实现）
    
    提供 fit/transform 接口，支持特征自动分组和多时间框架归一化。
    底层使用 normalize_by_group 实现，确保滚动归一化防止未来信息泄露。
    
    Example:
        >>> # 创建归一化器（默认使用滚动归一化）
        >>> normalizer = UnifiedNormalizer(
        ...     method="zscore",
        ...     window=252,  # 滚动窗口
        ...     group_col="_symbol"  # 多资产分组
        ... )
        >>> 
        >>> # 拟合（训练时）
        >>> normalizer.fit(train_df)
        >>> 
        >>> # 转换（测试时）
        >>> test_normalized = normalizer.transform(test_df)
        >>> 
        >>> # 或者一次性完成
        >>> train_normalized = normalizer.fit_transform(train_df)
    """
    
    def __init__(
        self,
        method: str = "zscore",
        window: Optional[int] = 252,  # 默认滚动归一化
        group_col: str = "_symbol",
        clip: bool = True,
        fillna: bool = True,
        feature_groups: Optional[Dict[str, List[str]]] = None,
        warn_global: bool = True,
    ):
        """
        Args:
            method: 归一化方法 ("zscore", "minmax", "robust_minmax")
            window: 滚动窗口大小（None = 全局归一化，不推荐）
                   推荐值：252（日频），52（周频），168（小时）
            group_col: 分组列（用于多资产归一化）
            clip: 是否裁剪异常值
            fillna: 是否填充NaN
            feature_groups: 特征分组字典（如果为None，会自动分组）
            warn_global: 是否在全局归一化时警告
        """
        self.method = method
        self.window = window
        self.group_col = group_col
        self.clip = clip
        self.fillna = fillna
        self.feature_groups = feature_groups
        self.warn_global = warn_global
        
        # 状态
        self.is_fitted = False
        self.fitted_features: List[str] = []
        
        # 警告：如果使用全局归一化
        if window is None and warn_global:
            warnings.warn(
                "⚠️ UnifiedNormalizer 使用全局归一化（window=None）可能导致未来信息泄露！\n"
                "强烈建议使用滚动归一化：window=252（日频）或 window=52（周频）",
                UserWarning,
                stacklevel=2
            )
    
    def _group_features(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """自动分组特征"""
        if self.feature_groups:
            return self.feature_groups
        
        # 自动分组逻辑
        feature_cols = [
            col
            for col in data.columns
            if col not in ["open", "high", "low", "close", "volume", self.group_col]
        ]
        
        # 定义特征分组
        price_features = [
            col
            for col in feature_cols
            if any(x in col.lower() for x in ["sma", "ema", "wma", "tema", "kama", "sar"])
        ]
        ratio_features = [
            col
            for col in feature_cols
            if any(x in col.lower() for x in ["ratio", "position", "normalized"])
        ]
        volatility_features = [
            col
            for col in feature_cols
            if any(x in col.lower() for x in ["atr", "natr", "trange", "bb_", "volatility"])
        ]
        volume_features = [
            col
            for col in feature_cols
            if any(x in col.lower() for x in ["volume", "obv", "ad", "vpt", "cmf"])
        ]
        momentum_features = [
            col
            for col in feature_cols
            if any(x in col.lower() for x in ["rsi", "stoch", "willr", "mom", "roc", "cci", "ultosc", "tsi"])
        ]
        index_features = [
            col
            for col in feature_cols
            if any(x in col.lower() for x in ["maxindex", "minindex", "max", "min"])
        ]
        
        # 其他特征
        other_features = [
            col
            for col in feature_cols
            if col not in price_features + ratio_features + volatility_features 
                + volume_features + momentum_features + index_features
        ]
        
        groups = {
            "price": price_features,
            "ratio": ratio_features,
            "volatility": volatility_features,
            "volume": volume_features,
            "momentum": momentum_features,
            "index": index_features,
            "other": other_features,
        }
        
        return groups
    
    def fit(self, data: pd.DataFrame) -> "UnifiedNormalizer":
        """
        拟合归一化器（记录特征列表）
        
        注意：由于使用滚动归一化，不需要预先计算统计量。
        此方法主要用于记录特征列表，确保 transform 时使用相同的特征。
        """
        # 获取特征分组
        feature_groups = self._group_features(data)
        
        # 收集所有特征
        all_features = []
        for group_features in feature_groups.values():
            all_features.extend(group_features)
        
        # 去重并排序
        self.fitted_features = sorted(list(set(all_features)))
        
        # 验证特征是否存在
        missing_features = [f for f in self.fitted_features if f not in data.columns]
        if missing_features:
            warnings.warn(
                f"以下特征在数据中不存在，将被忽略: {missing_features}",
                UserWarning
            )
            self.fitted_features = [f for f in self.fitted_features if f in data.columns]
        
        self.is_fitted = True
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """应用归一化"""
        if not self.is_fitted:
            raise ValueError("Must fit the normalizer first. Call fit() before transform().")
        
        df = data.copy()
        
        # 获取特征分组
        feature_groups = self._group_features(df)
        
        # 为每个特征组应用归一化
        for group_name, group_features in feature_groups.items():
            if not group_features:
                continue
            
            # 只归一化存在的特征
            existing_features = [f for f in group_features if f in df.columns]
            if not existing_features:
                continue
            
            # 使用 normalize_dataframe 批量归一化
            # 如果 group_col 不在数据中，使用 None（单资产模式）
            group_col_to_use = self.group_col if self.group_col in df.columns else None
            
            df = normalize_dataframe(
                df=df,
                value_cols=existing_features,
                group_col=group_col_to_use,
                method=self.method,
                window=self.window,
                clip=self.clip,
                fillna=self.fillna,
                suffix=None,  # 直接替换原列
                warn_global=False,  # 已在 __init__ 中警告
            )
        
        return df
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """拟合并应用归一化"""
        return self.fit(data).transform(data)
    
    def save(self, filepath: str):
        """保存归一化器配置"""
        normalizer_data = {
            "method": self.method,
            "window": self.window,
            "group_col": self.group_col,
            "clip": self.clip,
            "fillna": self.fillna,
            "feature_groups": self.feature_groups,
            "fitted_features": self.fitted_features,
            "is_fitted": self.is_fitted,
        }
        
        with open(filepath, "wb") as f:
            pickle.dump(normalizer_data, f)
        
        print(f"UnifiedNormalizer saved to {filepath}")
    
    def load(self, filepath: str):
        """加载归一化器配置"""
        with open(filepath, "rb") as f:
            normalizer_data = pickle.load(f)
        
        self.method = normalizer_data["method"]
        self.window = normalizer_data["window"]
        self.group_col = normalizer_data["group_col"]
        self.clip = normalizer_data["clip"]
        self.fillna = normalizer_data["fillna"]
        self.feature_groups = normalizer_data.get("feature_groups")
        self.fitted_features = normalizer_data.get("fitted_features", [])
        self.is_fitted = normalizer_data.get("is_fitted", False)
        
        print(f"UnifiedNormalizer loaded from {filepath}")
    
    def get_feature_stats(self, data: pd.DataFrame) -> Dict:
        """获取特征统计信息"""
        if not self.is_fitted:
            raise ValueError("Must fit the normalizer first")
        
        df = data.copy()
        feature_groups = self._group_features(df)
        
        stats = {}
        for group_name, group_features in feature_groups.items():
            if not group_features:
                continue
            
            existing_features = [f for f in group_features if f in df.columns]
            if not existing_features:
                continue
            
            group_data = df[existing_features].values
            group_data = np.nan_to_num(group_data, nan=0.0, posinf=0.0, neginf=0.0)
            
            stats[group_name] = {
                "count": len(existing_features),
                "features": existing_features,
                "mean": np.mean(group_data, axis=0).tolist(),
                "std": np.std(group_data, axis=0).tolist(),
                "min": np.min(group_data, axis=0).tolist(),
                "max": np.max(group_data, axis=0).tolist(),
            }
        
        return stats


def create_unified_normalizer(
    method: str = "zscore",
    window: Optional[int] = 252,
    group_col: str = "_symbol",
    **kwargs
) -> UnifiedNormalizer:
    """创建统一归一化器的便利函数"""
    return UnifiedNormalizer(
        method=method,
        window=window,
        group_col=group_col,
        **kwargs
    )


def normalize_features_unified(
    data: pd.DataFrame,
    method: str = "zscore",
    window: Optional[int] = 252,
    group_col: str = "_symbol",
    fit: bool = True,
    **kwargs,
) -> Tuple[pd.DataFrame, Optional[UnifiedNormalizer]]:
    """统一归一化特征的便利函数"""
    normalizer = UnifiedNormalizer(
        method=method,
        window=window,
        group_col=group_col,
        **kwargs
    )
    
    if fit:
        normalized_data = normalizer.fit_transform(data)
        return normalized_data, normalizer
    else:
        if not normalizer.is_fitted:
            raise ValueError("Normalizer must be fitted first when fit=False")
        normalized_data = normalizer.transform(data)
        return normalized_data, None

