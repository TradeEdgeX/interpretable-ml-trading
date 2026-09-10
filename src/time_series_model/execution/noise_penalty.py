"""
执行噪声惩罚计算器

核心思想：
- 不用于Gate/Evidence决策（避免数据泄露）
- 只用于Execution层的风险预算调整
- 基于Hurst/WPT/Spectrum/Hilbert特征的连续惩罚因子
"""

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class NoisePenaltyConfig:
    """噪声惩罚配置"""

    short_window: int = 20  # 短期窗口（响应）
    long_window: int = 120  # 长期窗口（背景基线）
    weights: Dict[str, float] = None  # 特征权重
    max_penalty: float = 0.8  # 最大惩罚值（永远<1，避免完全阻断）
    evt_threshold: float = 0.8  # EVT尾部风险阈值
    evt_penalty_addon: float = 0.2  # EVT触发时增加的惩罚

    def __post_init__(self):
        if self.weights is None:
            # 权重分配基于功能分工：
            # - WPT/Spectrum: 结构破碎度主信号 (0.35+0.30=0.65)
            # - Hilbert: 短期相位不稳定 (0.20)
            # - Hurst: 背景环境 (0.15)
            self.weights = {
                "wpt_entropy": 0.35,  # WPT: 多尺度分解，结构破碎度主信号
                "spectrum_width": 0.30,  # Spectrum: 频域分析，频谱稳定性
                "hilbert_phase_var": 0.20,  # Hilbert: 相位/包络不稳定性
                "hurst_inverse": 0.15,  # Hurst: 长期记忆性，背景环境指标
            }


class ExecutionNoisePenalty:
    """执行噪声惩罚计算器"""

    def __init__(self, config: NoisePenaltyConfig):
        self.config = config

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        计算执行噪声惩罚因子

        Args:
            df: 包含数学特征的DataFrame

        Returns:
            0-0.8之间的连续值序列（永远不到1，避免完全deny）
        """
        # 验证必需的特征列
        required_features = [
            "wpt_price_fluctuation",  # 用波动分量代表混乱度
            "spectrum_price_entropy",  # 谱熵
            "hilbert_price_env",  # 包络强度
            "hurst_price_rolling",  # Hurst指数
        ]

        missing_features = [f for f in required_features if f not in df.columns]
        if missing_features:
            raise ValueError(f"缺失必要的数学特征: {missing_features}")

        # 计算每个特征的相对化得分
        wpt_score = self._calculate_feature_score(
            df["wpt_price_fluctuation"], inverse=False
        )
        spectrum_score = self._calculate_feature_score(
            df["spectrum_price_entropy"], inverse=False
        )
        hilbert_score = self._calculate_feature_score(
            df["hilbert_price_env"], inverse=False
        )
        hurst_score = self._calculate_feature_score(
            df["hurst_price_rolling"], inverse=True  # Hurst反向：越低越噪
        )

        # 加权合成基础噪声惩罚
        noise_raw = (
            self.config.weights["wpt_entropy"] * wpt_score
            + self.config.weights["spectrum_width"] * spectrum_score
            + self.config.weights["hilbert_phase_var"] * hilbert_score
            + self.config.weights["hurst_inverse"] * hurst_score
        )

        # 钳制到基础范围
        noise_penalty = np.clip(noise_raw, 0.0, self.config.max_penalty)

        # EVT“保险丝”机制：在极端尾部风险时额外增加惩罚
        if "evt_tail_risk" in df.columns:
            evt_risk = df["evt_tail_risk"].fillna(0.5)  # 默认中性值
            # 当EVT检测到极端尾部风险时，增加惩罚
            high_evt_mask = evt_risk >= self.config.evt_threshold
            if high_evt_mask.any():
                evt_penalty_addon = np.where(
                    high_evt_mask, self.config.evt_penalty_addon, 0
                )
                noise_penalty = np.clip(
                    noise_penalty + evt_penalty_addon, 0.0, self.config.max_penalty
                )

        # 惯性滤波（防抖动）
        noise_penalty_smooth = pd.Series(noise_penalty).ewm(alpha=0.3).mean()

        return noise_penalty_smooth

    def _calculate_feature_score(
        self, series: pd.Series, inverse: bool = False
    ) -> np.ndarray:
        """
        计算单个特征的相对化得分

        Args:
            series: 特征序列
            inverse: 是否反向（如Hurst：越低越噪）

        Returns:
            [0,1]区间的得分数组
        """
        # 短期和长期统计
        short_mean = series.rolling(self.config.short_window).mean()
        long_mean = series.rolling(self.config.long_window).mean()
        long_std = series.rolling(self.config.long_window).std()

        # 相对偏离度
        rel_deviation = (short_mean - long_mean) / (long_std + 1e-8)

        # Sigmoid转换到[0,1]
        scores = 1 / (1 + np.exp(-rel_deviation))

        if inverse:
            scores = 1 - scores  # 反向：越小越噪

        # 填充NaN值（前期数据不足）
        scores = scores.fillna(0.5)  # 默认0.5（中性）

        return scores.values


def validate_noise_penalty_range(penalty_series: pd.Series, max_allowed: float = 0.8):
    """
    验证噪声惩罚值在合理范围内
    """
    if penalty_series.min() < 0.0:
        raise ValueError(f"噪声惩罚值不应小于0，实际最小值: {penalty_series.min()}")

    if penalty_series.max() > max_allowed:
        raise ValueError(
            f"噪声惩罚值不应大于{max_allowed}，实际最大值: {penalty_series.max()}"
        )

    return True
