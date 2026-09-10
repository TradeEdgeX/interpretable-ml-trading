"""
执行档位系统（支持噪声调整）

Evidence Score: 评估"alpha质量"（是否值得交易）
Noise Penalty: 评估"市场噪声"（如何执行）
"""

from typing import Dict, Optional, List
import pandas as pd
from dataclasses import dataclass


@dataclass
class ExecutionParams:
    """执行参数"""

    sl_r: float  # 止损R倍数
    tp_r: float  # 止盈R倍数
    size_multiplier: float  # 仓位倍数
    max_hold_time: Optional[int] = None  # 最大持仓时间
    trailing_enabled: bool = False  # 是否启用追踪止损
    trailing_threshold: Optional[float] = None  # 追踪止损阈值
    trailing_distance: Optional[float] = None  # 追踪止损距离

    def adjust_for_noise(self, noise_penalty: float) -> "ExecutionParams":
        """
        根据噪声惩罚调整参数

        Args:
            noise_penalty: [0, max_penalty]区间的噪声惩罚值

        Returns:
            调整后的参数
        """
        if noise_penalty <= 0:
            return self  # 无噪声调整

        # 调整规则：噪声越高，执行越保守
        adjusted = ExecutionParams(
            sl_r=self.sl_r * (1 + 0.5 * noise_penalty),  # 噪声高时，sl更紧
            tp_r=self.tp_r * (1 - 0.3 * noise_penalty),  # 噪声高时，tp更保守
            size_multiplier=self.size_multiplier
            * (1 - 0.7 * noise_penalty),  # 噪声高时，仓位大幅减小
            max_hold_time=self.max_hold_time,
            trailing_enabled=self.trailing_enabled,
            trailing_threshold=self.trailing_threshold,
            trailing_distance=self.trailing_distance,
        )

        # 如果调整后仓位过小，设置最小值
        min_size = 0.1  # 最小仓位倍数
        adjusted.size_multiplier = max(adjusted.size_multiplier, min_size)

        return adjusted


@dataclass
class ExecutionTier:
    """执行档位"""

    name: str
    score_min: float  # 证据分数下限
    params: ExecutionParams
    enabled: bool = True  # 是否启用


class TierSelector:
    """档位选择器（同时消费Evidence Score和Noise Penalty）"""

    def __init__(self, tiers: List[ExecutionTier]):
        self.tiers = sorted(
            tiers, key=lambda x: x.score_min, reverse=True
        )  # 从高到低排序

    def select_tier(self, evidence_score: float) -> Optional[ExecutionTier]:
        """根据证据分数选择档位"""
        for tier in self.tiers:
            if evidence_score >= tier.score_min and tier.enabled:
                return tier
        return None  # 没有符合条件的档位（相当于deny）

    def get_adjusted_params(
        self, evidence_score: float, noise_penalty: float
    ) -> Optional[ExecutionParams]:
        """
        获取调整后的执行参数

        Args:
            evidence_score: 证据分数（0-1），决定档位
            noise_penalty: 噪声惩罚（0-max_penalty），调整参数

        Returns:
            调整后的执行参数，如果被拒绝则返回None
        """
        tier = self.select_tier(evidence_score)
        if tier is None:
            return None  # 被拒绝

        # 先获取档位的基础参数，再根据噪声调整
        base_params = tier.params
        adjusted_params = base_params.adjust_for_noise(noise_penalty)

        return adjusted_params


def create_default_bpc_tiers() -> TierSelector:
    """创建默认的BPC执行档位"""
    tiers = [
        ExecutionTier(
            name="强证据",
            score_min=0.70,
            params=ExecutionParams(
                sl_r=0.8,
                tp_r=3.0,
                size_multiplier=1.2,
                max_hold_time=200,
                trailing_enabled=True,
                trailing_threshold=0.1,
                trailing_distance=0.05,
            ),
        ),
        ExecutionTier(
            name="中等证据",
            score_min=0.50,
            params=ExecutionParams(
                sl_r=1.0,
                tp_r=2.5,
                size_multiplier=1.0,
                max_hold_time=150,
                trailing_enabled=True,
                trailing_threshold=0.15,
                trailing_distance=0.08,
            ),
        ),
        ExecutionTier(
            name="弱证据",
            score_min=0.30,
            params=ExecutionParams(
                sl_r=1.2,
                tp_r=2.0,
                size_multiplier=0.8,
                max_hold_time=100,
                trailing_enabled=False,
            ),
        ),
        ExecutionTier(
            name="边缘证据",
            score_min=0.10,
            params=ExecutionParams(
                sl_r=1.5,
                tp_r=1.8,
                size_multiplier=0.5,
                max_hold_time=80,
                trailing_enabled=False,
            ),
        ),
    ]

    return TierSelector(tiers)
