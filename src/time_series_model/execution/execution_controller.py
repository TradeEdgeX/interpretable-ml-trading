"""
执行控制器：整合Evidence和Noise Penalty
"""

from typing import Optional, Tuple, Dict
import pandas as pd
from .noise_penalty import (
    ExecutionNoisePenalty,
    NoisePenaltyConfig,
    validate_noise_penalty_range,
)
from .tier import TierSelector, ExecutionParams, create_default_bpc_tiers


class ExecutionController:
    """执行控制器"""

    def __init__(
        self,
        tier_selector: TierSelector,
        noise_penalty_calculator: ExecutionNoisePenalty,
    ):
        self.tier_selector = tier_selector
        self.noise_penalty_calculator = noise_penalty_calculator

    def get_execution_params(
        self, df: pd.DataFrame, evidence_score: float
    ) -> Tuple[Optional[ExecutionParams], Dict[str, float]]:
        """
        获取执行参数（同时考虑Evidence Score和Noise Penalty）

        Args:
            df: 包含数学特征的DataFrame
            evidence_score: 证据分数（0-1）

        Returns:
            (调整后的执行参数, 调试信息)
        """
        # 1. 计算噪声惩罚
        noise_penalty_series = self.noise_penalty_calculator.compute(df)
        current_noise_penalty = noise_penalty_series.iloc[-1]  # 最新值

        # 2. 验证噪声惩罚范围
        validate_noise_penalty_range(noise_penalty_series)

        # 3. 获取调整后的参数
        adjusted_params = self.tier_selector.get_adjusted_params(
            evidence_score=evidence_score, noise_penalty=current_noise_penalty
        )

        # 4. 调试信息
        debug_info = {
            "evidence_score": evidence_score,
            "noise_penalty": current_noise_penalty,
            "final_params": adjusted_params.__dict__ if adjusted_params else None,
            "is_approved": adjusted_params is not None,
        }

        return adjusted_params, debug_info


def create_bpc_execution_controller() -> ExecutionController:
    """创建BPC执行控制器"""
    # 创建档位选择器
    tier_selector = create_default_bpc_tiers()

    # 创建噪声惩罚计算器
    noise_config = NoisePenaltyConfig(
        short_window=20,
        long_window=120,
        weights={
            "wpt_entropy": 0.35,
            "spectrum_width": 0.30,
            "hilbert_phase_var": 0.20,
            "hurst_inverse": 0.15,
        },
        max_penalty=0.8,
    )
    noise_calculator = ExecutionNoisePenalty(noise_config)

    return ExecutionController(tier_selector, noise_calculator)
