"""
ET (Exhaustion Turn) 作为条件式对冲模块

核心定位：
- ET 是保险，不是策略
- 没有 TC/TE 风险暴露，就不应该存在 ET
- ET 永远不创造风险，只减轻风险

设计原则：
1. 触发前置条件：必须有 TC/TE 方向性风险暴露
2. 渐进式开启：k = k_max * risk_score（不是 binary）
3. ET risk_score 公式：0.4 * ofci_p + 0.35 * shd_p + 0.25 * vol_spike_p
4. ET KPI：不看 Sharpe，只看 left-tail reduction、激活频率、长期成本
5. ET PnL 归类：risk_cost 而非 strategy_pnl
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List


@dataclass
class ETPositionPair:
    """
    ET仓位与TC/TE仓位的配对关系

    用于跟踪ET对冲订单与对应的TC/TE仓位的关联关系。
    """

    et_position_id: str  # ET订单的position_id
    tc_position_ids: List[str]  # 关联的TC仓位ID列表
    te_position_ids: List[str]  # 关联的TE仓位ID列表
    created_at_ns: int  # 创建时间戳（纳秒）
    directional_exposure: float  # 对冲的方向性暴露
    et_position_size: float  # ET仓位大小（有符号：正数=多，负数=空）
    risk_score: float  # 触发时的风险评分 [0, 1]
    ofci_p: float  # 触发时的OFCI percentile
    shd_p: float  # 触发时的SHD percentile
    vol_spike_p: float  # 触发时的vol_spike percentile

    def as_dict(self) -> Dict[str, Any]:
        """转换为字典格式（用于序列化/日志）"""
        return {
            "et_position_id": str(self.et_position_id),
            "tc_position_ids": list(self.tc_position_ids),
            "te_position_ids": list(self.te_position_ids),
            "created_at_ns": int(self.created_at_ns),
            "directional_exposure": float(self.directional_exposure),
            "et_position_size": float(self.et_position_size),
            "risk_score": float(self.risk_score),
            "ofci_p": float(self.ofci_p),
            "shd_p": float(self.shd_p),
            "vol_spike_p": float(self.vol_spike_p),
        }


def compute_et_risk_score(
    ofci_p: float,
    shd_p: float,
    vol_spike_p: float,
) -> float:
    """
    计算 ET 风险评分（无 book 版本）

    根据文档要求：0.4 * ofci_p + 0.35 * shd_p + 0.25 * vol_spike_p

    Args:
        ofci_p: 订单流一致性分位数 [0, 1]（基于abs(ofci)的percentile rank）
        shd_p: 策略同质化分位数 [0, 1]（SHD的percentile rank）
        vol_spike_p: 波动率爆发分位数 [0, 1]（可以使用atr_percentile作为proxy）

    Returns:
        risk_score: [0, 1] 的风险评分
    """
    # 权重分配：
    # - 0.4 * ofci_p: 群体方向一致（越高越危险）
    # - 0.35 * shd_p: 策略同质化（越高越危险）
    # - 0.25 * vol_spike_p: 能量释放（越高越危险）
    risk_score = 0.4 * float(ofci_p) + 0.35 * float(shd_p) + 0.25 * float(vol_spike_p)
    return float(np.clip(risk_score, 0.0, 1.0))


def should_activate_et(
    tc_position: float,
    te_position: float,
    ofci_p: float,
    shd_p: float,
    vol_spike_p: float,
    k_max: float = 0.8,
) -> Tuple[bool, float]:
    """
    判断是否应该激活 ET 对冲

    Args:
        tc_position: TC 仓位（正数=多，负数=空）
        te_position: TE 仓位（正数=多，负数=空）
        ofci_p: 订单流一致性分位数 [0, 1]
        shd_p: 策略同质化分位数 [0, 1]
        vol_spike_p: 波动率爆发分位数 [0, 1]
        k_max: 最大对冲比例（默认 0.8）

    Returns:
        (should_activate: bool, k: float)
        - should_activate: 是否应该激活 ET
        - k: 对冲比例 [0, k_max]
    """
    # 检查是否有方向性风险暴露
    directional_exposure = abs(float(tc_position)) + abs(float(te_position))

    if directional_exposure == 0:
        return False, 0.0

    # 计算风险评分
    risk_score = compute_et_risk_score(ofci_p, shd_p, vol_spike_p)

    # 渐进式开启
    k = k_max * risk_score

    return True, k


def compute_et_position(
    tc_position: float,
    te_position: float,
    ofci_p: float,
    shd_p: float,
    vol_spike_p: float,
    k_max: float = 0.8,
) -> float:
    """
    计算 ET 对冲仓位

    Args:
        tc_position: TC 仓位（正数=多，负数=空）
        te_position: TE 仓位（正数=多，负数=空）
        ofci_p: 订单流一致性分位数 [0, 1]
        shd_p: 策略同质化分位数 [0, 1]
        vol_spike_p: 波动率爆发分位数 [0, 1]
        k_max: 最大对冲比例（默认 0.8）

    Returns:
        et_position: ET 对冲仓位（负数，与方向性暴露相反）
    """
    should_activate, k = should_activate_et(
        tc_position=tc_position,
        te_position=te_position,
        ofci_p=ofci_p,
        shd_p=shd_p,
        vol_spike_p=vol_spike_p,
        k_max=k_max,
    )

    if not should_activate:
        return 0.0

    # 计算方向性暴露
    directional_exposure = abs(float(tc_position)) + abs(float(te_position))

    # 确定方向（TC/TE 如果是多，ET 应该做空）
    # 简化：如果 TC+TE 净多，ET 做空；如果 TC+TE 净空，ET 做多
    net_direction = np.sign(float(tc_position) + float(te_position))
    if net_direction == 0:
        # 如果 TC 和 TE 方向相反且大小相等，不做对冲
        return 0.0

    # ET 仓位 = -k * directional_exposure * net_direction
    et_position = -k * directional_exposure * net_direction

    return float(et_position)


def compute_et_risk_metrics(
    tc_position: float,
    te_position: float,
    ofci_p: float,
    shd_p: float,
    vol_spike_p: float,
    k_max: float = 0.8,
) -> Dict[str, Any]:
    """
    计算 ET 风险指标（用于监控和归因）

    Returns:
        Dict with keys:
        - should_activate: bool
        - risk_score: float [0, 1]
        - k: float [0, k_max]
        - directional_exposure: float
        - et_position: float
        - ofci_contribution: float
        - shd_contribution: float
        - vol_spike_contribution: float
    """
    directional_exposure = abs(float(tc_position)) + abs(float(te_position))

    if directional_exposure == 0:
        return {
            "should_activate": False,
            "risk_score": 0.0,
            "k": 0.0,
            "directional_exposure": 0.0,
            "et_position": 0.0,
            "ofci_contribution": 0.0,
            "shd_contribution": 0.0,
            "vol_spike_contribution": 0.0,
        }

    risk_score = compute_et_risk_score(ofci_p, shd_p, vol_spike_p)
    k = k_max * risk_score

    et_position = compute_et_position(
        tc_position=tc_position,
        te_position=te_position,
        ofci_p=ofci_p,
        shd_p=shd_p,
        vol_spike_p=vol_spike_p,
        k_max=k_max,
    )

    return {
        "should_activate": True,
        "risk_score": risk_score,
        "k": k,
        "directional_exposure": directional_exposure,
        "et_position": et_position,
        "ofci_contribution": 0.4 * float(ofci_p),
        "shd_contribution": 0.35 * float(shd_p),
        "vol_spike_contribution": 0.25 * float(vol_spike_p),
    }
