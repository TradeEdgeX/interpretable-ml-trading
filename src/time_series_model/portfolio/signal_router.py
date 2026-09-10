"""
信号路由器：archetype 优先级排序和 slot 分配

职责：
1. 收集多个 symbol + archetype 的候选信号
2. 计算 AOS (Archetype Opportunity Score)
3. 处理三类冲突场景：
   - 同 symbol 不同 archetype：选 AOS 最高的
   - 不同 symbol 同 archetype：按 AOS 排序
   - 不同 symbol 不同 archetype：按 AOS 排序
4. 输出前 N 个最优信号（N = max_slots）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class CandidateSignal:
    """
    候选交易信号

    Attributes:
        symbol: 交易对（如 'BTCUSDT'）
        archetype: 策略原型（如 'BPC', 'ME', 'Reversal'）
        evidence_score: 当前信号的 evidence 强度 [0, 1]
        side: 'LONG' | 'SHORT'
        entry_price: 建议入场价格
        stop_loss_price: 建议止损价格
        take_profit_price: 建议止盈价格
        notes: 备注信息
    """

    symbol: str
    archetype: str
    evidence_score: float
    side: str  # 'LONG' or 'SHORT'
    entry_price: float
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    notes: Optional[str] = None


@dataclass
class RankedSignal:
    """
    排序后的信号（带 AOS 评分）

    Attributes:
        signal: 原始候选信号
        aos: Archetype Opportunity Score
        edge: 该 archetype 的历史 Edge（Avg R-multiple）
        rank: 排名（1 = 最高）
    """

    signal: CandidateSignal
    aos: float
    edge: float
    rank: int


class SignalRouter:
    """
    信号路由器：处理多信号优先级排序

    算法：
        AOS = Edge_archetype × Evidence_score

    Edge_archetype 来自历史统计（需定期滚动更新）：
        - 最近 N 个月该 archetype 的平均 R-multiple
        - 例如：{'BPC': 0.62, 'ME': 0.85, 'Reversal': 0.55}

    冲突处理：
        1. 同 symbol 不同 archetype → 只保留 AOS 最高的
        2. 不同 symbol → 按 AOS 排序，取前 max_slots 个
    """

    def __init__(self, archetype_edges: Dict[str, float], max_slots: int = 2):
        """
        初始化信号路由器

        Args:
            archetype_edges: 各 archetype 的历史 Edge
                例如：{'BPC': 0.62, 'ME': 0.85, 'Reversal': 0.55}
            max_slots: 最大同时持仓数
                建议从 constitution.yaml → slots.slot_count 读取后传入
        """
        self.archetype_edges = archetype_edges
        self.max_slots = max_slots

    def compute_aos(self, signal: CandidateSignal) -> float:
        """
        计算 Archetype Opportunity Score

        Args:
            signal: 候选信号

        Returns:
            AOS = Edge × Evidence
        """
        edge = self.archetype_edges.get(signal.archetype, 0.5)  # 默认 0.5
        aos = edge * signal.evidence_score
        return aos

    def route_signals(
        self,
        candidates: List[CandidateSignal],
    ) -> List[RankedSignal]:
        """
        信号路由：处理冲突并排序

        逻辑：
        1. 同 symbol 不同 archetype → 只保留 AOS 最高的
        2. 所有信号按 AOS 排序
        3. 取前 max_slots 个

        Args:
            candidates: 所有候选信号

        Returns:
            排序后的信号列表（最多 max_slots 个）
        """
        if not candidates:
            return []

        # 计算所有信号的 AOS
        signal_aos_list: List[Tuple[CandidateSignal, float, float]] = []
        for signal in candidates:
            aos = self.compute_aos(signal)
            edge = self.archetype_edges.get(signal.archetype, 0.5)
            signal_aos_list.append((signal, aos, edge))

        # 按 symbol 分组，每组只保留 AOS 最高的
        symbol_best: Dict[str, Tuple[CandidateSignal, float, float]] = {}
        for signal, aos, edge in signal_aos_list:
            symbol = signal.symbol
            if symbol not in symbol_best or aos > symbol_best[symbol][1]:
                symbol_best[symbol] = (signal, aos, edge)

        # 收集所有 symbol 的最优信号
        filtered_signals = list(symbol_best.values())

        # 按 AOS 排序（降序）
        filtered_signals.sort(key=lambda x: x[1], reverse=True)

        # 取前 max_slots 个
        top_signals = filtered_signals[: self.max_slots]

        # 封装为 RankedSignal
        ranked_signals: List[RankedSignal] = []
        for rank, (signal, aos, edge) in enumerate(top_signals, start=1):
            ranked_signals.append(
                RankedSignal(
                    signal=signal,
                    aos=aos,
                    edge=edge,
                    rank=rank,
                )
            )

        return ranked_signals

    def get_edge_config(self) -> Dict[str, float]:
        """获取当前 archetype edges 配置"""
        return self.archetype_edges.copy()

    def update_edge(self, archetype: str, new_edge: float) -> None:
        """
        更新某个 archetype 的 Edge

        Args:
            archetype: archetype 名称
            new_edge: 新的 Edge 值（建议每月滚动更新）
        """
        self.archetype_edges[archetype] = new_edge


def load_archetype_edges_from_config(config_path: str) -> Dict[str, float]:
    """从 YAML 配置文件加载 archetype edges

    Args:
        config_path: 配置文件路径
            研究: 'config/strategies/bpc/archetypes/archetype_edges.yaml'
            实盘: 'live/highcap/config/strategies/bpc/archetypes/archetype_edges.yaml'

    Returns:
        archetype edges 字典

    示例：
        edges = load_archetype_edges_from_config('config/strategies/bpc/archetypes/archetype_edges.yaml')
        # {'BPC': 0.62, 'ME': 0.85, ...}
    """
    import yaml
    from pathlib import Path

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    edges = config.get("archetype_edges", {})
    if not edges:
        raise ValueError(f"配置文件中未找到 'archetype_edges': {config_path}")

    return edges


def compute_archetype_edges_from_trades(
    trades: List[Dict],
    lookback_months: int = 3,
) -> Dict[str, float]:
    """
    从历史交易记录统计各 archetype 的 Edge

    Args:
        trades: 历史交易记录列表
            每条记录需包含：
            - 'archetype': str
            - 'r_multiple': float（盈利/风险比）
            - 'closed_at': datetime（用于过滤时间窗口）
        lookback_months: 回看周期（月）

    Returns:
        各 archetype 的平均 R-multiple
        例如：{'BPC': 0.62, 'ME': 0.85, 'Reversal': 0.55}

    示例：
        trades = [
            {'archetype': 'BPC', 'r_multiple': 0.8, 'closed_at': datetime(...)},
            {'archetype': 'BPC', 'r_multiple': 0.4, 'closed_at': datetime(...)},
            {'archetype': 'ME', 'r_multiple': 1.2, 'closed_at': datetime(...)},
        ]
        edges = compute_archetype_edges_from_trades(trades, lookback_months=3)
        # {'BPC': 0.6, 'ME': 1.2}
    """
    from datetime import datetime, timedelta
    from collections import defaultdict

    # 计算时间窗口
    now = datetime.utcnow()
    cutoff = now - timedelta(days=lookback_months * 30)

    # 按 archetype 分组统计
    archetype_r_multiples: Dict[str, List[float]] = defaultdict(list)
    for trade in trades:
        closed_at = trade.get("closed_at")
        if closed_at and closed_at < cutoff:
            continue  # 跳过太旧的交易

        archetype = trade.get("archetype")
        r_multiple = trade.get("r_multiple")
        if archetype and r_multiple is not None:
            archetype_r_multiples[archetype].append(r_multiple)

    # 计算平均值
    edges: Dict[str, float] = {}
    for archetype, r_multiples in archetype_r_multiples.items():
        if r_multiples:
            edges[archetype] = sum(r_multiples) / len(r_multiples)

    return edges
