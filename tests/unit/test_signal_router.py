"""
测试 SignalRouter：archetype 优先级排序
"""

import pytest
from datetime import datetime, timedelta

from time_series_model.portfolio.signal_router import (
    CandidateSignal,
    RankedSignal,
    SignalRouter,
    compute_archetype_edges_from_trades,
)


def test_aos_calculation():
    """测试 AOS 计算"""
    edges = {
        "BPC": 0.62,
        "ME": 0.85,
        "Reversal": 0.55,
    }
    router = SignalRouter(archetype_edges=edges, max_slots=2)

    signal = CandidateSignal(
        symbol="BTCUSDT",
        archetype="ME",
        evidence_score=0.8,
        side="LONG",
        entry_price=50000.0,
    )

    aos = router.compute_aos(signal)
    expected = 0.85 * 0.8  # Edge × Evidence
    assert abs(aos - expected) < 0.001


def test_same_symbol_different_archetype():
    """测试：同 symbol 不同 archetype，选 AOS 最高的"""
    edges = {
        "BPC": 0.62,
        "ME": 0.85,
        "Reversal": 0.55,
    }
    router = SignalRouter(archetype_edges=edges, max_slots=2)

    candidates = [
        CandidateSignal(
            symbol="BTCUSDT",
            archetype="BPC",
            evidence_score=0.9,  # AOS = 0.62 * 0.9 = 0.558
            side="LONG",
            entry_price=50000.0,
        ),
        CandidateSignal(
            symbol="BTCUSDT",
            archetype="ME",
            evidence_score=0.7,  # AOS = 0.85 * 0.7 = 0.595（更高）
            side="LONG",
            entry_price=50000.0,
        ),
    ]

    ranked = router.route_signals(candidates)

    # 只保留 ME（AOS 更高）
    assert len(ranked) == 1
    assert ranked[0].signal.archetype == "ME"
    assert ranked[0].rank == 1


def test_different_symbol_ranking():
    """测试：不同 symbol，按 AOS 排序"""
    edges = {
        "BPC": 0.62,
        "ME": 0.85,
        "Reversal": 0.55,
    }
    router = SignalRouter(archetype_edges=edges, max_slots=2)

    candidates = [
        CandidateSignal(
            symbol="BTCUSDT",
            archetype="BPC",
            evidence_score=0.8,  # AOS = 0.62 * 0.8 = 0.496
            side="LONG",
            entry_price=50000.0,
        ),
        CandidateSignal(
            symbol="ETHUSDT",
            archetype="ME",
            evidence_score=0.9,  # AOS = 0.85 * 0.9 = 0.765（最高）
            side="LONG",
            entry_price=3000.0,
        ),
        CandidateSignal(
            symbol="SOLUSDT",
            archetype="Reversal",
            evidence_score=0.7,  # AOS = 0.55 * 0.7 = 0.385
            side="SHORT",
            entry_price=100.0,
        ),
    ]

    ranked = router.route_signals(candidates)

    # 只取前 2 个（max_slots=2）
    assert len(ranked) == 2
    assert ranked[0].signal.symbol == "ETHUSDT"  # AOS 最高
    assert ranked[0].signal.archetype == "ME"
    assert ranked[0].rank == 1

    assert ranked[1].signal.symbol == "BTCUSDT"  # AOS 第二
    assert ranked[1].signal.archetype == "BPC"
    assert ranked[1].rank == 2


def test_max_slots_limit():
    """测试：max_slots 限制"""
    edges = {"BPC": 0.6, "ME": 0.8}
    router = SignalRouter(archetype_edges=edges, max_slots=1)

    candidates = [
        CandidateSignal(
            symbol="BTCUSDT",
            archetype="BPC",
            evidence_score=0.9,
            side="LONG",
            entry_price=50000.0,
        ),
        CandidateSignal(
            symbol="ETHUSDT",
            archetype="ME",
            evidence_score=0.8,
            side="LONG",
            entry_price=3000.0,
        ),
    ]

    ranked = router.route_signals(candidates)

    # 只取 1 个
    assert len(ranked) == 1


def test_empty_candidates():
    """测试：空候选列表"""
    edges = {"BPC": 0.6}
    router = SignalRouter(archetype_edges=edges, max_slots=2)

    ranked = router.route_signals([])
    assert len(ranked) == 0


def test_update_edge():
    """测试：更新 Edge"""
    edges = {"BPC": 0.6, "ME": 0.8}
    router = SignalRouter(archetype_edges=edges, max_slots=2)

    router.update_edge("BPC", 0.75)

    assert router.archetype_edges["BPC"] == 0.75
    assert router.archetype_edges["ME"] == 0.8


def test_compute_archetype_edges_from_trades():
    """测试：从历史交易统计 Edge"""
    now = datetime.utcnow()
    trades = [
        {"archetype": "BPC", "r_multiple": 0.8, "closed_at": now - timedelta(days=10)},
        {"archetype": "BPC", "r_multiple": 0.4, "closed_at": now - timedelta(days=20)},
        {"archetype": "ME", "r_multiple": 1.2, "closed_at": now - timedelta(days=15)},
        {"archetype": "ME", "r_multiple": 0.5, "closed_at": now - timedelta(days=25)},
        # 太旧的交易（超过 3 个月）
        {"archetype": "BPC", "r_multiple": 2.0, "closed_at": now - timedelta(days=100)},
    ]

    edges = compute_archetype_edges_from_trades(trades, lookback_months=3)

    # BPC: (0.8 + 0.4) / 2 = 0.6
    assert abs(edges["BPC"] - 0.6) < 0.001

    # ME: (1.2 + 0.5) / 2 = 0.85
    assert abs(edges["ME"] - 0.85) < 0.001

    # 太旧的交易不参与统计
    assert "BPC" in edges


def test_default_edge_for_unknown_archetype():
    """测试：未知 archetype 使用默认 Edge"""
    edges = {"BPC": 0.6}
    router = SignalRouter(archetype_edges=edges, max_slots=2)

    signal = CandidateSignal(
        symbol="BTCUSDT",
        archetype="Unknown",  # 未知的 archetype
        evidence_score=0.8,
        side="LONG",
        entry_price=50000.0,
    )

    aos = router.compute_aos(signal)
    expected = 0.5 * 0.8  # 默认 Edge = 0.5
    assert abs(aos - expected) < 0.001
