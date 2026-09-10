"""
币安 Tick 聚合器测试

使用模拟数据测试聚合功能。
"""

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

import pandas as pd
import pytest

# 检查是否有异步测试支持
try:
    import pytest_asyncio

    HAS_ASYNC_SUPPORT = True
except ImportError:
    try:
        import anyio

        HAS_ASYNC_SUPPORT = True
    except ImportError:
        HAS_ASYNC_SUPPORT = False

from src.data_tools.binance_tick_aggregator import BinanceTickAggregator


class MockQuestDBClient:
    """模拟 QuestDB 客户端"""

    def __init__(self):
        self.inserted_data = []
        self.insert_calls = 0

    async def insert_aggregated_ticks(self, df: pd.DataFrame):
        """模拟插入聚合数据"""
        self.inserted_data.append(df.copy())
        self.insert_calls += 1
        print(f"[MockQuestDB] 插入 {len(df)} 条数据")


class TestBinanceTickAggregator:
    """币安 Tick 聚合器测试"""

    @pytest.fixture
    def questdb_client(self):
        """创建模拟 QuestDB 客户端"""
        return MockQuestDBClient()

    @pytest.fixture
    def aggregator(self, questdb_client):
        """创建聚合器实例"""
        return BinanceTickAggregator(
            ws_url="wss://fstream.binance.com/ws/btcusdt@trade",
            symbol="BTCUSDT",
            questdb_client=questdb_client,
            aggregation_ms=100,  # 100ms 聚合窗口
            batch_size=10,  # 小批量用于测试
            flush_seconds=1,  # 1秒刷新
        )

    def test_window_start_calculation(self, aggregator):
        """测试窗口开始时间计算"""
        # 测试 100ms 窗口
        assert aggregator._get_window_start(100) == 100
        assert aggregator._get_window_start(150) == 100
        assert aggregator._get_window_start(199) == 100
        assert aggregator._get_window_start(200) == 200
        assert aggregator._get_window_start(250) == 200

        # 测试边界情况
        assert aggregator._get_window_start(0) == 0
        assert aggregator._get_window_start(99) == 0

    def test_aggregate_single_tick(self, aggregator):
        """测试单个 tick 聚合"""
        tick_data = {
            "ts_ms": 1000,
            "price": 50000.0,
            "qty": 0.1,
            "is_buyer_maker": False,  # 买方主动
            "trade_id": 12345,
        }

        aggregator._aggregate_tick(tick_data)

        # 检查窗口数据
        window = aggregator.aggregation_buffer[1000]
        assert window["open"] == 50000.0
        assert window["high"] == 50000.0
        assert window["low"] == 50000.0
        assert window["close"] == 50000.0
        assert window["volume"] == 0.1
        assert window["trade_count"] == 1
        assert window["buy_volume"] == 0.1
        assert window["sell_volume"] == 0.0
        assert window["buy_count"] == 1
        assert window["sell_count"] == 0
        assert window["first_trade_id"] == 12345
        assert window["last_trade_id"] == 12345

    def test_aggregate_multiple_ticks_same_window(self, aggregator):
        """测试同一窗口内的多个 tick 聚合"""
        ticks = [
            {
                "ts_ms": 1000,
                "price": 50000.0,
                "qty": 0.1,
                "is_buyer_maker": False,
                "trade_id": 1,
            },
            {
                "ts_ms": 1050,
                "price": 50050.0,
                "qty": 0.2,
                "is_buyer_maker": True,
                "trade_id": 2,
            },
            {
                "ts_ms": 1090,
                "price": 49950.0,
                "qty": 0.15,
                "is_buyer_maker": False,
                "trade_id": 3,
            },
        ]

        for tick in ticks:
            aggregator._aggregate_tick(tick)

        # 检查聚合结果
        window = aggregator.aggregation_buffer[1000]
        assert window["open"] == 50000.0  # 第一个价格
        assert window["high"] == 50050.0  # 最高价
        assert window["low"] == 49950.0  # 最低价
        assert window["close"] == 49950.0  # 最后一个价格
        assert window["volume"] == pytest.approx(0.45, rel=1e-6)  # 总成交量
        assert window["trade_count"] == 3
        assert window["buy_volume"] == 0.25  # 0.1 + 0.15
        assert window["sell_volume"] == 0.2  # 0.2
        assert window["buy_count"] == 2
        assert window["sell_count"] == 1
        assert window["first_trade_id"] == 1
        assert window["last_trade_id"] == 3

    def test_aggregate_multiple_windows(self, aggregator):
        """测试多个窗口的聚合"""
        ticks = [
            {
                "ts_ms": 1000,
                "price": 50000.0,
                "qty": 0.1,
                "is_buyer_maker": False,
                "trade_id": 1,
            },
            {
                "ts_ms": 1100,
                "price": 50100.0,
                "qty": 0.2,
                "is_buyer_maker": False,
                "trade_id": 2,
            },
            {
                "ts_ms": 1200,
                "price": 50200.0,
                "qty": 0.3,
                "is_buyer_maker": True,
                "trade_id": 3,
            },
        ]

        for tick in ticks:
            aggregator._aggregate_tick(tick)

        # 应该有 3 个窗口
        assert len(aggregator.aggregation_buffer) == 3
        assert 1000 in aggregator.aggregation_buffer
        assert 1100 in aggregator.aggregation_buffer
        assert 1200 in aggregator.aggregation_buffer

        # 检查每个窗口
        window1 = aggregator.aggregation_buffer[1000]
        assert window1["volume"] == 0.1
        assert window1["trade_count"] == 1

        window2 = aggregator.aggregation_buffer[1100]
        assert window2["volume"] == 0.2
        assert window2["trade_count"] == 1

        window3 = aggregator.aggregation_buffer[1200]
        assert window3["volume"] == 0.3
        assert window3["trade_count"] == 1

    def test_finalize_completed_windows(self, aggregator):
        """测试完成窗口的逻辑"""
        # 添加一些 tick 到不同窗口
        ticks = [
            {
                "ts_ms": 1000,
                "price": 50000.0,
                "qty": 0.1,
                "is_buyer_maker": False,
                "trade_id": 1,
            },
            {
                "ts_ms": 1050,
                "price": 50050.0,
                "qty": 0.2,
                "is_buyer_maker": True,
                "trade_id": 2,
            },
            {
                "ts_ms": 1100,
                "price": 50100.0,
                "qty": 0.3,
                "is_buyer_maker": False,
                "trade_id": 3,
            },
        ]

        for tick in ticks:
            aggregator._aggregate_tick(tick)

        # 当前时间是 1200ms，应该完成 1000ms 窗口（窗口结束时间是 1100ms）
        aggregator._finalize_completed_windows(1200)

        # 1000ms 窗口应该被完成（因为窗口结束时间 1100 < 当前窗口结束时间 1200）
        assert 1000 not in aggregator.aggregation_buffer
        # 注意：1100ms 窗口也会被完成（因为窗口结束时间 1200 < 当前窗口结束时间 1200+100=1300）
        # 实际上，当前窗口是 1200ms，所以 1200ms 之前的窗口都应该完成
        assert len(aggregator.completed_aggregates) >= 1

        # 检查完成的数据
        completed = aggregator.completed_aggregates[0]
        assert completed["timestamp"] == 1000
        assert completed["symbol"] == "BTCUSDT"
        assert completed["volume"] == pytest.approx(0.3, rel=1e-6)  # 0.1 + 0.2
        assert completed["buy_volume"] == 0.1
        assert completed["sell_volume"] == 0.2
        assert "buy_ratio" in completed
        assert "sell_ratio" in completed
        assert "delta" in completed

        # 1100ms 窗口也会被完成（因为窗口结束时间 1200 < 当前窗口结束时间 1300）
        # 所以缓冲区中不应该有 1100ms 窗口
        assert 1100 not in aggregator.aggregation_buffer

    def test_finalize_multiple_windows(self, aggregator):
        """测试完成多个窗口"""
        # 添加多个窗口的 tick
        for i in range(5):
            tick = {
                "ts_ms": 1000 + i * 100,
                "price": 50000.0 + i * 10,
                "qty": 0.1,
                "is_buyer_maker": False,
                "trade_id": i + 1,
            }
            aggregator._aggregate_tick(tick)

        # 当前时间是 1600ms，应该完成前 4 个窗口（1000, 1100, 1200, 1300）
        # 当前窗口是 1600ms，窗口结束时间是 1700ms
        # 所以窗口结束时间 < 1700ms 的都应该完成
        aggregator._finalize_completed_windows(1600)

        # 应该完成 4 个窗口（1000, 1100, 1200, 1300）
        # 1400ms 窗口的结束时间是 1500ms < 1700ms，所以也会完成
        assert len(aggregator.completed_aggregates) >= 4
        # 1400ms 窗口也会完成，所以缓冲区可能为空或只有 1500ms 窗口

        # 检查完成的数据按时间排序（1400ms 窗口也会完成）
        timestamps = [agg["timestamp"] for agg in aggregator.completed_aggregates]
        assert timestamps[:4] == [1000, 1100, 1200, 1300]  # 前4个窗口

    def test_flush_aggregates(self, aggregator, questdb_client):
        """测试刷新聚合数据"""
        # 添加一些完成的聚合数据
        aggregator.completed_aggregates = [
            {
                "timestamp": 1000,
                "symbol": "BTCUSDT",
                "open": 50000.0,
                "high": 50050.0,
                "low": 49950.0,
                "close": 50025.0,
                "volume": 1.0,
                "trade_count": 10,
                "buy_volume": 0.6,
                "sell_volume": 0.4,
                "buy_count": 6,
                "sell_count": 4,
                "buy_ratio": 0.6,
                "sell_ratio": 0.4,
                "delta": 0.2,
                "first_trade_id": 1,
                "last_trade_id": 10,
            },
            {
                "timestamp": 1100,
                "symbol": "BTCUSDT",
                "open": 50025.0,
                "high": 50075.0,
                "low": 50000.0,
                "close": 50050.0,
                "volume": 1.5,
                "trade_count": 15,
                "buy_volume": 0.9,
                "sell_volume": 0.6,
                "buy_count": 9,
                "sell_count": 6,
                "buy_ratio": 0.6,
                "sell_ratio": 0.4,
                "delta": 0.3,
                "first_trade_id": 11,
                "last_trade_id": 25,
            },
        ]

        # 刷新数据（不依赖 pytest-asyncio/anyio）
        asyncio.run(aggregator._flush_aggregates())

        # 检查是否调用了插入方法
        assert questdb_client.insert_calls == 1
        assert len(questdb_client.inserted_data) == 1

        # 检查 DataFrame
        df = questdb_client.inserted_data[0]
        assert len(df) == 2
        assert "ts" in df.columns
        assert "symbol" in df.columns
        assert "open" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns

        # 检查时间戳格式
        assert pd.api.types.is_datetime64_any_dtype(df["ts"])

        # 检查数据值
        assert df.iloc[0]["open"] == 50000.0
        assert df.iloc[1]["volume"] == 1.5

        # 检查已完成列表已清空
        assert len(aggregator.completed_aggregates) == 0

    def test_buy_sell_direction(self, aggregator):
        """测试买卖方向判断"""
        # 买方主动（is_buyer_maker=False）
        buy_tick = {
            "ts_ms": 1000,
            "price": 50000.0,
            "qty": 0.1,
            "is_buyer_maker": False,
            "trade_id": 1,
        }

        # 卖方主动（is_buyer_maker=True）
        sell_tick = {
            "ts_ms": 1000,
            "price": 50000.0,
            "qty": 0.2,
            "is_buyer_maker": True,
            "trade_id": 2,
        }

        aggregator._aggregate_tick(buy_tick)
        aggregator._aggregate_tick(sell_tick)

        window = aggregator.aggregation_buffer[1000]
        assert window["buy_volume"] == 0.1
        assert window["sell_volume"] == 0.2
        assert window["buy_count"] == 1
        assert window["sell_count"] == 1

    def test_empty_window_handling(self, aggregator):
        """测试空窗口处理"""
        # 完成一个不存在的窗口（应该没有影响）
        aggregator._finalize_completed_windows(1000)

        assert len(aggregator.aggregation_buffer) == 0
        assert len(aggregator.completed_aggregates) == 0

    def test_flush_empty_aggregates(self, aggregator, questdb_client):
        """测试刷新空聚合数据"""
        # 没有完成的聚合数据
        aggregator.completed_aggregates = []

        asyncio.run(aggregator._flush_aggregates())

        # 不应该调用插入方法
        assert questdb_client.insert_calls == 0

    def test_ohlc_calculation(self, aggregator):
        """测试 OHLC 计算"""
        ticks = [
            {
                "ts_ms": 1000,
                "price": 50000.0,
                "qty": 0.1,
                "is_buyer_maker": False,
                "trade_id": 1,
            },  # open
            {
                "ts_ms": 1050,
                "price": 50100.0,
                "qty": 0.1,
                "is_buyer_maker": False,
                "trade_id": 2,
            },  # high
            {
                "ts_ms": 1080,
                "price": 49900.0,
                "qty": 0.1,
                "is_buyer_maker": False,
                "trade_id": 3,
            },  # low
            {
                "ts_ms": 1090,
                "price": 50050.0,
                "qty": 0.1,
                "is_buyer_maker": False,
                "trade_id": 4,
            },  # close
        ]

        for tick in ticks:
            aggregator._aggregate_tick(tick)

        window = aggregator.aggregation_buffer[1000]
        assert window["open"] == 50000.0
        assert window["high"] == 50100.0
        assert window["low"] == 49900.0
        assert window["close"] == 50050.0

    def test_order_flow_metrics(self, aggregator):
        """测试订单流指标计算"""
        ticks = [
            {
                "ts_ms": 1000,
                "price": 50000.0,
                "qty": 0.1,
                "is_buyer_maker": False,
                "trade_id": 1,
            },  # buy
            {
                "ts_ms": 1050,
                "price": 50010.0,
                "qty": 0.2,
                "is_buyer_maker": True,
                "trade_id": 2,
            },  # sell
            {
                "ts_ms": 1080,
                "price": 50020.0,
                "qty": 0.3,
                "is_buyer_maker": False,
                "trade_id": 3,
            },  # buy
        ]

        for tick in ticks:
            aggregator._aggregate_tick(tick)

        # 完成窗口
        aggregator._finalize_completed_windows(1100)

        completed = aggregator.completed_aggregates[0]
        assert completed["volume"] == pytest.approx(0.6, rel=1e-6)
        assert completed["buy_volume"] == 0.4  # 0.1 + 0.3
        assert completed["sell_volume"] == 0.2
        assert completed["buy_ratio"] == pytest.approx(0.4 / 0.6, rel=1e-6)
        assert completed["sell_ratio"] == pytest.approx(0.2 / 0.6, rel=1e-6)
        assert completed["delta"] == 0.2  # 0.4 - 0.2

    def test_simulate_websocket_messages(self, aggregator, questdb_client):
        """模拟 WebSocket 消息处理"""
        # 模拟币安 WebSocket 消息
        messages = [
            json.dumps(
                {
                    "e": "trade",
                    "E": 1000,
                    "T": 1000,
                    "s": "BTCUSDT",
                    "t": 1,
                    "p": "50000.0",
                    "q": "0.1",
                    "m": False,  # 买方主动
                }
            ),
            json.dumps(
                {
                    "e": "trade",
                    "E": 1050,
                    "T": 1050,
                    "s": "BTCUSDT",
                    "t": 2,
                    "p": "50050.0",
                    "q": "0.2",
                    "m": True,  # 卖方主动
                }
            ),
            json.dumps(
                {
                    "e": "trade",
                    "E": 1100,
                    "T": 1100,
                    "s": "BTCUSDT",
                    "t": 3,
                    "p": "50100.0",
                    "q": "0.3",
                    "m": False,  # 买方主动
                }
            ),
        ]

        # 处理消息
        last_ts = None
        for msg in messages:
            data = json.loads(msg)
            if data.get("e") != "trade":
                continue

            timestamp_ms = int(data.get("T") or data.get("E"))
            last_ts = timestamp_ms
            tick_data = {
                "ts_ms": timestamp_ms,
                "symbol": data.get("s", aggregator.symbol),
                "price": float(data["p"]),
                "qty": float(data["q"]),
                "is_buyer_maker": bool(data.get("m", False)),
                "trade_id": int(data.get("t", 0)),
            }

            aggregator._aggregate_tick(tick_data)
            aggregator._finalize_completed_windows(timestamp_ms)

        # Finalize the last (current) window by advancing time one full window.
        # Otherwise the last window_end == current_window_end would not be considered "completed".
        if last_ts is not None:
            aggregator._finalize_completed_windows(last_ts + aggregator.aggregation_ms)

        # 刷新数据（不依赖 pytest-asyncio/anyio）
        asyncio.run(aggregator._flush_aggregates())

        # 检查结果
        assert questdb_client.insert_calls == 1
        df = questdb_client.inserted_data[0]
        # 应该至少有 2 个完成的窗口（1000ms 和 1100ms）
        assert len(df) >= 2

        # 检查第一个窗口
        window1 = df[df["timestamp_ms"] == 1000].iloc[0]
        assert window1["volume"] == pytest.approx(0.3, rel=1e-9)  # 0.1 + 0.2
        assert window1["buy_volume"] == 0.1
        assert window1["sell_volume"] == pytest.approx(0.2, rel=1e-9)

        # 检查第二个窗口
        window2 = df[df["timestamp_ms"] == 1100].iloc[0]
        assert window2["volume"] == pytest.approx(0.3, rel=1e-9)
        assert window2["buy_volume"] == pytest.approx(0.3, rel=1e-9)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
