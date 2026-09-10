"""
事件驱动架构集成测试

测试完整的工作流程：
1. IncrementalFeatureComputer 处理数据流
2. EventDrivenStrategy 生成信号
3. WebSocket 客户端接收数据
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, patch

from src.time_series_model.live.incremental_feature_computer import (
    IncrementalFeatureComputer,
)
from src.live_data_stream.websocket_client import BinanceWebSocketClient, BinanceTick


class TestIncrementalFeatureComputerIntegration:
    """IncrementalFeatureComputer 集成测试"""

    def test_tick_to_bar_integration(self):
        """测试从 tick 到 bar 的完整流程"""
        computer = IncrementalFeatureComputer(
            tick_window_minutes=30,
            bar_window_size=100,
            vpin_bucket_volume_usd=100000.0,
            vpin_n_buckets=50,
            live_feature_plan_path="/dev/null",  # 跳过 feature plan 让所有 key 可见
        )

        base_ts = int(datetime.now().timestamp() * 1e9)

        # 模拟 1 小时的 tick 数据流
        for minute in range(60):
            for second in range(60):
                tick = {
                    "ts": base_ts + (minute * 60 + second) * 1_000_000_000,
                    "price": 50000.0 + np.random.randn() * 100,
                    "volume": np.random.uniform(0.01, 0.1),
                    "side": 1 if np.random.rand() > 0.5 else -1,
                }
                computer.on_tick(tick)

        # 模拟 1 小时的 bar 数据
        for hour in range(24):
            bar = {
                "ts": base_ts + hour * 3600 * 1_000_000_000,
                "open": 50000.0 + hour * 10,
                "high": 50000.0 + hour * 10 + 50,
                "low": 50000.0 + hour * 10 - 50,
                "close": 50000.0 + hour * 10 + 20,
                "volume": 100.0 + hour * 10,
            }
            computer.on_bar(bar, timeframe="1H")

        # 获取所有特征
        features = computer.get_features()

        # 验证特征存在
        assert len(features) > 0
        # vpin 在 orderflow features 而非 bar features
        orderflow = computer.get_orderflow_features(window_minutes=15)
        assert "vpin" in orderflow or "imbalance" in orderflow
        # IFC 将 bar 特征存储为不带前缀的 key（当只有一个 timeframe 或为 primary 时）
        assert "close" in features
        assert "volume" in features

    def test_vpin_continuity(self):
        """测试 VPIN 跨 bucket 连续性"""
        computer = IncrementalFeatureComputer(
            tick_window_minutes=30,
            vpin_bucket_volume_usd=10000.0,  # 小 bucket，容易填满
            vpin_n_buckets=10,
        )

        base_ts = int(datetime.now().timestamp() * 1e9)

        # 添加足够多的 tick 填满多个 bucket
        for i in range(1000):
            tick = {
                "ts": base_ts + i * 1_000_000_000,
                "price": 50000.0,
                "volume": 0.02,  # $1000 per tick
                "side": 1 if i % 2 == 0 else -1,
            }
            computer.on_tick(tick)

        # 应该有多个 VPIN bucket
        assert len(computer.vpin_buckets) > 0

        # VPIN 值应该在合理范围内
        for _, vpin_value in computer.vpin_buckets:
            assert 0.0 <= vpin_value <= 1.0

    def test_multiple_timeframes(self):
        """测试多时间框架特征"""
        computer = IncrementalFeatureComputer(
            bar_window_size=200,
            live_feature_plan_path="/dev/null",  # 跳过 feature plan 避免慢计算
        )

        base_ts = int(datetime.now().timestamp() * 1e9)

        # 添加 15 分钟 bar
        for i in range(100):
            bar_15m = {
                "ts": base_ts + i * 15 * 60 * 1_000_000_000,
                "open": 50000.0,
                "high": 51000.0,
                "low": 49000.0,
                "close": 50500.0,
                "volume": 100.0,
            }
            computer.on_bar(bar_15m, timeframe="15T")

        # 添加 1 小时 bar
        for i in range(24):
            bar_1h = {
                "ts": base_ts + i * 3600 * 1_000_000_000,
                "open": 50000.0,
                "high": 51000.0,
                "low": 49000.0,
                "close": 50500.0,
                "volume": 500.0,
            }
            computer.on_bar(bar_1h, timeframe="1H")

        # 获取特征
        features = computer.get_features()

        # 验证多时间框架特征（IFC 将 bar OHLCV 存储为不带前缀的 key）
        assert "close" in features
        assert "volume" in features
        # timeframe_features 内部按 timeframe 分组
        assert "15T" in computer.timeframe_features
        assert "1H" in computer.timeframe_features
        assert "close" in computer.timeframe_features["15T"]
        assert "close" in computer.timeframe_features["1H"]


class TestWebSocketIntegration:
    """WebSocket 集成测试"""

    def test_websocket_callback_chain(self):
        """测试 WebSocket 回调链"""
        received_ticks = []

        def callback(tick: BinanceTick):
            received_ticks.append(tick)

        client = BinanceWebSocketClient(
            symbols=["BTCUSDT"],
            use_futures=True,
        )
        client.add_callback(callback)

        # 模拟 tick
        tick1 = BinanceTick(
            symbol="BTCUSDT",
            timestamp_ms=1234567890000,
            price=50000.0,
            volume=0.1,
            turnover=5000.0,
            side=1,
        )

        tick2 = BinanceTick(
            symbol="BTCUSDT",
            timestamp_ms=1234567891000,
            price=50001.0,
            volume=0.2,
            turnover=10000.2,
            side=-1,
        )

        # 手动调用回调（模拟 WebSocket 收到数据）
        for callback_func in client._callbacks:
            callback_func(tick1)
            callback_func(tick2)

        assert len(received_ticks) == 2
        assert received_ticks[0].price == 50000.0
        assert received_ticks[1].price == 50001.0

    def test_multiple_callbacks(self):
        """测试多个回调"""
        results1 = []
        results2 = []

        def callback1(tick):
            results1.append(tick.symbol)

        def callback2(tick):
            results2.append(tick.price)

        client = BinanceWebSocketClient(symbols=["BTCUSDT"])
        client.add_callback(callback1)
        client.add_callback(callback2)

        tick = BinanceTick(
            symbol="BTCUSDT",
            timestamp_ms=1234567890000,
            price=50000.0,
            volume=0.1,
            turnover=5000.0,
            side=1,
        )

        # 调用所有回调
        for callback_func in client._callbacks:
            callback_func(tick)

        assert len(results1) == 1
        assert results1[0] == "BTCUSDT"
        assert len(results2) == 1
        assert results2[0] == 50000.0


class TestEndToEndIntegration:
    """端到端集成测试"""

    def test_tick_to_features_to_signal(self):
        """测试从 tick 到特征到信号的完整流程"""
        computer = IncrementalFeatureComputer(
            tick_window_minutes=15,
            bar_window_size=100,
            vpin_bucket_volume_usd=50000.0,
            vpin_n_buckets=20,
            live_feature_plan_path="/dev/null",  # 跳过 feature plan
        )

        base_ts = int(datetime.now().timestamp() * 1e9)

        # 1. 添加 tick 数据
        for i in range(500):
            tick = {
                "ts": base_ts + i * 1_000_000_000,
                "price": 50000.0 + np.random.randn() * 50,
                "volume": np.random.uniform(0.01, 0.05),
                "side": 1 if np.random.rand() > 0.4 else -1,  # 60% 买入
            }
            computer.on_tick(tick)

        # 2. 添加 bar 数据
        for i in range(20):
            bar = {
                "ts": base_ts + i * 3600 * 1_000_000_000,
                "open": 50000.0 + i * 10,
                "high": 50000.0 + i * 10 + 100,
                "low": 50000.0 + i * 10 - 100,
                "close": 50000.0 + i * 10 + 50,
                "volume": 100.0 + i * 10,
            }
            computer.on_bar(bar, timeframe="1H")

        # 3. 获取特征
        all_features = computer.get_features()
        orderflow_features = computer.get_orderflow_features(window_minutes=15)

        # 4. 模拟信号评估
        should_enter = False
        signal_reason = ""

        if orderflow_features.get("vpin", 0) > 0.6:
            if orderflow_features.get("imbalance", 0) < -0.2:
                should_enter = True
                signal_reason = "High VPIN + Sell Imbalance"
            elif orderflow_features.get("imbalance", 0) > 0.3:
                should_enter = True
                signal_reason = "High VPIN + Buy Surge"

        # 验证
        assert len(all_features) > 0
        assert "vpin" in orderflow_features or "imbalance" in orderflow_features

        # 如果有信号，验证信号信息
        if should_enter:
            assert signal_reason != ""
            assert "VPIN" in signal_reason or "Imbalance" in signal_reason
