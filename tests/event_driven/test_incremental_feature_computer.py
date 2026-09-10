"""
IncrementalFeatureComputer 单元测试
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import deque

from src.time_series_model.live.incremental_feature_computer import (
    IncrementalFeatureComputer,
)


class TestIncrementalFeatureComputer:
    """IncrementalFeatureComputer 单元测试"""

    def test_init(self):
        """测试初始化"""
        computer = IncrementalFeatureComputer(
            tick_window_minutes=30,
            bar_window_size=1000,
            vpin_bucket_volume=1000.0,
            vpin_n_buckets=50,
        )

        assert computer.tick_window_minutes == 30
        assert computer.bar_window_size == 1000
        assert computer.vpin_bucket_volume == 1000.0
        assert computer.vpin_n_buckets == 50
        assert len(computer.tick_buffer) == 0
        assert len(computer.bar_buffer) == 0
        assert len(computer.vpin_buckets) == 0

    def test_on_tick_dict(self):
        """测试处理字典格式的 tick"""
        computer = IncrementalFeatureComputer(
            tick_window_minutes=30,
            vpin_bucket_volume=100.0,
        )

        tick = {
            "ts": int(datetime.now().timestamp() * 1e9),
            "price": 50000.0,
            "volume": 1.0,
            "side": 1,  # BUY
        }

        computer.on_tick(tick)

        assert len(computer.tick_buffer) == 1
        assert computer.tick_buffer[0]["price"] == 50000.0
        assert computer.tick_buffer[0]["volume"] == 1.0
        assert computer.tick_buffer[0]["side"] == 1

    def test_on_tick_nautilus(self):
        """测试处理 Nautilus TradeTick（如果可用）"""
        computer = IncrementalFeatureComputer(
            tick_window_minutes=30,
            vpin_bucket_volume=100.0,
        )

        try:
            from nautilus_trader.model import TradeTick
            from nautilus_trader.model.enums import AggressorSide
            from nautilus_trader.model.identifiers import TradeId, InstrumentId
            from nautilus_trader.model.objects import Price, Quantity

            # 创建模拟的 TradeTick
            instrument_id = InstrumentId.from_str("BTCUSDT-PERP.BINANCE")
            trade_id = TradeId("12345")
            ts_event = int(datetime.now().timestamp() * 1e9)

            tick = TradeTick(
                instrument_id=instrument_id,
                price=Price.from_str("50000.00"),
                size=Quantity.from_str("1.0"),
                aggressor_side=AggressorSide.BUYER,
                trade_id=trade_id,
                ts_event=ts_event,
                ts_init=ts_event,
            )

            computer.on_tick(tick)

            assert len(computer.tick_buffer) == 1
            assert computer.tick_buffer[0]["price"] == 50000.0
            assert computer.tick_buffer[0]["side"] == 1

        except ImportError:
            pytest.skip("Nautilus Trader not available")
        except (TypeError, AttributeError) as e:
            # Nautilus Trader API 可能不同，跳过测试
            pytest.skip(f"Nautilus Trader API not compatible: {e}")

    def test_on_bar_dict(self):
        """测试处理字典格式的 bar"""
        computer = IncrementalFeatureComputer(
            bar_window_size=100,
        )

        bar = {
            "ts": int(datetime.now().timestamp() * 1e9),
            "open": 50000.0,
            "high": 51000.0,
            "low": 49000.0,
            "close": 50500.0,
            "volume": 100.0,
        }

        computer.on_bar(bar, timeframe="1H")

        assert len(computer.bar_buffer) == 1
        assert computer.bar_buffer[0]["close"] == 50500.0
        assert "1H" in computer.timeframe_features

    def test_vpin_calculation(self):
        """测试 VPIN 计算"""
        computer = IncrementalFeatureComputer(
            tick_window_minutes=30,
            vpin_bucket_volume=100.0,  # 每个 bucket 100 个币
            vpin_n_buckets=10,
        )

        base_ts = int(datetime.now().timestamp() * 1e9)

        # 添加买入 tick（填满一个 bucket）
        for i in range(100):
            tick = {
                "ts": base_ts + i * 1_000_000_000,  # 每秒一个
                "price": 50000.0,
                "volume": 1.0,
                "side": 1,  # BUY
            }
            computer.on_tick(tick)

        # 应该有一个 VPIN bucket
        assert len(computer.vpin_buckets) >= 1

        # VPIN 值应该在 0-1 之间
        if len(computer.vpin_buckets) > 0:
            vpin_value = computer.vpin_buckets[0][1]
            assert 0.0 <= vpin_value <= 1.0

    def test_vpin_usd_mode(self):
        """测试 VPIN USD 模式"""
        computer = IncrementalFeatureComputer(
            tick_window_minutes=30,
            vpin_bucket_volume_usd=100000.0,  # $100k per bucket
            vpin_n_buckets=10,
        )

        base_ts = int(datetime.now().timestamp() * 1e9)

        # 添加 tick（每个 tick $1000，需要 100 个 tick 填满一个 bucket）
        for i in range(100):
            tick = {
                "ts": base_ts + i * 1_000_000_000,
                "price": 50000.0,
                "volume": 0.02,  # 0.02 * 50000 = $1000 per tick
                "side": 1 if i % 2 == 0 else -1,  # 交替买卖
            }
            computer.on_tick(tick)

        # 应该有至少一个 VPIN bucket
        assert len(computer.vpin_buckets) >= 1

    def test_orderflow_features(self):
        """测试订单流特征计算"""
        computer = IncrementalFeatureComputer(
            tick_window_minutes=15,
        )

        base_ts = int(datetime.now().timestamp() * 1e9)

        # 添加买入和卖出 tick
        for i in range(50):
            tick = {
                "ts": base_ts + i * 1_000_000_000,
                "price": 50000.0,
                "volume": 1.0,
                "side": 1 if i < 30 else -1,  # 前 30 个买入，后 20 个卖出
            }
            computer.on_tick(tick)

        # 更新订单流特征
        computer._update_orderflow_features()

        # 检查特征
        assert "orderflow_imbalance" in computer.current_features
        assert "orderflow_total_vol" in computer.current_features

        # 不平衡度应该是正的（买入更多）
        imbalance = computer.current_features["orderflow_imbalance"]
        assert imbalance > 0

    def test_timeframe_features(self):
        """测试时间框架特征计算"""
        computer = IncrementalFeatureComputer(
            bar_window_size=100,
            live_feature_plan_path="/dev/null",  # 跳过 feature plan 避免耗时计算
        )

        base_ts = int(datetime.now().timestamp() * 1e9)

        # 添加多个 bar（用于计算技术指标）
        closes = [50000 + i * 100 for i in range(20)]  # 上升趋势

        for i, close in enumerate(closes):
            bar = {
                "ts": base_ts + i * 3600 * 1_000_000_000,  # 每小时一个
                "open": close - 50,
                "high": close + 50,
                "low": close - 100,
                "close": close,
                "volume": 100.0,
            }
            computer.on_bar(bar, timeframe="1H")

        # 检查时间框架特征
        assert "1H" in computer.timeframe_features
        features = computer.get_features()
        # IFC 将 OHLCV 存储为不带前缀的 key（通过 _want 过滤）
        assert "close" in features
        assert "volume" in features

    def test_get_features(self):
        """测试获取所有特征"""
        computer = IncrementalFeatureComputer(
            tick_window_minutes=30,
            bar_window_size=100,
            live_feature_plan_path="/dev/null",  # 跳过 feature plan
        )

        # 添加一些数据
        base_ts = int(datetime.now().timestamp() * 1e9)

        tick = {
            "ts": base_ts,
            "price": 50000.0,
            "volume": 1.0,
            "side": 1,
        }
        computer.on_tick(tick)

        # 需要至少 2 个 bar 才能触发 _update_timeframe_features 完整流程
        for i in range(3):
            bar = {
                "ts": base_ts + i * 3600 * 1_000_000_000,
                "open": 50000.0 + i * 10,
                "high": 51000.0 + i * 10,
                "low": 49000.0 + i * 10,
                "close": 50500.0 + i * 10,
                "volume": 100.0,
            }
            computer.on_bar(bar, timeframe="1H")

        # 获取特征
        features = computer.get_features()

        assert isinstance(features, dict)
        # timeframe_features 应该有 OHLCV 数据
        assert "1H" in computer.timeframe_features
        assert "close" in computer.timeframe_features["1H"]

    def test_get_orderflow_features(self):
        """测试获取订单流特征（指定时间窗口）"""
        computer = IncrementalFeatureComputer(
            tick_window_minutes=30,
        )

        base_ts = int(datetime.now().timestamp() * 1e9)

        # 添加最近 10 分钟的 tick
        for i in range(100):
            tick = {
                "ts": base_ts - (10 - i) * 60 * 1_000_000_000,  # 最近 10 分钟
                "price": 50000.0,
                "volume": 1.0,
                "side": 1 if i < 60 else -1,
            }
            computer.on_tick(tick)

        # 获取最近 5 分钟的订单流特征
        flow_features = computer.get_orderflow_features(window_minutes=5)

        assert "vpin" in flow_features
        assert "imbalance" in flow_features
        assert "total_vol" in flow_features
        assert isinstance(flow_features["vpin"], float)
        assert isinstance(flow_features["imbalance"], float)
        assert isinstance(flow_features["total_vol"], float)

    def test_reset(self):
        """测试重置状态"""
        computer = IncrementalFeatureComputer(
            tick_window_minutes=30,
            bar_window_size=100,
        )

        # 添加一些数据
        base_ts = int(datetime.now().timestamp() * 1e9)

        tick = {
            "ts": base_ts,
            "price": 50000.0,
            "volume": 1.0,
            "side": 1,
        }
        computer.on_tick(tick)

        bar = {
            "ts": base_ts,
            "open": 50000.0,
            "high": 51000.0,
            "low": 49000.0,
            "close": 50500.0,
            "volume": 100.0,
        }
        computer.on_bar(bar, timeframe="1H")

        # 重置
        computer.reset()

        assert len(computer.tick_buffer) == 0
        assert len(computer.bar_buffer) == 0
        assert len(computer.vpin_buckets) == 0
        assert len(computer.current_features) == 0
        assert len(computer.timeframe_features) == 0

    def test_tick_buffer_maxlen(self):
        """测试 tick 缓冲区最大长度"""
        computer = IncrementalFeatureComputer(
            tick_window_minutes=30,
        )

        base_ts = int(datetime.now().timestamp() * 1e9)

        # 添加超过 maxlen 的 tick
        for i in range(25000):  # 超过 maxlen=20000
            tick = {
                "ts": base_ts + i * 1_000_000_000,
                "price": 50000.0,
                "volume": 1.0,
                "side": 1,
            }
            computer.on_tick(tick)

        # 缓冲区应该不超过 maxlen
        assert len(computer.tick_buffer) <= 20000

    def test_bar_buffer_maxlen(self):
        """测试 bar 缓冲区最大长度"""
        computer = IncrementalFeatureComputer(
            bar_window_size=100,
            live_feature_plan_path="/dev/null",  # 跳过 feature plan 避免耗时计算
        )

        base_ts = int(datetime.now().timestamp() * 1e9)

        # 添加超过 maxlen 的 bar
        for i in range(150):  # 超过 maxlen=100
            bar = {
                "ts": base_ts + i * 3600 * 1_000_000_000,
                "open": 50000.0,
                "high": 51000.0,
                "low": 49000.0,
                "close": 50500.0,
                "volume": 100.0,
            }
            computer.on_bar(bar, timeframe="1H")

        # 缓冲区应该不超过 maxlen
        assert len(computer.bar_buffer) <= 100
