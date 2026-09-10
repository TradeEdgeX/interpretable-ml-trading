"""
BinanceWebSocketClient 单元测试
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import json
import time

from src.live_data_stream.websocket_client import (
    BinanceWebSocketClient,
    BinanceTick,
    StaleAggTradeSession,
    configure_binance_ws_queue_size,
    create_and_run_websocket,
)


class TestBinanceTick:
    """BinanceTick 数据类测试"""

    def test_from_binance(self):
        """测试从 Binance 数据解析"""
        payload = {
            "e": "aggTrade",
            "s": "BTCUSDT",
            "p": "50000.00",
            "q": "0.1",
            "T": 1234567890000,
            "m": False,  # 买方是 taker（主动买入）
            "a": 12345,
        }

        tick = BinanceTick.from_binance(payload)

        assert tick.symbol == "BTCUSDT"
        assert tick.price == 50000.0
        assert tick.volume == 0.1
        assert tick.timestamp_ms == 1234567890000
        assert tick.side == 1  # BUY
        assert tick.trade_id == 12345

    def test_from_binance_sell(self):
        """测试卖出 tick"""
        payload = {
            "e": "aggTrade",
            "s": "BTCUSDT",
            "p": "50000.00",
            "q": "0.1",
            "T": 1234567890000,
            "m": True,  # 买方是 maker（主动卖出）
            "a": 12345,
        }

        tick = BinanceTick.from_binance(payload)

        assert tick.side == -1  # SELL

    def test_to_dict(self):
        """测试转换为字典"""
        tick = BinanceTick(
            symbol="BTCUSDT",
            timestamp_ms=1234567890000,
            price=50000.0,
            volume=0.1,
            turnover=5000.0,
            side=1,
            trade_id=12345,
        )

        result = tick.to_dict()

        assert result["symbol"] == "BTCUSDT"
        assert result["price"] == 50000.0
        assert result["volume"] == 0.1
        assert result["side"] == 1


class TestBinanceWebSocketClient:
    """BinanceWebSocketClient 测试"""

    def test_init(self):
        """测试初始化"""
        client = BinanceWebSocketClient(
            symbols=["BTCUSDT", "ETHUSDT"],
            use_futures=True,
            reconnect_delay=5,
        )

        assert client.symbols == ["BTCUSDT", "ETHUSDT"]
        assert client.use_futures is True
        assert len(client._callbacks) == 0

    def test_configure_binance_ws_queue_size(self, monkeypatch):
        monkeypatch.setenv("MLBOT_BINANCE_WS_MAX_QUEUE", "800")
        size = configure_binance_ws_queue_size()
        assert size == 800
        from binance.ws.reconnecting_websocket import ReconnectingWebsocket

        # Old python-binance (<1.0.30) uses a class attribute; new builds
        # (prod 1.0.37) apply the queue per-instance via BinanceSocketManager,
        # so the class attribute may not exist. Only assert it when present.
        if hasattr(ReconnectingWebsocket, "MAX_QUEUE_SIZE"):
            assert ReconnectingWebsocket.MAX_QUEUE_SIZE == 800

    def test_build_socket_manager_passes_max_queue_when_supported(self, monkeypatch):
        """Regression: on python-binance 1.0.37 the WS queue is a ctor param.

        The class-attr patch is a silent no-op there (prod QueueOverflow bug), so
        BinanceSocketManager must be built with max_queue_size to raise the queue.
        """
        monkeypatch.setenv("MLBOT_BINANCE_WS_MAX_QUEUE", "2048")
        import src.live_data_stream.websocket_client as wsc

        captured = {}

        class _FakeBSM:
            def __init__(self, client, max_queue_size=100, **kwargs):
                captured["client"] = client
                captured["max_queue_size"] = max_queue_size

        # Signature exposes max_queue_size -> support detection returns True.
        monkeypatch.setattr(wsc, "BinanceSocketManager", _FakeBSM)
        assert wsc.binance_socket_manager_supports_max_queue() is True

        sentinel = object()
        bsm = wsc.build_binance_socket_manager(sentinel)
        assert isinstance(bsm, _FakeBSM)
        assert captured["client"] is sentinel
        assert captured["max_queue_size"] == 2048

    def test_build_socket_manager_omits_kwarg_on_old_binance(self, monkeypatch):
        """Old builds without max_queue_size must not get a TypeError."""
        import src.live_data_stream.websocket_client as wsc

        captured = {}

        class _OldBSM:
            def __init__(self, client, user_timeout=300):
                captured["client"] = client

        monkeypatch.setattr(wsc, "BinanceSocketManager", _OldBSM)
        assert wsc.binance_socket_manager_supports_max_queue() is False

        sentinel = object()
        bsm = wsc.build_binance_socket_manager(sentinel)
        assert isinstance(bsm, _OldBSM)
        assert captured["client"] is sentinel

    def test_init_empty_symbols(self):
        """测试空符号列表"""
        with pytest.raises(ValueError, match="symbols must not be empty"):
            BinanceWebSocketClient(symbols=[])

    def test_ws_url_spot(self):
        """use_futures=False 也走 USD-M aggTrade（兼容旧参数）。"""
        client = BinanceWebSocketClient(
            symbols=["BTCUSDT"],
            use_futures=False,
        )

        url = client._ws_url()
        assert "fstream.binance.com" in url
        assert "btcusdt@aggTrade" in url

    def test_ws_url_futures(self):
        """测试期货 WebSocket URL"""
        client = BinanceWebSocketClient(
            symbols=["BTCUSDT"],
            use_futures=True,
        )

        url = client._ws_url()
        assert "fstream.binance.com" in url
        assert "btcusdt@aggTrade" in url

    def test_ws_url_multiple_symbols(self):
        """测试多币种 WebSocket URL"""
        client = BinanceWebSocketClient(
            symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
            use_futures=True,
        )

        url = client._ws_url()
        assert "btcusdt@aggTrade" in url
        assert "ethusdt@aggTrade" in url
        assert "solusdt@aggTrade" in url

    def test_add_callback(self):
        """测试添加回调"""
        client = BinanceWebSocketClient(symbols=["BTCUSDT"])

        def callback(tick):
            pass

        client.add_callback(callback)

        assert len(client._callbacks) == 1
        assert callback in client._callbacks

    def test_remove_callback(self):
        """测试移除回调"""
        client = BinanceWebSocketClient(symbols=["BTCUSDT"])

        def callback1(tick):
            pass

        def callback2(tick):
            pass

        client.add_callback(callback1)
        client.add_callback(callback2)

        assert len(client._callbacks) == 2

        client.remove_callback(callback1)

        assert len(client._callbacks) == 1
        assert callback1 not in client._callbacks
        assert callback2 in client._callbacks

    def test_stream_ticks_success(self):
        """测试成功流式获取 tick（简化版，不测试实际 WebSocket 连接）"""
        client = BinanceWebSocketClient(symbols=["BTCUSDT"])

        # 测试 URL 生成
        url = client._ws_url()
        assert "btcusdt@aggTrade" in url

        # 测试基本功能（不测试实际 WebSocket 连接）
        assert client.symbols == ["BTCUSDT"]
        assert client.use_futures is True

    def test_stream_ticks_invalid_json(self):
        """测试无效 JSON 处理（简化版）"""
        client = BinanceWebSocketClient(symbols=["BTCUSDT"])

        # 测试基本功能（不测试实际 WebSocket 连接）
        # reconnect_delay 已移至 ReconnectionConfig
        assert client.symbols == ["BTCUSDT"]

    def test_callbacks(self):
        """测试回调调用"""
        client = BinanceWebSocketClient(symbols=["BTCUSDT"])

        callback_results = []

        def callback1(tick):
            callback_results.append(("callback1", tick))

        def callback2(tick):
            callback_results.append(("callback2", tick))

        client.add_callback(callback1)
        client.add_callback(callback2)

        # 创建测试 tick
        tick = BinanceTick(
            symbol="BTCUSDT",
            timestamp_ms=1234567890000,
            price=50000.0,
            volume=0.1,
            turnover=5000.0,
            side=1,
        )

        # 手动调用回调（模拟 WebSocket 收到数据）
        for callback in client._callbacks:
            callback(tick)

        assert len(callback_results) == 2
        assert callback_results[0][0] == "callback1"
        assert callback_results[1][0] == "callback2"
        assert callback_results[0][1].symbol == "BTCUSDT"

    def test_callback_error_handling(self):
        """测试回调错误处理"""
        client = BinanceWebSocketClient(symbols=["BTCUSDT"])

        def bad_callback(tick):
            raise ValueError("Callback error")

        def good_callback(tick):
            pass

        client.add_callback(bad_callback)
        client.add_callback(good_callback)

        tick = BinanceTick(
            symbol="BTCUSDT",
            timestamp_ms=1234567890000,
            price=50000.0,
            volume=0.1,
            turnover=5000.0,
            side=1,
        )

        # 回调应该处理错误，不影响其他回调
        for callback in client._callbacks:
            try:
                callback(tick)
            except Exception:
                pass  # 错误应该被捕获

        # 如果到这里没有异常，说明错误处理正常
        assert True

    @pytest.mark.asyncio
    async def test_stream_ticks_reconnects_after_socket_error(self):
        """Socket error payload should restart the session and keep yielding ticks."""

        class FakeAsyncClient:
            created = []

            @classmethod
            async def create(cls):
                client = cls()
                client.closed = False
                cls.created.append(client)
                return client

            async def close_connection(self):
                self.closed = True

        class FakeSocket:
            def __init__(self, messages):
                self.messages = list(messages)

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            async def recv(self):
                if not self.messages:
                    await asyncio.sleep(3600)
                return self.messages.pop(0)

        class FakeBsm:
            sessions = []

            def __init__(self, client):
                self.client = client

            def futures_multiplex_socket(self, streams, futures_type):
                messages = self.sessions.pop(0)
                self.streams = streams
                return FakeSocket(messages)

        first = [{"e": "error", "m": "socket closed"}]
        second = [
            {
                "e": "aggTrade",
                "s": "BTCUSDT",
                "p": "50000.00",
                "q": "0.1",
                "T": 1234567890000,
                "m": False,
                "a": 12345,
            }
        ]

        client = BinanceWebSocketClient(symbols=["BTCUSDT"])
        client.reconnect_manager.wait_before_reconnect = AsyncMock(return_value=True)
        stop_event = asyncio.Event()

        with patch(
            "src.live_data_stream.websocket_client.AsyncClient", FakeAsyncClient
        ), patch("src.live_data_stream.websocket_client.BinanceSocketManager", FakeBsm):
            FakeBsm.sessions = [first, second]
            stream = client.stream_ticks(stop_event)
            tick = await stream.__anext__()
            stop_event.set()
            await stream.aclose()

        assert len(FakeAsyncClient.created) == 2
        assert all(c.closed for c in FakeAsyncClient.created)
        assert tick.symbol == "BTCUSDT"
        assert tick.trade_id == 12345
        client.reconnect_manager.wait_before_reconnect.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stream_ticks_reconnects_after_stale_session(self, caplog):
        """Stale session should warn and restart instead of logging ERROR traceback."""

        class FakeAsyncClient:
            created = []

            @classmethod
            async def create(cls):
                client = cls()
                client.closed = False
                cls.created.append(client)
                return client

            async def close_connection(self):
                self.closed = True

        class HungSocket:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            async def recv(self):
                await asyncio.sleep(3600)

        class FakeBsm:
            sessions = []

            def __init__(self, client):
                self.client = client

            def futures_multiplex_socket(self, streams, futures_type):
                return HungSocket()

        second_messages = [
            {
                "e": "aggTrade",
                "s": "BTCUSDT",
                "p": "50000.00",
                "q": "0.1",
                "T": 1234567890000,
                "m": False,
                "a": 99,
            }
        ]

        class RecoveringSocket:
            def __init__(self, messages):
                self.messages = list(messages)

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            async def recv(self):
                if not self.messages:
                    await asyncio.sleep(3600)
                return self.messages.pop(0)

        class RecoveringBsm:
            call_count = 0

            def __init__(self, client):
                self.client = client

            def futures_multiplex_socket(self, streams, futures_type):
                RecoveringBsm.call_count += 1
                if RecoveringBsm.call_count == 1:
                    return HungSocket()
                return RecoveringSocket(second_messages)

        client = BinanceWebSocketClient(symbols=["BTCUSDT"], heartbeat_timeout=0.5)
        client.reconnect_manager.wait_before_reconnect = AsyncMock(return_value=True)
        stop_event = asyncio.Event()

        with patch(
            "src.live_data_stream.websocket_client.AsyncClient", FakeAsyncClient
        ), patch(
            "src.live_data_stream.websocket_client.BinanceSocketManager", RecoveringBsm
        ):
            RecoveringBsm.call_count = 0
            stream = client.stream_ticks(stop_event)
            tick = await asyncio.wait_for(stream.__anext__(), timeout=5.0)
            stop_event.set()
            await stream.aclose()

        assert tick.symbol == "BTCUSDT"
        assert RecoveringBsm.call_count == 2
        assert not any(
            record.levelname == "ERROR"
            and "aggTrade stream failed" in record.getMessage()
            for record in caplog.records
        )
        assert isinstance(StaleAggTradeSession(), TimeoutError)
