"""Hyperliquid订单流数据获取和分析模块.

基于Hyperliquid的链上订单流数据，提供：
1. 实时L3订单簿数据获取
2. 逐笔成交数据分析
3. 清算事件监控
4. 巨鲸地址追踪
5. 订单流不平衡分析
"""

import asyncio
import websockets
import json
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
import time
import sqlite3
from dataclasses import dataclass
from collections import defaultdict, deque
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OrderBookUpdate:
    """订单簿更新数据结构."""

    coin: str
    side: str  # 'buy' or 'sell'
    price: float
    size: float
    timestamp: int
    user: str  # 钱包地址
    order_id: str


@dataclass
class Trade:
    """成交数据结构."""

    coin: str
    price: float
    size: float
    timestamp: int
    maker: str
    taker: str
    side: str  # 'buy' or 'sell'


@dataclass
class Liquidation:
    """清算数据结构."""

    coin: str
    user: str
    price: float
    size: float
    timestamp: int
    pnl: float
    leverage: float


class HyperliquidDataCollector:
    """Hyperliquid数据收集器."""

    def __init__(self, db_path: str = "hyperliquid_data.db"):
        """
        初始化数据收集器.

        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        self.websocket = None
        self.is_running = False

        # 订阅管理：记录已订阅的流，用于重连后重新订阅
        self._subscriptions = {
            "order_book": [],  # [(coin, ...), ...]
            "trades": [],  # [(coin, ...), ...]
            "liquidations": [],  # [(coin, ...), ...]
        }

        # 数据缓存
        self.order_book = defaultdict(lambda: {"buy": {}, "sell": {}})
        self.recent_trades = deque(maxlen=10000)
        self.recent_liquidations = deque(maxlen=1000)

        # 巨鲸地址库
        self.whale_addresses = set()
        self.hlp_addresses = set()  # Hyperliquid做市商地址

        # 初始化数据库
        self._init_database()

    def _init_database(self):
        """初始化数据库表."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 订单簿更新表
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS order_book_updates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                coin TEXT,
                side TEXT,
                price REAL,
                size REAL,
                timestamp INTEGER,
                user TEXT,
                order_id TEXT
            )
        """
        )

        # 成交表
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                coin TEXT,
                price REAL,
                size REAL,
                timestamp INTEGER,
                maker TEXT,
                taker TEXT,
                side TEXT
            )
        """
        )

        # 清算表
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS liquidations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                coin TEXT,
                user TEXT,
                price REAL,
                size REAL,
                timestamp INTEGER,
                pnl REAL,
                leverage REAL
            )
        """
        )

        # 巨鲸地址表
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS whale_addresses (
                address TEXT PRIMARY KEY,
                label TEXT,
                total_volume REAL,
                last_seen INTEGER
            )
        """
        )

        conn.commit()
        conn.close()

    async def connect_websocket(self):
        """连接Hyperliquid WebSocket."""
        uri = "wss://api.hyperliquid.xyz/ws"

        try:
            self.websocket = await websockets.connect(uri)
            logger.info("✅ 已连接到Hyperliquid WebSocket")

            # 重连后重新订阅所有之前的订阅
            await self._resubscribe_all()

            return True
        except Exception as e:
            logger.error(f"❌ WebSocket连接失败: {e}")
            return False

    async def _resubscribe_all(self):
        """重新订阅所有之前的订阅（用于重连后）"""
        if not self.websocket:
            return

        # 重新订阅订单簿
        for coin_tuple in self._subscriptions["order_book"]:
            coin = coin_tuple[0]
            subscribe_msg = {
                "method": "subscribe",
                "subscription": {"type": "l3Book", "coin": coin},
            }
            try:
                await self.websocket.send(json.dumps(subscribe_msg))
                logger.info(f"🔄 重新订阅 {coin} L3订单簿")
            except Exception as e:
                logger.error(f"重新订阅订单簿失败 {coin}: {e}")

        # 重新订阅成交
        for coin_tuple in self._subscriptions["trades"]:
            coin = coin_tuple[0]
            subscribe_msg = {
                "method": "subscribe",
                "subscription": {"type": "trades", "coin": coin},
            }
            try:
                await self.websocket.send(json.dumps(subscribe_msg))
                logger.info(f"🔄 重新订阅 {coin} 逐笔成交")
            except Exception as e:
                logger.error(f"重新订阅成交失败 {coin}: {e}")

        # 重新订阅清算
        for coin_tuple in self._subscriptions["liquidations"]:
            coin = coin_tuple[0]
            subscribe_msg = {
                "method": "subscribe",
                "subscription": {"type": "clearings", "coin": coin},
            }
            try:
                await self.websocket.send(json.dumps(subscribe_msg))
                logger.info(f"🔄 重新订阅 {coin} 清算事件")
            except Exception as e:
                logger.error(f"重新订阅清算失败 {coin}: {e}")

    async def subscribe_order_book(self, coin: str = "BTC"):
        """订阅L3订单簿数据."""
        if not self.websocket:
            await self.connect_websocket()

        subscribe_msg = {
            "method": "subscribe",
            "subscription": {"type": "l3Book", "coin": coin},
        }

        await self.websocket.send(json.dumps(subscribe_msg))

        # 记录订阅
        if (coin,) not in self._subscriptions["order_book"]:
            self._subscriptions["order_book"].append((coin,))

        logger.info(f"📡 已订阅 {coin} L3订单簿")

    async def subscribe_trades(self, coin: str = "BTC"):
        """订阅逐笔成交数据."""
        if not self.websocket:
            await self.connect_websocket()

        subscribe_msg = {
            "method": "subscribe",
            "subscription": {"type": "trades", "coin": coin},
        }

        await self.websocket.send(json.dumps(subscribe_msg))

        # 记录订阅
        if (coin,) not in self._subscriptions["trades"]:
            self._subscriptions["trades"].append((coin,))

        logger.info(f"📡 已订阅 {coin} 逐笔成交")

    async def subscribe_liquidations(self, coin: str = "BTC"):
        """订阅清算事件."""
        if not self.websocket:
            await self.connect_websocket()

        subscribe_msg = {
            "method": "subscribe",
            "subscription": {"type": "clearings", "coin": coin},
        }

        await self.websocket.send(json.dumps(subscribe_msg))

        # 记录订阅
        if (coin,) not in self._subscriptions["liquidations"]:
            self._subscriptions["liquidations"].append((coin,))

        logger.info(f"📡 已订阅 {coin} 清算事件")

    async def listen_data(self):
        """监听WebSocket数据."""
        self.is_running = True

        while self.is_running:
            try:
                message = await self.websocket.recv()
                data = json.loads(message)

                # 处理订单簿更新
                if data.get("channel") == "l3Book":
                    await self._handle_order_book_update(data)

                # 处理成交数据
                elif data.get("channel") == "trades":
                    await self._handle_trade(data)

                # 处理清算事件
                elif data.get("channel") == "clearings":
                    await self._handle_liquidation(data)

            except websockets.exceptions.ConnectionClosed:
                logger.warning("⚠️ WebSocket连接断开，尝试重连...")
                # 重连会自动调用_resubscribe_all来重新订阅
                await self.connect_websocket()
            except Exception as e:
                logger.error(f"❌ 数据处理错误: {e}")
                # 如果是连接错误，尝试重连
                if "connection" in str(e).lower() or "closed" in str(e).lower():
                    logger.warning("⚠️ 检测到连接错误，尝试重连...")
                    await self.connect_websocket()

    async def _handle_order_book_update(self, data: dict):
        """处理订单簿更新."""
        try:
            update_data = data.get("data", {})

            # 解析订单簿更新
            for side in ["buy", "sell"]:
                for price, size in update_data.get(side, {}).items():
                    if size > 0:  # 新增或修改订单
                        update = OrderBookUpdate(
                            coin=update_data.get("coin", ""),
                            side=side,
                            price=float(price),
                            size=float(size),
                            timestamp=int(time.time() * 1000),
                            user=update_data.get("user", ""),
                            order_id=update_data.get("orderId", ""),
                        )

                        # 更新内存中的订单簿
                        self.order_book[update.coin][side][price] = size

                        # 保存到数据库
                        await self._save_order_book_update(update)

                        # 检查是否为巨鲸地址
                        if update.user in self.whale_addresses:
                            logger.info(
                                f"🐋 巨鲸订单更新: {update.user} {side} {price}@{size}"
                            )

        except Exception as e:
            logger.error(f"❌ 订单簿更新处理错误: {e}")

    async def _handle_trade(self, data: dict):
        """处理成交数据."""
        try:
            trade_data = data.get("data", {})

            trade = Trade(
                coin=trade_data.get("coin", ""),
                price=float(trade_data.get("price", 0)),
                size=float(trade_data.get("size", 0)),
                timestamp=int(time.time() * 1000),
                maker=trade_data.get("maker", ""),
                taker=trade_data.get("taker", ""),
                side=trade_data.get("side", ""),
            )

            # 添加到缓存
            self.recent_trades.append(trade)

            # 保存到数据库
            await self._save_trade(trade)

            # 检查巨鲸交易
            if (
                trade.maker in self.whale_addresses
                or trade.taker in self.whale_addresses
            ):
                logger.info(
                    f"🐋 巨鲸交易: {trade.maker} -> {trade.taker} {trade.side} {trade.price}@{trade.size}"
                )

        except Exception as e:
            logger.error(f"❌ 成交数据处理错误: {e}")

    async def _handle_liquidation(self, data: dict):
        """处理清算事件."""
        try:
            liq_data = data.get("data", {})

            liquidation = Liquidation(
                coin=liq_data.get("coin", ""),
                user=liq_data.get("user", ""),
                price=float(liq_data.get("price", 0)),
                size=float(liq_data.get("size", 0)),
                timestamp=int(time.time() * 1000),
                pnl=float(liq_data.get("pnl", 0)),
                leverage=float(liq_data.get("leverage", 0)),
            )

            # 添加到缓存
            self.recent_liquidations.append(liquidation)

            # 保存到数据库
            await self._save_liquidation(liquidation)

            logger.warning(
                f"💥 清算事件: {liquidation.user} {liquidation.coin} {liquidation.price}@{liquidation.size} PnL:{liquidation.pnl}"
            )

        except Exception as e:
            logger.error(f"❌ 清算事件处理错误: {e}")

    async def _save_order_book_update(self, update: OrderBookUpdate):
        """保存订单簿更新到数据库."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO order_book_updates 
            (coin, side, price, size, timestamp, user, order_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                update.coin,
                update.side,
                update.price,
                update.size,
                update.timestamp,
                update.user,
                update.order_id,
            ),
        )

        conn.commit()
        conn.close()

    async def _save_trade(self, trade: Trade):
        """保存成交数据到数据库."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO trades 
            (coin, price, size, timestamp, maker, taker, side)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                trade.coin,
                trade.price,
                trade.size,
                trade.timestamp,
                trade.maker,
                trade.taker,
                trade.side,
            ),
        )

        conn.commit()
        conn.close()

    async def _save_liquidation(self, liquidation: Liquidation):
        """保存清算数据到数据库."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO liquidations 
            (coin, user, price, size, timestamp, pnl, leverage)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                liquidation.coin,
                liquidation.user,
                liquidation.price,
                liquidation.size,
                liquidation.timestamp,
                liquidation.pnl,
                liquidation.leverage,
            ),
        )

        conn.commit()
        conn.close()

    def add_whale_address(self, address: str, label: str = ""):
        """添加巨鲸地址."""
        self.whale_addresses.add(address)

        # 保存到数据库
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO whale_addresses 
            (address, label, last_seen) VALUES (?, ?, ?)
        """,
            (address, label, int(time.time() * 1000)),
        )

        conn.commit()
        conn.close()

        logger.info(f"🐋 已添加巨鲸地址: {address} ({label})")

    def get_historical_data(self, coin: str, start_time: int, end_time: int) -> Dict:
        """获取历史数据."""
        conn = sqlite3.connect(self.db_path)

        # 获取历史成交
        trades_df = pd.read_sql_query(
            """
            SELECT * FROM trades 
            WHERE coin = ? AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp
        """,
            conn,
            params=(coin, start_time, end_time),
        )

        # 获取历史清算
        liquidations_df = pd.read_sql_query(
            """
            SELECT * FROM liquidations 
            WHERE coin = ? AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp
        """,
            conn,
            params=(coin, start_time, end_time),
        )

        # 获取历史订单簿更新
        order_book_df = pd.read_sql_query(
            """
            SELECT * FROM order_book_updates 
            WHERE coin = ? AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp
        """,
            conn,
            params=(coin, start_time, end_time),
        )

        conn.close()

        return {
            "trades": trades_df,
            "liquidations": liquidations_df,
            "order_book": order_book_df,
        }

    async def start_collection(self, coins: List[str] = ["BTC", "ETH"]):
        """开始数据收集."""
        logger.info("🚀 开始Hyperliquid数据收集...")

        # 连接WebSocket
        if not await self.connect_websocket():
            return False

        # 订阅数据
        for coin in coins:
            await self.subscribe_order_book(coin)
            await self.subscribe_trades(coin)
            await self.subscribe_liquidations(coin)

        # 开始监听
        await self.listen_data()

    async def stop_collection(self):
        """停止数据收集."""
        self.is_running = False
        if self.websocket:
            await self.websocket.close()
        logger.info("🛑 数据收集已停止")


class OrderFlowAnalyzer:
    """订单流分析器."""

    def __init__(self, data_collector: HyperliquidDataCollector):
        """
        初始化订单流分析器.

        Args:
            data_collector: 数据收集器实例
        """
        self.data_collector = data_collector
        self.analysis_results = {}

    def calculate_order_flow_imbalance(
        self, coin: str, window_minutes: int = 5
    ) -> Dict:
        """计算订单流不平衡."""
        # 获取最近的数据
        end_time = int(time.time() * 1000)
        start_time = end_time - window_minutes * 60 * 1000

        data = self.data_collector.get_historical_data(coin, start_time, end_time)
        trades_df = data["trades"]

        if trades_df.empty:
            return {"imbalance": 0, "buy_volume": 0, "sell_volume": 0}

        # 计算买卖不平衡
        buy_trades = trades_df[trades_df["side"] == "buy"]
        sell_trades = trades_df[trades_df["side"] == "sell"]

        buy_volume = buy_trades["size"].sum()
        sell_volume = sell_trades["size"].sum()
        total_volume = buy_volume + sell_volume

        if total_volume == 0:
            imbalance = 0
        else:
            imbalance = (buy_volume - sell_volume) / total_volume

        return {
            "imbalance": imbalance,
            "buy_volume": buy_volume,
            "sell_volume": sell_volume,
            "total_volume": total_volume,
        }

    def detect_whale_activity(self, coin: str, window_minutes: int = 10) -> Dict:
        """检测巨鲸活动."""
        end_time = int(time.time() * 1000)
        start_time = end_time - window_minutes * 60 * 1000

        data = self.data_collector.get_historical_data(coin, start_time, end_time)
        trades_df = data["trades"]

        if trades_df.empty:
            return {"whale_trades": 0, "whale_volume": 0, "whale_addresses": []}

        # 识别巨鲸交易（大额交易）
        large_trades = trades_df[trades_df["size"] > trades_df["size"].quantile(0.95)]

        # 统计巨鲸活动
        whale_trades = len(large_trades)
        whale_volume = large_trades["size"].sum()
        whale_addresses = set(
            large_trades["maker"].tolist() + large_trades["taker"].tolist()
        )

        return {
            "whale_trades": whale_trades,
            "whale_volume": whale_volume,
            "whale_addresses": list(whale_addresses),
            "large_trades": large_trades,
        }

    def analyze_liquidation_clusters(self, coin: str, window_minutes: int = 30) -> Dict:
        """分析清算集群."""
        end_time = int(time.time() * 1000)
        start_time = end_time - window_minutes * 60 * 1000

        data = self.data_collector.get_historical_data(coin, start_time, end_time)
        liquidations_df = data["liquidations"]

        if liquidations_df.empty:
            return {"liquidation_count": 0, "total_pnl": 0, "clusters": []}

        # 按时间窗口分组清算
        liquidations_df["time_window"] = liquidations_df["timestamp"] // (
            5 * 60 * 1000
        )  # 5分钟窗口

        clusters = []
        for window, group in liquidations_df.groupby("time_window"):
            if len(group) >= 3:  # 至少3个清算事件
                clusters.append(
                    {
                        "time_window": window,
                        "count": len(group),
                        "total_pnl": group["pnl"].sum(),
                        "avg_leverage": group["leverage"].mean(),
                    }
                )

        return {
            "liquidation_count": len(liquidations_df),
            "total_pnl": liquidations_df["pnl"].sum(),
            "clusters": clusters,
            "recent_liquidations": liquidations_df.tail(10),
        }

    def generate_trading_signals(self, coin: str) -> Dict:
        """生成交易信号."""
        # 订单流不平衡
        flow_imbalance = self.calculate_order_flow_imbalance(coin)

        # 巨鲸活动
        whale_activity = self.detect_whale_activity(coin)

        # 清算集群
        liquidation_clusters = self.analyze_liquidation_clusters(coin)

        # 生成信号
        signals = {
            "timestamp": int(time.time() * 1000),
            "coin": coin,
            "order_flow_imbalance": flow_imbalance["imbalance"],
            "whale_activity_score": whale_activity["whale_trades"],
            "liquidation_pressure": liquidation_clusters["liquidation_count"],
            "signal_strength": 0,
            "signal_direction": "neutral",
        }

        # 计算信号强度
        signal_strength = 0

        # 订单流不平衡信号
        if abs(flow_imbalance["imbalance"]) > 0.3:
            signal_strength += abs(flow_imbalance["imbalance"]) * 2

        # 巨鲸活动信号
        if whale_activity["whale_trades"] > 5:
            signal_strength += min(whale_activity["whale_trades"] / 10, 1)

        # 清算压力信号
        if liquidation_clusters["liquidation_count"] > 10:
            signal_strength += min(liquidation_clusters["liquidation_count"] / 20, 1)

        signals["signal_strength"] = min(signal_strength, 1)

        # 确定信号方向
        if flow_imbalance["imbalance"] > 0.2 and whale_activity["whale_trades"] > 3:
            signals["signal_direction"] = "bullish"
        elif (
            flow_imbalance["imbalance"] < -0.2
            and liquidation_clusters["liquidation_count"] > 5
        ):
            signals["signal_direction"] = "bearish"

        return signals


async def main():
    """主函数 - 演示Hyperliquid数据收集和分析."""
    # 创建数据收集器
    collector = HyperliquidDataCollector()

    # 添加一些巨鲸地址（示例）
    collector.add_whale_address("0x1234...", "巨鲸1")
    collector.add_whale_address("0x5678...", "巨鲸2")

    # 创建分析器
    analyzer = OrderFlowAnalyzer(collector)

    # 开始数据收集
    try:
        await collector.start_collection(["BTC", "ETH"])
    except KeyboardInterrupt:
        logger.info("🛑 用户中断，停止数据收集")
    finally:
        await collector.stop_collection()


if __name__ == "__main__":
    asyncio.run(main())
