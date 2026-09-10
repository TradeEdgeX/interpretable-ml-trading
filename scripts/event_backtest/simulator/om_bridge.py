from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from scripts.event_backtest._bootstrap import logger

try:
    from src.order_management.mock_binance_api import MockBinanceAPI
    from src.order_management.storage import Storage as OMStorage
    from src.order_management.order_manager import OrderManager
    from src.order_management.position_manager import PositionManager
    from src.order_management.models import (
        PositionSide as OMPositionSide,
        OrderSide as OMOrderSide,
        OrderType as OMOrderType,
    )

    OM_AVAILABLE = True
except ImportError:
    OM_AVAILABLE = False


class OMBridge:
    """将回测交易写入 order_management SQLite DB。

    创建时初始化 MockBinanceAPI + Storage + OrderManager + PositionManager。
    PositionSimulator 在开仓/平仓时调用 record_open / record_close。
    """

    def __init__(self, db_path: str):
        if not OM_AVAILABLE:
            raise RuntimeError(
                "order_management 模块不可用, 请检查 src/order_management"
            )
        self.db_path = db_path
        self.mock_api = MockBinanceAPI()
        self.storage = OMStorage(db_path)
        self.order_manager = OrderManager(self.storage, self.mock_api)
        self.position_manager = PositionManager(self.storage, self.mock_api)
        # pid → om_position_id 映射
        self._pid_map: Dict[str, str] = {}
        logger.info(f"OMBridge initialized: {db_path}")

    def record_open(
        self,
        pid: str,
        symbol: str,
        side: str,
        entry_price: float,
        size: float,
        atr: float,
        stop_loss: Optional[float],
        take_profit: Optional[float],
        archetype: str,
        entry_time: datetime,
    ) -> None:
        """开仓时写入 DB (order + position)。"""
        try:
            self.mock_api.set_price(symbol, entry_price)
            om_side = (
                OMPositionSide.LONG if side in ("LONG", "BUY") else OMPositionSide.SHORT
            )
            order_side = (
                OMOrderSide.BUY if side in ("LONG", "BUY") else OMOrderSide.SELL
            )

            # 下单
            order = self.order_manager.place_order(
                symbol=symbol,
                side=order_side,
                order_type=OMOrderType.MARKET,
                quantity=size,
                price=entry_price,
            )

            # 创建仓位
            position = self.position_manager.create_position(
                symbol=symbol,
                side=om_side,
                entry_price=entry_price,
                size=size,
                stop_loss_price=stop_loss,
                take_profit_price=take_profit,
                strategy_id=archetype,
                archetype=archetype,
                notes=f"backtest|atr={atr:.4f}",
            )
            self._pid_map[pid] = position.position_id
        except Exception as e:
            logger.warning(f"OMBridge.record_open failed: {e}")

    def record_close(
        self,
        pid: str,
        exit_price: float,
        exit_time: datetime,
        exit_reason: str,
        pnl_r: float,
    ) -> None:
        """平仓时写入 DB (close position + exit order)。"""
        om_pid = self._pid_map.get(pid)
        if not om_pid:
            return
        try:
            pos = self.position_manager.get_position(om_pid)
            if not pos or pos.status.value == "closed":
                return
            self.mock_api.set_price(pos.symbol, exit_price)
            self.position_manager.close_position(
                position_id=om_pid,
                price=exit_price,
                reason=f"{exit_reason}|pnl_r={pnl_r:.3f}",
            )
        except Exception as e:
            logger.warning(f"OMBridge.record_close failed: {e}")
