"""
Backward-compatible import path for BinanceTickAggregator.

The implementation lives in `src/live_data_stream/binance_tick_aggregator.py`,
but some tests and older code import it from `src.data_tools`.
"""

from __future__ import annotations

from src.live_data_stream.binance_tick_aggregator import BinanceTickAggregator

__all__ = ["BinanceTickAggregator"]
