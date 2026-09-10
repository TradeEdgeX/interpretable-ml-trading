"""Stub: event_backtest uses StorageManager only when --data-path is omitted."""

from __future__ import annotations

from typing import Any


class StorageManager:
    def __init__(self, root: str) -> None:
        self.root = root

    def load_ohlcv(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("pass --data-path; live StorageManager is not in this extract")
