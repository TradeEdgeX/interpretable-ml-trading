"""Repo path + logger for event_backtest."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

logger = logging.getLogger("event_backtest")
