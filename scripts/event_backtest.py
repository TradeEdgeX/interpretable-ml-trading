#!/usr/bin/env python3
"""Compatibility CLI — delegates to ``scripts.event_backtest`` package."""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from scripts.event_backtest.engine import main

if __name__ == "__main__":
    sys.exit(main())
